import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torch.nn.utils.rnn import pad_sequence
from scipy.signal import medfilt
from tqdm import tqdm
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Dataset Definition
class ExoplanetDataset(Dataset):
    def __init__(self, hdf5_path):
        self.hdf5_path = hdf5_path
        with h5py.File(hdf5_path, 'r') as f:
            self.keys = [
                f"{iteration}/{system}"
                for iteration in f.keys()
                for system in f[iteration].keys()
            ]

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        length_of_data = np.arange(0, 1600, 0.02043357)
        with h5py.File(self.hdf5_path, 'r') as f:
            data = f[key]
            time = np.array(data['time'])
            flux_with_noise = np.array(data['flux_with_noise'])
            detected_count = data['num_detectable_planets'][()]
            periods = [
                data[f'planets/planet_{i}/period'][()]
                for i in range(detected_count)
            ]

        full_flux_with_noise = np.ones(len(length_of_data), dtype=np.float32)
        time_to_index = {t: i for i, t in enumerate(length_of_data)}

        # Insert `flux_with_noise` at corresponding indices in `full_flux_with_noise`
        for t, flux in zip(time, flux_with_noise):
            if t in time_to_index:
                full_flux_with_noise[time_to_index[t]] = flux

        # Apply median filter
        filtered_flux = medfilt(full_flux_with_noise, kernel_size=5)

        # Normalize flux
        full_flux_with_noise /= filtered_flux

        return (
            torch.tensor(flux_with_noise, dtype=torch.float32),
            torch.tensor(periods, dtype=torch.float32),
        )

# Collate Function
def collate_fn(batch):
    fluxes, periods = zip(*batch)
    fluxes_padded = pad_sequence(fluxes, batch_first=True, padding_value=0)
    periods_padded = pad_sequence(periods, batch_first=True, padding_value=0)
    return fluxes_padded, periods_padded

class TransitModel(nn.Module):
    def __init__(self):
        super(TransitModel, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.pool = nn.MaxPool1d(2)
        self.fc1 = nn.Linear(64 * 39151, 128)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(128, 10)  # Predict up to 10 periods
        self.fc3 = nn.Linear(128, 1)  # Predict the number of planets

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = self.pool(x)
        x = x.flatten(start_dim=1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        
        periods = self.fc2(x)  # Predict up to 10 periods
        num_planets = torch.sigmoid(self.fc3(x)) * 10  # Predict the number of planets (scaled to 0-10)
        
        # Mask periods based on predicted number of planets
        max_num_planets = periods.size(1)
        predicted_planet_count = num_planets.round().long().clamp(0, max_num_planets)  # Clamp to valid range
        
        masked_periods = [
            periods[i, :predicted_planet_count[i, 0]]  # Only include up to predicted count
            for i in range(periods.size(0))
        ]
        
        # Pad periods to match batch size for consistency
        masked_periods_padded = pad_sequence(masked_periods, batch_first=True, padding_value=0)
        
        return masked_periods_padded, num_planets


def masked_mse_loss(predicted_periods, target_periods, predicted_num_planets, target_num_planets):
    batch_size = target_periods.size(0)
    period_loss = 0
    valid_entries = 0  # Keep track of valid entries to avoid division by zero

    for i in range(batch_size):
        num_predicted = predicted_periods[i].size(0)
        num_target = int(target_num_planets[i].item())
        
        # Skip if both predicted and target periods are empty
        if num_predicted == 0 or num_target == 0:
            continue
        
        # Calculate loss for valid predictions
        valid_periods = min(num_predicted, num_target)  # Compare only valid entries
        period_loss += ((predicted_periods[i, :valid_periods] - target_periods[i, :valid_periods]) ** 2).mean()
        valid_entries += 1

    # Average period loss only if there are valid entries
    if valid_entries > 0:
        period_loss /= valid_entries
    else:
        period_loss = torch.tensor(0.0, device=predicted_periods.device)

    # Calculate the loss for the number of planets
    num_planets_loss = ((predicted_num_planets - target_num_planets) ** 2).mean()
    return period_loss + num_planets_loss


def train_model(model, hdf5_path, device, epochs=10, batch_size=16, patience=5, data_percentage=1.0):
    dataset = ExoplanetDataset(hdf5_path)
    if data_percentage < 1.0:
        subset_size = int(len(dataset) * data_percentage)
        dataset = Subset(dataset, range(subset_size))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)  
    model.train()


    losses = []
    batch_losses = []
    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        total_loss = 0
        with tqdm(dataloader, unit="batch") as tepoch:
            for flux, periods in tepoch:
                tepoch.set_description(f"Epoch {epoch + 1}/{epochs}")
                flux, periods = flux.to(device), periods.to(device)
                optimizer.zero_grad()
                pred_periods, pred_num_planets = model(flux)
                target_num_planets = (periods > 0).sum(dim=1, keepdim=True).float()
                loss = masked_mse_loss(pred_periods, periods, pred_num_planets, target_num_planets)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                batch_losses.append(loss.item())
                tepoch.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(dataloader)
        losses.append(avg_loss)

        print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss:.4f}")

        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            save_model(model, "best_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

    # Plot loss
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(0, epochs), losses, label='Average Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss per Epoch')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(0, len(batch_losses)), batch_losses, label='Loss per Batch')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.title('Training Loss per Batch')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Main Function
def main(hdf5_path, data_percentage=1.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransitModel().to(device)
    train_model(model, hdf5_path=hdf5_path, device=device, epochs=1, batch_size=2, data_percentage=data_percentage)

    save_model(model, "AlignedPeriodAndPlanetOnePercentTheGoodModeltransit_model_5_percent_weight_decay.pth")

# Save Model
def save_model(model, path):
    torch.save({
        'model_state_dict': model.state_dict(),
    }, path)
    print(f"Model saved to {path}")

# Load Model
def load_model(path, device):
    checkpoint = torch.load(path, map_location=device)
    model = TransitModel().to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

if __name__ == "__main__":
    hdf5_path = "TransitFindOneForGap.hdf5"
    main(hdf5_path, data_percentage=1)