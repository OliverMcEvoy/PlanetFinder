import h5py
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torch.nn.utils.rnn import pad_sequence
from scipy.signal import medfilt
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.sparse import coo_matrix
import torch.nn.functional as F

class ExoplanetDataset(Dataset):
    def __init__(self, hdf5_path, max_planets=5):
        self.hdf5_path = hdf5_path
        self.period_max = 50
        self.max_planets = max_planets
        with h5py.File(hdf5_path, 'r') as f:
            self.keys = [f"{iteration}/{system}" for iteration in f.keys() for system in f[iteration].keys()]

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        length_of_data = np.arange(0, 1600, 0.02043357)
        
        with h5py.File(self.hdf5_path, 'r') as f:
            data = f[key]
            time = np.array(data['time'])
            flux_with_noise = np.array(data['flux_with_noise'], dtype=np.float32)
            detected_count = data['num_detectable_planets'][()]
            periods = sorted([data[f'planets/planet_{i}/period'][()] for i in range(detected_count)])


        # Initialize normalized flux array
        full_flux_with_noise = np.ones(len(length_of_data), dtype=np.float32)
        time_to_index = {t: i for i, t in enumerate(length_of_data)}
        for t, flux in zip(time, flux_with_noise):
            if t in time_to_index:
                full_flux_with_noise[time_to_index[t]] = flux

        # Normalize flux to range [0, 1]
        min_flux = full_flux_with_noise.min()
        max_flux = full_flux_with_noise.max()
        if max_flux - min_flux > 0:  # Avoid division by zero
            full_flux_with_noise = (full_flux_with_noise - min_flux) / (max_flux - min_flux)

        return (
            torch.tensor(full_flux_with_noise, dtype=torch.float32),
            torch.tensor(periods, dtype=torch.float32),
        )

def collate_fn(batch):
    fluxes, periods = zip(*batch)
    fluxes_padded = pad_sequence(fluxes, padding_value=0)
    periods_padded = pad_sequence(periods, padding_value=0)
    
    # Reshape periods to (batch_size, number_of_planets, 1)
    periods_padded = periods_padded.permute(1, 0).unsqueeze(2)
    
    return fluxes_padded, periods_padded


class BayesianDense(nn.Module):
    def __init__(self, in_features, out_features, p=0.3):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(p)

    def forward(self, x):
        return self.dropout(self.linear(x))

class TransitModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=1024, period_max=200):
        super().__init__()
        self.hidden_size = hidden_size
        self.period_max = period_max

        self.rnn = nn.RNN(input_size, hidden_size, num_layers=3, dropout=0.3)
        self.fc_period = BayesianDense(hidden_size, 5, p=0.3)  # Change output size to 5

    def forward(self, x):
        x = x.unsqueeze(2)  # [B, L, I]
        _, hidden = self.rnn(x)  # [num_layers, B, H]
        periods = self.fc_period(hidden[-1])  # [B, 5]
        periods = periods.unsqueeze(2)  # [B, 5, 1]

        return periods

    def _resolve_close_periods(self, periods, rnn_out):
        periods = torch.sort(periods)[0]
        min_distance = 0.1  # Minimum distance between periods

        for k in range(10):
            conflict = False
            for j in range(1, len(periods)):
                if periods[j] - periods[j - 1] < min_distance:
                    periods[j] = self._repredict_period(rnn_out[j])
                    conflict = True
            if not conflict:
                break
            else:
                periods = torch.sort(periods)[0]  # Re-sort periods after potential changes
        return torch.sort(periods)[0]
    
    def _repredict_period(self, rnn_out):
        # just return a random value between 0 nad 1 to punish the model for guessing periods so close togther,
        # This seems to work so much better than repredicting the period
        return torch.rand(1).item()
        # return torch.sigmoid(self.fc_period(rnn_out)).squeeze(-1)

def masked_mse_loss(predicted_periods, target_periods):
    # scale factors 
    alpha = 1
    beta = 100
    loss = 0
    batch_size ,_ ,_ = target_periods.shape

    # print(f"predicted_periods shape: {predicted_periods.shape}")
    # print(f"target_periods shape: {target_periods.shape}")

    for i in range(batch_size):
        # print(f"predicted_periods[{i}] shape: {predicted_periods[i].shape}")
        # print(f"predicted_periods[{i}]: {predicted_periods[i]}")
        # print(f"target_periods[{i}] shape: {target_periods[i].shape}")
        # print(f"target_periods[{i}]: {target_periods[i]}")

        pred = predicted_periods[i]
        target = target_periods[i]
        residual = (pred[:,None] - target[None,:]).abs()
        closest = residual.min(1).indices.unique()

        non_zero_count = (target != 0).sum().item()

        loss += alpha * (residual ** 2).sum()
        loss += torch.tensor(beta * (non_zero_count - len(closest)) ** 2)

    return loss

def train_model(model, hdf5_path, device, epochs=10, batch_size=16, patience=5, data_percentage=1.0, period_max=50):
    dataset = ExoplanetDataset(hdf5_path)
    if data_percentage < 1.0:
        subset_size = int(len(dataset) * data_percentage)
        dataset = Subset(dataset, range(subset_size))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4, pin_memory=True)
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
                flux, periods = flux.to(device), periods.to(device),
                optimizer.zero_grad()
                pred_periods = model(flux)
                
                loss = masked_mse_loss(pred_periods, periods)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                batch_losses.append(loss.item())
                tepoch.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(dataloader)
        losses.append(avg_loss)

        print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss:.4f}")

        save_model(model, f"{epoch}best_model_stuf.pth")
        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(0, epochs), losses,color='indigo')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('RNN Training Loss per Epoch')
    plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
    plt.gca().xaxis.set_minor_locator(ticker.AutoMinorLocator())
    plt.gca().yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
    plt.gca().yaxis.set_minor_locator(ticker.AutoMinorLocator())
    plt.gca().tick_params(axis='both', which='both', direction='in')

    plt.subplot(1, 2, 2)
    plt.plot(range(0, len(batch_losses)), batch_losses, color ='navy')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.title('RNN Training Loss per Batch')
    plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
    plt.gca().xaxis.set_minor_locator(ticker.AutoMinorLocator())
    plt.gca().yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
    plt.gca().yaxis.set_minor_locator(ticker.AutoMinorLocator())
    plt.gca().tick_params(axis='both', which='both', direction='in')

    plt.tight_layout()
    plt.savefig('Images/RnnLoss.png')
    plt.show()


# Main Function
def main(hdf5_path, data_percentage=1.0, period_max=200):
    torch.backends.cudnn.enabled = False
    print("cuDNN enabled:", torch.backends.cudnn.enabled)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == 'cuda':
        torch.cuda.empty_cache()
    model = TransitModel().to(device)
    train_model(model, hdf5_path=hdf5_path, device=device, epochs=3, batch_size=3, data_percentage=data_percentage, period_max=period_max,patience=3)

    save_model(model, "Models/RnnModel_Adjusted.pth")

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
    hdf5_path = "TrainingData/improved_randomness.hdf5"
    main(hdf5_path, data_percentage=0.008, period_max=200)
