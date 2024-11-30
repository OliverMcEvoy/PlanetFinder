import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from scipy.signal import medfilt
from torch.nn.utils.rnn import pad_sequence

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
        length_of_data = np.arange(0,1600,0.02043357);
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

def collate_fn(batch):
    fluxes, periods = zip(*batch)
    fluxes_padded = pad_sequence(fluxes, batch_first=True, padding_value=0)
    periods_padded = pad_sequence(periods, batch_first=True, padding_value=0)
    return fluxes_padded, periods_padded


class TransitModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(TransitModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, flux, max_planets):
        flux = flux.unsqueeze(-1)  # Add a channel dimension
        h0 = torch.zeros(self.num_layers, flux.size(0), self.hidden_size).to(flux.device)
        c0 = torch.zeros(self.num_layers, flux.size(0), self.hidden_size).to(flux.device)
        out, _ = self.lstm(flux, (h0, c0))
        lstm_output = out[:, -1, :]  # Take the last LSTM output
        pred_periods = torch.relu(self.fc(lstm_output))  # Ensure positive periods
        return pred_periods.repeat(1, max_planets)

def masked_mse_loss(predicted, target):
    """Compute MSE loss, ignoring zero-padded targets."""
    mask = target > 0
    return ((predicted[mask] - target[mask]) ** 2).mean()

def train_model(model, hdf5_path, device, epochs=10, batch_size=16, patience=5):
    dataset = ExoplanetDataset(hdf5_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        for flux, periods in dataloader:
            flux, periods = flux.to(device), periods.to(device)
            optimizer.zero_grad()
            max_planets = periods.size(1)
            pred_periods = model(flux, max_planets)
            loss = masked_mse_loss(pred_periods, periods)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader):.4f}")

def infer_detectable_planets(predicted_periods):
    """Infer the number of detectable planets by counting non-zero periods."""
    return (predicted_periods > 0).sum(dim=-1).int()

def main(hdf5_path):
    input_size = 1
    hidden_size = 64
    num_layers = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TransitModel(input_size, hidden_size, num_layers).to(device)
    train_model(model, hdf5_path=hdf5_path, device=device, epochs=10, batch_size=32)

    # Save the model
    save_model(model, "transit_model_period_finder.pth")

if __name__ == "__main__":
    hdf5_path = "PeriodIncluded.hdf5"
    main(hdf5_path)

def save_model(model, path):
    torch.save({
        'model_state_dict': model.state_dict(),
    }, path)
    print(f"Model saved to {path}")

def load_model(path, device):
    checkpoint = torch.load(path, map_location=device)
    input_size = 2
    hidden_size = 128
    num_layers = 2
    model = TransitModel(input_size, hidden_size, num_layers).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model