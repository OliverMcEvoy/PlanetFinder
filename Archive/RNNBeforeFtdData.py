import h5py
import numpy as np
import torch
import os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch.nn.functional as F

class ExoplanetDataset(Dataset):
    '''
    Dataset class for the exoplanet data stored in an HDF5 file.
    The dataset contains light curves with simulated exoplanet transits.
    Each item in the dataset is a tuple containing the flux values and the periods of the transits.
    '''
    def __init__(self, hdf5_path, max_planets=10, max_period=50):
        self.hdf5_path = hdf5_path
        self.max_period = max_period
        self.max_planets = max_planets
        with h5py.File(hdf5_path, 'r') as f:
            self.keys = [f"{iteration}/{system}" for iteration in f.keys() for system in f[iteration].keys()]

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        length_of_data = np.arange(0, 1600, 1)
        
        with h5py.File(self.hdf5_path, 'r') as f:
            data = f[key]
            detected_count = data['num_detectable_planets'][()]
            time = np.array(data['time'])
            flux_with_noise = np.array(data['flux_with_noise'])
            periods = [data[f'planets/planet_{i}/period'][()] for i in range(detected_count)]

        # Normalize periods to range [0, 1]
        periods = torch.tensor(periods, dtype=torch.float32)
        periods = periods.sort(descending=True).values
        periods = periods / self.max_period
        periods = F.pad(periods, pad=(0,int(self.max_planets - detected_count)))

        # Initialize normalized flux array
        flux_with_noise = torch.tensor(flux_with_noise, dtype=torch.float32)
        full_flux = torch.ones(len(length_of_data), dtype=torch.float32)
        time_to_index = {t: i for i, t in enumerate(length_of_data)}
        for t, f in zip(time, flux_with_noise):
            if t in time_to_index:
                full_flux[time_to_index[t]] = f

        # Normalize flux to range [0, 1]
        min_flux = full_flux.min()
        max_flux = full_flux.max()
        if max_flux - min_flux > 0:  # Avoid division by zero
            full_flux = (full_flux - min_flux) / (max_flux - min_flux)

        full_flux = full_flux.unsqueeze(1) # Input size of 1

        # Normalize detected_count to range [0, 1]
        detected_count = torch.tensor(detected_count, dtype=torch.float32)
        detected_count = detected_count / self.max_planets

        return full_flux, periods, detected_count

def collate_fn(batch):
    fluxes, periods, detected_counts = zip(*batch)
    fluxes = pad_sequence(fluxes, padding_value=0, batch_first=True)
    periods = torch.stack(periods)
    detected_counts = torch.stack(detected_counts)
    return fluxes, periods, detected_counts


class LinearDropout(nn.Module):
    def __init__(self, input_size, output_size, p=0.3):
        super().__init__()
        self.norm = nn.BatchNorm1d(input_size)
        self.linear = nn.Linear(input_size, output_size)
        if p < 1:
            self.dropout = nn.Dropout(p)
        else:
            self.dropout = nn.Identity()

    def forward(self, x):
        #x = self.norm(x)
        x = self.dropout(x)
        x = self.linear(x)
        return x

class TransitModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=1024, output_size=10, num_layers=3):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers=num_layers, dropout=0, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(0)

    def forward(self, x):
        rnn_out, hidden = self.rnn(x)
        logits = self.fc(rnn_out[:,-1])
        periods = self.sigmoid(logits)
        conf = self.softmax(logits)
        return periods, conf
    

def split_dataset(hdf5_path, dataset_size=1.0, train_size=0.8, batch_size=16, max_period=50, max_planets=10, seed=42):
    dataset = ExoplanetDataset(hdf5_path, max_planets=max_planets, max_period=max_period)
    print("Number of Light Curves", len(dataset))
    if dataset_size < 1.0:
        subset_size = int(len(dataset) * dataset_size)
        dataset = Subset(dataset, range(subset_size))

    if train_size < 1.0:
        train_size = int(len(dataset) * train_size)
        train_set, test_set = random_split(
            dataset, 
            [train_size, (len(dataset) - train_size)],
            generator=torch.Generator().manual_seed(seed)
        )
 
    print(len(train_set), len(test_set))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4, pin_memory=True)

    return train_loader, test_loader

def train_model(model, train_loader, test_loader, device=torch.device("cuda"), epochs=10, patience=5):
    loss_fn = torch.nn.MSELoss(reduction="mean")
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    model.train()

    train_losses = []
    batch_losses = []
    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        train_loss = 0
        with tqdm(train_loader) as tqdm_loader:
            for flux, periods, detected_counts in tqdm_loader:
                tqdm_loader.set_description(f"Train Epoch {epoch + 1}/{epochs}")
                flux, periods, detected_counts = flux.to(device), periods.to(device), detected_counts.to(device)
                optimizer.zero_grad()
                pred_periods, _ = model(flux)
                
                print(pred_periods[0])
                print(periods[0])
                loss = loss_fn(pred_periods, periods)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                batch_losses.append(loss.item())
                tqdm_loader.set_postfix(loss=loss.item())

        train_loss = train_loss / len(train_loader)
        train_losses.append(train_loss)

        test_loss = 0
        with tqdm(test_loader) as tqdm_loader:
            with torch.no_grad():
                for flux, periods, detected_counts in tqdm_loader:
                    tqdm_loader.set_description(f"Test Epoch {epoch + 1}/{epochs}")
                    flux, periods, detected_counts = flux.to(device), periods.to(device), detected_counts.to(device)
                    pred_periods, _ = model(flux)
                    
                    loss = loss_fn(pred_periods, periods)
                    test_loss += loss.item()
                    tqdm_loader.set_postfix(loss=loss.item())

        test_loss = test_loss / len(test_loader)
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.5f}, Test Loss: {test_loss:.5f}")

        # Early stopping
        if test_loss < best_loss:
            #save_model(model, f"{epoch + 1:04d}.pth")
            best_loss = test_loss
            patience_counter = 0

        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

        #plot_losses(epochs, train_losses, batch_losses)


def plot_losses(epochs, losses, batch_losses):
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
def main(
        hdf5_path='LightCurves.hdf5', 
        dataset_size=1.0,
        train_size=0.8,
        batch_size=32,
        max_period=200,
        max_planets=10,
        epochs=100,
        patience=5,
    ):

    torch.backends.cudnn.enabled = False
    print("cuDNN enabled:", torch.backends.cudnn.enabled)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == 'cuda':
        torch.cuda.empty_cache()

    model = TransitModel(output_size=max_planets, hidden_size=512, num_layers=1).to(device)
    train_loader, test_loader = split_dataset(
        hdf5_path, 
        dataset_size=dataset_size, 
        train_size=train_size,
        batch_size=batch_size,
        max_period=max_period,
        max_planets=max_planets
    )
    train_model(model, train_loader, test_loader, device=device, epochs=epochs, patience=patience)
    #save_model(model, "Models/RnnModel_Adjusted.pth")

# Save Model
def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

# Load Model
def load_model(path, device):
    checkpoint = torch.load(path, map_location=device)
    model = TransitModel().to(device)
    model.load_state_dict(checkpoint)
    model.eval()
    return model

if __name__ == '__main__':
    os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
    hdf5_path = 'LightCurves.hdf5'
    main(hdf5_path=hdf5_path, dataset_size=1.0, max_period=1600, max_planets=8, batch_size=8, epochs=1000, patience=100)
