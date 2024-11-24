import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

from torch.utils.tensorboard import SummaryWriter

# Lazy loading dataset
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
        with h5py.File(self.hdf5_path, 'r') as f:
            data = f[key]
            time = np.array(data['time'])
            flux_with_noise = np.array(data['flux_with_noise'])
            detected_count = data['num_detectable_planets'][()]

            median_flux = np.median(flux_with_noise)
            mad = np.median(np.abs(flux_with_noise - median_flux))
            flux_with_noise = (flux_with_noise - median_flux) / mad

        return (
            torch.tensor(flux_with_noise, dtype=torch.float32),
            torch.tensor(time, dtype=torch.float32),
            torch.tensor(detected_count, dtype=torch.long),
        )

# Custom collate function
def collate_fn(batch):
    flux_with_noise = [item[0] for item in batch]
    time = [item[1] for item in batch]
    detected_count = [item[2] for item in batch]

    flux_with_noise_tensor = torch.stack(flux_with_noise)
    time_tensor = torch.stack(time)
    detected_count_tensor = torch.tensor(detected_count, dtype=torch.long)

    return flux_with_noise_tensor, time_tensor, detected_count_tensor

# Define the LSTM-based Model
class TransitModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(TransitModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, flux, time):
        combined = torch.cat((flux.unsqueeze(-1), time.unsqueeze(-1)), dim=-1)
        h0 = torch.zeros(self.num_layers, combined.size(0), self.hidden_size).to(combined.device)
        c0 = torch.zeros(self.num_layers, combined.size(0), self.hidden_size).to(combined.device)
        out, _ = self.lstm(combined, (h0, c0))
        out = self.fc(out[:, -1, :])
        probabilities = F.softmax(out, dim=-1)
        return probabilities

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.counter = 0

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
            return False
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
        if self.counter >= self.patience:
            return True
        return False

def train_model(model, hdf5_path, device, epochs=3, batch_size=1, patience=5):
    dataset = ExoplanetDataset(hdf5_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    criterion = nn.CrossEntropyLoss()
    early_stopping = EarlyStopping(patience=patience)

    writer = SummaryWriter()
    model.train()

    for epoch in range(epochs):
        epoch_loss = 0
        all_preds = []
        all_labels = []
        batch_count = 0

        for flux_with_noise, time, detected_count in dataloader:
            flux_with_noise, time, detected_count = (
                flux_with_noise.to(device),
                time.to(device),
                detected_count.to(device),
            )

            optimizer.zero_grad()
            probabilities = model(flux_with_noise, time)
            loss = criterion(probabilities, detected_count)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batch_count += 1
            preds = torch.argmax(probabilities, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(detected_count.cpu().numpy())

            writer.add_scalar('Loss/batch', loss.item(), batch_count)

            print(f"Epoch {epoch + 1}/{epochs}, Batch {batch_count}, Loss: {loss.item():.4f}", end='\r')

        epoch_loss /= batch_count
        f1 = f1_score(all_labels, all_preds, average='weighted')
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='weighted')
        recall = recall_score(all_labels, all_preds, average='weighted')

        writer.add_scalar('Loss/epoch', epoch_loss, epoch)
        writer.add_scalar('F1/epoch', f1, epoch)
        writer.add_scalar('Accuracy/epoch', accuracy, epoch)
        writer.add_scalar('Precision/epoch', precision, epoch)
        writer.add_scalar('Recall/epoch', recall, epoch)

        print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {epoch_loss:.4f}, F1 Score: {f1:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
        print()

    writer.close()

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
    num_classes = 12
    model = TransitModel(input_size, hidden_size, num_layers, num_classes).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def main(hdf5_path):
    input_size = 2
    hidden_size = 128
    num_layers = 2
    num_classes = 12
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print("CUDA is available")

    model = TransitModel(input_size, hidden_size, num_layers, num_classes).to(device)
    train_model(model, hdf5_path=hdf5_path, device=device, epochs=10, batch_size=8, patience=3)
    save_model(model, "theBigModeldifferntModel.pth")

if __name__ == "__main__":
    hdf5_path = "thisIsABigFile.hdf5"
    main(hdf5_path)