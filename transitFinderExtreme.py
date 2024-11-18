import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.metrics import f1_score
from torch.utils.tensorboard import SummaryWriter  # Import TensorBoard

# Bayesian Dropout Layer
class BayesianDropout(nn.Module):
    def __init__(self, p=0.1):
        super(BayesianDropout, self).__init__()
        self.p = p

    def forward(self, x):
        return F.dropout(x, p=self.p, training=True)

# Advanced Bayesian Dense Layer
class AdvancedBayesianDense(nn.Module):
    def __init__(self, in_features, out_features, activation=None):
        super(AdvancedBayesianDense, self).__init__()
        self.weight_mean = nn.Parameter(torch.randn(out_features, in_features))
        self.weight_log_std = nn.Parameter(torch.randn(out_features, in_features))
        self.bias_mean = nn.Parameter(torch.randn(out_features))
        self.bias_log_std = nn.Parameter(torch.randn(out_features))
        self.activation = activation
        self.dropout = BayesianDropout(p=0.2)

    def forward(self, x, n_samples=5):
        weights = []
        biases = []
        for _ in range(n_samples):
            weight = self.weight_mean + torch.exp(self.weight_log_std) * torch.randn_like(self.weight_mean)
            bias = self.bias_mean + torch.exp(self.bias_log_std) * torch.randn_like(self.bias_mean)
            weights.append(weight)
            biases.append(bias)
        
        out_samples = [self.dropout(F.linear(x, weight, bias)) for weight, bias in zip(weights, biases)]
        out = torch.mean(torch.stack(out_samples), dim=0)
        
        if self.activation:
            out = self.activation(out)
        
        return out

#Lazy loading dataset
class ExoplanetDataset(Dataset):
    def __init__(self, hdf5_path):
        self.hdf5_path = hdf5_path
        with h5py.File(hdf5_path, 'r') as f:
            # Pre-load metadata.
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

            # Load data lazily
            time = np.array(data['time'])
            flux_with_noise = np.array(data['flux_with_noise'])
            detected_count = data['num_detectable_planets'][()]

            # Normalize flux using magnitude
            flux_with_noise /= np.abs(flux_with_noise).max()

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

# Define the Complex Bayesian Neural Network Model
class TransitModel(nn.Module):
    def __init__(self, max_flux_len, max_time_len):
        super(TransitModel, self).__init__()
        self.num_classes = 12  
        self.flux_input = nn.Linear(max_flux_len, 256)  # Increased units
        self.time_input = nn.Linear(max_time_len, 256)  # Increased units
        
        self.concat = nn.Sequential(
            nn.Linear(512, 256),  # Increased units
            nn.ReLU(),
            BayesianDropout(p=0.3)
        )
        
        self.bayesian1 = AdvancedBayesianDense(256, 128, activation=nn.ReLU())  # Increased units
        self.bayesian2 = AdvancedBayesianDense(128, 64, activation=nn.ReLU())
        self.bayesian3 = AdvancedBayesianDense(64, 32, activation=nn.ReLU())  # New layer
        
        self.dropout = BayesianDropout(p=0.3)  # Additional dropout layer
        
        self.detected_count_output = nn.Linear(32, self.num_classes)  # Use num_classes

    def forward(self, flux, time, n_samples=10):
        flux_out = self.flux_input(flux)
        time_out = self.time_input(time)
        
        combined = torch.cat((flux_out, time_out), dim=-1)
        
        x = self.concat(combined)
        x = self.bayesian1(x, n_samples=n_samples)
        x = self.bayesian2(x, n_samples=n_samples)
        x = self.bayesian3(x, n_samples=n_samples)  # Forward pass through new layer
        
        x = self.dropout(x)  # Apply additional dropout
        
        detected_count = self.detected_count_output(x)
        probabilities = F.softmax(detected_count, dim=-1)  # Softmax to get probabilities
        return probabilities

# Early Stopping Mechanism
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
    criterion = nn.CrossEntropyLoss()  # Multi-class classification loss
    early_stopping = EarlyStopping(patience=patience)

    writer = SummaryWriter()  # Initialize TensorBoard writer
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

            # Forward pass
            optimizer.zero_grad()
            probabilities = model(flux_with_noise, time, n_samples=5)
            loss = criterion(probabilities, detected_count)
            loss.backward()
            optimizer.step()

            # Track metrics
            epoch_loss += loss.item()
            batch_count += 1
            preds = torch.argmax(probabilities, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(detected_count.cpu().numpy())

            # Log batch loss to TensorBoard
            writer.add_scalar('Loss/batch', loss.item(), batch_count)

            print(f"Epoch {epoch + 1}/{epochs}, Batch {batch_count}, Loss: {loss.item():.4f}", end='\r')

        epoch_loss /= batch_count
        f1 = f1_score(all_labels, all_preds, average='weighted')
        print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {epoch_loss:.4f}, F1 Score: {f1:.4f}")
        print()

        # Log epoch metrics to TensorBoard
        writer.add_scalar('Loss/train', epoch_loss, epoch)
        writer.add_scalar('F1_Score/train', f1, epoch)

        if early_stopping(epoch_loss):
            print("Early stopping")
            break

    writer.close()  # Close the TensorBoard writer

# Function to save the model and input dimensions
def save_model(model, path):
    torch.save({
        'model_state_dict': model.state_dict(),
    }, path)
    print(f"Model saved to {path}")

# Function to load the model and input dimensions
def load_model(path, device):
    checkpoint = torch.load(path, map_location=device)
    max_flux_len = 26427
    max_time_len = 26427
    model = TransitModel(max_flux_len, max_time_len).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])  # Correctly load the model state dict
    model.eval()
    return model

def predict_planets(model, flux, time, device, n_samples=5):
    flux_tensor = torch.tensor(flux, dtype=torch.float32).unsqueeze(0).to(device)
    time_tensor = torch.tensor(time, dtype=torch.float32).unsqueeze(0).to(device)
    
    probabilities = model(flux_tensor, time_tensor, n_samples)
    
    planet_counts = torch.argmax(probabilities, dim=-1)  # The most probable class
    return planet_counts.item(), probabilities.squeeze().cpu().numpy()

# Main Function
def main(hdf5_path):
    max_len = 26427
    print(f"Max length: {max_len}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print("CUDA is available")

    # Initialize and train the model
    model = TransitModel(max_flux_len=max_len, max_time_len=max_len).to(device)
    train_model(model, hdf5_path=hdf5_path, device=device, epochs=10, batch_size=128, patience=10)

    # Save the trained model
    save_model(model, "theBigModel.pth")

# Call main function
if __name__ == "__main__":
    hdf5_path = "thisIsABigFile.hdf5"
    main(hdf5_path)