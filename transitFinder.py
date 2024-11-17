import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
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

# Load and Normalize Data in Batches
def load_data_in_batches(hdf5_path, batch_size):
    with h5py.File(hdf5_path, 'r') as f:
        inputs = []
        outputs = []
        for iteration in f.keys():
            for system in f[iteration].keys():
                data = f[f"{iteration}/{system}"]
                
                time = np.array(data['time'])
                flux_with_noise = np.array(data['flux_with_noise'])
                detected_count = data['num_detectable_planets'][()]

                # Normalize flux using magnitude
                flux_with_noise_magnitude = np.abs(flux_with_noise).max()
                flux_with_noise = flux_with_noise / flux_with_noise_magnitude

                inputs.append([flux_with_noise, time])
                outputs.append(detected_count)

                if len(inputs) == batch_size:
                    yield inputs, outputs
                    inputs, outputs = [], []

        if inputs:
            yield inputs, outputs

def collate_fn(batch):
    flux_with_noise = [item[0] for item in batch]
    time = [item[1] for item in batch]
    detected_count = [item[2] for item in batch]

    flux_with_noise_tensor = torch.stack([torch.tensor(f, dtype=torch.float32).clone().detach() for f in flux_with_noise])
    time_tensor = torch.stack([torch.tensor(t, dtype=torch.float32).clone().detach() for t in time])
    detected_count_tensor = torch.tensor(detected_count, dtype=torch.float32).clone().detach()

    return flux_with_noise_tensor, time_tensor, detected_count_tensor

# Define the Complex Bayesian Neural Network Model
class TransitModel(nn.Module):
    def __init__(self, max_flux_len, max_time_len):
        super(TransitModel, self).__init__()
        self.num_classes = 12  # 9 classes (0 to 8 planets)
        self.flux_input = nn.Linear(max_flux_len, max_flux_len)
        self.time_input = nn.Linear(max_time_len, max_time_len)
        
        self.concat = nn.Sequential(
            nn.Linear(max_flux_len + max_time_len, 256),
            nn.ReLU(),
            BayesianDropout(p=0.3)
        )
        
        self.bayesian1 = AdvancedBayesianDense(256, 128, activation=nn.ReLU())
        self.bayesian2 = AdvancedBayesianDense(128, 64, activation=nn.ReLU())
        self.bayesian3 = AdvancedBayesianDense(64, 32, activation=nn.ReLU())
        
        self.detected_count_output = nn.Linear(32, self.num_classes)  # Use num_classes

    def forward(self, flux, time, n_samples=10):
        flux_out = self.flux_input(flux)
        time_out = self.time_input(time)
        
        combined = torch.cat((flux_out, time_out), dim=-1)
        
        x = self.concat(combined)
        x = self.bayesian1(x, n_samples=n_samples)
        x = self.bayesian2(x, n_samples=n_samples)
        x = self.bayesian3(x, n_samples=n_samples)
        
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

def train_model(model, hdf5_path, device, epochs=3, batch_size=128, patience=5):
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
    criterion = nn.CrossEntropyLoss()  # Using CrossEntropyLoss for multi-class classification
    early_stopping = EarlyStopping(patience=patience)
    
    writer = SummaryWriter()  # Initialize TensorBoard writer
    
    model.train()
    
    for epoch in range(epochs):
        epoch_loss = 0
        batch_count = 0
        all_preds = []
        all_labels = []
        
        for inputs, outputs in load_data_in_batches(hdf5_path, batch_size):
            flux_with_noise, time = zip(*inputs)
            detected_count = outputs
            
            flux_with_noise_tensor = torch.tensor(np.array(flux_with_noise), dtype=torch.float32).to(device)
            time_tensor = torch.tensor(np.array(time), dtype=torch.float32).to(device)
            detected_count_tensor = torch.tensor(detected_count, dtype=torch.long).to(device)  # Use long for class indices

            dataset = TensorDataset(flux_with_noise_tensor, time_tensor, detected_count_tensor)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
            
            for flux, time, detected_count in dataloader:
                flux, time, detected_count = flux.to(device), time.to(device), detected_count.to(device)
                
                # Check if detected_count is within the valid range
                if detected_count.min() < 0 or detected_count.max() >= model.num_classes:
                    raise ValueError(f"Invalid label detected: {detected_count}")
                
                optimizer.zero_grad()
                
                probabilities = model(flux, time, n_samples=5)
                
                loss = criterion(probabilities, detected_count.long())  # Ensure detected_count is long
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
        print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {epoch_loss:.4f}, F1 Score: {f1:.4f}")
        print()
        
        # Log loss and F1 score to TensorBoard
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
    max_flux_len = 120*20
    max_time_len = 120*20
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
    max_len = 120*20

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print("CUDA is available. Using GPU.")
    else:
        print("CUDA is not available. Using CPU.")
    
    torch.cuda.empty_cache()

    model = TransitModel(max_len, max_len).to(device)
    train_model(model, hdf5_path, device)
    
    # Save the trained model
    save_model(model, "10_epoch_model_SNR_3.pth")

    # Example prediction for a new flux and time
    flux_example = np.random.rand(max_len)  # Replace with actual data
    time_example = np.arange(max_len)      # Replace with actual time data
    planet_count, probabilities = predict_planets(model, flux_example, time_example, device)
    
    print(f"Detected planet count: {planet_count}")
    print(f"Probabilities for 0-8 planets: {probabilities}")
    
# Run the main function
if __name__ == "__main__":
    main("no_exo_module_planet_systems_SNR_3.hdf5")