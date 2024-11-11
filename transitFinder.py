import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

# Custom Bayesian Dense Layer (simplified)
class BayesianDense(nn.Module):
    def __init__(self, in_features, out_features, activation=None):
        super(BayesianDense, self).__init__()
        self.weight_mean = nn.Parameter(torch.randn(out_features, in_features))
        self.weight_std = nn.Parameter(torch.randn(out_features, in_features))
        self.bias_mean = nn.Parameter(torch.randn(out_features))
        self.bias_std = nn.Parameter(torch.randn(out_features))
        self.activation = activation

    def forward(self, x):
        weight = self.weight_mean + self.weight_std * torch.randn_like(self.weight_mean)
        bias = self.bias_mean + self.bias_std * torch.randn_like(self.bias_mean)
        out = F.linear(x, weight, bias)
        
        if self.activation:
            out = self.activation(out)
        
        return out

# Step 1: Load and Process Data from HDF5
def load_data(hdf5_path):
    inputs = []
    outputs = []
    with h5py.File(hdf5_path, 'r') as f:
        for iteration in f.keys():
            for system in f[iteration].keys():
                data = f[f"{iteration}/{system}"]
                
                time = np.array(data['time'])
                flux_with_noise = np.array(data['flux_with_noise'])
                detected_count = data['detected_count'][()]
                
                individual_light_curves = [np.array(curve) for curve in data['individual_light_curves']]

                # Ensure exactly 8 light curves
                if len(individual_light_curves) < 5:
                    individual_light_curves += [np.zeros_like(time)] * (5 - len(individual_light_curves))
                elif len(individual_light_curves) > 5:
                    individual_light_curves = individual_light_curves[:5]

                # print(f"System: {system}, Detected Planets: {detected_count}")
                # print(f"Time Shape: {time.shape}, Flux Shape: {flux_with_noise.shape}")
                # print(f"Individual Light Curves: {[curve.shape for curve in individual_light_curves]}")
                # print()
                
                inputs.append([flux_with_noise, time])
                outputs.append((detected_count, individual_light_curves))

    return inputs, outputs

def preprocess_data(inputs, outputs):
    flux_with_noise, time = zip(*inputs)
    detected_count, individual_light_curves = zip(*outputs)

    # Pad flux and time sequences
    max_flux_len = max(len(f) for f in flux_with_noise)
    max_time_len = max(len(t) for t in time)

    flux_with_noise_padded = np.array([np.pad(f, (0, max_flux_len - len(f)), mode='constant') for f in flux_with_noise])
    time_padded = np.array([np.pad(t, (0, max_time_len - len(t)), mode='constant') for t in time])

    # Process individual light curves
    max_light_curve_length = max(len(curve) for curves in individual_light_curves for curve in curves)
    individual_light_curves_padded = []
    for curves in individual_light_curves:
        padded_curves = [np.pad(curve, (0, max_light_curve_length - len(curve)), mode='constant') for curve in curves]
        individual_light_curves_padded.append(np.array(padded_curves))

    return [flux_with_noise_padded, time_padded], [detected_count, individual_light_curves_padded]

def collate_fn(batch):
    flux_with_noise = [item[0] for item in batch]
    time = [item[1] for item in batch]
    detected_count = [item[2] for item in batch]
    individual_flux = [item[3] for item in batch]

    # Stack the lists into tensors
    flux_with_noise_tensor = torch.stack([torch.tensor(f, dtype=torch.float32).clone().detach() for f in flux_with_noise])
    time_tensor = torch.stack([torch.tensor(t, dtype=torch.float32).clone().detach() for t in time])
    detected_count_tensor = torch.tensor(detected_count, dtype=torch.float32).clone().detach()
    individual_flux_tensor = torch.stack([torch.tensor(f, dtype=torch.float32).clone().detach() for f in individual_flux])

    return flux_with_noise_tensor, time_tensor, detected_count_tensor, individual_flux_tensor



# Define Bayesian Neural Network Model
class TransitModel(nn.Module):
    def __init__(self, max_flux_len, max_time_len):
        super(TransitModel, self).__init__()
        self.flux_input = nn.Linear(max_flux_len, max_flux_len)
        self.time_input = nn.Linear(max_time_len, max_time_len)
        
        self.concat = nn.Sequential(
            nn.Linear(max_flux_len + max_time_len, 128),
            nn.ReLU()
        )
        
        self.bayesian1 = BayesianDense(128, 64, activation=nn.ReLU())
        self.bayesian2 = BayesianDense(64, 32, activation=nn.ReLU())
        
        self.detected_count_output = nn.Linear(32, 1)
        self.individual_light_curves_flux_output = nn.Linear(32, max_flux_len * 5)
        self.individual_light_curves_time_output = nn.Linear(32, max_time_len * 5)

    def forward(self, flux, time):
        flux_out = self.flux_input(flux)
        time_out = self.time_input(time)
        
        combined = torch.cat((flux_out, time_out), dim=-1)
        
        x = self.concat(combined)
        x = self.bayesian1(x)
        x = self.bayesian2(x)
        
        detected_count = self.detected_count_output(x)
        individual_flux = self.individual_light_curves_flux_output(x).view(-1, 5, flux.size(1))
        individual_time = self.individual_light_curves_time_output(x).view(-1, 5, time.size(1))
        
        return detected_count, individual_flux, individual_time

def train_model(model, x_train, y_train, device, epochs=1000, batch_size=16):
    # Convert data to tensors
    flux_with_noise_tensor = torch.tensor(np.array(x_train[0]), dtype=torch.float32).clone().detach().to(device)
    time_tensor = torch.tensor(np.array(x_train[1]), dtype=torch.float32).clone().detach().to(device)

    # Ensure y_train is properly converted to tensors
    detected_count_tensor = torch.tensor(np.array(y_train[0]), dtype=torch.float32).clone().detach().to(device)
    individual_light_curves_tensor = torch.tensor(np.array(y_train[1]), dtype=torch.float32).clone().detach().to(device)

    # Dataset and DataLoader with custom collate function
    dataset = TensorDataset(flux_with_noise_tensor, time_tensor, detected_count_tensor, individual_light_curves_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    
    optimizer = optim.Adam(model.parameters())
    criterion = nn.MSELoss()
    
    model.train()
    for epoch in range(epochs):
        for flux, time, detected_count, individual_flux in dataloader:
            flux, time, detected_count, individual_flux = flux.to(device), time.to(device), detected_count.to(device), individual_flux.to(device)
            
            optimizer.zero_grad()
            
            detected_count_pred, individual_flux_pred, _ = model(flux, time)
            
            loss = (criterion(detected_count_pred.squeeze(), detected_count.squeeze()) + 
                    criterion(individual_flux_pred, individual_flux))
            loss.backward()
            optimizer.step()
            
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

# Function to plot light curves
def plot_light_curves(detected_count, individual_light_curves_flux, individual_light_curves_time):
    num_planets = int(detected_count[0][0])
    for i in range(num_planets):
        plt.figure()
        plt.plot(individual_light_curves_time[0][i], individual_light_curves_flux[0][i], label=f'Planet {i+1}')
        plt.xlabel('Time')
        plt.ylabel('Flux')
        plt.title(f'Light Curve of Planet {i+1}')
        plt.legend()
        plt.show()

# Function to save the model and input dimensions
def save_model(model, path):
    torch.save({
        'model_state_dict': model.state_dict(),
    }, path)
    print(f"Model saved to {path}")

# Function to load the model and input dimensions
def load_model(path, device):
    checkpoint = torch.load(path, map_location=device)
    max_flux_len = 16384
    max_time_len = 16384
    model = TransitModel(max_flux_len, max_time_len).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])  # Correctly load the model state dict
    model.eval()
    return model

# Main Function to Run
def main(hdf5_path):
    max_len = 16384

    # Check for CUDA
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    # if torch.cuda.is_available():
    #     print("CUDA is available. Using GPU.")
    # else:
    #     print("CUDA is not available. Using CPU.")
    
    # Load data
    inputs, outputs = load_data(hdf5_path)
    
    # Preprocess data
    x_train, y_train = preprocess_data(inputs, outputs)
    
    # Manually set max lengths for input shapes
    max_flux_len = max_len
    max_time_len = max_len
    
    # Pad the training data to the set max lengths
    x_train[0] = np.array([np.pad(f, (0, max_flux_len - len(f)), mode='constant') for f in x_train[0]])
    x_train[1] = np.array([np.pad(t, (0, max_time_len - len(t)), mode='constant') for t in x_train[1]])
    y_train[1] = np.array([[np.pad(curve, (0, max_flux_len - len(curve)), mode='constant') for curve in curves] for curves in y_train[1]])
    
    torch.cuda.empty_cache()

    # Build model
    model = TransitModel(max_flux_len, max_time_len).to(device)
    
    # Train model
    train_model(model, x_train, y_train, device)
    
    # Save the trained model
    save_model(model, "transit_model.pth")
    
    # Example prediction and plotting
    model.eval()
    with torch.no_grad():
        detected_count, individual_light_curves_flux, individual_light_curves_time = model(
            torch.tensor(np.array(x_train[0][:1]), dtype=torch.float32).to(device), 
            torch.tensor(np.array(x_train[1][:1]), dtype=torch.float32).to(device)
        )
    
    plot_light_curves(detected_count.cpu(), individual_light_curves_flux.cpu(), individual_light_curves_time.cpu())

# Run the main function
if __name__ == "__main__":
    import os
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    main("planet_systems.hdf5")