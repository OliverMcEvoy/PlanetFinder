import torch
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import mean_squared_error
import numpy as np
import RNN

def evaluate_model(model, dataloader, device):
    model.eval()
    all_predicted_periods = []
    all_true_periods = []

    with torch.no_grad():
        for flux, periods, detected_counts in dataloader:
            flux, periods, detected_counts = flux.to(device), periods.to(device), detected_counts.to(device)
            pred_periods, _ = model(flux)
            
            # Convert tensor to numpy and get non-zero values
            true_period = periods[0].cpu().numpy()
            true_period = true_period[true_period != 0]
            
            # Get predicted periods as numpy array
            pred_period = pred_periods[0].cpu().numpy()
            
            # Append only if both arrays have values
            if len(true_period) > 0 and len(pred_period) > 0:
                # Pad shorter array to match length of longer array
                max_len = max(len(true_period), len(pred_period))
                true_period = np.pad(true_period, (0, max_len - len(true_period)))
                pred_period = np.pad(pred_period, (0, max_len - len(pred_period)))
                
                all_predicted_periods.append(pred_period)
                all_true_periods.append(true_period)

    return all_predicted_periods, all_true_periods

def calculate_period_metrics(predicted_periods, true_periods):
    # Flatten the lists of periods
    flat_predicted_periods = [item for sublist in predicted_periods for item in sublist]
    flat_true_periods = [item for sublist in true_periods for item in sublist]

    # Calculate Mean Squared Error (MSE)
    mse = mean_squared_error(flat_true_periods, flat_predicted_periods)

    return mse

def main(data_percentage=0.1):
    torch.backends.cudnn.enabled = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RNN.load_model("Models/RnnModel.pth", device)

    test_dataset = RNN.ExoplanetDataset("TrainingData/LightCurveTrainingData.hdf5")
    if data_percentage < 1.0:
        subset_size = int(len(test_dataset) * data_percentage)
        test_dataset = Subset(test_dataset, range(subset_size))
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=RNN.collate_fn)

    predicted_periods, true_periods = evaluate_model(model, test_dataloader, device)
    mse = calculate_period_metrics(predicted_periods, true_periods)

    print(f"Mean Squared Error (MSE) for periods: {mse:.4f}")

    # Save the MSE score to a file
    with open("mse_score.txt", "w") as f:
        f.write(f"Mean Squared Error (MSE) for periods: {mse:.4f}\n")

if __name__ == "__main__":
    main(data_percentage=0.002)