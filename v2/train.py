import torch
import os
from tqdm import tqdm
#from models import RNN as TransitModel
from models import LSTM as TransitModel
#from models import SequenceModel as TransitModel
#from models import TransitModel
from dataset import create_dataloaders
import argparse

# Save model
def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

# Load model
def load_model(path, device):
    checkpoint = torch.load(path, map_location=device)
    model = TransitModel().to(device)
    model.load_state_dict(checkpoint)
    model.eval()
    return model

def loss_function(y_pred, y_true, epoch):
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()
    #if epoch > 3:
    mask = y_true.nonzero()
    loss = ((y_pred[mask] - y_true[mask]).abs() ** 2).mean()
    #else:
        #loss = ((y_pred - y_true).abs() ** 2).mean()
    return loss

# Train model
def train_model(
        model, 
        train_loader, 
        test_loader, 
        device=torch.device("cuda"), 
        epochs=10, 
        patience=5, 
        max_period=400
    ):

    metric = torch.nn.MSELoss(reduction="mean")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-5)

    best_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        train_loss = 0
        model.train()

        with tqdm(train_loader) as tqdm_loader:
            for power, periods in tqdm_loader:
                tqdm_loader.set_description(f"Train Epoch {epoch + 1}/{epochs}")
                power, periods = power.to(device), periods.to(device)
                optimizer.zero_grad()
                pred_periods = model(power)
                
                print(pred_periods[0])
                print(periods[0])
                loss = metric(pred_periods, periods)
                #loss = loss_function(pred_periods, periods, epoch)
                loss.backward()
                optimizer.step()

                scaled_loss = max_period * loss.item()
                train_loss += scaled_loss
                tqdm_loader.set_postfix(loss=scaled_loss)

        train_loss = train_loss / len(train_loader)
        test_loss = 0
        model.eval()

        with tqdm(test_loader) as tqdm_loader:
            with torch.no_grad():
                for power, periods in tqdm_loader:
                    tqdm_loader.set_description(f"Test Epoch {epoch + 1}/{epochs}")
                    power, periods = power.to(device), periods.to(device)
                    pred_periods = model(power)
                    
                    loss = metric(pred_periods, periods)
                    scaled_loss = max_period * loss.item()
                    test_loss += scaled_loss
                    tqdm_loader.set_postfix(loss=scaled_loss)

        test_loss = test_loss / len(test_loader)
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.5f}, Test Loss: {test_loss:.5f}")

        # Early stopping
        if test_loss < best_loss:
            #save_model(model, f"checkpoints/{epoch + 1:04d}.pth")
            best_loss = test_loss
            patience_counter = 0

        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

# Main function
def main(
        dataset_size=1.0,
        train_size=0.8,
        batch_size=16,
        max_planets=8,
        max_period=400,
        epochs=100,
        patience=5,
    ):

    torch.backends.cudnn.enabled = False
    print("cuDNN enabled:", torch.backends.cudnn.enabled)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda":
        torch.cuda.empty_cache()

    model = TransitModel(
        output_size=max_planets
    ).to(device)

    train_loader, test_loader = create_dataloaders(
        dataset_size=dataset_size, 
        train_size=train_size,
        batch_size=batch_size
    )

    train_model(
        model, 
        train_loader, 
        test_loader, 
        device=device, 
        epochs=epochs, 
        patience=patience,
        max_period=max_period
    )

    #save_model(model, "checkpoints/RNN_final.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model for exoplanet detection.")
    parser.add_argument("--dataset_size", type=float, default=1.0, help="Proportion of dataset to utilise for testing and training.")
    parser.add_argument("--train_size", type=float, default=0.8, help="Proportion of dataset to utilise for training.")
    parser.add_argument("--max_planets", type=int, default=8, help="Maximum number of model output periods.")
    parser.add_argument("--max_period", type=int, default=400, help="Maximum period to normalise data.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--patience", type=int, default=10, help="Patience of early stopping during training.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size of gradient descent.")

    args = vars(parser.parse_args())
    print(args)
    main(**args)