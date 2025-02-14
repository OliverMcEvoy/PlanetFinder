import torch
import zarr
import os

def minmax(x):
    x_min = x.min(0, keepdim=True)[0]
    x_max = x.max(0, keepdim=True)[0]
    return (x - x_min) / (x_max - x_min)

class ExoplanetDataset(torch.utils.data.Dataset):
    def __init__(self, power_path="power", periods_path="periods"):
        self.power_path = power_path
        self.periods_path = periods_path

    def __len__(self):
        return len(os.listdir(self.power_path))

    def __getitem__(self, idx):
        power = zarr.load(f"{self.power_path}/{idx}_power.zip")
        periods = zarr.load(f"{self.periods_path}/{idx}_periods.zip")
        power = torch.from_numpy(power).to(torch.float32)
        periods = torch.from_numpy(periods).to(torch.float32)
        return power, periods[:1]
    
class TessDataset(torch.utils.data.Dataset):
    def __init__(self, path="tess2"):
        self.path = path
        self.files = os.listdir(self.path)
        self.periods = zarr.load("periods.zip")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        flux = zarr.load(os.path.join(self.path, self.files[idx]))
        periods = self.periods[idx]

        flux = torch.from_numpy(flux).to(torch.float32)
        periods = torch.from_numpy(periods).to(torch.float32)

        flux = flux[:,1].unsqueeze(1)
        periods = periods[:1]
        flux = minmax(flux)

        return flux, periods

def collate_fn(batch):
    power, periods = zip(*batch)
    power = torch.nn.utils.rnn.pad_sequence(power, padding_value=0, batch_first=True)
    periods = torch.stack(periods)
    return power, periods

def create_dataloaders(dataset_size=1.0, train_size=0.8, batch_size=16, seed=42):
    #dataset = ExoplanetDataset()
    dataset = TessDataset()

    print("Number of Light Curves", len(dataset))
    if dataset_size < 1.0:
        subset_size = int(len(dataset) * dataset_size)
        dataset = torch.utils.data.Subset(dataset, range(subset_size))

    if train_size < 1.0:
        train_size = int(len(dataset) * train_size)
        train_set, test_set = torch.utils.data.random_split(
            dataset, 
            [train_size, (len(dataset) - train_size)],
            generator=torch.Generator().manual_seed(seed)
        )
    
    print("Training Size", len(train_set), "Testing Size", len(test_set))
    train_loader = torch.utils.data.DataLoader(
        train_set, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_fn, 
        num_workers=4,
        pin_memory=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_set, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_fn, 
        num_workers=4, 
        pin_memory=True
    )

    return train_loader, test_loader