import torch

from torch.utils.data import DataLoader, TensorDataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dummy dataset
inputs = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device)
targets = torch.tensor([[1.0], [0.0]], device=device)

dataset = TensorDataset(inputs, targets)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# Iterate through the DataLoader
for batch_inputs, batch_targets in dataloader:
    print(batch_inputs, batch_targets)
