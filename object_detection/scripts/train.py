import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.utils.data
from utils.dataset import CustomDataset, get_transform
from models.model import get_object_detection_model

def train():
    dataset = CustomDataset(root='data/images', annotation_file='data/annotations/annotations.json', transforms=get_transform(train=True))
    dataset_test = CustomDataset(root='data/images', annotation_file='data/annotations/annotations.json', transforms=get_transform(train=False))

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=4,
        collate_fn=lambda x: tuple(zip(*x)))

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=lambda x: tuple(zip(*x)))

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = get_object_detection_model(num_classes=2)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        for images, targets in data_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

        lr_scheduler.step()
        print(f'Epoch #{epoch} Loss: {losses.item()}')

        torch.save(model.state_dict(), os.path.join('checkpoints', f'model_{epoch}.pth'))

if __name__ == "__main__":
    train()
