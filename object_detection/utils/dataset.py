import os
import torch
import torchvision.transforms as T
from PIL import Image
import json

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, root, annotation_file, transforms=None):
        self.root = root
        self.transforms = transforms
        with open(annotation_file) as f:
            self.annotations = json.load(f)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.annotations[idx]['image'])
        img = Image.open(img_path).convert("RGB")
        boxes = torch.tensor(self.annotations[idx]['boxes'])
        labels = torch.tensor(self.annotations[idx]['labels'])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels

        if self.transforms:
            img = self.transforms(img)

        return img, target

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)
