import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class ImgCaptionConceptDataset(Dataset):
    """Dataset dùng cho concept detection."""

    def __init__(self, dataframe, img_dir, name_list, mlb, image_size=(224, 224), mode="train", transform=None):
        self.df = dataframe
        self.img_dir = img_dir
        self.name_list = name_list
        self.mlb = mlb
        self.image_size = image_size
        self.mode = mode

        # Nếu transform được cung cấp, sử dụng nó
        if transform is not None:
            self.transform = transform
        else:
            if self.mode == "train":
                self.transform = transforms.Compose([
                    transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(15),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                    transforms.ToTensor(),
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize(image_size),
                    transforms.ToTensor(),
                ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_path = os.path.join(self.img_dir, f"{row['ID']}.jpg")
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        encoding = {
            "pixel_values": image,
            "id": row["ID"]
        }

        if self.mode != "test":
            concept_names = row["Concept_Names"] if isinstance(row["Concept_Names"], list) else []
            concept_vector = self.mlb.transform([concept_names])[0]
            encoding["labels_concept"] = torch.tensor(concept_vector, dtype=torch.float)

        return encoding
