import os
import pandas as pd
import torch
import cv2
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class ImgCaptionConceptDataset(Dataset):
    """Dataset dùng cho đồng thời caption prediction và concept detection."""

    def __init__(self, dataframe, img_dir, processor, name_list, mlb, image_size=(224, 224), max_length=100, mode="train"):
        """
        Args:
            dataframe (pd.DataFrame): Dataframe đã merge từ caption.csv và concept.csv.
            img_dir (str): Đường dẫn chứa ảnh.
            processor: BLIP processor.
            name_list (list): Danh sách các concept Name đã map từ cui_names.csv.
            mlb: MultiLabelBinarizer fitted trên Name.
            image_size (tuple): Kích thước resize ảnh.
            max_length (int): Độ dài tối đa của caption tokenized.
            mode (str): 'train', 'valid', 'test'.
        """
        self.df = dataframe
        self.img_dir = img_dir
        self.processor = processor
        self.name_list = name_list
        self.mlb = mlb
        self.image_size = image_size
        self.max_length = max_length
        self.mode = mode

        # Định nghĩa transform cho data augmentation
        if self.mode == "train":
            self.transform = transforms.Compose([
                transforms.RandomRotation(10),  # Xoay ngẫu nhiên ±10 độ
                transforms.RandomHorizontalFlip(p=0.5),  # Lật ngang với xác suất 50%
                transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Thay đổi độ sáng, độ tương phản
                transforms.Resize(self.image_size),  # Resize ảnh
                transforms.ToTensor(),  # Chuyển thành tensor
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Chuẩn hóa
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_path = os.path.join(self.img_dir, f"{row['ID']}.jpg")
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Ảnh {img_path} không tồn tại hoặc lỗi khi đọc.")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)  # Chuyển sang PIL Image để dùng torchvision transforms

        # Áp dụng transform
        image = self.transform(image)

        pixel_values = self.processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)

        encoding = {
            "pixel_values": pixel_values,
            "id": row["ID"]
        }

        if self.mode != "test":
            caption = row["Caption"]
            input_ids = self.processor.tokenizer(
                caption,
                padding="max_length",
                max_length=self.max_length,
                truncation=True,
                return_tensors="pt"
            ).input_ids.squeeze(0)

            attention_mask = self.processor.tokenizer(
                caption,
                padding="max_length",
                max_length=self.max_length,
                truncation=True,
                return_tensors="pt"
            ).attention_mask.squeeze(0)

            labels_caption = input_ids.clone()
            labels_caption[labels_caption == self.processor.tokenizer.pad_token_id] = -100

            # Chuyển concept Name thành multi-hot vector
            concept_names = row["Concept_Names"] if isinstance(row["Concept_Names"], list) else []
            concept_vector = self.mlb.transform([concept_names])[0]

            encoding.update({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels_caption": labels_caption,
                "labels_concept": torch.tensor(concept_vector, dtype=torch.float)
            })

        return encoding
