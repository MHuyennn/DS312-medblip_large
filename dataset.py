
import cv2
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MultiLabelBinarizer

class ImgCaptionConceptDataset(Dataset):
    """Dataset hỗ trợ đồng thời caption prediction và concept detection."""
    def __init__(self, 
                 df, 
                 path,
                 processor=None,
                 image_size=(224, 224),
                 max_length=100):
        self.df = df
        self.image_size = image_size 
        self.max_length = max_length
        self.processor = processor
        self.path = path
        self.ids = list(self.df["ID"])

        self.df["CUI_list"] = self.df["CUIs"]

        # Dùng MultiLabelBinarizer để chuyển thành vector multi-hot
        self.mlb = MultiLabelBinarizer()
        self.label_matrix = self.mlb.fit_transform(self.df["CUI_list"])

        # Lưu lại tất cả CUI để dùng sau này nếu cần
        self.classes_ = self.mlb.classes_

    def get_caption_by_id(self, idx):
        return self.df["Caption"].iloc[idx]

    def get_image_by_id(self, idx):
        iid = self.df["ID"].iloc[idx]
        img_path = os.path.join(self.path, str(iid) + ".jpg")
        image = cv2.imread(img_path)
        return image
    
    def get_multilabel_vector(self, idx):
        return torch.tensor(self.label_matrix[idx], dtype=torch.float)

    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, idx):
        image = self.get_image_by_id(idx)
        image = cv2.resize(image, self.image_size)
        caption = self.get_caption_by_id(idx)

        # Tokenize input image + caption
        encoding = self.processor(
            images=image,
            text=caption,
            padding="max_length",
            truncation="longest_first",
            max_length=self.max_length,
            return_tensors="pt"
        )
        encoding = {k: v.squeeze() for k, v in encoding.items()}

        # Thêm nhãn concept (multi-label)
        encoding["labels_concept"] = self.get_multilabel_vector(idx)

        return encoding
