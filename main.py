import os
import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torchvision import transforms

# Tắt parallelism để tránh cảnh báo
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Bật debug CUDA
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

from dataset import ImgCaptionConceptDataset
from model import MedCSRAModel

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce_loss = nn.BCEWithLogitsLoss(reduction='none')(logits, targets)
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()

def evaluate_threshold(model, valid_loader, name_list, device, thresholds=np.arange(0.01, 0.41, 0.01)):
    """Tìm ngưỡng tối ưu cho concept prediction dựa trên F1-score trên tập valid."""
    model.eval()
    all_true_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in valid_loader:
            pixel_values = batch["pixel_values"].to(device)
            labels_concept = batch["labels_concept"].to(device)

            outputs = model(pixel_values)
            logits = outputs["logits_concept"]
            probs = torch.sigmoid(logits).cpu().numpy()

            all_true_labels.append(labels_concept.cpu().numpy())
            all_probs.append(probs)

    all_true_labels = np.concatenate(all_true_labels, axis=0)
    all_probs = np.concatenate(all_probs, axis=0)

    best_threshold = 0.3
    best_f1 = 0.0

    for threshold in thresholds:
        pred_labels = (all_probs > threshold).astype(int)
        f1 = f1_score(all_true_labels, pred_labels, average="micro")
        print(f"Threshold {threshold:.2f}: F1-score = {f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    print(f"Best threshold: {best_threshold:.2f} with F1-score = {best_f1:.4f}")
    return best_threshold, best_f1

def train(root_path, batch_size=16, num_epochs=50, lr=0.001, save_path="./model_best.pth", patience=10, min_delta=0.001, start_epoch=0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count()
    print(f"Using {num_gpus} GPUs for training: {list(range(num_gpus))}")

    if num_gpus > 1:
        device_ids = list(range(num_gpus))  # Sử dụng tất cả GPU (0 và 1)
        print(f"Assigning device IDs: {device_ids}")

    # Load data
    train_img_dir = os.path.join(root_path, "train/train")
    valid_img_dir = os.path.join(root_path, "valid/valid")
    concept_train_csv = os.path.join(train_img_dir, "train_concepts.csv")
    concept_valid_csv = os.path.join(valid_img_dir, "valid_concepts.csv")
    cui_names_csv = os.path.join(root_path, "cui_names.csv")

    df_con_train = pd.read_csv(concept_train_csv)
    df_con_valid = pd.read_csv(concept_valid_csv)
    df_cui = pd.read_csv(cui_names_csv)

    # Xử lý CUIs không hợp lệ
    df_cui = df_cui[df_cui["Name"] != "Name nicht gefunden"].drop_duplicates(subset=["Name"]).reset_index(drop=True)
    
    cui2name = dict(zip(df_cui["CUI"], df_cui["Name"]))
    name2cui = {v: k for k, v in cui2name.items()}
    df_train = df_con_train
    df_valid = df_con_valid

    df_train["Concept_Names"] = df_train["CUIs"].apply(
        lambda x: [cui2name[cui] for cui in str(x).split(";") if cui in cui2name]
    )
    df_valid["Concept_Names"] = df_valid["CUIs"].apply(
        lambda x: [cui2name[cui] for cui in str(x).split(";") if cui in cui2name]
    )

    # Lấy danh sách concept names từ tập train
    concept_names_in_train = set()
    for concept_list in df_train["Concept_Names"]:
        concept_names_in_train.update(concept_list)
    concept_names_in_train = list(concept_names_in_train)

    df_cui_train = df_cui[df_cui["Name"].isin(concept_names_in_train)].drop_duplicates(subset=["Name"]).reset_index(drop=True)
    name_list = df_cui_train["Name"].tolist()

    if len(name_list) != len(set(name_list)):
        raise ValueError(f"Duplicate names found in name_list: {[name for name in set(name_list) if name_list.count(name) > 1]}")

    # Tính num_classes dựa trên name_list
    num_classes = len(name_list)
    print(f"Number of classes: {num_classes}")

    # Khởi tạo MultiLabelBinarizer
    mlb = MultiLabelBinarizer(classes=name_list)
    mlb.fit(df_train["Concept_Names"])

    # Data augmentation và transforms
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    valid_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Datasets
    train_dataset = ImgCaptionConceptDataset(
        df_train, train_img_dir, name_list, mlb, mode="train", transform=train_transforms
    )
    valid_dataset = ImgCaptionConceptDataset(
        df_valid, valid_img_dir, name_list, mlb, mode="valid", transform=valid_transforms
    )

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Model
    model = MedCSRAModel(num_classes=num_classes, num_heads=1, lam=0.1, dropout=0.5).to(device)
    if num_gpus > 1:
        model = nn.DataParallel(model, device_ids=device_ids)
        print("Model wrapped in DataParallel for multi-GPU training with specified device IDs")

    # Kiểm tra phân phối GPU qua bộ nhớ được cấp phát
    if num_gpus > 1:
        for i in range(num_gpus):
            mem = torch.cuda.memory_allocated(i)
            print(f"Memory allocated on GPU {i}: {mem / 1024**2:.2f} MiB")

    # Loss và optimizer
    criterion_concept = FocalLoss(alpha=0.25, gamma=2.0)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    # Load checkpoint nếu có
    best_f1 = 0.0
    best_threshold = 0.3
    if os.path.exists(save_path) and start_epoch > 0:
        checkpoint = torch.load(save_path, map_location=device)
        state_dict = checkpoint["model_state_dict"]
        if list(state_dict.keys())[0].startswith("module."):
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        if num_gpus > 1:
            model.module.load_state_dict(state_dict)
        else:
            model.load_state_dict(state_dict)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        best_f1 = checkpoint["best_f1"]
        best_threshold = checkpoint["best_threshold"]
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resuming training from epoch {start_epoch} with best F1-score: {best_f1:.4f}")

    # Khởi tạo biến theo dõi
    epochs_no_improve = 0
    early_stop = False

    # Training loop
    for epoch in range(start_epoch, num_epochs):
        if early_stop:
            print(f"Early stopping triggered at epoch {epoch}")
            break

        model.train()
        running_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Training"):
            optimizer.zero_grad()

            # Loại bỏ .to(device) để DataParallel tự quản lý
            pixel_values = batch["pixel_values"]
            labels_concept = batch["labels_concept"]

            outputs = model(pixel_values)
            logits = outputs["logits_concept"]
            preds = torch.sigmoid(logits)

            loss = criterion_concept(logits, labels_concept)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()

            # Kiểm tra bộ nhớ GPU sau mỗi batch
            if num_gpus > 1:
                for i in range(num_gpus):
                    mem = torch.cuda.memory_allocated(i)
                    print(f"After batch - Memory allocated on GPU {i}: {mem / 1024**2:.2f} MiB")

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}] Loss: {avg_loss:.4f}")

        # Đánh giá trên tập valid
        model.eval()
        valid_loss = 0.0
        all_true_labels = []
        all_probs = []

        with torch.no_grad():
            for batch in valid_loader:
                pixel_values = batch["pixel_values"].to(device)
                labels_concept = batch["labels_concept"].to(device)

                outputs = model(pixel_values)
                logits = outputs["logits_concept"]
                preds = torch.sigmoid(logits)

                loss = criterion_concept(logits, labels_concept)
                valid_loss += loss.item()

                probs = preds.cpu().numpy()
                all_true_labels.append(labels_concept.cpu().numpy())
                all_probs.append(probs)

        avg_valid_loss = valid_loss / len(valid_loader)
        all_true_labels = np.concatenate(all_true_labels, axis=0)
        all_probs = np.concatenate(all_probs, axis=0)
        
        # Cập nhật ngưỡng tối ưu và F1-score
        best_threshold, valid_f1 = evaluate_threshold(model, valid_loader, name_list, device)
        valid_precision = precision_score(all_true_labels, (all_probs > best_threshold).astype(int), average="micro")
        valid_recall = recall_score(all_true_labels, (all_probs > best_threshold).astype(int), average="micro")
        print(f"Validation Loss: {avg_valid_loss:.4f}, Precision: {valid_precision:.4f}, Recall: {valid_recall:.4f}, F1-score (threshold {best_threshold:.2f}): {valid_f1:.4f}")

        # Đánh giá trên tập train
        model.eval()
        all_train_labels = []
        all_train_probs = []
        with torch.no_grad():
            for batch in train_loader:
                pixel_values = batch["pixel_values"].to(device)
                labels_concept = batch["labels_concept"].to(device)
                outputs = model(pixel_values)
                logits = outputs["logits_concept"]
                probs = torch.sigmoid(logits).cpu().numpy()
                all_train_labels.append(labels_concept.cpu().numpy())
                all_train_probs.append(probs)
        all_train_labels = np.concatenate(all_train_labels, axis=0)
        all_train_probs = np.concatenate(all_train_probs, axis=0)
        train_f1 = f1_score(all_train_labels, (all_train_probs > best_threshold).astype(int), average="micro")
        print(f"Train F1-score (threshold {best_threshold:.2f}): {train_f1:.4f}")

        # Save best model nếu F1-score cải thiện
        if valid_f1 > best_f1 + min_delta:
            best_f1 = valid_f1
            best_threshold = best_threshold
            epochs_no_improve = 0
            state_dict = model.module.state_dict() if num_gpus > 1 else model.state_dict()
            torch.save({
                "model_state_dict": state_dict,
                "optimizer_state_dict": optimizer.state_dict(),
                "best_f1": best_f1,
                "best_threshold": best_threshold,
                "epoch": epoch
            }, save_path)
            print(f"✅ Saved best model to {save_path} with F1-score: {best_f1:.4f}")
        else:
            epochs_no_improve += 1
            print(f"No improvement in F1-score. Epochs without improvement: {epochs_no_improve}/{patience}")

        # Early stopping
        if epochs_no_improve >= patience:
            early_stop = True
            print(f"Early stopping: No improvement in F1-score for {patience} epochs.")
        scheduler.step(valid_f1)

    print(f"Final best threshold: {best_threshold:.2f} with F1-score: {best_f1:.4f}")
    return best_threshold

def predict(split="test", batch_size=8, model_path="./model_best.pth", threshold=0.3):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # Chỉ dùng GPU 0
    num_gpus = torch.cuda.device_count()
    print(f"Detected {num_gpus} GPUs, but using only 1 GPU (cuda:0) for prediction to avoid DataParallel issues")

    # Đường dẫn dữ liệu
    if split == "test":
        img_dir = "/kaggle/input/oggy-ds312/test/test"
    else:
        img_dir = os.path.join("/kaggle/input/if-u-know-u-know", split, split)
    cui_names_csv = "/kaggle/input/if-u-know-u-know/cui_names.csv"

    # Tải dữ liệu CUI
    df_cui = pd.read_csv(cui_names_csv)
    df_cui = df_cui[df_cui["Name"] != "Name nicht gefunden"].drop_duplicates(subset=["Name"]).reset_index(drop=True)
    name_list = list(df_cui["Name"])

    name2cui = dict(zip(df_cui["Name"], df_cui["CUI"]))

    if len(name_list) != len(set(name_list)):
        raise ValueError(f"Duplicate names found in name_list: {[name for name in set(name_list) if name_list.count(name) > 1]}")

    # Tính num_classes
    num_classes = len(name_list)
    print(f"Number of classes: {num_classes}")

    # Tải mô hình
    try:
        model = MedCSRAModel(num_classes=num_classes, num_heads=1, lam=0.1, dropout=0.5).to(device)
        checkpoint = torch.load(model_path, map_location=device)
        # Kiểm tra xem checkpoint có được lưu từ DataParallel không
        state_dict = checkpoint["model_state_dict"]
        if list(state_dict.keys())[0].startswith("module."):
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        print("Using single GPU (cuda:0) for prediction")
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy tệp mô hình tại {model_path}")
        return
    except Exception as e:
        print(f"Lỗi khi tải mô hình: {e}")
        return
    model.eval()

    # Kiểm tra định dạng mô hình
    print(f"Model architecture: {model}")

    # Khởi tạo MultiLabelBinarizer
    mlb = MultiLabelBinarizer(classes=name_list)
    mlb.fit([])

    # Tạo dataset
    test_ids = [f.split(".")[0] for f in os.listdir(img_dir) if f.endswith(".jpg")]
    print(f"Number of test images: {len(test_ids)}")
    df_test = pd.DataFrame({"ID": test_ids})
    df_test["Concept_Names"] = [[] for _ in range(len(df_test))]

    dataset = ImgCaptionConceptDataset(df_test, img_dir, name_list, mlb, mode="test")
    # Sử dụng drop_last=True để bỏ batch cuối nếu không đủ mẫu
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=True)

    # Dự đoán
    concept_preds = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Dự đoán"):
            pixel_values = batch["pixel_values"].to(device)
            ids = batch["id"]

            # Kiểm tra định dạng đầu vào
            print(f"Batch pixel values shape: {pixel_values.shape}, device: {pixel_values.device}, dtype: {pixel_values.dtype}")

            outputs = model(pixel_values)
            logits = outputs["logits_concept"]
            probs = torch.sigmoid(logits).cpu().numpy()

            for i in range(len(ids)):
                top_k = 5
                top_probs, top_indices = torch.topk(torch.tensor(probs[i]), k=top_k)
                top_names = [name_list[idx] for idx in top_indices]
                print(f"Top {top_k} concepts for ID {ids[i]}: {list(zip(top_names, top_probs.tolist()))}")

                concept_names = [name_list[j] for j in range(len(name_list)) if probs[i][j] > threshold]
                print(f"Concept names for ID {ids[i]}: {concept_names}")
                cuis = [name2cui[n] for n in concept_names if n in name2cui]
                print(f"CUIs for ID {ids[i]}: {cuis}")
                concept_preds.append({"ID": ids[i], "CUIs": ";".join(cuis)})

    # Xử lý các mẫu còn lại (nếu có) do drop_last=True
    remaining_samples = len(test_ids) % batch_size
    if remaining_samples > 0:
        print(f"Processing remaining {remaining_samples} samples individually")
        remaining_ids = test_ids[-remaining_samples:]
        remaining_df = pd.DataFrame({"ID": remaining_ids, "Concept_Names": [[] for _ in range(len(remaining_ids))]})
        remaining_dataset = ImgCaptionConceptDataset(remaining_df, img_dir, name_list, mlb, mode="test")
        remaining_loader = torch.utils.data.DataLoader(remaining_dataset, batch_size=1, shuffle=False, num_workers=2)

        with torch.no_grad():
            for batch in tqdm(remaining_loader, desc="Dự đoán các mẫu còn lại"):
                pixel_values = batch["pixel_values"].to(device)
                ids = batch["id"]

                print(f"Remaining batch pixel values shape: {pixel_values.shape}, device: {pixel_values.device}, dtype: {pixel_values.dtype}")

                outputs = model(pixel_values)
                logits = outputs["logits_concept"]
                probs = torch.sigmoid(logits).cpu().numpy()

                for i in range(len(ids)):
                    top_k = 5
                    top_probs, top_indices = torch.topk(torch.tensor(probs[i]), k=top_k)
                    top_names = [name_list[idx] for idx in top_indices]
                    print(f"Top {top_k} concepts for ID {ids[i]}: {list(zip(top_names, top_probs.tolist()))}")

                    concept_names = [name_list[j] for j in range(len(name_list)) if probs[i][j] > threshold]
                    print(f"Concept names for ID {ids[i]}: {concept_names}")
                    cuis = [name2cui[n] for n in concept_names if n in name2cui]
                    print(f"CUIs for ID {ids[i]}: {cuis}")
                    concept_preds.append({"ID": ids[i], "CUIs": ";".join(cuis)})

    # Lưu kết quả
    os.makedirs("outputs", exist_ok=True)
    pd.DataFrame(concept_preds).to_csv("outputs/concept_predictions.csv", index=False)
    print("✅ Đã lưu dự đoán khái niệm vào outputs/concept_predictions.csv")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", type=str, choices=["train", "predict"])
    parser.add_argument("--root_path", type=str, help="Root path chứa dữ liệu (chỉ cần cho train)")
    parser.add_argument("--batch_size", type=int, default=16)  # Mặc định cho 2 GPU
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--split", type=str, choices=["valid", "test"], default="test")
    parser.add_argument("--model_path", type=str, default="./model_best.pth")
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--start_epoch", type=int, default=0, help="Epoch để tiếp tục huấn luyện (dùng khi resume)")
    args = parser.parse_args()

    if args.mode == "train":
        if not args.root_path:
            raise ValueError("root_path là bắt buộc cho mode train")
        best_threshold = train(
            args.root_path,
            args.batch_size,
            args.num_epochs,
            args.lr,
            args.model_path,
            patience=10,
            start_epoch=args.start_epoch
        )
        print(f"Using best threshold {best_threshold:.2f} for prediction")
        args.threshold = best_threshold
    elif args.mode == "predict":
        predict(
            args.split,
            args.batch_size,
            args.model_path,
            args.threshold
        )

if __name__ == "__main__":
    main()
