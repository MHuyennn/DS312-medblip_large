import os
import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score
import numpy as np
from transformers import AutoProcessor, BlipForConditionalGeneration
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms

from dataset import ImgCaptionConceptDataset
from model import MedBLIPMultitask, FocalLoss
from evaluate import evaluate_caption, evaluate_concept

def load_cui_name_embeddings(df_cui, processor, device, expected_num_concepts=2468):
    """Tạo embeddings từ danh sách Name concepts, đảm bảo đúng số lượng concepts."""
    names = df_cui["Name"].drop_duplicates().tolist()
    print(f"Initial number of unique names: {len(names)}")

    # Nếu số concept ít hơn expected_num_concepts, thêm dummy concepts
    if len(names) < expected_num_concepts:
        print(f"Adding {expected_num_concepts - len(names)} dummy concepts to match checkpoint")
        dummy_names = [f"dummy_concept_{i}" for i in range(expected_num_concepts - len(names))]
        names.extend(dummy_names)
    elif len(names) > expected_num_concepts:
        print(f"Warning: Truncating to {expected_num_concepts} names to match checkpoint")
        names = names[:expected_num_concepts]

    inputs = processor.tokenizer(names, padding=True, truncation=True, return_tensors="pt").to(device)

    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
    
    input_embeddings = model.text_decoder.get_input_embeddings()(inputs.input_ids)
    name_embeddings = input_embeddings.mean(dim=1)
    print(f"Final number of names: {len(names)}, Embeddings shape: {name_embeddings.shape}")
    
    if name_embeddings.shape[0] != expected_num_concepts:
        raise ValueError(f"Expected {expected_num_concepts} concepts, but got {name_embeddings.shape[0]}")
    
    return name_embeddings, names

def evaluate_threshold(model, valid_loader, name_list, device, thresholds=np.arange(0.1, 0.6, 0.1)):
    """Tìm ngưỡng tối ưu cho concept prediction dựa trên F1-score trên tập valid."""
    model.eval()
    all_true_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in valid_loader:
            pixel_values = batch["pixel_values"].to(device)
            labels_concept = batch["labels_concept"].to(device)

            outputs = model(pixel_values, mode="predict_concept")
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
        print(f"Threshold {threshold:.1f}: F1-score = {f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    print(f"Best threshold: {best_threshold:.1f} with F1-score: {best_f1:.4f}")
    return best_threshold

def train(root_path, batch_size=4, num_epochs=5, lr=1e-5, save_path="./model_best.pth", checkpoint_path=None, start_epoch=2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    train_img_dir = os.path.join(root_path, "train/train")
    valid_img_dir = os.path.join(root_path, "valid/valid")
    caption_train_csv = os.path.join(train_img_dir, "train_captions.csv")
    concept_train_csv = os.path.join(train_img_dir, "train_concepts.csv")
    caption_valid_csv = os.path.join(valid_img_dir, "valid_captions.csv")
    concept_valid_csv = os.path.join(valid_img_dir, "valid_concepts.csv")
    cui_names_csv = os.path.join(root_path, "cui_names.csv")

    df_cap_train = pd.read_csv(caption_train_csv)
    df_con_train = pd.read_csv(concept_train_csv)
    df_cap_valid = pd.read_csv(caption_valid_csv)
    df_con_valid = pd.read_csv(concept_valid_csv)
    df_cui = pd.read_csv(cui_names_csv)

    # Xử lý CUIs không hợp lệ
    df_cui = df_cui[df_cui["Name"] != "Name nicht gefunden"].drop_duplicates(subset=["Name"]).reset_index(drop=True)
    
    cui2name = dict(zip(df_cui["CUI"], df_cui["Name"]))
    name2cui = {v: k for k, v in cui2name.items()}
    df_train = pd.merge(df_cap_train, df_con_train, on="ID")
    df_valid = pd.merge(df_cap_valid, df_con_valid, on="ID")

    df_train["Concept_Names"] = df_train["CUIs"].apply(
        lambda x: [cui2name[cui] for cui in str(x).split(";") if cui in cui2name]
    )
    df_valid["Concept_Names"] = df_valid["CUIs"].apply(
        lambda x: [cui2name[cui] for cui in str(x).split(";") if cui in cui2name]
    )

    processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

    # Lấy danh sách concept names từ tập train
    concept_names_in_train = set()
    for concept_list in df_train["Concept_Names"]:
        concept_names_in_train.update(concept_list)
    concept_names_in_train = list(concept_names_in_train)

    df_cui_train = df_cui[df_cui["Name"].isin(concept_names_in_train)].drop_duplicates(subset=["Name"]).reset_index(drop=True)
    name_list = df_cui_train["Name"].tolist()

    if len(name_list) != len(set(name_list)):
        raise ValueError(f"Duplicate names found in name_list: {[name for name in set(name_list) if name_list.count(name) > 1]}")

    # Tải embeddings với số concept khớp checkpoint
    name_embeddings, name_list = load_cui_name_embeddings(df_cui_train, processor, device, expected_num_concepts=2468)

    # Kiểm tra số lượng concept names
    print(f"Final concept embeddings shape: {name_embeddings.shape}")

    # Khởi tạo MultiLabelBinarizer
    mlb = MultiLabelBinarizer(classes=name_list)
    mlb.fit(df_train["Concept_Names"])

    # Data augmentation mạnh hơn
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor()
    ])

    # Datasets
    train_dataset = ImgCaptionConceptDataset(
        df_train, train_img_dir, processor, name_list, mlb, mode="train", transform=train_transform
    )
    valid_dataset = ImgCaptionConceptDataset(
        df_valid, valid_img_dir, processor, name_list, mlb, mode="valid"
    )

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Model
    blip_base = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    model = MedBLIPMultitask(
        vision_encoder=blip_base.vision_model,
        text_decoder=blip_base.text_decoder,
        processor=processor
    ).to(device)
    model.set_concept_embeddings(name_embeddings)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion_concept = FocalLoss(alpha=1.0, gamma=2.0)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Khởi tạo biến theo dõi
    best_f1 = 0.0
    best_threshold = 0.3

    # Tải checkpoint nếu có
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)  # Bỏ qua tham số không khớp
        best_f1 = checkpoint.get("best_f1", 0.0)  # Lấy best_f1, mặc định 0.0 nếu không có
        best_threshold = checkpoint.get("best_threshold", 0.3)  # Lấy best_threshold, mặc định 0.3 nếu không có
        print(f"✅ Loaded checkpoint from {checkpoint_path} (best_f1: {best_f1:.4f}, best_threshold: {best_threshold:.1f})")
        print("Note: Some parameters (e.g., ConceptHead, MultiheadAttention) were not loaded and will be randomly initialized.")
        print("Note: Optimizer state was not loaded due to parameter mismatch. Using new optimizer.")
    else:
        print("No checkpoint found. Starting training with new optimizer.")

    # Training loop
    for epoch in range(start_epoch, start_epoch + num_epochs):
        model.train()
        running_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{start_epoch + num_epochs - 1} - Training"):
            optimizer.zero_grad()

            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_caption = batch["labels_caption"].to(device)
            labels_concept = batch["labels_concept"].to(device)

            outputs = model(
                pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels_caption=labels_caption,
                mode="train"
            )
            loss_caption = outputs["loss_caption"]
            logits_concept = outputs["logits_concept"]

            if loss_caption is None:
                raise ValueError("loss_caption is None. Check model output or input data.")

            loss_concept = criterion_concept(logits_concept, labels_concept)
            loss = loss_caption + loss_concept
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()

        scheduler.step()
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch}/{start_epoch + num_epochs - 1}] Loss: {avg_loss:.4f}")

        # Đánh giá trên tập valid
        model.eval()
        valid_loss = 0.0
        all_true_labels = []
        all_probs = []
        all_captions = []
        all_true_captions = []

        with torch.no_grad():
            for batch in valid_loader:
                pixel_values = batch["pixel_values"].to(device)
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels_caption = batch["labels_caption"].to(device)
                labels_concept = batch["labels_concept"].to(device)

                outputs = model(
                    pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels_caption=labels_caption,
                    mode="train"
                )
                loss_caption = outputs["loss_caption"]
                logits_concept = outputs["logits_concept"]

                loss = loss_caption + criterion_concept(logits_concept, labels_concept)
                valid_loss += loss.item()

                probs = torch.sigmoid(logits_concept).cpu().numpy()
                all_true_labels.append(labels_concept.cpu().numpy())
                all_probs.append(probs)

                # Đánh giá caption
                caption_outputs = model(pixel_values, mode="predict_caption")
                all_captions.extend(caption_outputs["generated_captions"])
                all_true_captions.extend(
                    processor.tokenizer.batch_decode(labels_caption, skip_special_tokens=True)
                )

        avg_valid_loss = valid_loss / len(valid_loader)
        all_true_labels = np.concatenate(all_true_labels, axis=0)
        all_probs = np.concatenate(all_probs, axis=0)
        valid_f1 = f1_score(all_true_labels, (all_probs > best_threshold).astype(int), average="micro")

        # Đánh giá caption (giả sử evaluate_caption trả về BLEU score)
        bleu_score = evaluate_caption(all_true_captions, all_captions)
        print(f"Validation Loss: {avg_valid_loss:.4f}, F1-score (threshold {best_threshold:.1f}): {valid_f1:.4f}, BLEU: {bleu_score:.4f}")

        # Cập nhật ngưỡng tối ưu
        best_threshold = evaluate_threshold(model, valid_loader, name_list, device)

        # Save best model dựa trên tổng hợp F1 và BLEU
        combined_score = 0.7 * valid_f1 + 0.3 * bleu_score
        if combined_score > best_f1:
            best_f1 = combined_score
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_f1": best_f1,
                "best_threshold": best_threshold,
                "epoch": epoch
            }, save_path)
            print(f"✅ Saved best model to {save_path} with Combined Score: {best_f1:.4f}")

        # Lưu checkpoint mỗi epoch
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_f1": best_f1,
            "best_threshold": best_threshold,
            "epoch": epoch
        }
        checkpoint_path_epoch = f"/kaggle/working/checkpoint_epoch{epoch}.pth"
        torch.save(checkpoint, checkpoint_path_epoch)
        print(f"✅ Saved checkpoint to {checkpoint_path_epoch}")

    print(f"Final best threshold: {best_threshold:.1f}")
    return best_threshold

def predict(root_path, split="test", task="both", batch_size=4, cui_path=None, model_path="./model_best.pth", threshold=0.3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Đường dẫn dữ liệu
    img_dir = os.path.join(root_path, split, split)
    cui_path = cui_path if cui_path else root_path
    cui_names_csv = os.path.join(cui_path, "cui_names.csv")

    # Tải processor và dữ liệu CUI
    processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    df_cui = pd.read_csv(cui_names_csv)
    df_cui = df_cui[df_cui["Name"] != "Name nicht gefunden"].drop_duplicates(subset=["Name"]).reset_index(drop=True)
    name_list = list(df_cui["Name"])

    name2cui = dict(zip(df_cui["Name"], df_cui["CUI"]))

    if len(name_list) != len(set(name_list)):
        raise ValueError(f"Duplicate names found in name_list: {[name for name in set(name_list) if name_list.count(name) > 1]}")

    # Tải mô hình
    try:
        blip_base = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        model = MedBLIPMultitask(
            vision_encoder=blip_base.vision_model,
            text_decoder=blip_base.text_decoder,
            processor=processor
        ).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy tệp mô hình tại {model_path}")
        return
    model.eval()

    # Tải embeddings
    embeddings, _ = load_cui_name_embeddings(df_cui, processor, device)
    model.set_concept_embeddings(embeddings)
    print(f"Concept embeddings shape: {embeddings.shape}")

    # Khởi tạo MultiLabelBinarizer
    mlb = MultiLabelBinarizer(classes=name_list)
    mlb.fit([])

    # Tạo dataset
    test_ids = [f.split(".")[0] for f in os.listdir(img_dir) if f.endswith(".jpg")]
    print(f"Number of test images: {len(test_ids)}")
    df_test = pd.DataFrame({"ID": test_ids})
    df_test["Caption"] = [""] * len(df_test)
    df_test["Concept_Names"] = [[] for _ in range(len(df_test))]

    dataset = ImgCaptionConceptDataset(df_test, img_dir, processor, name_list, mlb, mode="test")
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Dự đoán
    caption_preds = []
    concept_preds = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Dự đoán"):
            pixel_values = batch["pixel_values"].to(device)
            ids = batch["id"]

            outputs = model(pixel_values, mode="predict_both" if task == "both" else f"predict_{task}")

            if task in ["caption", "both"]:
                captions = outputs["generated_captions"]
                for i, id in enumerate(ids):
                    caption_preds.append({"ID": id, "Caption": captions[i]})

            if task in ["concept", "both"]:
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
                    concept_preds.append({"ID": id, "CUIs": ";".join(cuis)})

    # Lưu kết quả
    os.makedirs("outputs", exist_ok=True)
    if task in ["caption", "both"]:
        pd.DataFrame(caption_preds).to_csv("outputs/caption_predictions.csv", index=False)
        print("✅ Đã lưu dự đoán chú thích vào outputs/caption_predictions.csv")
    if task in ["concept", "both"]:
        pd.DataFrame(concept_preds).to_csv("outputs/concept_predictions.csv", index=False)
        print("✅ Đã lưu dự đoán khái niệm vào outputs/concept_predictions.csv")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", type=str, choices=["train", "predict"])
    parser.add_argument("--root_path", type=str, required=True)
    parser.add_argument("--cui_path", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--task", type=str, choices=["caption", "concept", "both"], default="both")
    parser.add_argument("--split", type=str, choices=["valid", "test"], default="test")
    parser.add_argument("--model_path", type=str, default="./model_best.pth")
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--start_epoch", type=int, default=2, choices=[1, 2])
    args = parser.parse_args()

    if args.mode == "train":
        best_threshold = train(
            args.root_path,
            args.batch_size,
            args.num_epochs,
            args.lr,
            args.model_path,
            args.checkpoint_path,
            args.start_epoch
        )
        print(f"Using best threshold {best_threshold:.1f} for prediction")
        args.threshold = best_threshold
    elif args.mode == "predict":
        predict(
            args.root_path,
            args.split,
            args.task,
            args.batch_size,
            args.cui_path,
            args.model_path,
            args.threshold
        )

if __name__ == "__main__":
    main()
