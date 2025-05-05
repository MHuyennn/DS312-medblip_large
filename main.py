import os
import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import AutoProcessor, BlipForConditionalGeneration

from dataset import ImgCaptionConceptDataset
from model import MedBLIPMultitask
from evaluate import evaluate_caption, evaluate_concept

def load_cui_name_embeddings(df_cui, processor, device):
    """Tạo embeddings từ danh sách Name concepts."""
    names = df_cui["Name"].drop_duplicates().tolist()  # Loại bỏ trùng lặp
    inputs = processor.tokenizer(names, padding=True, truncation=True, return_tensors="pt").to(device)

    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
    
    input_embeddings = model.text_decoder.get_input_embeddings()(inputs.input_ids)
    name_embeddings = input_embeddings.mean(dim=1)
    print(f"Number of names: {len(names)}, Embeddings shape: {name_embeddings.shape}")
    return name_embeddings, names

def train(root_path, batch_size=4, num_epochs=5, lr=1e-5, save_path="./model_best.pth"):
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

    cui2name = dict(zip(df_cui["CUI"], df_cui["Name"]))
    df_train = pd.merge(df_cap_train, df_con_train, on="ID")
    df_valid = pd.merge(df_cap_valid, df_con_valid, on="ID")

    df_train["Concept_Names"] = df_train["CUIs"].apply(lambda x: [cui2name[cui] for cui in str(x).split(";") if cui in cui2name])
    df_valid["Concept_Names"] = df_valid["CUIs"].apply(lambda x: [cui2name[cui] for cui in str(x).split(";") if cui in cui2name])

    processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

    # Lấy danh sách concept names từ tập train
    concept_names_in_train = set()
    for concept_list in df_train["Concept_Names"]:
        concept_names_in_train.update(concept_list)
    concept_names_in_train = list(concept_names_in_train)

    # Lọc df_cui và loại bỏ trùng lặp dựa trên Name
    df_cui_train = df_cui[df_cui["Name"].isin(concept_names_in_train)].drop_duplicates(subset=["Name"]).reset_index(drop=True)
    
    # Tạo name_list từ df_cui_train
    name_list = df_cui_train["Name"].tolist()

    # Kiểm tra trùng lặp
    if len(name_list) != len(set(name_list)):
        raise ValueError(f"Duplicate names found in name_list: {[name for name in set(name_list) if name_list.count(name) > 1]}")

    name_embeddings, _ = load_cui_name_embeddings(df_cui_train, processor, device)

    # Khởi tạo MultiLabelBinarizer
    mlb = MultiLabelBinarizer(classes=name_list)
    mlb.fit(df_train["Concept_Names"])

    # Datasets
    train_dataset = ImgCaptionConceptDataset(df_train, train_img_dir, processor, name_list, mlb, mode="train")
    valid_dataset = ImgCaptionConceptDataset(df_valid, valid_img_dir, processor, name_list, mlb, mode="valid")

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    # Model
    blip_base = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    model = MedBLIPMultitask(
        vision_encoder=blip_base.vision_model,
        text_decoder=blip_base.text_decoder,
        processor=processor
    ).to(device)
    model.set_concept_embeddings(name_embeddings)

    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion_concept = nn.BCEWithLogitsLoss()

    # Training loop
    best_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
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

            # Kiểm tra loss_caption
            if loss_caption is None:
                raise ValueError("loss_caption is None. Check model output or input data.")

            # Loss concept
            loss_concept = criterion_concept(logits_concept, labels_concept)

            loss = loss_caption + loss_concept
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {avg_loss:.4f}")

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model, save_path)
            print(f"✅ Saved best model to {save_path}")

def predict(root_path, split="test", task="both", batch_size=4, cui_path=None, model_path="./model_best.pth"):
    """Predict caption and/or concept detection."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Đường dẫn dữ liệu
    img_dir = os.path.join(root_path, split, split)
    cui_path = cui_path if cui_path else root_path
    cui_names_csv = os.path.join(cui_path, "cui_names.csv")

    # Tải processor và dữ liệu CUI
    processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    df_cui = pd.read_csv(cui_names_csv)
    name_list = list(df_cui["Name"].drop_duplicates())
    print(f"Number of unique names in df_cui: {len(name_list)}")

    # Kiểm tra trùng lặp
    if len(name_list) != len(set(name_list)):
        raise ValueError(f"Duplicate names found in name_list: {[name for name in set(name_list) if name_list.count(name) > 1]}")

    # Kiểm tra khớp giữa name_list và df_cui
    missing_names = [n for n in name_list if n not in df_cui["Name"].values]
    print(f"Missing names in df_cui: {missing_names}")

    # Tải mô hình
    try:
        model = torch.load(model_path, map_location=device)
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy tệp mô hình tại {model_path}")
        return
    model.eval()
    model.to(device)

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
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Dự đoán
    caption_preds = []
    concept_preds = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Dự đoán"):
            pixel_values = batch["pixel_values"].to(device)
            ids = batch["id"]
            print(f"Batch pixel_values shape: {pixel_values.shape}, IDs: {ids}")

            # Shared vision embedding
            vision_out = model.vision_encoder(pixel_values=pixel_values)
            vision_embeds = vision_out.last_hidden_state[:, 0, :]

            # Caption prediction
            if task in ["caption", "both"]:
                gen_ids = model.text_decoder.generate(
                    encoder_hidden_states=vision_out.last_hidden_state,
                    max_length=100
                )
                captions = processor.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

            # Concept prediction
            if task in ["concept", "both"]:
                logits = torch.matmul(vision_embeds, embeddings.T)
                probs = torch.sigmoid(logits).cpu().numpy()
                print(f"Probs max: {probs.max()}, min: {probs.min()}, mean: {probs.mean()}")

                # In top 5 khái niệm có xác suất cao nhất
                top_k = 5
                for i in range(len(ids)):
                    top_probs, top_indices = torch.topk(torch.tensor(probs[i]), k=top_k)
                    top_names = [name_list[idx] for idx in top_indices]
                    print(f"Top {top_k} concepts for ID {ids[i]}: {list(zip(top_names, top_probs.tolist()))}")

            for i in range(len(ids)):
                id = ids[i]

                if task in ["caption", "both"]:
                    caption_preds.append({"ID": id, "Caption": captions[i]})

                if task in ["concept", "both"]:
                    concept_names = [name_list[j] for j in range(len(name_list)) if probs[i][j] > 0.2]
                    print(f"Concept names for ID {id}: {concept_names}")
                    cuis = [df_cui[df_cui["Name"] == n]["CUI"].values[0] for n in concept_names if n in df_cui["Name"].values]
                    print(f"CUIs for ID {id}: {cuis}")
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
    parser.add_argument("--cui_path", type=str, default=None, help="Path to folder containing cui_names.csv")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--task", type=str, choices=["caption", "concept", "both"], default="both")
    parser.add_argument("--split", type=str, choices=["valid", "test"], default="test")
    parser.add_argument("--model_path", type=str, default="./model_best.pth", help="Path to the trained model")
    args = parser.parse_args()

    if args.mode == "train":
        train(args.root_path, args.batch_size, args.num_epochs, args.lr, args.model_path)
    elif args.mode == "predict":
        predict(args.root_path, args.split, args.task, args.batch_size, args.cui_path, args.model_path)

if __name__ == "__main__":
    main()
