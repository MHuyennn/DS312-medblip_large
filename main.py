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
    names = df_cui["Name"].tolist()
    inputs = processor.tokenizer(names, padding=True, truncation=True, return_tensors="pt").to(device)

    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
    
    input_embeddings = model.text_decoder.get_input_embeddings()(inputs.input_ids)  # (batch_size, seq_len, hidden_dim)

    name_embeddings = input_embeddings.mean(dim=1)  # Trung bình các token

    return name_embeddings, names

def train(root_path, batch_size=4, num_epochs=5, lr=1e-5, save_path="./model_best.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load data
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

    # 2. Map CUIs thành Name
    cui2name = dict(zip(df_cui["CUI"], df_cui["Name"]))
    df_train = pd.merge(df_cap_train, df_con_train, on="ID")
    df_valid = pd.merge(df_cap_valid, df_con_valid, on="ID")

    df_train["Concept_Names"] = df_train["CUIs"].apply(lambda x: [cui2name[cui] for cui in x.split(";") if cui in cui2name])
    df_valid["Concept_Names"] = df_valid["CUIs"].apply(lambda x: [cui2name[cui] for cui in x.split(";") if cui in cui2name])

    processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    name_embeddings, name_list = load_cui_name_embeddings(df_cui, processor, device)

    # 3. MultiLabelBinarizer
    mlb = MultiLabelBinarizer(classes=name_list)
    mlb.fit(df_train["Concept_Names"])

    # 4. Dataset và DataLoader
    train_dataset = ImgCaptionConceptDataset(df_train, train_img_dir, processor, name_list, mlb, mode="train")
    valid_dataset = ImgCaptionConceptDataset(df_valid, valid_img_dir, processor, name_list, mlb, mode="valid")

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    # 5. Model
    blip_base = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    model = MedBLIPMultitask(
        vision_encoder=blip_base.vision_model,
        text_decoder=blip_base.text_decoder,
        processor=processor
    ).to(device)

    model.set_concept_embeddings(name_embeddings)

    # 6. Optimizer và Loss
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion_concept = nn.BCEWithLogitsLoss()

    # 7. Training loop
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

            outputs = model(pixel_values, input_ids=input_ids, attention_mask=attention_mask, mode="train")
            loss_caption = outputs["loss_caption"]
            logits_concept = outputs["logits_concept"]

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
            print(f"Saved best model to {save_path}")

def predict(root_path, split="test", task="caption", batch_size=4):
    """Predict caption hoặc concept."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img_dir = os.path.join(root_path, split, split)
    cui_names_csv = os.path.join(root_path, "cui_names.csv")
    df_cui = pd.read_csv(cui_names_csv)

    processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

    name_list = list(set(df_cui["Name"].tolist()))
    mlb = MultiLabelBinarizer(classes=name_list)
    mlb.fit([])  # Empty fit để giữ đúng thứ tự classes

    # Load model
    model = torch.load(os.path.join(root_path, "model_best.pth"), map_location=device)
    model.to(device)

    # Load image ID
    ids = [os.path.splitext(f)[0] for f in os.listdir(img_dir) if f.endswith(".jpg")]
    df_test = pd.DataFrame({"ID": ids})
    df_test["Concept_Names"] = [[] for _ in range(len(df_test))]
    df_test["Caption"] = [""] * len(df_test)

    dataset = ImgCaptionConceptDataset(df_test, img_dir, processor, name_list, mlb, mode="test")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    if task == "caption":
        result_df, _ = evaluate_caption(model, dataloader, processor, device)
        save_path = os.path.join(root_path, f"{split}_captions_pred.csv")
    elif task == "concept":
        name2cui = dict(zip(df_cui["Name"], df_cui["CUI"]))
        result_df = evaluate_concept(model, dataloader, device, mlb, name2cui)
        save_path = os.path.join(root_path, f"{split}_concepts_pred.csv")
    
    result_df.to_csv(save_path, index=False)
    print(f"Saved predictions to {save_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", type=str, choices=["train", "predict"])
    parser.add_argument("--root_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--task", type=str, choices=["caption", "concept"], default="caption")
    parser.add_argument("--split", type=str, choices=["valid", "test"], default="test")
    args = parser.parse_args()

    if args.mode == "train":
        train(args.root_path, args.batch_size, args.num_epochs, args.lr)
    elif args.mode == "predict":
        predict(args.root_path, split=args.split, task=args.task)

if __name__ == "__main__":
    main()
