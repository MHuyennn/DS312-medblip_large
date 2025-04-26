import os
import argparse
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModel, AutoProcessor, BlipForConditionalGeneration
from torch.utils.data import DataLoader
from dataset import ImgCaptionConceptDataset
from model import MedBLIPMultitask
import torch.nn.functional as F

def load_cui_name_embeddings(cui_names_path, device="cuda"):
    df = pd.read_csv(cui_names_path)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    encoder = AutoModel.from_pretrained("bert-base-uncased").to(device)
    encoder.eval()

    vectors, cui_list = [], []

    for _, row in df.iterrows():
        inputs = tokenizer(row["name"], return_tensors="pt", truncation=True, padding=True).to(device)
        with torch.no_grad():
            outputs = encoder(**inputs)
            emb = outputs.last_hidden_state[:, 0, :]
        vectors.append(emb.squeeze(0).cpu())
        cui_list.append(row["CUI"])

    emb_tensor = torch.stack(vectors)
    return emb_tensor, cui_list

def train(root_path, batch_size=4, num_epochs=2, lr=1e-5, load_weights=False, path_weights="./checkpoints"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dir = os.path.join(root_path, "train/train")
    train_captions = os.path.join(train_dir, "train_captions.csv")
    train_concepts = os.path.join(train_dir, "train_concepts.csv")
    cui_names = os.path.join("cui_names.csv") 

    df_cap = pd.read_csv(train_captions)
    df_con = pd.read_csv(train_concepts)
    df_train = pd.merge(df_cap, df_con, on="ID")

    model_blip = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
    processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = MedBLIPMultitask(model_blip.vision_model, model_blip.text_decoder, processor).to(device)

    emb_tensor, cui_list = load_cui_name_embeddings(cui_names, device)
    model.set_concept_embeddings(emb_tensor.to(device))

    from sklearn.preprocessing import MultiLabelBinarizer
    mlb = MultiLabelBinarizer(classes=cui_list)
    df_train["CUIs"] = df_train["CUIs"].apply(lambda x: x.split(";"))
    mlb.fit(df_train["CUIs"])

    dataset = ImgCaptionConceptDataset(df_train, train_dir, processor, mlb)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in dataloader:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(pixel_values, input_ids=input_ids, attention_mask=attention_mask, mode="caption")
            loss_caption = outputs.loss

            logits = model(pixel_values, mode="concept")
            loss_concept = F.binary_cross_entropy_with_logits(logits, labels.float())

            loss = loss_caption + loss_concept
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1} Loss: {total_loss/len(dataloader):.4f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "eval_caption", "eval_concept"])
    parser.add_argument("--root_path", required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--load_weights", action="store_true")
    parser.add_argument("--path_weights", type=str, default="./checkpoints")
    parser.add_argument("--score", type=str, default="rouge") 
    args = parser.parse_args()

    if args.mode == "train":
        train(args.root_path, args.batch_size, args.num_epochs, args.lr, args.load_weights, args.path_weights)

    elif args.mode == "eval_caption":
        # Gọi evaluate caption
        os.system(f"python evaluate.py --root {args.root_path} --task caption --score {args.score}")

    elif args.mode == "eval_concept":
        # Gọi evaluate concept
        os.system(f"python evaluate.py --root {args.root_path} --task concept")

if __name__ == "__main__":
    main()
