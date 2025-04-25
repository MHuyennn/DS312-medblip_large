import os
import glob
import argparse
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoProcessor
from tqdm import tqdm

from dataset import ImgCaptionConceptDataset
from model import MedBLIPMultiTaskModel

def train(root_path, batch_size=4, num_epochs=2, lr=1e-5, load_weights=False, path_weights="./checkpoints/"):
    train_dir = os.path.join(root_path, "train/train")
    train_captions = os.path.join(root_path, "train_captions.csv")
    train_concepts = os.path.join(root_path, "train_concepts.csv")

    df_cap = pd.read_csv(train_captions)
    df_con = pd.read_csv(train_concepts)
    df_train = pd.merge(df_cap, df_con, on="ID")

    processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    dataset = ImgCaptionConceptDataset(df=df_train, path=train_dir, processor=processor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = MedBLIPMultiTaskModel(num_concepts=len(dataset.classes_))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    global_step, start_epoch = 0, 0

    if load_weights:
        ckpts = glob.glob(os.path.join(path_weights, "medblip_multitask_step_*.pth"))
        if ckpts:
            latest_ckpt = max(ckpts, key=lambda x: int(x.split("_step_")[-1].split(".")[0]))
            checkpoint = torch.load(latest_ckpt)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            global_step = checkpoint['global_step']
            start_epoch = checkpoint['epoch'] + 1
            print(f"Loaded checkpoint: {latest_ckpt}")

    model.train()
    for epoch in range(start_epoch, start_epoch + num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        for batch in tqdm(dataloader, desc=f"Training"):
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            labels_caption = input_ids
            labels_concept = batch["labels_concept"].to(device)

            out = model(pixel_values=pixel_values,
                        input_ids=input_ids,
                        attention_mask=attn,
                        labels_caption=labels_caption,
                        labels_concept=labels_concept)

            loss = out["loss"]
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

        # Lưu checkpoint
        os.makedirs(path_weights, exist_ok=True)
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "global_step": global_step,
            "epoch": epoch,
            "loss": loss.item()
        }, os.path.join(path_weights, f"medblip_multitask_step_{global_step}.pth"))
        print(f"[Checkpoint] Epoch {epoch+1}, Step {global_step}, Loss: {loss.item():.4f}")

@torch.no_grad()
def predict(root_path, path_weights="./checkpoints/", output_dir="./results/"):
    from transformers import AutoProcessor
    from sklearn.preprocessing import MultiLabelBinarizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-large")

    for split in ["valid", "test"]:
        cap_path = os.path.join(root_path, f"{split}/{split}_captions.csv")
        con_path = os.path.join(root_path, f"{split}/{split}_concepts.csv")
        img_dir = os.path.join(root_path, f"{split}/{split}")

        df_cap = pd.read_csv(cap_path)
        df_con = pd.read_csv(con_path)
        df = pd.merge(df_cap, df_con, on="ID")

        dataset = ImgCaptionConceptDataset(df=df, path=img_dir, processor=processor)
        dataloader = DataLoader(dataset, batch_size=1)

        model = MedBLIPMultiTaskModel(num_concepts=len(dataset.classes_))
        ckpts = glob.glob(os.path.join(path_weights, "medblip_multitask_step_*.pth"))
        assert ckpts, f"Không tìm thấy checkpoint trong {path_weights}"
        ckpt = torch.load(max(ckpts, key=lambda x: int(x.split("_step_")[-1].split(".")[0])))
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device).eval()

        cap_results, con_results = [], []
        for batch in tqdm(dataloader, desc=f"Inferencing on {split}"):
            idx = 0  # vì batch_size = 1
            ID = dataset.ids[idx]
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)

            out = model(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attn)

            # Caption
            gen_ids = model.blip.generate(pixel_values=pixel_values, max_new_tokens=100)
            caption = processor.batch_decode(gen_ids, skip_special_tokens=True)[0]

            # Concept
            logits = out["logits_concept"].cpu().numpy()[0]
            preds = (logits > 0.5).astype(int)
            cui_list = dataset.mlb.inverse_transform([preds])[0]
            cui_str = ";".join(cui_list)

            cap_results.append([ID, caption])
            con_results.append([ID, cui_str])

        os.makedirs(output_dir, exist_ok=True)
        pd.DataFrame(cap_results, columns=["ID", "Caption"]).to_csv(f"{output_dir}/{split}_captions.csv", index=False)
        pd.DataFrame(con_results, columns=["ID", "CUIs"]).to_csv(f"{output_dir}/{split}_concepts.csv", index=False)
        print(f"[{split}] ✅ Xuất file CSV tại {output_dir}")

def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command')

    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--root_path', type=str, default='./data')
    parser_train.add_argument('--batch_size', type=int, default=4)
    parser_train.add_argument('--num_epochs', type=int, default=2)
    parser_train.add_argument('--lr', type=float, default=1e-5)
    parser_train.add_argument('--load_weights', action='store_true')
    parser_train.add_argument('--path_weights', type=str, default='./checkpoints/')

    parser_predict = subparsers.add_parser('predict')
    parser_predict.add_argument('--root_path', type=str, default='./data')
    parser_predict.add_argument('--path_weights', type=str, default='./checkpoints/')
    parser_predict.add_argument('--output_dir', type=str, default='./results/')

    args = parser.parse_args()

    if args.command == 'train':
        train(args.root_path, args.batch_size, args.num_epochs, args.lr, args.load_weights, args.path_weights)
    elif args.command == 'predict':
        predict(args.root_path, args.path_weights, args.output_dir)

if __name__ == "__main__":
    main()
