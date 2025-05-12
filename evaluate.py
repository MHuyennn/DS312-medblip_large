import os
import torch
import pandas as pd
from tqdm import tqdm
from dataset import ImgCaptionConceptDataset
from model import MedCSRAModel
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score
import numpy as np

def evaluate_concept(model, dataloader, device, mlb, name2cui, thresholds=np.arange(0.1, 0.6, 0.1)):
    model.eval()
    preds = []
    ids = []
    all_true_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating Concept"):
            pixel_values = batch["pixel_values"].to(device)
            id_list = batch["id"]

            outputs = model(pixel_values)
            logits = outputs["logits_concept"]
            probs = torch.sigmoid(logits).cpu().numpy()

            if "labels_concept" in batch:
                all_true_labels.append(batch["labels_concept"].numpy())
                all_probs.append(probs)

            for i, id_val in enumerate(id_list):
                preds.append(probs[i])
                ids.append(id_val)

    # Tìm ngưỡng tối ưu nếu có ground truth
    best_threshold = 0.3
    best_f1 = 0.0
    if all_true_labels:
        all_true_labels = np.concatenate(all_true_labels, axis=0)
        all_probs = np.concatenate(all_probs, axis=0)
        for threshold in thresholds:
            pred_labels = (all_probs > threshold).astype(int)
            f1 = f1_score(all_true_labels, pred_labels, average="micro")
            print(f"Threshold {threshold:.1f}: F1-score = {f1:.4f}")
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        print(f"Best threshold: {best_threshold:.1f} with F1-score: {best_f1:.4f}")

    # Dự đoán với ngưỡng tốt nhất
    result_preds = []
    for i, id_val in enumerate(ids):
        concept_names = [mlb.classes_[j] for j in range(len(mlb.classes_)) if preds[i][j] > best_threshold]
        cuis = [name2cui.get(name, "Unknown") for name in concept_names]
        result_preds.append(";".join(cuis))

    result_df = pd.DataFrame({"ID": ids, "CUIs": result_preds})
    return result_df, best_f1, best_threshold

def main_evaluate(root_path, split="valid"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count()
    print(f"Using {num_gpus} GPUs")

    img_dir = os.path.join(root_path, split, split)
    concept_csv = os.path.join(img_dir, f"{split}_concepts.csv")
    cui_names_csv = os.path.join(root_path, "cui_names.csv")

    df_con = pd.read_csv(concept_csv)
    df_cui = pd.read_csv(cui_names_csv)

    df_cui = df_cui[df_cui["Name"] != "Name nicht gefunden"].drop_duplicates(subset=["Name"]).reset_index(drop=True)
    cui2name = dict(zip(df_cui["CUI"], df_cui["Name"]))
    name2cui = dict(zip(df_cui["Name"], df_cui["CUI"]))

    df = df_con
    df["Concept_Names"] = df["CUIs"].apply(lambda x: [cui2name[cui] for cui in x.split(";") if cui in cui2name])

    name_list = list(df_cui["Name"])
    if len(name_list) != len(set(name_list)):
        raise ValueError(f"Duplicate names found in name_list: {[name for name in set(name_list) if name_list.count(name) > 1]}")

    mlb = MultiLabelBinarizer(classes=name_list)
    mlb.fit(df["Concept_Names"])

    dataset = ImgCaptionConceptDataset(df, img_dir, name_list, mlb, mode=split)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=2)

    model = MedCSRAModel(num_classes=2468, num_heads=1, lam=0.1).to(device)
    try:
        checkpoint = torch.load(os.path.join(root_path, "model_best.pth"), map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        if num_gpus > 1:
            model = torch.nn.DataParallel(model)
    except FileNotFoundError:
        print(f"Error: Model file not found at {os.path.join(root_path, 'model_best.pth')}")
        return

    result_df, best_f1, best_threshold = evaluate_concept(model, dataloader, device, mlb, name2cui)
    save_path = os.path.join(root_path, f"{split}_concepts_pred.csv")
    result_df.to_csv(save_path, index=False)
    print(f"Saved concept predictions at: {save_path}")
    print(f"Best F1-score: {best_f1:.4f}, Best threshold: {best_threshold:.1f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True, help="Root path chứa dữ liệu")
    parser.add_argument("--split", type=str, default="valid", help="valid hoặc test")
    args = parser.parse_args()

    main_evaluate(args.root, split=args.split)
