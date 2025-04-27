import os
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoProcessor
from dataset import ImgCaptionConceptDataset
from sklearn.preprocessing import MultiLabelBinarizer
import evaluate

def compute_caption_scores(predictions, references):
    """Tính các chỉ số đánh giá caption prediction."""
    scorer_rouge = evaluate.load("rouge")
    scorer_bleu = evaluate.load("bleu")
    scorer_bertscore = evaluate.load("bertscore")
    scorer_meteor = evaluate.load("meteor")
    scorer_cider = evaluate.load("cider")
    scorer_spice = evaluate.load("spice")

    results = {}
    results.update(scorer_rouge.compute(predictions=predictions, references=references))
    results.update(scorer_bleu.compute(predictions=predictions, references=references))
    results.update(scorer_bertscore.compute(predictions=predictions, references=references, lang="en"))
    results.update(scorer_meteor.compute(predictions=predictions, references=references))
    results.update(scorer_cider.compute(predictions=predictions, references=references))
    results.update(scorer_spice.compute(predictions=predictions, references=references))

    return results

def evaluate_caption(model, dataloader, processor, device):
    model.eval()
    preds = []
    gts = []
    ids = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating Caption"):
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)

            outputs = model(
                pixel_values,
                input_ids=input_ids,
                attention_mask=batch["attention_mask"].to(device),
                mode="predict_caption"  # Sửa mode để khớp với model.py
            )
            generated_ids = outputs["generated_ids"]

            captions = processor.batch_decode(generated_ids, skip_special_tokens=True)
            references = processor.batch_decode(batch["labels_caption"], skip_special_tokens=True)

            preds.extend(captions)
            gts.extend(references)
            ids.extend(batch["id"])

    result_df = pd.DataFrame({"ID": ids, "Caption": preds})
    metrics = compute_caption_scores(preds, gts)

    return result_df, metrics

def evaluate_concept(model, dataloader, device, mlb, name2cui):
    model.eval()
    preds = []
    ids = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating Concept"):
            pixel_values = batch["pixel_values"].to(device)
            id_list = batch["id"]

            outputs = model(pixel_values, mode="predict_concept")
            logits = outputs["logits_concept"]
            probs = torch.sigmoid(logits).cpu()

            pred_labels = (probs > 0.5).int()

            pred_concepts = mlb.inverse_transform(pred_labels.numpy())

            for id_val, concept_names in zip(id_list, pred_concepts):
                cuis = [name2cui.get(name, "Unknown") for name in concept_names]
                preds.append(";".join(cuis))
                ids.append(id_val)

    result_df = pd.DataFrame({"ID": ids, "CUIs": preds})
    return result_df

def main_evaluate(root_path, mode="caption", split="valid"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img_dir = os.path.join(root_path, split, split)
    caption_csv = os.path.join(img_dir, f"{split}_captions.csv")
    concept_csv = os.path.join(img_dir, f"{split}_concepts.csv")
    cui_names_csv = os.path.join(root_path, "cui_names.csv")

    df_cap = pd.read_csv(caption_csv)
    df_con = pd.read_csv(concept_csv)
    df_cui = pd.read_csv(cui_names_csv)

    cui2name = dict(zip(df_cui["CUI"], df_cui["Name"]))
    name2cui = dict(zip(df_cui["Name"], df_cui["CUI"]))

    # Merge caption + concepts
    df = pd.merge(df_cap, df_con, on="ID")
    df["Concept_Names"] = df["CUIs"].apply(lambda x: [cui2name[cui] for cui in x.split(";") if cui in cui2name])

    processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

    # Lấy danh sách tên duy nhất từ df_cui
    name_list = list(df_cui["Name"].drop_duplicates())

    # Kiểm tra trùng lặp
    if len(name_list) != len(set(name_list)):
        raise ValueError(f"Duplicate names found in name_list: {[name for name in set(name_list) if name_list.count(name) > 1]}")

    mlb = MultiLabelBinarizer(classes=name_list)
    mlb.fit(df["Concept_Names"])

    dataset = ImgCaptionConceptDataset(df, img_dir, processor, name_list, mlb, mode=split)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False)

    model = torch.load(os.path.join(root_path, "model_best.pth"), map_location=device)
    model.to(device)

    if mode == "caption":
        result_df, metrics = evaluate_caption(model, dataloader, processor, device)
        save_path = os.path.join(root_path, f"{split}_captions_pred.csv")
        result_df.to_csv(save_path, index=False)
        print(f" Saved caption predictions at: {save_path}")
        print(" Caption Evaluation Metrics:")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")

    elif mode == "concept":
        result_df = evaluate_concept(model, dataloader, device, mlb, name2cui)
        save_path = os.path.join(root_path, f"{split}_concepts_pred.csv")
        result_df.to_csv(save_path, index=False)
        print(f" Saved concept predictions at: {save_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True, help="Root path chứa dữ liệu")
    parser.add_argument("--task", type=str, choices=["caption", "concept"], required=True, help="Chọn task")
    parser.add_argument("--split", type=str, default="valid", help="valid hoặc test")
    args = parser.parse_args()

    main_evaluate(args.root, mode=args.task, split=args.split)