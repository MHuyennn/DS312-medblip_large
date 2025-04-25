import os
import csv
import evaluate
import numpy as np
import pandas as pd
import re
import warnings
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import MultiLabelBinarizer
import argparse

warnings.filterwarnings('ignore')

### ====== PHẦN CHẤM CAPTION (GIỮ NGUYÊN) ====== ###
def preprocess_sentences(sentences):
    processed_sentences = []
    for sentence in sentences:
        sentence = sentence.lower()
        sentence = re.sub(r'[^\w\s]', '', sentence)
        sentence = re.sub(r'\b\d+\b', '<n>', sentence)
        sentence_words = sentence.split()
        cleaned_words = []
        previous_word = None
        for word in sentence_words:
            if word != previous_word:
                cleaned_words.append(word)
            previous_word = word
        cleaned_sentence = ' '.join(cleaned_words)
        processed_sentences.append(cleaned_sentence)
    return processed_sentences

def preprocess_df(df):
    df['processed_Caption'] = preprocess_sentences(df['Caption'])
    return df

def BERTscore(bertscore, valid_captions, cands):
    scores = []
    for i in tqdm(range(len(cands["Caption"]))):
        bert = bertscore.compute(
            predictions=[cands["Caption"][i]],
            lang="en",
            model_type="microsoft/deberta-xlarge-mnli",
            references=[valid_captions["Caption"][i]]
        )
        scores.append(bert["f1"][0])
    return scores

def evaluate_caption(root, score_type):
    print(f" Đang đánh giá caption với phương pháp {score_type.upper()}")

    valid_captions = pd.read_csv(os.path.join(root, "valid_captions.csv"))
    valid_captions = preprocess_df(valid_captions)

    predictions = pd.read_csv(os.path.join(root, "results/valid_captions.csv"))
    predictions = preprocess_df(predictions)

    bleu = evaluate.load("bleu")
    rouge = evaluate.load("rouge")
    meteor = evaluate.load("meteor")
    bertscore = evaluate.load("bertscore")

    if score_type == 'rouge':
        result = rouge.compute(predictions=predictions["Caption"], references=valid_captions["Caption"])
        print("ROUGE-1:", round(result["rouge1"], 6))
        print("ROUGE-L:", round(result["rougeL"], 6))

    elif score_type == 'bleu':
        result = bleu.compute(predictions=predictions["Caption"], references=valid_captions["Caption"])
        print("BLEU Precision-1:", round(result["precisions"][0], 6))

    elif score_type == 'meteor':
        result = meteor.compute(predictions=predictions["Caption"], references=valid_captions["Caption"])
        print("METEOR:", round(result["meteor"], 6))

    elif score_type == 'bertscore':
        result = BERTscore(bertscore, valid_captions, predictions)
        print("BERTScore (avg F1):", round(np.average(result), 6))

### ====== PHẦN CHẤM CONCEPT CHUẨN IMAGECLEF ====== ###
class ConceptEvaluator:
    def __init__(self, ground_truth_path, secondary_ground_truth_path):
        self.ground_truth_path = ground_truth_path
        self.ground_truth_path_secondary = secondary_ground_truth_path
        self.gt = self.load_gt(self.ground_truth_path)
        self.gt_secondary = self.load_gt(self.ground_truth_path_secondary)

    def _evaluate(self, prediction_path):
        predictions = self.load_predictions(prediction_path)
        score = self.compute_primary_score(predictions)
        score_secondary = self.compute_secondary_score(predictions)
        return score, score_secondary

    def load_gt(self, path):
        gt = {}
        with open(path) as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if "ID" not in row[0]:
                    image_id = row[0]
                    concepts = tuple(concept.strip() for concept in row[1].split(";")) if len(row) > 1 else tuple()
                    gt[image_id] = concepts
        return gt

    def load_predictions(self, submission_file_path):
        predictions = {}
        image_ids_gt = tuple(self.gt.keys())
        max_num_concepts = 100
        with open(submission_file_path) as csvfile:
            reader = csv.reader(csvfile)
            lineCnt = 0
            for row in reader:
                if "ID" not in row[0]:
                    lineCnt += 1
                    if not 1 <= len(row) <= 2:
                        raise Exception(f"Wrong format at line {lineCnt}")
                    image_id = row[0]
                    if image_id not in image_ids_gt:
                        raise Exception(f"Image ID {image_id} không tồn tại trong tập test tại dòng {lineCnt}")
                    if image_id in predictions:
                        raise Exception(f"Image ID {image_id} bị trùng tại dòng {lineCnt}")
                    concepts = tuple(con.strip() for con in row[1].split(";")) if len(row) > 1 else tuple()
                    if len(concepts) > max_num_concepts:
                        raise Exception(f"Dòng {lineCnt} có quá nhiều concept (>100)")
                    if len(concepts) != len(set(concepts)):
                        raise Exception(f"Dòng {lineCnt} có concept bị lặp")
                    predictions[image_id] = concepts
            if len(predictions) != len(image_ids_gt):
                raise Exception(f"Số lượng ảnh không khớp ground truth: {len(predictions)} vs {len(image_ids_gt)}")
        return predictions

    def compute_primary_score(self, predictions):
        max_score = len(self.gt)
        current_score = 0
        for image_id in predictions:
            pred = tuple(con.upper() for con in predictions[image_id])
            gt = tuple(con.upper() for con in self.gt[image_id])
            if len(gt) == 0:
                max_score -= 1
                continue
            all_concepts = sorted(list(set(gt + pred)))
            y_true = [int(c in gt) for c in all_concepts]
            y_pred = [int(c in pred) for c in all_concepts]
            f1score = f1_score(y_true, y_pred, average="binary")
            current_score += f1score
        return current_score / max_score

    def compute_secondary_score(self, predictions):
        max_score = len(self.gt_secondary)
        current_score = 0
        allowed_concepts = {
            "C0002978", "C0040405", "C0024485", "C0032743", "C0041618", "C1306645",
            "C1140618", "C0037949", "C0030797", "C0023216", "C0037303", "C0817096",
            "C0006141", "C0000726", "C0920367"
        }
        for image_id in predictions:
            pred = tuple(con.upper() for con in predictions[image_id])
            gt = tuple(con.upper() for con in self.gt_secondary[image_id])
            pred = tuple(c for c in pred if c in allowed_concepts)
            gt = tuple(c for c in gt if c in allowed_concepts)
            if len(gt) == 0:
                max_score -= 1
                continue
            all_concepts = sorted(list(set(gt + pred)))
            y_true = [int(c in gt) for c in all_concepts]
            y_pred = [int(c in pred) for c in all_concepts]
            f1score = f1_score(y_true, y_pred, average="binary")
            current_score += f1score
        return current_score / max_score

def evaluate_concept_official(root):
    gt_path = os.path.join(root, "valid/valid_concepts.csv")
    secondary_gt_path = os.path.join(root, "valid/concepts_manual.csv")
    pred_path = os.path.join(root, "results/valid_concepts.csv")
    evaluator = ConceptEvaluator(gt_path, secondary_gt_path)
    score, score_secondary = evaluator._evaluate(pred_path)
    print(f" Concept Detection:")
    print(f" - Primary Score (Official F1):     {score:.4f}")
    print(f" - Secondary Score (Allowed only):  {score_secondary:.4f}")

### ====== MAIN CLI ====== ###
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='./')
    parser.add_argument('--task', type=str, choices=['caption', 'concept'], default='caption')
    parser.add_argument('--score', type=str, default='rouge', help="Chỉ dùng cho caption: rouge/bleu/meteor/bertscore")
    args = parser.parse_args()

    if args.task == "caption":
        evaluate_caption(args.root, args.score)
    elif args.task == "concept":
        evaluate_concept_official(args.root)

if __name__ == "__main__":
    main()
