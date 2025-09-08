import csv
from sklearn.metrics import f1_score
import os
import argparse

class ConceptEvaluator:
    def __init__(self, ground_truth_path, secondary_ground_truth_path, **kwargs):
        """
        Evaluator class for ImageCLEF 2025 Caption - Concept Detection.
        
        Args:
            ground_truth_path (str): Path to primary ground truth CSV (concepts.csv).
            secondary_ground_truth_path (str): Path to secondary ground truth CSV (concepts_manual.csv).
        """
        self.ground_truth_path = ground_truth_path
        self.ground_truth_path_secondary = secondary_ground_truth_path
        # Ground truth dict => gt[image_id] = tuple of concepts
        self.gt = self.load_gt(self.ground_truth_path)
        self.gt_secondary = self.load_gt(self.ground_truth_path_secondary)

    def _evaluate(self, submission_file_path, _context={}):
        """
        Evaluate the submission file against ground truth.
        
        Args:
            submission_file_path (str): Path to the submission CSV file.
            _context (dict): Optional context (not used).
        
        Returns:
            dict: Contains 'score' (F1-score) and 'score_secondary' (F1-secondary).
        """
        print("Evaluating...")
        predictions = self.load_predictions(submission_file_path)
        score = self.compute_primary_score(predictions)
        score_secondary = self.compute_secondary_score(predictions)

        _result_object = {"score": score, "score_secondary": score_secondary}
        assert "score" in _result_object
        assert "score_secondary" in _result_object
        return _result_object

    def load_gt(self, path):
        """
        Load ground truth data from CSV.
        
        Args:
            path (str): Path to ground truth CSV.
        
        Returns:
            dict: Mapping of image_id to tuple of concepts.
        """
        print(f"Loading ground truth from {path}...")
        gt = {}
        with open(path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if "ID" not in row[0]:  # Skip header
                    image_id = row[0]
                    if len(row) > 1:
                        concepts = tuple(concept.strip().upper() for concept in row[1].split(";") if concept.strip())
                        gt[image_id] = concepts
                    else:
                        raise Exception(f"Invalid format in {path}: Row for ID {image_id} has no concepts.")
        return gt

    def load_predictions(self, submission_file_path):
        """
        Load and validate predictions from submission CSV.
        
        Args:
            submission_file_path (str): Path to submission CSV.
        
        Returns:
            dict: Mapping of image_id to tuple of predicted concepts.
        """
        print(f"Loading predictions from {submission_file_path}...")
        predictions = {}
        image_ids_gt = tuple(self.gt.keys())
        max_num_concepts = 100
        with open(submission_file_path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            line_cnt = 0
            for row in reader:
                if "ID" not in row[0]:  # Skip header
                    line_cnt += 1
                    if not 1 <= len(row) <= 2:
                        self.raise_exception(
                            "Wrong format: Each line must have an image ID, optionally followed by semicolon-separated concepts.",
                            line_cnt,
                            "<image_id>,<concept_1>;<concept_2>;<concept_n>"
                        )
                    image_id = row[0]
                    if image_id not in image_ids_gt:
                        self.raise_exception(
                            f"Image ID '{image_id}' not in ground truth.",
                            line_cnt,
                            image_id
                        )
                    if image_id in predictions:
                        self.raise_exception(
                            f"Image ID '{image_id}' specified multiple times.",
                            line_cnt,
                            image_id
                        )
                    concepts = tuple()
                    if len(row) > 1 and row[1].strip() and row[1].strip().lower() != "unknown_cui":
                        concepts = tuple(concept.strip().upper() for concept in row[1].split(";") if concept.strip())
                        if len(concepts) > max_num_concepts:
                            self.raise_exception(
                                f"Too many concepts for ID '{image_id}'. Max allowed: {max_num_concepts}.",
                                line_cnt,
                                max_num_concepts
                            )
                        if len(concepts) != len(set(concepts)):
                            self.raise_exception(
                                f"Duplicate concepts for ID '{image_id}'.",
                                line_cnt,
                                image_id
                            )
                    predictions[image_id] = concepts
            if len(predictions) != len(image_ids_gt):
                self.raise_exception(
                    f"Submission file has {len(predictions)} IDs, expected {len(image_ids_gt)}.",
                    line_cnt
                )
        return predictions

    def raise_exception(self, message, record_count, *args):
        """
        Raise an exception with formatted message and line number.
        """
        raise Exception(f"{message.format(*args)} Error at line {record_count}.")

    def compute_primary_score(self, predictions):
        """
        Compute primary F1-score.
        
        Args:
            predictions (dict): Mapping of image_id to tuple of predicted concepts.
        
        Returns:
            float: Average F1-score across images.
        """
        print("Computing primary F1-score...")
        max_score = len(self.gt)
        current_score = 0
        for image_id in predictions:
            predicted_concepts = predictions[image_id]
            gt_concepts = self.gt[image_id]
            if len(gt_concepts) == 0:
                max_score -= 1
                continue
            all_concepts = sorted(list(set(gt_concepts + predicted_concepts)))
            y_true = [int(concept in gt_concepts) for concept in all_concepts]
            y_pred = [int(concept in predicted_concepts) for concept in all_concepts]
            f1score = f1_score(y_true, y_pred, average="binary")
            current_score += f1score
        return current_score / max_score if max_score > 0 else 0

    def compute_secondary_score(self, predictions):
        """
        Compute secondary F1-score using a filtered set of concepts.
        
        Args:
            predictions (dict): Mapping of image_id to tuple of predicted concepts.
        
        Returns:
            float: Average F1-secondary score across images.
        """
        print("Computing secondary F1-score...")
        max_score = len(self.gt_secondary)
        current_score = 0
        allowed_concepts = {
            "C0002978", "C0040405", "C0024485", "C0032743", "C0041618",
            "C1306645", "C1140618", "C0037949", "C0030797", "C0023216",
            "C0037303", "C0817096", "C0006141", "C0000726", "C0920367"
        }
        for image_id in predictions:
            predicted_concepts = tuple(con for con in predictions[image_id] if con in allowed_concepts)
            gt_concepts = tuple(con for con in self.gt_secondary[image_id] if con in allowed_concepts)
            if len(gt_concepts) == 0:
                max_score -= 1
                continue
            all_concepts = sorted(list(set(gt_concepts + predicted_concepts)))
            y_true = [int(concept in gt_concepts) for concept in all_concepts]
            y_pred = [int(concept in predicted_concepts) for concept in all_concepts]
            f1score = f1_score(y_true, y_pred, average="binary")
            current_score += f1score
        return current_score / max_score if max_score > 0 else 0

def main():
    """Main function for command-line evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate concept detection submissions.")
    parser.add_argument('--root_path', type=str, default='/kaggle/input/oggyyy-dataset/', help='Root path to dataset')
    parser.add_argument('--submission_file_path', type=str, default='/kaggle/working/submission.csv', help='Path to submission CSV')
    args = parser.parse_args()

    # Construct ground truth paths based on root_path
    ground_truth_path = os.path.join(args.root_path, "valid/valid/concepts.csv")
    secondary_ground_truth_path = os.path.join(args.root_path, "valid/valid/concepts_manual.csv")

    # Instantiate evaluator
    evaluator = ConceptEvaluator(ground_truth_path, secondary_ground_truth_path)
    
    # Evaluate
    result = evaluator._evaluate(args.submission_file_path)
    print(f"Evaluation Results: {result}")

if __name__ == "__main__":
    main()
