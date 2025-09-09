# Med CSRA for ImageClef Medical 2025 at Concept Detection task

## Introduction
In the ImageCLEFmedical 2025 Concept Detection competition, our team built a MedCSRA model and achieved significant results. Below are the methods we used for training, predicting, and evaluating our model.

## Training

To train the model, use the following command:

```
python main.py train
    --root_path: Root path containing the dataset (required, default: ./).
    --batch_size: Number of samples per batch (default: 8).
    --num_epochs: Number of training epochs (default: 20).
    --lr: Learning rate for the Adam optimizer (default: 0.001).
    --save_path: Path to save the best model checkpoint (default: ./model_best.pth).
    --patience: Number of epochs with no improvement after which training will stop (default: 5).
    --min_delta: Minimum change in F1-score to qualify as an improvement (default: 0.001).
    --start_epoch: Epoch to resume training from (default: 0, used with checkpoint loading).
```

Note: 
- The model checkpoint is saved automatically as model_best.pth in the save_path, which also serves as the load path for resuming training.
- Training uses the Adam optimizer with a weight decay of 0.00001 and a CosineAnnealingLR scheduler.
- Early stopping is implemented with a patience of 5 epochs and a minimum F1-score improvement of 0.001.
- The best threshold for prediction is determined based on validation F1-score and printed at the end.

## Prediction

To generate predictions, use the following command:

```
python main.py predict
    --root_path: Root path containing the dataset (default: ./).
    --model_path: Path to the pre-trained model weights (default: ./model_best.pth).
    --split: Dataset split to predict on (default: test, options: valid, test).
    --batch_size: Number of samples per batch (default: 8).
    --threshold: Threshold for concept prediction (default: 0.3, updated to best threshold from training if applicable).
```

Note:
- Predictions are saved to outputs/concept_predictions.csv.
- The script loads the model from model_path and processes images from the specified split 
- Top 5 concepts and their probabilities are printed for each image, along with the final CUIs.

## Evaluation

To evaluate the MedCSRA model, use the following command:

```
python evaluate.py 
    --root: Root path containing the dataset (required).
    --split: Dataset split to evaluate (default: valid, options: valid, test).
```
Notes:
- The script loads the pre-trained model from model_best.pth in the root path.
- For the valid split, it computes the optimal threshold (ranging from 0.1 to 0.5 with a step of 0.1) based on the F1-score and saves predictions to {split}_concepts_pred.csv.
- For the test split, it assumes no ground truth and generates predictions using the default threshold of 0.3, saving to test_concepts_pred.csv.
- The model uses a batch size of 8 and supports multi-GPU if available.
- Evaluation results include the best F1-score and threshold, printed to the console.
