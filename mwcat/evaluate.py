from functools import partial

import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from datasets import load_dataset
from torch.nn import BCEWithLogitsLoss


from mwcat.utils import category_to_id, tokenize_and_format


def load_model():
    model = DistilBertForSequenceClassification.from_pretrained(
        "./fine_tuned_distilbert"
    )
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    return model, tokenizer


def calculate_accuracy(logits, true_labels, threshold=0.5):
    """
    Calculate the accuracy of predictions using a specified threshold.

    Args:
    logits (numpy.ndarray): The logits output by the model.
    true_labels (numpy.ndarray): The true labels.
    threshold (float): The threshold for converting logits to binary predictions.

    Returns:
    float: The accuracy score.
    """
    # Apply sigmoid to logits and convert to binary predictions
    preds = 1 / (1 + np.exp(-logits))
    binary_preds = (preds > threshold).astype(int)

    # Calculate accuracy
    correct_predictions = np.sum((binary_preds == true_labels).all(axis=1))
    total_predictions = true_labels.shape[0]

    accuracy = correct_predictions / total_predictions
    return accuracy


def convert_to_tensors(batch):
    batch = {k: torch.tensor(v) for k, v in batch.items()}
    return batch


def main():
    model, tokenizer = load_model()

    # Load and preprocess your test dataset
    test_dataset = load_dataset("tarekziade/wikipedia-topics", split="test")
    test_dataset = test_dataset.map(
        partial(tokenize_and_format, tokenizer), batched=True
    )
    test_dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "labels"]
    )
    test_dataloader = DataLoader(test_dataset, batch_size=8)

    # Put model in evaluation mode
    model.eval()

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Evaluation loop
    total_eval_accuracy = 0
    total_eval_loss = 0
    for batch in test_dataloader:
        b_input_ids = batch["input_ids"].to(device)
        b_labels = batch["labels"].to(device)

        with torch.no_grad():
            outputs = model(b_input_ids)

        logits = outputs.logits
        loss_fct = BCEWithLogitsLoss()
        loss = loss_fct(logits, b_labels.float())

        total_eval_loss += loss.item()

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to("cpu").numpy()

        # Calculate the accuracy for this batch of test sentences
        # You can define your own function to calculate accuracy based on your requirements
        batch_accuracy = calculate_accuracy(logits, label_ids)
        total_eval_accuracy += batch_accuracy

    # Report the final accuracy and loss for this validation run.
    avg_val_accuracy = total_eval_accuracy / len(test_dataloader)
    print("Accuracy: {0:.2f}".format(avg_val_accuracy))
    avg_val_loss = total_eval_loss / len(test_dataloader)
    print("Average testing loss: {0:.2f}".format(avg_val_loss))


if __name__ == "__main__":
    main()
