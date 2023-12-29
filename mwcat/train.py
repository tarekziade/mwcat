"""
    Fine-tune DistilBert on Wikipedia Topics
"""
from functools import partial

from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from datasets import load_dataset
from torch.nn import BCEWithLogitsLoss

from mwcat.utils import NUM_CATEGORIES, tokenize_and_format


def load_data():
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=NUM_CATEGORIES
    )
    dataset = load_dataset("tarekziade/wikipedia-topics")
    tokenized_dataset = dataset.map(
        partial(tokenize_and_format, tokenizer), batched=True
    )
    return tokenizer, model, tokenized_dataset


training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    evaluation_strategy="steps",
    eval_steps=500,
    save_steps=500,
    learning_rate=5e-5,
)


class CatTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )
        loss = BCEWithLogitsLoss()(outputs["logits"].float(), inputs["labels"].float())
        return (loss, outputs) if return_outputs else loss


def main():
    tokenizer, model, tokenized_dataset = load_data()

    trainer = CatTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
    )

    trainer.train()

    model.save_pretrained("./fine_tuned_distilbert")
    model.push_to_hub("tarekziade/wikipedia-topics-distilbert")
