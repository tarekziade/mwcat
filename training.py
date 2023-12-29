from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from datasets import load_dataset
import torch
from torch.nn import BCEWithLogitsLoss

categories = [
    "Academic_disciplines",
    "Business",
    "Communication",
    "Concepts",
    "Culture",
    "Economy",
    "Education",
    "Energy",
    "Engineering",
    "Entertainment",
    "Entities",
    "Ethics",
    "Food_and_drink",
    "Geography",
    "Government",
    "Health",
    "History",
    "Human_behavior",
    "Humanities",
    "Information",
    "Internet",
    "Knowledge",
    "Language",
    "Law",
    "Life",
    "Lists",
    "Mass media",
    "Mathematics",
    "Military",
    "Nature",
    "People",
    "Philosophy",
    "Politics",
    "Religion",
    "Science",
    "Society",
    "Sports",
    "Technology",
    "Time",
    "Universe",
]
category_to_id = {cat: i for i, cat in enumerate(categories)}
num_labels = len(categories)

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=num_labels
)


def tokenize_and_format(examples):
    inputs = [
        f"cat: {title}::{summary}".lower()
        for (title, summary) in zip(examples["title"], examples["summary"])
    ]

    tokenized_inputs = tokenizer(
        inputs,
        padding="max_length",
        truncation=True,
        max_length=512,
    )

    labels = []
    for cats in examples["categories"]:
        vec = [0] * num_labels
        for cat in cats:
            vec[category_to_id[cat]] = 1
        labels.append(vec)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


dataset = load_dataset("tarekziade/wikipedia-topics")
tokenized_dataset = dataset.map(tokenize_and_format, batched=True)

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
