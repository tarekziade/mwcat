"""
    Fine-tune
    - DistilBert on Wikipedia Topics
    - t5-small on Wikipedia summaries
"""
from functools import partial

from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    T5Tokenizer,
    T5ForConditionalGeneration,
    Trainer,
    TrainingArguments,
)
from datasets import load_dataset
from torch.nn import BCEWithLogitsLoss

from mwcat.utils import NUM_CATEGORIES, tokenize_and_format


class CatTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )
        loss = BCEWithLogitsLoss()(outputs["logits"].float(), inputs["labels"].float())
        return (loss, outputs) if return_outputs else loss


class WikipediaClassifier:
    trainer_klass = CatTrainer

    def __init__(self, save_path, hub_name, model_name, dataset_name):
        self.training_args = TrainingArguments(
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
        self.save_path = save_path
        self.hub_name = hub_name
        self.model_name = model_name
        self.dataset_name = dataset_name

    def load_trainer(self, train_dataset, eval_dataset):
        return self.trainer_klass(
            model=self.model,
            args=self.training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
        )

    def _load(self, tokenizer, split, process=tokenize_and_format):
        dataset = load_dataset(self.dataset_name, split=split)
        return dataset.map(partial(process, tokenizer), batched=True)

    def load_data(self, dry=False):
        tokenizer = DistilBertTokenizer.from_pretrained(self.model_name)
        model = DistilBertForSequenceClassification.from_pretrained(
            self.model_name, num_labels=NUM_CATEGORIES
        )
        train_dataset = self._load(tokenizer, "train[:1%]" if dry else "train")
        test_dataset = self._load(tokenizer, "test[:1%]" if dry else "test")

        return tokenizer, model, train_dataset, test_dataset

    def run(self, dry=False):
        self.tokenizer, self.model, train_dataset, eval_dataset = self.load_data(
            dry=dry
        )
        self.trainer = self.load_trainer(train_dataset, eval_dataset)
        self.trainer.train()
        self.model.save_pretrained(self.save_path)

        if not dry:
            self.model.push_to_hub(self.hub_name)


class WikipediaSummarizer(WikipediaClassifier):
    trainer_klass = Trainer

    def load_data(self, dry=False):
        tokenizer = T5Tokenizer.from_pretrained(self.model_name)
        model = T5ForConditionalGeneration.from_pretrained(self.model_name)

        def preprocess_function(tokenizer, examples):
            input_text = [f"summarize: {text}" for text in examples["text"]]
            target_text = examples["summary"]
            model_inputs = tokenizer(
                input_text, max_length=512, padding="max_length", truncation=True
            )
            with tokenizer.as_target_tokenizer():
                labels = tokenizer(
                    target_text, max_length=128, padding="max_length", truncation=True
                )
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        train_dataset = self._load(
            tokenizer, "train[:2%]" if dry else "train", preprocess_function
        )
        test_dataset = self._load(
            tokenizer, "test[:2%]" if dry else "test", preprocess_function
        )

        return tokenizer, model, train_dataset, test_dataset


def main():
    # trainer = WikipediaClassifier(
    #    "./fine_tuned_distilbert",
    #    "tarekziade/wikipedia-topics-distilbert",
    #    "distilbert-base-uncased",
    #    "tarekziade/wikipedia-topics",
    # )
    # trainer.run()

    trainer = WikipediaSummarizer(
        "./fine_tuned_t5",
        "tarekziade/wikipedia-summaries-t5-small",
        "t5-small",
        "tarekziade/wikipedia-topics",
    )
    trainer.run(dry=True)


if __name__ == "__main__":
    main()
