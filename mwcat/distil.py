import functools

import torch
from torch.utils.data import DataLoader

from transformers import (
    LongT5ForConditionalGeneration,
    T5Tokenizer,
    AutoConfig,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from datasets import load_dataset
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch import mps
import evaluate


# TEXT_MAX_SIZE = 16384
# SUMMARY_MAX_SIZE = 1024
TEXT_MAX_SIZE = 512
SUMMARY_MAX_SIZE = 40


def create_models():
    teacher_model_name = "pszemraj/long-t5-tglobal-base-16384-book-summary"
    teacher_model = LongT5ForConditionalGeneration.from_pretrained(teacher_model_name)
    tokenizer = T5Tokenizer.from_pretrained(teacher_model_name)
    teacher_model.eval()

    student_model_name = teacher_model_name
    config = AutoConfig.from_pretrained(
        student_model_name,
        d_model=256,
        d_ff=1024,
        d_kv=32,  # d_model / num_heads
        num_layers=6,
        num_decoder_layers=6,
        num_heads=8,
    )
    student_model = LongT5ForConditionalGeneration(config)

    student_model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher_model.to(device)
    student_model.to(device)
    return teacher_model, student_model, tokenizer


class BookSumDataset:
    def __init__(self, dataset_id, split, tokenizer):
        self.tokenizer = tokenizer
        self.dataset_id = dataset_id
        data = load_dataset(dataset_id, split=split)

        def check_line(line):
            # if line["summary_length"] > SUMMARY_MAX_SIZE:
            #    return False
            if line["summary_text"] is None:
                return False
            return line["chapter"] is not None

        data = data.filter(check_line)
        assert len(data) > 100, f"Not enough data, got {len(data)}"
        data = data.select_columns(["summary_length", "summary_text", "chapter"])
        self.data = data.map(self.tokenize_function, batched=True)

    def tokenize_function(self, example):
        inputs = self.tokenizer(
            example["chapter"],
            padding="max_length",
            truncation=True,
            max_length=TEXT_MAX_SIZE,
        )
        targets = self.tokenizer(
            example["summary_text"],
            padding="max_length",
            truncation=True,
            max_length=SUMMARY_MAX_SIZE,
        )
        inputs["labels"] = targets["input_ids"]
        return inputs


class DistillationTrainingArguments(Seq2SeqTrainingArguments):
    def __init__(self, *args, alpha=0.5, temperature=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.temperature = temperature


class DistillationTrainer(Seq2SeqTrainer):
    def __init__(self, *args, teacher_model=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher_model
        # place teacher on same device as student
        self._move_model_to_device(self.teacher, self.model.device)
        self.teacher.eval()

    def compute_loss(self, student, inputs, return_outputs=False):
        # compute student output
        outputs_student = student(**inputs)
        student_loss = outputs_student.loss
        # compute teacher output
        with torch.no_grad():
            outputs_teacher = self.teacher(**inputs)

        # assert size
        assert outputs_student.logits.size() == outputs_teacher.logits.size()

        # Soften probabilities and compute distillation loss
        loss_function = nn.KLDivLoss(reduction="batchmean")
        loss_logits = loss_function(
            F.log_softmax(outputs_student.logits / self.args.temperature, dim=-1),
            F.softmax(outputs_teacher.logits / self.args.temperature, dim=-1),
        ) * (self.args.temperature**2)
        # Return weighted student loss
        loss = self.args.alpha * student_loss + (1.0 - self.args.alpha) * loss_logits
        return (loss, outputs_student) if return_outputs else loss


rouge = evaluate.load("rouge")


def compute_metrics(tokenizer, eval_pred):
    predictions, labels = eval_pred
    # predictions is 2x134 and labels 134... why?
    # looks like second is crap
    predictions = predictions[0]

    predictions = tokenizer.batch_decode(
        np.argmax(predictions, axis=-1),
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )

    labels = tokenizer.batch_decode(
        labels, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )

    results = rouge.compute(predictions=predictions, references=labels)
    return results


def main():
    training_args = DistillationTrainingArguments(
        output_dir="./distillation_output",  # Output directory for model checkpoints and logs
        overwrite_output_dir=True,  # Overwrite the output directory if it exists
        num_train_epochs=2,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        learning_rate=6e-5,
        seed=33,
        # logging & evaluation strategies
        # logging_strategy="epoch",  # to get more information to TB
        evaluation_strategy="steps",
        eval_steps=30,
        save_strategy="epoch",
        save_total_limit=2,
        # load_best_model_at_end=True,
        # push to hub parameters
        push_to_hub=False,
        # distilation parameters
        alpha=0.5,
        temperature=4.0,
        # predict_with_generate=True,  # super slow!!
    )

    print("Loading models")
    percent = "50"
    teacher_model, student_model, tokenizer = create_models()
    train_dataset = BookSumDataset(
        "kmfoda/booksum", f"train[:{percent}%]", tokenizer
    ).data
    eval_dataset = BookSumDataset(
        "kmfoda/booksum", f"validation[:{percent}%]", tokenizer
    ).data

    trainer = DistillationTrainer(
        student_model,
        training_args,
        teacher_model=teacher_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer),
        tokenizer=tokenizer,
        # compute_metrics=functools.partial(compute_metrics, tokenizer),
    )

    mps.empty_cache()
    trainer.train()

    student_model.save_pretrained("distilled_t5_model")

    print("Distillation complete. Distilled model saved as 'distilled_t5_model'.")


if __name__ == "__main__":
    main()
