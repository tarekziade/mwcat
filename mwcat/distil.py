import torch
from torch.utils.data import DataLoader

from transformers import (
    LongT5ForConditionalGeneration,
    T5Tokenizer,
    AutoConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)
from datasets import load_dataset
import torch.nn as nn
import torch.nn.functional as F
from torch import mps


# TEXT_MAX_SIZE = 16384
# SUMMARY_MAX_SIZE = 1024
TEXT_MAX_SIZE = 1024
SUMMARY_MAX_SIZE = 400


def create_models():
    teacher_model_name = "pszemraj/long-t5-tglobal-base-16384-book-summary"
    teacher_model = LongT5ForConditionalGeneration.from_pretrained(teacher_model_name)
    tokenizer = T5Tokenizer.from_pretrained(teacher_model_name)
    teacher_model.eval()

    student_model_name = teacher_model_name
    config = AutoConfig.from_pretrained(
        student_model_name,
        d_model=128,
        d_ff=512,
        d_kv=16,  # d_model / num_heads
        num_layers=4,
        num_decoder_layers=4,
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
        data = load_dataset("kmfoda/booksum", split="train[:2%]")
        data = data.filter(self.check_line)
        data = data.select_columns(["summary_length", "summary_text", "chapter"])
        self.data = data.map(self.tokenize_function, batched=True)

    def check_line(self, line):
        if line["summary_length"] > SUMMARY_MAX_SIZE:
            return False
        if line["summary_text"] is None:
            return False
        return line["chapter"] is not None

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


class DistillationTrainingArguments(TrainingArguments):
    def __init__(self, *args, alpha=0.5, temperature=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.temperature = temperature


class DistillationTrainer(Trainer):
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


def main():
    training_args = DistillationTrainingArguments(
        output_dir="./distillation_output",  # Output directory for model checkpoints and logs
        overwrite_output_dir=True,  # Overwrite the output directory if it exists
        num_train_epochs=7,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=6e-5,
        seed=33,
        # logging & evaluation strategies
        logging_strategy="epoch",  # to get more information to TB
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        # push to hub parameters
        push_to_hub=False,
        # distilation parameters
        alpha=0.5,
        temperature=4.0,
    )

    print("Loading models")
    teacher_model, student_model, tokenizer = create_models()
    train_dataset = BookSumDataset("kmfoda/booksum", "train[:2%]", tokenizer).data
    eval_dataset = BookSumDataset("kmfoda/booksum", "validation[:2%]", tokenizer).data

    trainer = DistillationTrainer(
        student_model,
        training_args,
        teacher_model=teacher_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer),
        tokenizer=tokenizer,
        # compute_metrics=compute_metrics,
    )

    mps.empty_cache()
    trainer.train()

    student_model.save_pretrained("distilled_t5_model")

    print("Distillation complete. Distilled model saved as 'distilled_t5_model'.")


if __name__ == "__main__":
    main()
