import os

from transformers import (
    LongT5ForConditionalGeneration,
    T5Tokenizer,
    AutoConfig,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    AdamW,
)
from datasets import load_dataset
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import mps
import evaluate


TEXT_MAX_SIZE = 3000  # 16384
SUMMARY_MAX_SIZE = 512  # 1024

# student model config
model_config = {
    "d_ff": 1024,  # 2048,
    "d_kv": 128,  # 64,
    "d_model": 384,  # 768,
    "num_decoder_layers": 3,  # 12
    "num_heads": 3,  # 12
    "num_layers": 3,  # 12
    "decoder_start_token_id": 0,
    "dense_act_fn": "gelu_new",
    "dropout_rate": 0.1,
    "early_stopping": True,
    "encoder_attention_type": "transient-global",
    "encoder_no_repeat_ngram_size": 4,
    "eos_token_id": 1,
    "feed_forward_proj": "gated-gelu",
    "global_block_size": 16,
    "initializer_factor": 1.0,
    "is_encoder_decoder": True,
    "is_gated_act": True,
    "layer_norm_epsilon": 1e-06,
    "length_penalty": 0.8,
    "local_radius": 127,
    "max_length": 512,
    "min_length": 8,
    "model_type": "longt5",
    "n_positions": 4096,
    "no_repeat_ngram_size": 3,
    "num_beams": 2,
    "output_past": True,
    "pad_token_id": 0,
    "relative_attention_max_distance": 128,
    "relative_attention_num_buckets": 32,
    "repetition_penalty": 3.5,
    "tie_word_embeddings": False,
    "torch_dtype": "float32",
    "use_cache": True,
    "vocab_size": 32128,
}

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


def create_models():
    teacher_model_name = "pszemraj/long-t5-tglobal-base-16384-book-summary"
    teacher_model = LongT5ForConditionalGeneration.from_pretrained(teacher_model_name)
    tokenizer = T5Tokenizer.from_pretrained(teacher_model_name)
    teacher_model.eval()
    torch.compile(teacher_model)
    config = AutoConfig.from_pretrained(teacher_model_name, **model_config)
    student_model = LongT5ForConditionalGeneration(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    teacher_model.to(device)
    student_model.to(device)
    return teacher_model, student_model, tokenizer


class BookSumDataset:
    def __init__(self, dataset_id, split, tokenizer):
        self.tokenizer = tokenizer
        self.dataset_id = dataset_id
        data = load_dataset(dataset_id, split=split)

        def check_line(line):
            if line["summary_length"] > 1024:
                return False
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
        self._move_model_to_device(self.teacher, self.model.device)

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
    print("Loading models")
    percent = "100"
    teacher_model, student_model, tokenizer = create_models()

    training_args = DistillationTrainingArguments(
        output_dir="./distillation_output",  # Output directory for model checkpoints and logs
        overwrite_output_dir=True,  # Overwrite the output directory if it exists
        num_train_epochs=2,
        learning_rate=0.0005,
        seed=42,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=128,
        # total_train_batch_size=128,
        lr_scheduler_type="cosine",
        warmup_ratio=0.01,
        # logging & evaluation strategies
        # logging_strategy="epoch",  # to get more information to TB
        # evaluation_strategy="steps",
        # eval_steps=30,
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

    def get_parameter_count(model):
        num_params = sum(p.numel() for p in model.parameters())
        return num_params

    print(
        f"teacher model has {(get_parameter_count(teacher_model)/1000000):.2f}M parameters"
    )
    print(
        f"student model has {(get_parameter_count(student_model)/1000000):.2f}M parameters"
    )
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
        # optimizer=AdamW(student_model.parameters(), betas=(0.9, 0.999), eps=1e-08),
        # compute_metrics=functools.partial(compute_metrics, tokenizer),  <== M1 Killer :)
    )

    mps.empty_cache()
    trainer.train()

    distilled_model_name = "distilled-long-t5-tglobal-base-16384-book-summary"
    student_model.save_pretrained(f"./{distilled_model_name}")
    tokenizer.save_pretrained(f"./{distilled_model_name}")
    # student_model.push_to_hub(f"tarekziade/{distilled_model_name}")
    # tokenizer.push_to_hub(f"tarekziade/{distilled_model_name}")

    print(f"Distillation complete. Distilled model saved as '{distilled_model_name}'")


if __name__ == "__main__":
    main()
