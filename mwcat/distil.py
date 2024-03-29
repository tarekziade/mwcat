"""
T5 and Booksum-based models distillation
"""
import os
import argparse
import json

from transformers import (
    AutoModelForSeq2SeqLM,
    T5TokenizerFast,
    AutoConfig,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    LongT5ForConditionalGeneration,
    T5ForConditionalGeneration,
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


os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


def get_pytorch_device():
    """
    Returns the most appropriate device for PyTorch operations,
    trying CUDA, MPS, and then CPU in that order.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Model distillation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--percent",
        type=int,
        help="Percent of data used.",
        default=100,
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run the script with 1%% corpus and no upload",
        default=False,
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate using ROUGE",
        default=False,
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force model downloads",
        default=False,
    )
    parser.add_argument(
        "--student-model-id",
        type=str,
        help="Name of the student model.",
        default="tarekziade/t5-small-booksum-distilled",
    )
    parser.add_argument(
        "--teacher-model-id",
        type=str,
        help="Name of the teacher model for fine-tuning.",
        default="cnicu/t5-small-booksum",
    )
    parser.add_argument(
        "--input-max-size",
        type=int,
        help="Max input size for the model.",
        default=16384,
    )
    parser.add_argument(
        "--summary-max-size",
        type=int,
        help="Max output size for the model.",
        default=1024,
    )
    parser.add_argument(
        "--model-config",
        type=str,
        help="Path to the model configuration for the student",
        default=os.path.join(os.path.dirname(__file__), "t5-distillation.json"),
    )
    parser.add_argument(
        "--device",
        type=str,
        help="Device used",
        default=get_pytorch_device(),
    )

    args = parser.parse_args()
    return args


TEST_DATA = """\
The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building, and the tallest structure in Paris. Its base is square, measuring 125 metres (410 ft) on each side. During its construction, the Eiffel Tower surpassed the Washington Monument to become the tallest man-made structure in the world, a title it held for 41 years until the Chrysler Building in New York City was finished in 1930. It was the first structure to reach a height of 300 metres. Due to the addition of a broadcasting aerial at the top of the tower in 1957, it is now taller than the Chrysler Building by 5.2 metres (17 ft). Excluding transmitters, the Eiffel Tower is the second tallest free-standing structure in France after the Millau Viaduct.
"""


def test_model(model, tokenizer):
    input_ids = tokenizer.encode(
        "summarize: " + TEST_DATA,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=512,
        add_special_tokens=False,
    )

    input_ids = input_ids.to(model.device)

    generated_ids = model.generate(input_ids, max_length=120)[0]
    print(generated_ids)
    return tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
        remove_invalid_values=True,
    )


def check(student_model, teacher_model, tokenizer):
    student_model.eval()
    print("Student: " + test_model(student_model, tokenizer))
    student_model.train()
    print()
    print("Teacher: " + test_model(teacher_model, tokenizer))
    print()


def create_models(args):
    with open(args.model_config) as f:
        model_config = json.loads(f.read())

    arch = model_config["architectures"][0]

    if arch == "LongT5ForConditionalGeneration":
        klass = LongT5ForConditionalGeneration
    else:
        klass = T5ForConditionalGeneration

    teacher_model = klass.from_pretrained(args.teacher_model_id)
    tokenizer = T5TokenizerFast.from_pretrained(args.teacher_model_id)
    teacher_model.eval()
    torch.compile(teacher_model)

    config = AutoConfig.from_pretrained(args.teacher_model_id)  # , **model_config)
    # config.num_layers = 3
    # config.vocab_size = 16064
    # config.num_heads = 4
    # config.d_model = 256
    # config.d_ff = 1024
    # config.num_decoder_layers = 6

    student_model = klass(config)
    # student_model = klass.from_pretrained(
    #    args.teacher_model_id, config=config, ignore_mismatched_sizes=True
    # )
    # student_model = klass.from_pretrained(args.teacher_model_id)

    teacher_model.to(args.device)
    student_model.to(args.device)
    return teacher_model, student_model, tokenizer


class BookSumDataset:
    def __init__(self, args, dataset_id, train_split, eval_split, tokenizer):
        self.args = args
        self.tokenizer = tokenizer
        self.dataset_id = dataset_id
        if args.force_download:
            self.mode = "force_redownload"
        else:
            self.mode = "reuse_dataset_if_exists"

        self.train = self._load_data(train_split)
        self.eval = self._load_data(eval_split)

    def _load_data(self, split):
        data = load_dataset(self.dataset_id, split=split, download_mode=self.mode)

        def check_line(line):
            if line["summary_length"] > 1024:
                return False
            if line["summary_text"] is None:
                return False
            return line["chapter"] is not None

        data = data.filter(check_line)
        # if not self.args.dry_run:
        #    assert len(data) > 100, f"Not enough data, got {len(data)}"

        data = data.select_columns(["summary_length", "summary_text", "chapter"])

        def tokenize_function(example):
            inputs = self.tokenizer(
                ["summarize: " + item for item in example["chapter"]],
                padding="max_length",
                truncation=True,
                max_length=self.args.input_max_size,
            )
            targets = self.tokenizer(
                example["summary_text"],
                padding="max_length",
                truncation=True,
                max_length=self.args.summary_max_size,
            )
            inputs["labels"] = targets["input_ids"]
            return inputs

        return data.map(tokenize_function, batched=True)


class DistillationTrainingArguments(Seq2SeqTrainingArguments):
    def __init__(self, *args, alpha=0.5, temperature=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.temperature = temperature


class DistillationTrainer(Seq2SeqTrainer):
    def __init__(self, *args, teacher_model=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher_model
        # self._move_model_to_device(self.teacher, self.model.device)
        self.loss_function = nn.KLDivLoss(reduction="batchmean")

    def compute_loss(self, student, inputs, return_outputs=False):
        """
        Runs the input against the student and the teacher and compare results.
        Returns the student's loss to refine its weights.
        """

        outputs_student = student(**inputs)

        student_loss = outputs_student.loss
        with torch.no_grad():
            outputs_teacher = self.teacher(**inputs)

        assert outputs_student.logits.size() == outputs_teacher.logits.size()

        loss_logits = self.loss_function(
            F.log_softmax(outputs_student.logits / self.args.temperature, dim=-1),
            F.softmax(outputs_teacher.logits / self.args.temperature, dim=-1),
        ) * (self.args.temperature**2)

        loss = self.args.alpha * student_loss + (1.0 - self.args.alpha) * loss_logits
        return (loss, outputs_student) if return_outputs else loss


class Metrics:
    """
    Evaluates the model using ROUGE.
    """

    def __init__(self, tokenizer, name="rouge"):
        self.evaluator = evaluate.load(name)
        self.tokenizer = tokenizer

    def compute_metrics(self, eval_pred):
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
        results = self.evaluator.compute(predictions=predictions, references=labels)
        return results


def params_count(model):
    num_params = sum(p.numel() for p in model.parameters())
    return num_params / 1000000


def main():
    args = parse_arguments()

    print("Loading models")

    if args.dry_run:
        percent = 1
    else:
        percent = args.percent

    teacher_model, student_model, tokenizer = create_models(args)
    distilled_model_name = args.teacher_model_id + "-distilled"
    local_name = f"./{distilled_model_name.replace('/', '-')}"

    training_args = {
        "output_dir": f"{local_name}-output",  # Output directory for model checkpoints and logs
        "overwrite_output_dir": True,  # Overwrite the output directory if it exists
        "num_train_epochs": 3,
        "learning_rate": 0.0005,
        "seed": 42,
        "per_device_train_batch_size": 5,
        "per_device_eval_batch_size": 5,
        "gradient_accumulation_steps": 50,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.05,
        "save_strategy": "epoch",
        "save_total_limit": 2,
        "push_to_hub": False,
        "alpha": 0.5,
        "temperature": 4.0,
    }

    if args.evaluate:
        metrics = Metrics(tokenizer)
        training_args["evaluation_strategy"] = "steps"
        training_args["eval_steps"] = 30

    training_args = DistillationTrainingArguments(**training_args)

    print(f"Teacher model has {(params_count(teacher_model)):.2f}M parameters")
    print(f"Student model has {(params_count(student_model)):.2f}M parameters")

    dataset = BookSumDataset(
        args,
        "kmfoda/booksum",
        f"train[:{percent}%]",
        f"validation[:{percent}%]",
        tokenizer,
    )

    trainer_args = {
        "teacher_mode": teacher_model,
        "train_datase": dataset.train,
        "eval_datase": dataset.eval,
        "data_collator": DataCollatorForSeq2Seq(tokenizer),
        "tokenizer": tokenizer
        # optimizer=AdamW(student_model.parameters(), betas=(0.9, 0.999), eps=1e-08),
    }

    if args.evaluate:
        trainer_args["compute_metrics"] = metrics  # M1 Killer !! :)

    trainer = DistillationTrainer(
        student_model,
        training_args,
        teacher_model=teacher_model,
        train_dataset=dataset.train,
        eval_dataset=dataset.eval,
        data_collator=DataCollatorForSeq2Seq(tokenizer),
        tokenizer=tokenizer,
        # optimizer=AdamW(student_model.parameters(), betas=(0.9, 0.999), eps=1e-08),
        # compute_metrics=functools.partial(compute_metrics, tokenizer),  <== M1 Killer :)
    )

    mps.empty_cache()

    trainer.train()

    student_model.save_pretrained(local_name)
    tokenizer.save_pretrained(local_name)

    check(student_model, teacher_model, tokenizer)

    if not args.dry_run:
        # student_model.push_to_hub(f"tarekziade/{distilled_model_name}")
        # tokenizer.push_to_hub(f"tarekziade/{distilled_model_name}")
        pass

    print(f"Distillation complete. Distilled model saved as '{distilled_model_name}'")


if __name__ == "__main__":
    main()
