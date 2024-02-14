import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import T5TokenizerFast, AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset
from tqdm import tqdm
import time

from mwcat.utils import id_to_category


def summarize_text(text_to_summarize):
    print(text_to_summarize)

    # model_name = "t5-base"
    # model_name = "t5-small"
    # model_name = "tarekziade/wikipedia-summaries-t5-small"
    # "google/t5-efficient-tiny",
    # "tarekziade/wikipedia-summaries-t5-efficient-tiny"
    # model_name = "pszemraj/long-t5-tglobal-base-16384-book-summary"
    # model_name = "./distilled-long-t5-tglobal-base-16384-book-summary"
    # model_name = "./cnicu-t5-small-booksum-distilled"
    # model_name = "./distilled_t5_model"
    # tokenizer_name = "pszemraj/long-t5-tglobal-base-16384-book-summary"
    model_name = ""
    tokenizer_name = model_name

    tokenizer = T5TokenizerFast.from_pretrained(tokenizer_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.eval()
    # model.to("mps")
    # torch.compile(model)

    # Prepare the input text
    input_ids = tokenizer.encode(
        "summary: " + text_to_summarize,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=512,  # len(text_to_summarize),
        add_special_tokens=False,
    )

    start = time.time()
    # summary_ids = model.generate(
    #    input_ids, output_scores=True, return_dict_in_generate=True, max_length=120
    # )

    # summary = tokenizer.batch_decode(
    #    summary_ids.sequences,
    #    skip_special_tokens=True,
    #    remove_invalid_values=True,
    # )
    summary = tokenizer.decode(
        model.generate(input_ids, max_length=120)[0],
        skip_special_tokens=True,
        remove_invalid_values=True,
    )
    print(f"Took {time.time() - start:.2f} seconds")
    return summary


def load_model():
    print("Loading models")
    model = DistilBertForSequenceClassification.from_pretrained(
        "./fine_tuned_distilbert"
    )
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    return tokenizer, model


def dataset_iterator(dataset_name, split="train"):
    dataset = load_dataset(dataset_name, split=split)
    import pdb

    pdb.set_trace()
    return len(dataset), iter(dataset)


def classify(input_text, tokenizer, model):
    inputs = tokenizer(
        input_text,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    probabilities = torch.nn.functional.softmax(logits, dim=1)
    # predicted_class = torch.argmax(probabilities, dim=1).item()
    # category = id_to_category[predicted_class]
    _, sorted_indices = torch.sort(probabilities, descending=True)
    sorted_categories = [id_to_category[int(index)] for index in sorted_indices[0]][:3]
    return sorted_categories


def main():
    input_text = "cat: {title}::{summary}".lower()
    tokenizer, model = load_model()

    goods = 0
    bads = 0
    total = 0
    dataset_len, iter = dataset_iterator("tarekziade/wikipedia-topics", split="test")

    with tqdm(total=dataset_len, desc="Testing pages") as pbar:
        for row in iter:
            total += 1
            original_categories = [cat.replace(" ", "_") for cat in row["categories"]]
            input_text = f"cat: {row['title']}::{row['summary']}".lower()
            sorted_categories = classify(input_text, tokenizer, model)
            bad = False
            for cat in original_categories:
                if cat not in sorted_categories:
                    bad = True

            if bad:
                print(f"Bad result on {row['title']}")
                print(f"Original categories {original_categories}")
                print(f"Predicted categories: {','.join(sorted_categories)}")
                print()
                bads += 1
            else:
                goods += 1
            pbar.update(1)

    print(f"Total: {total}, good: {goods}, bad: {bads}, ratio: {bads/goods/100}")


if __name__ == "__main__":
    text = open("sandman.txt").read().strip()
    summary = summarize_text(text.strip())
    print("Summary:", summary)

    # main()
