import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from nltk.tokenize import sent_tokenize
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from datasets import load_dataset
from tqdm import tqdm


def load_model():
    print("Loading models")
    model = DistilBertForSequenceClassification.from_pretrained(
        "./fine_tuned_distilbert"
    )
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    return tokenizer, model


def dataset_iterator(dataset_name, split="train"):
    dataset = load_dataset(dataset_name, split=split)
    return iter(dataset)


categories = [
    "Academic disciplines",
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
    "Food and drink",
    "Geography",
    "Government",
    "Health",
    "History",
    "Human behavior",
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
categories = [cat.replace(" ", "_") for cat in categories]

id_to_category = {i: cat for i, cat in enumerate(categories)}


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
    predicted_class = torch.argmax(probabilities, dim=1).item()
    category = id_to_category[predicted_class]
    sorted_probabilities, sorted_indices = torch.sort(probabilities, descending=True)
    sorted_categories = [id_to_category[int(index)] for index in sorted_indices[0]][:3]

    # print("Class probabilities sorted from highest to lowest:")
    # for category, probability in zip(sorted_categories, sorted_probabilities[0]):
    #    print(f"{category}: {probability.item()}")
    return sorted_categories


def main():
    input_text = "cat: {title}::{summary}".lower()
    tokenizer, model = load_model()

    goods = 0
    bads = 0
    total = 0
    with tqdm(total=763, desc="Testing pages") as pbar:
        for row in dataset_iterator("tarekziade/wikipedia-topics", split="test"):
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
    main()
