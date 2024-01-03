import nltk
from nltk.tokenize import sent_tokenize

nltk.download("punkt")


CATEGORIES = (
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
    "Mass_media",
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
)

category_to_id = {cat: i for i, cat in enumerate(CATEGORIES)}
id_to_category = {i: cat for i, cat in enumerate(CATEGORIES)}


NUM_CATEGORIES = len(CATEGORIES)


def sentences(text, num_sentences=5):
    sentences = sent_tokenize(text)[:num_sentences]
    return " ".join(sentences)


def tokenize_and_format(tokenizer, use_summary, examples):
    def _convert(title, summary):
        if use_summary:
            return f"cat: {title}::{summary}".lower()
        return f"cat: {title}"

    inputs = [
        _convert(title, summary)
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
        vec = [0] * NUM_CATEGORIES
        for cat in cats:
            cat = cat.replace(" ", "_")
            vec[category_to_id[cat]] = 1
        labels.append(vec)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs
