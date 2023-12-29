import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from nltk.tokenize import sent_tokenize
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification


def load_model():
    model = DistilBertForSequenceClassification.from_pretrained(
        "./fine_tuned_distilbert"
    )
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    return tokenizer, model


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


summary = """\
Biological organisation is the organisation of complex biological structures and systems that define life using a reductionistic approach.[1] The traditional hierarchy, as detailed below, extends from atoms to biospheres. The higher levels of this scheme are often referred to as an ecological organisation concept, or as the field, hierarchical ecology.

Each level in the hierarchy represents an increase in organisational complexity, with each "object" being primarily composed of the previous level's basic unit.[2] The basic principle behind the organisation is the concept of emergence—the properties and functions found at a hierarchical level are not present and irrelevant at the lower levels.
"""

sentences = sent_tokenize(summary)[:5]
summary = " ".join(sentences)
print(summary)


def classify(input_text, tokenizer, model):
    # Tokenize the input text
    inputs = tokenizer(input_text, padding=True, truncation=True, return_tensors="pt")

    # Generate the output
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    probabilities = torch.nn.functional.softmax(logits, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1).item()
    category = categories[predicted_class]
    sorted_probabilities, sorted_indices = torch.sort(probabilities, descending=True)
    sorted_categories = [categories[index] for index in sorted_indices[0]][:5]

    print(f"Predicted category: {category}")
    print("Class probabilities sorted from highest to lowest:")
    for category, probability in zip(sorted_categories, sorted_probabilities[0]):
        print(f"{category}: {probability.item()}")


def main():
    input_text = (
        "cat: Biological organisation::" + summary
    )  # Update this with your input

    tokenizer, model = load_model()

    classify(input_text, tokenizer, model)


if __name__ == "__main__":
    main()
