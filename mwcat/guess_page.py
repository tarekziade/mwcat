import requests
from bs4 import BeautifulSoup
import nltk
import sys
import time

import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

from mwcat.utils import id_to_category


category_to_intents_map = {
    "Information": [
        "Academic_disciplines",
        "Concepts",
        "Information",
        "Knowledge",
        "Science",
        "Education",
        "Geography",
        "History",
        "Mathematics",
        "Engineering",
        "Technology",
        "Health",
        "Law",
        "Economy",
        "Government",
    ],
    "Communication": [
        "Communication",
        "Language",
        "Human_behavior",
        "Business",
        "Society",
        "Politics",
        "Ethics",
        "Philosophy",
    ],
    "Social Networking": [
        "Culture",
        "Entertainment",
        "Humanities",
        "People",
        "Society",
        "Religion",
        "Ethics",
        "Philosophy",
        "Life",
        "Entities",
    ],
    "Entertainment": [
        "Entertainment",
        "Food_and_drink",
        "Sports",
        "Mass_media",
        "Music",
        "Culture",
        "Arts",
        "History",
        "Human_behavior",
    ],
    "Online Media": [
        "Internet",
        "Technology",
        "Mass_media",
        "Engineering",
        "Computers",
        "Software",
        "Information",
        "Business",
        "Economy",
        "Education",
    ],
}


def get_most_likely_intent(categories):
    intents_count = {
        "Information": 0,
        "Communication": 0,
        "Social Networking": 0,
        "Entertainment": 0,
        "Online Media": 0,
    }

    for category in categories:
        for intent, mapped_categories in category_to_intents_map.items():
            if category in mapped_categories:
                intents_count[intent] += 1

    print(intents_count)
    most_likely_intent = max(intents_count, key=intents_count.get)
    return most_likely_intent


def load_model():
    model = DistilBertForSequenceClassification.from_pretrained(
        "tarekziade/wikipedia-topics-tinybert"
    )
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    return tokenizer, model


def extract_webpage_content(url):
    response = requests.get(url)
    # Check if the request was successful
    if response.status_code != 200:
        return "Failed to retrieve the webpage"
    soup = BeautifulSoup(response.content, "html.parser")
    title = soup.title.string if soup.title else "No title found"
    paragraphs = soup.find_all("p")
    content = " ".join([para.get_text() for para in paragraphs])
    sentences = nltk.sent_tokenize(content)
    first_five_sentences = " ".join(sentences[:10])
    return {"title": title, "summary": first_five_sentences.strip()}


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
    _, sorted_indices = torch.sort(probabilities, descending=True)
    sorted_categories = [id_to_category[int(index)] for index in sorted_indices[0]][:3]
    return sorted_categories


result = extract_webpage_content(sys.argv[-1])
print(f"Working on page `{result['title']}`")
input_text = f"cat: {result['title']}::{result['summary']}".lower()
tokenizer, model = load_model()
start = time.time()
sorted_categories = classify(input_text, tokenizer, model)
print(f"Classified in {time.time() - start:.2f} seconds")

most_probable_intent = get_most_likely_intent(sorted_categories)

print(f"The most likely categories are: {sorted_categories}")
print(f"The most probable user intent is: {most_probable_intent}")
