import requests
from bs4 import BeautifulSoup
import nltk
import sys
import time

import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

from mwcat.utils import id_to_category


def get_most_probable_intent(categories):
    category_to_intent = {
        "Academic_disciplines": "Professional or Educational Development",
        "Business": "Professional or Educational Development",
        "Communication": "Social Interaction and Networking",
        "Concepts": "Information Seeking",
        "Culture": "Entertainment and Leisure",
        "Economy": "Information Seeking",
        "Education": "Professional or Educational Development",
        "Energy": "Information Seeking",
        "Engineering": "Professional or Educational Development",
        "Entertainment": "Entertainment and Leisure",
        "Entities": "Information Seeking",
        "Ethics": "Professional or Educational Development",
        "Food_and_drink": "Entertainment and Leisure",
        "Geography": "Information Seeking",
        "Government": "Information Seeking",
        "Health": "Professional or Educational Development",
        "History": "Information Seeking",
        "Human_behavior": "Social Interaction and Networking",
        "Humanities": "Professional or Educational Development",
        "Information": "Information Seeking",
        "Internet": "Information Seeking",
        "Knowledge": "Information Seeking",
        "Language": "Professional or Educational Development",
        "Law": "Information Seeking",
        "Life": "Information Seeking",
        "Lists": "Information Seeking",
        "Mass_media": "Entertainment and Leisure",
        "Mathematics": "Professional or Educational Development",
        "Military": "Information Seeking",
        "Nature": "Information Seeking",
        "People": "Social Interaction and Networking",
        "Philosophy": "Professional or Educational Development",
        "Politics": "Information Seeking",
        "Religion": "Social Interaction and Networking",
        "Science": "Professional or Educational Development",
        "Society": "Social Interaction and Networking",
        "Sports": "Entertainment and Leisure",
        "Technology": "Professional or Educational Development",
        "Time": "Information Seeking",
        "Universe": "Information Seeking",
    }

    # Count the occurrence of each intent
    intent_count = {}
    for category in categories:
        intent = category_to_intent.get(category, "Other")
        intent_count[intent] = intent_count.get(intent, 0) + 1

    # Determine the most probable intent
    most_probable_intent = max(intent_count, key=intent_count.get)
    return most_probable_intent


def load_model():
    model = DistilBertForSequenceClassification.from_pretrained(
        "./fine_tuned_distilbert"
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

most_probable_intent = get_most_probable_intent(sorted_categories)

print(f"The most likely categories are: {sorted_categories}")
print(f"The most probable user intent is: {most_probable_intent}")
