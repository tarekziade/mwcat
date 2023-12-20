import sys
import re
from collections import defaultdict
import csv
import time

import mwxml
import mwparserfromhell
import nltk
from nltk.tokenize import sent_tokenize


nltk.download("punkt")

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


def extract_categories(text):
    pattern = r"\[\[Category:(.*?)\]\]"
    return re.findall(pattern, text)


found_cats = defaultdict(int)


def clean_cat(cat):
    cat = cat.replace("|", "").replace(",", "")

    return cat.strip()


def clean_text(text):
    text = re.sub(r"^Category:.*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\|", " ", text)
    return re.sub(r"\s{2,}", " ", text).strip()


pages = 0
hits = 0
start_time = time.time()

with open("output.csv", mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["Page ID", "Page Title", "Categories", "Page Text"])
    dump = mwxml.Dump.from_file(open(sys.argv[-1]))

    for page in dump:
        if pages % 1000 == 0:
            file.flush()

        current_time = time.time()
        elapsed_time = current_time - start_time
        pages_per_second = pages / elapsed_time if elapsed_time > 0 else 0
        sys.stdout.write(
            f"{pages} processed, {hits} hits, {pages_per_second:.2f} pages/second    \r"
        )
        sys.stdout.flush()
        pages += 1
        cats = []

        # only looking at the most recent revision
        for revision in page:
            text = clean_text(mwparserfromhell.parse(revision.text).strip_code())
            text = sent_tokenize(text)[:5]
            try:
                cats = extract_categories(revision.text)
            except Exception:
                break
            if cats != []:
                cats = [clean_cat(cat) for cat in cats]

                # only extracting top categories
                # XXX how can I find the parent category from each cat. I need to build the tree...

                cats = [cat for cat in cats if cat in categories]

                if cats != []:
                    hits += 1
                    for cat in cats:
                        found_cats[cat] += 1
                    writer.writerow(
                        [page.id, page.title, ",".join(cats), " ".join(text)]
                    )

            break


for name, num_articles in found_cats.items():
    if num_articles > 1:
        print(f"Category `{name}` found {num_articles} times")
