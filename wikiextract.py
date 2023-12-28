"""
    Creates a text/category dataset using Wikipedia.

    Explores the 40 root categories and their sub-categories to collect pages that are seen only on
    each root category. The produced dataset provides 200 pages per category.

    Author: Tarek Ziadé / Mozilla

"""
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import wikipediaapi
from datasets import Dataset, DatasetDict
import nltk
from nltk.tokenize import sent_tokenize
import pandas as pd
from tqdm import tqdm


_LIMIT_PER_CAT = 200
_ROOT_CATS = [
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


class WikiExtractor:
    def __init__(self):
        self.visited_page_ids = defaultdict(set)
        self.all_ids = set()
        self.client = wikipediaapi.Wikipedia("MediaWikiCat Project", "en", timeout=30)
        self.data_lock = Lock()
        self.pbar = None

    def fetch_pages_from_category(
        self,
        root_category_name,
        category_name,
        limit_per_category=_LIMIT_PER_CAT,
        depth=0,
        max_depth=10,
    ):
        if len(self.visited_page_ids[root_category_name]) >= limit_per_category:
            return []

        if depth > max_depth:  # Limit the recursion depth
            return []

        cat = self.client.page(category_name)
        pages = []

        # Fetch pages from the current category
        for c in cat.categorymembers.values():
            if (
                c.ns == wikipediaapi.Namespace.MAIN
                and c.pageid not in self.visited_page_ids
            ):
                if c.pageid in self.all_ids:
                    continue
                pages.append(c)

                with self.data_lock:  # Ensure thread-safe updates
                    self.visited_page_ids[root_category_name].add(c.pageid)
                    self.all_ids.add(c.pageid)

                if len(self.visited_page_ids[root_category_name]) >= limit_per_category:
                    break

        # Fetch pages from subcategories
        for subcat in cat.categorymembers.values():
            if subcat.ns == wikipediaapi.Namespace.CATEGORY:
                pages += self.fetch_pages_from_category(
                    root_category_name,
                    subcat.title,
                    limit_per_category,
                    depth + 1,
                    max_depth,
                )

        return pages

    def preprocess_content(self, text):
        sentences = sent_tokenize(text)[:5]
        return " ".join(sentences)

    def process_page(self, page):
        if page.summary:
            summary = self.preprocess_content(page.summary)
        else:
            summary = self.preprocess_content(page.text)

        summary = self.preprocess_content(summary)
        return {
            "title": page.title,
            "id": page.pageid,
            "summary": summary,
        }

    def process_category(self, category):
        category_data = []
        category = f"Category:{category}"
        pages = self.fetch_pages_from_category(category, category)

        for page in pages:
            data = self.process_page(page)
            data["category"] = category.removeprefix("Category:")
            category_data.append(data)
            if self.pbar is not None:
                self.pbar.update(1)

        return category_data

    def __call__(self):
        with tqdm(
            total=len(_ROOT_CATS) * _LIMIT_PER_CAT, desc="Processing Categories"
        ) as pbar:
            self.pbar = pbar
            with ThreadPoolExecutor(max_workers=15) as executor:
                future_to_category = {
                    executor.submit(self.process_category, category): category
                    for category in _ROOT_CATS
                }

                for future in as_completed(future_to_category):
                    category_data = future.result()
                    for item in category_data:
                        yield item


def main():
    nltk.download("punkt")
    extractor = WikiExtractor()
    pages = list(extractor())

    def gen():
        for page in pages:
            yield page

    dataset = Dataset.from_generator(gen)
    train_test_split = dataset.train_test_split(test_size=0.1)
    dataset_dict = DatasetDict(
        {"train": train_test_split["train"], "test": train_test_split["test"]}
    )

    dataset_dict.push_to_hub("tarekziade/wikipedia-topics")


if __name__ == "__main__":
    main()
