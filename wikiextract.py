"""
    Creates a text/category dataset using Wikipedia.

    Explores the 40 root categories and their sub-categories to collect pages.
    The produced dataset returns 40k pages.

    Author: Tarek Ziadé / Mozilla

"""
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import shelve

import wikipediaapi
from datasets import Dataset, DatasetDict
import nltk
from nltk.tokenize import sent_tokenize
import pandas as pd
from tqdm import tqdm


_LIMIT_PER_CAT = 1000
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
]


class WikiExtractor:
    def __init__(self):
        self.visited_page_ids = defaultdict(set)
        self.ids_count = defaultdict(int)
        self.client = wikipediaapi.Wikipedia("MediaWikiCat Project", "en", timeout=30)
        self.data_lock = Lock()
        self.pbar = None
        self.page_cache = shelve.open("pages.shelve", writeback=True)

    def process_category(
        self,
        root_category_name,
        category_name,
        limit_per_category=_LIMIT_PER_CAT,
        depth=0,
        max_depth=10,
    ):
        if len(self.visited_page_ids[root_category_name]) >= limit_per_category:
            return

        if depth > max_depth:  # Limit the recursion depth
            return

        cat = self.client.page(category_name)

        # TODO: feed a queue with pages to collect so this loop is not blocking
        for c in cat.categorymembers.values():
            if (
                c.ns == wikipediaapi.Namespace.MAIN
                and c.pageid not in self.visited_page_ids
            ):
                self.process_page(root_category_name, c)

                with self.data_lock:  # Ensure thread-safe updates
                    self.visited_page_ids[root_category_name].add(c.pageid)
                    self.ids_count[c.pageid] += 1

                if self.pbar is not None:
                    self.pbar.update(1)

                if len(self.visited_page_ids[root_category_name]) >= limit_per_category:
                    break

        # Fetch pages from subcategories
        for subcat in cat.categorymembers.values():
            if subcat.ns == wikipediaapi.Namespace.CATEGORY:
                self.process_category(
                    root_category_name,
                    subcat.title,
                    limit_per_category,
                    depth + 1,
                    max_depth,
                )

    def extract_summary(self, page):
        if page.summary:
            text = page.summary
        else:
            text = page.text
        sentences = sent_tokenize(text)[:5]
        return " ".join(sentences)

    def process_page(self, category_name, page):
        page_id = str(page.pageid)
        category_name = category_name.removeprefix("Category:")

        with self.data_lock:
            page_data = self.page_cache.get(
                page_id,
                {
                    "title": page.title,
                    "id": page.pageid,
                    "summary": self.extract_summary(page),
                    "categories": set(),
                },
            )
            page_data["categories"].add(category_name)
            self.page_cache[page_id] = page_data
            self.page_cache.sync()
            return page_data

    def __call__(self):
        with tqdm(
            total=len(_ROOT_CATS) * _LIMIT_PER_CAT, desc="Processing Categories"
        ) as pbar:
            self.pbar = pbar
            with ThreadPoolExecutor(max_workers=15) as executor:
                for category in _ROOT_CATS:
                    category = f"Category:{category}"
                    executor.submit(self.process_category, category, category)
                executor.shutdown(wait=True)

            for page_data in self.page_cache.values():
                yield page_data

        self.page_cache.close()


def main():
    nltk.download("punkt")
    extractor = WikiExtractor()

    # TODO: use the iterator to stream to Dataset avoid loading all the pages in memory
    pages = list(extractor())

    def gen():
        for page in pages:
            yield page

    dataset = Dataset.from_generator(gen)
    train_test_split = dataset.train_test_split(test_size=0.1)
    dataset_dict = DatasetDict(
        {"train": train_test_split["train"], "test": train_test_split["test"]}
    )

    print(dataset_dict)
    dataset_dict.push_to_hub("tarekziade/wikipedia-topics")


if __name__ == "__main__":
    main()
