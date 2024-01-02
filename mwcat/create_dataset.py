"""
    Creates a text/category dataset using Wikipedia.

    Explores the 40 root categories and their sub-categories to collect pages.
    The produced dataset returns 40k pages.

    Author: Tarek Ziadé / Mozilla

"""
import queue
import threading
import sqlite3
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import wikipediaapi
from datasets import Dataset, DatasetDict
import pandas as pd
from tqdm import tqdm

from mwcat.utils import CATEGORIES, NUM_CATEGORIES, sentences


_LIMIT_PER_CAT = 2000


class WikiExtractor:
    def __init__(self):
        self.visited_page_ids = defaultdict(set)
        self.ids_count = defaultdict(int)
        self.client = wikipediaapi.Wikipedia("MediaWikiCat Project", "en", timeout=30)
        self.data_lock = Lock()
        self.pbar = None
        self.db_connections = threading.local()
        self.create_tables()
        self.task_queue = queue.Queue()
        self.workers = []

    def get_db_connection(self):
        if not hasattr(self.db_connections, "conn"):
            self.db_connections.conn = sqlite3.connect("pages.db")
        return self.db_connections.conn

    def start_workers(self, num_workers):
        for _ in range(num_workers):
            thread = threading.Thread(target=self.worker)
            thread.start()
            self.workers.append(thread)

    def stop_workers(self):
        for _ in range(len(self.workers)):
            self.task_queue.put(None)
        for worker in self.workers:
            worker.join()

    def worker(self):
        while True:
            task = self.task_queue.get()
            if task is None:  # Sentinel to indicate end of tasks
                self.task_queue.task_done()
                break
            self.process_category(*task)
            self.task_queue.task_done()

    def create_tables(self):
        create_table_query = """
        CREATE TABLE IF NOT EXISTS pages (
            id INTEGER PRIMARY KEY,
            title TEXT,
            summary TEXT,
            text TEXT,
            categories TEXT
        );
        """
        conn = self.get_db_connection()
        conn.execute(create_table_query)
        conn.commit()

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
        return sentences(text)

    def process_page(self, category_name, page):
        page_id = str(page.pageid)
        category_name = category_name.removeprefix("Category:")

        with self.data_lock:
            conn = self.get_db_connection()
            page_query = (
                "SELECT title, summary, text, categories FROM pages WHERE id = ?"
            )
            cursor = conn.execute(page_query, (page_id,))
            row = cursor.fetchone()
            if row:
                title = row[0]
                summary = row[1]
                text = row[2]
                existing_categories = set(row[3].split(","))
                existing_categories.add(category_name)
                categories = existing_categories

                update_query = "UPDATE pages SET categories = ?, text = ? WHERE id = ?"
                conn.execute(
                    update_query, (",".join(existing_categories), text, page_id)
                )
            else:
                # Insert new page
                title = page.title
                text = page.text
                summary = self.extract_summary(page)
                insert_query = "INSERT INTO pages (id, title, text, summary, categories) VALUES (?, ?, ?, ?, ?)"
                conn.execute(
                    insert_query, (page_id, title, text, summary, category_name)
                )
                categories = [category_name]

            conn.commit()

            return {
                "id": page_id,
                "summary": summary,
                "title": title,
                "categories": categories,
                "text": text,
            }

    def __call__(self):
        with tqdm(
            total=NUM_CATEGORIES * _LIMIT_PER_CAT, desc="Extracting data from wikipedia"
        ) as pbar:
            self.pbar = pbar
            self.start_workers(15)

            for category in CATEGORIES:
                category = f"Category:{category}"
                self.task_queue.put((category, category))

            self.task_queue.join()  # Wait for all tasks to be completed
            self.stop_workers()

        with tqdm(
            total=NUM_CATEGORIES * _LIMIT_PER_CAT, desc="Creating dataset"
        ) as pbar:
            select_query = "SELECT id, title, summary, text, categories FROM pages"
            cursor = self.get_db_connection().execute(select_query)

            while True:
                row = cursor.fetchone()
                if row is None:
                    break

                page_data = {
                    "id": row[0],
                    "title": row[1],
                    "summary": row[2],
                    "text": row[3],
                    "categories": set(row[4].split(",")),
                }
                yield page_data
                pbar.update(1)

        if hasattr(self.db_connections, "conn"):
            self.db_connections.conn.close()


def main():
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
