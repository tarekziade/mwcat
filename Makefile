
bin/python:
	python3 -m venv .

.PHONY: install
install: bin/python
	bin/pip install -r requirements.txt

data/enwiki-latest-pages-articles.xml:
	mkdir -p data
	wget https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2
	bzip2 -d enwiki-latest-pages-articles.xml
	mv enwiki-latest-pages-articles.xml data/


data/enwiki-latest-page.sql:
	mkdir -p data
	wget https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-page.sql.gz
	gzip -d enwiki-latest-page.sql.gz
	mv enwiki-latest-page.sql data/

data/enwiki-latest-categorylinks.sql:
	mkdir -p data
	wget https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-categorylinks.sql.gz
	gzip -d enwiki-latest-categorylinks.sql.gz
	mv enwiki-latest-categorylinks.sql data/

.PHONY: extract
extract: data/enwiki-latest-pages-articles.xml data/enwiki-latest-categorylinks.sql data/enwiki-latest-page.sql
	bin/python extract.py data/enwiki-latest-pages-articles.xml


