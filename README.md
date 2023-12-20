# mwcat

Generates a dataset containing Wikipedia (english) pages with:

- id
- title
- categories: top-most category from each category found on the page
- text: 5 first sentences of the page (cleaned)

This dataset can be used to train a text classification model.

It uses Wikipedia dumps and runs a dockerized mysql server to query for
the page categories hierarchy.

Wikipedia has 40 top categories.

Requirements: Python 3 and a good internet connection.

Run `make install` and then `make extract`

**WARNING**: This dataset will download over 100GiB of data from Wikipedia (once).
