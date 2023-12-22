# mwcat

Generates a Dataset containing Wikipedia (english) pages with categories

```mermaid
graph LR
    Z[Wikipedia Dumps] --> A
    Z[Wikipedia Dumps] --> F
     
    A[articles.xml] -->|Processing| B(Extraction Process)
    B -->|page_id| C[page_id]
    B -->|title| D[title]
    B -->|text| E[text]
    F[categories.sql] -->|Import| G[MySQL DB]
    G -->|categories| H[categories]

    C --> I(Dataset)
    D --> I(Dataset)
    E --> I(Dataset)
    H --> I(Dataset)
   
```


Fields:
- page_id
- title
- categories: root category from each category found for the page
- text: 5 first sentences of the page (cleaned)

This dataset can be used to train a text classification model.

It uses Wikipedia dumps and runs a dockerized mysql server to query for
the page categories hierarchy.

Wikipedia has 40 top categories.

Requirements: Python 3, a good internet connection and a lot of time.

Run `make install` and then `make extract`

**WARNING**: This dataset will download over 100GiB of data from Wikipedia (once).
