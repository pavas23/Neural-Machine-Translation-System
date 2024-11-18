### Download Dataset

- From this link: https://data.statmt.org/news-commentary/v18.1/ download parallel datasets for english, french and german languages.

- While sourcing the monlingual datasets keep in mind that they should be from different domains for each language, as this website has parallel sentences language wise if they are extracted from the same domain.

- Download Spacy libraries for tokenizing

python3.9 -m pip install sapcy

python3.9 -m spacy download en_core_web_lg

python3.9 -m spacy download fr_core_news_lg

python3.9 -m spacy download de_core_news_lg


- The CLI tool has three main features described below:
    - The CLI tool can train a new translation model using two independent monolingual
    corpora and save the trained encoder and decoder models.

    - It can translate text or a single sentence between the source and target languages
    using the trained encoder and decoder models as input.

    - The tool can evaluate the trained model using parallel datasets and display metrics
    such as BLEU and TER scores.
