### Download Monolingual Dataset

- From this link: https://data.statmt.org/news-commentary/v18.1/ download monolingual datasets for english, french and german languages.

- While sourcing the monlingual datasets keep in mind that they should be from different domains for each language, as this website has parallel sentences language wise if they are extracted from the same domain.

- Download Spacy libraries for tokenizing

python3.9 -m pip install sapcy

python3.9 -m spacy download en_core_web_sm

python3.9 -m spacy download fr_core_news_sm
