import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import re
import spacy
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from nltk.tokenize import word_tokenize
import torch.nn.functional as F
import matplotlib.pyplot as plt
from nltk.translate.bleu_score import corpus_bleu

from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics.pairwise import cosine_similarity

from model import Encoder, Decoder, Discriminator
import helper

SIZE_OF_DATASET = 2

# Hyperparameters
BATCH_SIZE = 32
NUM_LSTM_LAYERS = 3
LEARNING_RATE = 0.0003
LANG_DIM = 32

# Encoder
ENCODER_INPUT_DIM = 768 
ENCODER_HIDDEN_DIM = 300
ENCODER_OUTPUT_DIM = 100 

# Decoder
DECODER_OUTPUT_DIM = ENCODER_INPUT_DIM
DECODER_LATENT_DIM = ENCODER_OUTPUT_DIM
DECODER_HIIDEN_DIM = 300

NUM_LANGS = 2
MAX_SEQ_LEN = 50

# Define save paths for the models
ENCODER_SAVE_PATH = "./Saved Models/encoder_model.pth"
DECODER_SAVE_PATH = "./Saved Models/decoder_model.pth"
DISCRIMINATOR_SAVE_PATH = "./Saved Models/discriminator_model.pth"

# Load French Monolingual Dataset
sentences = []

with open('./Dataset/Monolingual/fra_news_2023_1M-sentences.txt', 'r', encoding='utf-8') as file:
    for line in file:
        if not line:
            continue

        parts = line.split('\t', 1)
        if len(parts) > 1:
            sentence = parts[1]
            sentences.append(sentence)

french_df = pd.DataFrame(sentences, columns=['Sentence'])
french_df

# Load English Monolingual Dataset
with open('./Dataset/Monolingual/news-commentary-v18.txt', 'r', encoding='utf-8') as file:
    sentences = file.readlines()

sentences = [sentence.strip() for sentence in sentences]
english_df = pd.DataFrame(sentences, columns=['Sentence'])
english_df

# Remove empty sentences if any
english_df = english_df[english_df['Sentence'].notna() & (english_df['Sentence'] != '')]
french_df = french_df[french_df['Sentence'].notna() & (french_df['Sentence'] != '')]

lang_idx_mapping = {
    'english': 0,
    'french': 1,
}

french_df = french_df[:SIZE_OF_DATASET]
english_df = english_df[:SIZE_OF_DATASET]

english_df.loc[:, 'Sentence'] = english_df['Sentence'].apply(lambda x: helper.preprocess_text(x))
french_df.loc[:, 'Sentence'] = french_df['Sentence'].apply(lambda x: helper.preprocess_text(x))

nlp_en = spacy.load('en_core_web_lg')
nlp_fr = spacy.load('fr_core_news_lg')

# Tokenize
english_df['Tokens'] = english_df['Sentence'].apply(lambda x: helper.tokenize_with_spacy(x, nlp_en))
french_df['Tokens'] = french_df['Sentence'].apply(lambda x: helper.tokenize_with_spacy(x, nlp_fr))

# Drop sentences longer than 50 words
english_df = english_df[english_df['Tokens'].apply(len) <= MAX_SEQ_LEN]
french_df = french_df[french_df['Tokens'].apply(len) <= MAX_SEQ_LEN]

if len(english_df) != len(french_df):
    min_length = min(len(english_df), len(french_df))
    english_df = english_df.iloc[:min_length]
    french_df = french_df.iloc[:min_length]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the pre-trained BERT tokenizer and model
# Load the fast version of the tokenizer
tokenizer_en = AutoTokenizer.from_pretrained('bert-base-uncased', use_fast=True)
bert_model_en = AutoModel.from_pretrained('bert-base-uncased')
bert_model_en = bert_model_en.to(device)
vocab_embeddings_en = bert_model_en.embeddings.word_embeddings.weight  # shape: (vocab_size, 768)

tokenizer_fr = AutoTokenizer.from_pretrained('camembert-base', use_fast=True)
bert_model_fr = AutoModel.from_pretrained('camembert-base')
bert_model_fr = bert_model_fr.to(device)
vocab_embeddings_fr = bert_model_fr.embeddings.word_embeddings.weight  # shape: (vocab_size, 768)

english_embeddings = helper.generate_embeddings(english_df, tokenizer_en, bert_model_en)
english_embeddings[0].shape

helper.generate_padded_embeddings(english_df, english_embeddings)

french_embeddings = helper.generate_embeddings(french_df, tokenizer_fr, bert_model_fr)
french_embeddings[0].shape

helper.generate_padded_embeddings(french_df, french_embeddings)

# x and y are same because of auto-encoding
english_dataset = TensorDataset(
    torch.stack(english_df['Embeddings'].tolist()),
    torch.stack(english_df['Embeddings'].tolist()),
    torch.tensor([lang_idx_mapping['english']] * len(english_df))
)
english_loader = DataLoader(
    english_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

french_dataset = TensorDataset(
    torch.stack(french_df['Embeddings'].tolist()),
    torch.stack(french_df['Embeddings'].tolist()),
    torch.tensor([lang_idx_mapping['french']] * len(french_df))
)
french_loader = DataLoader(
    french_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

# Combine both datasets
combined_dataset = torch.utils.data.ConcatDataset([english_dataset, french_dataset])
combined_loader = DataLoader(
    combined_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

# Initialize models
encoder_model = Encoder(
    ENCODER_INPUT_DIM,
    ENCODER_HIDDEN_DIM, 
    ENCODER_OUTPUT_DIM,
    NUM_LSTM_LAYERS,
)

decoder_model = Decoder(
    DECODER_LATENT_DIM,
    DECODER_HIIDEN_DIM,
    DECODER_OUTPUT_DIM,
    NUM_LSTM_LAYERS,
)

# Set the models to train mode
encoder_model.train()
decoder_model.train()

# Initialize optimizer (Adam)
enc_dec_optimizer = optim.Adam(
    list(encoder_model.parameters()) + list(decoder_model.parameters()), 
    lr=LEARNING_RATE,
    betas=(0.5, 0.999),
)

# Generate text for english
helper.generate_text_using_dae(english_df, tokenizer_en, vocab_embeddings_en, lang_idx_mapping['english'], encoder_model, decoder_model)

# Generate text for french
helper.generate_text_using_dae(french_df, tokenizer_fr, vocab_embeddings_fr, lang_idx_mapping['french'], encoder_model, decoder_model)

source_loader = english_loader
target_loader = french_loader
source_vocab = vocab_embeddings_en
target_vocab = vocab_embeddings_fr

source_to_target_mapping, target_to_source_mapping = helper.generate_word_by_word_mapping(source_loader, target_loader, source_vocab, target_vocab)

disc_model = Discriminator(ENCODER_OUTPUT_DIM)
disc_optimizer = optim.Adam(disc_model.parameters(), lr=LEARNING_RATE)

helper.train_cross_domain_non_parallel(
    encoder_model,
    decoder_model,
    disc_model,
    enc_dec_optimizer,
    disc_optimizer,
    source_loader,
    target_loader,
    source_vocab,
    target_vocab,
    source_to_target_mapping,
    target_to_source_mapping,
    clip=1.0,
)

# Save models after training
helper.save_model(encoder_model, enc_dec_optimizer, ENCODER_SAVE_PATH)
helper.save_model(decoder_model, enc_dec_optimizer, DECODER_SAVE_PATH)
helper.save_model(disc_model, disc_optimizer, DISCRIMINATOR_SAVE_PATH)

# Example DataFrame with source domain embeddings
source_df = english_df 

# Source and target language indices
source_lang_idx = lang_idx_mapping['english']  
target_lang_idx = lang_idx_mapping['french']  

# Generate cross-domain sentences from English to French
helper.generate_cross_domain_text(
    source_df,
    encoder_model,
    decoder_model,
    tokenizer_en,
    vocab_embeddings_en,
    source_lang_idx,
    target_lang_idx
)
