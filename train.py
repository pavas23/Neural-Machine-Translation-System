import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import re
import spacy
import random
import numpy as np
import os

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
CLIP = 1.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer_en = AutoTokenizer.from_pretrained('bert-base-uncased', use_fast=True)
bert_model_en = AutoModel.from_pretrained('bert-base-uncased')
bert_model_en = bert_model_en.to(device)
vocab_embeddings_en = bert_model_en.embeddings.word_embeddings.weight  # shape: (vocab_size, 768)

tokenizer_fr = AutoTokenizer.from_pretrained('camembert-base', use_fast=True)
bert_model_fr = AutoModel.from_pretrained('camembert-base')
bert_model_fr = bert_model_fr.to(device)
vocab_embeddings_fr = bert_model_fr.embeddings.word_embeddings.weight  # shape: (vocab_size, 768)

tokenizer_de = AutoTokenizer.from_pretrained('bert-base-german-cased', use_fast=True)
bert_model_de = AutoModel.from_pretrained('bert-base-german-cased')
bert_model_de = bert_model_de.to(device)
vocab_embeddings_de = bert_model_de.embeddings.word_embeddings.weight  # shape: (vocab_size, 768)

nlp_en = spacy.load('en_core_web_lg')
nlp_fr = spacy.load('fr_core_news_lg')
nlp_de = spacy.load('de_core_news_lg')

lang_idx_mapping = {
    'english': 0,
    'french': 1,
    'german': 2,
}

def train_cross_domain_autoencoder(
    lang1_name,
    lang2_name,
    lang1_data_path,
    lang2_data_path,
    encoder_save_path,
    decoder_save_path,
    discriminator_save_path
):
    # Load datasets
    def load_dataset(file_path, size):
        with open(file_path, 'r', encoding='utf-8') as file:
            sentences = file.readlines()
        sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
        df = pd.DataFrame(sentences[:size], columns=['Sentence'])
        return df

    lang1_df = load_dataset(lang1_data_path, SIZE_OF_DATASET)
    lang2_df = load_dataset(lang2_data_path, SIZE_OF_DATASET)

    # Pre-defined models and tokenizers
    lang_settings = {
        'en': {
            'nlp': nlp_en,
            'tokenizer': tokenizer_en,
            'bert_model': bert_model_en,
            'vocab_embeddings': vocab_embeddings_en,
            'index': 0
        },
        'fr': {
            'nlp': nlp_fr,
            'tokenizer': tokenizer_fr,
            'bert_model': bert_model_fr,
            'vocab_embeddings': vocab_embeddings_fr,
            'index': 1
        },
        'de': {
            'nlp': nlp_de,
            'tokenizer': tokenizer_de,
            'bert_model': bert_model_de,
            'vocab_embeddings': vocab_embeddings_de,
            'index': 2
        },
    }

    if lang1_name not in lang_settings or lang2_name not in lang_settings:
        raise ValueError("Unsupported language. Supported languages are 'en', 'fr', 'de'.")
    
    # Assign configurations for lang1 and lang2
    lang1_config = lang_settings[lang1_name]
    lang2_config = lang_settings[lang2_name]

    # Preprocess datasets
    lang1_df['Sentence'] = lang1_df['Sentence'].apply(lambda x: helper.preprocess_text(x))
    lang2_df['Sentence'] = lang2_df['Sentence'].apply(lambda x: helper.preprocess_text(x))

    # Tokenize using spaCy models
    lang1_df['Tokens'] = lang1_df['Sentence'].apply(lambda x: helper.tokenize_with_spacy(x, lang1_config['nlp']))
    lang2_df['Tokens'] = lang2_df['Sentence'].apply(lambda x: helper.tokenize_with_spacy(x, lang2_config['nlp']))

    # Drop sentences longer than max_seq_len
    lang1_df = lang1_df[lang1_df['Tokens'].apply(len) <= MAX_SEQ_LEN]
    lang2_df = lang2_df[lang2_df['Tokens'].apply(len) <= MAX_SEQ_LEN]

    # Match dataset sizes
    min_length = min(len(lang1_df), len(lang2_df))
    lang1_df = lang1_df.iloc[:min_length]
    lang2_df = lang2_df.iloc[:min_length]

    # Generate embeddings
    lang1_embeddings = helper.generate_embeddings(lang1_df, lang1_config['tokenizer'], lang1_config['bert_model'])
    lang2_embeddings = helper.generate_embeddings(lang2_df, lang2_config['tokenizer'], lang2_config['bert_model'])

    helper.generate_padded_embeddings(lang1_df, lang1_embeddings)
    helper.generate_padded_embeddings(lang2_df, lang2_embeddings)

    lang1_dataset = TensorDataset(
        torch.stack(lang1_df['Embeddings'].tolist()),
        torch.stack(lang1_df['Embeddings'].tolist()),
        torch.tensor([lang1_config['index']] * len(lang1_df))
    )
    lang2_dataset = TensorDataset(
        torch.stack(lang2_df['Embeddings'].tolist()),
        torch.stack(lang2_df['Embeddings'].tolist()),
        torch.tensor([lang2_config['index']] * len(lang2_df))
    )

    lang1_loader = DataLoader(
        lang1_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    lang2_loader = DataLoader(
        lang2_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    # Initialize models
    encoder_model = Encoder(ENCODER_INPUT_DIM, ENCODER_HIDDEN_DIM, ENCODER_OUTPUT_DIM, NUM_LSTM_LAYERS)
    decoder_model = Decoder(DECODER_LATENT_DIM, DECODER_HIIDEN_DIM, DECODER_OUTPUT_DIM, NUM_LSTM_LAYERS)
    disc_model = Discriminator(ENCODER_OUTPUT_DIM)

    # Optimizers
    enc_dec_optimizer = optim.Adam(
        list(encoder_model.parameters()) + list(decoder_model.parameters()), 
        lr=LEARNING_RATE,
        betas=(0.5, 0.999),
    )
    disc_optimizer = optim.Adam(disc_model.parameters(), lr=LEARNING_RATE)

    # Train the model
    helper.train_cross_domain_non_parallel(
        encoder_model, decoder_model, disc_model,
        enc_dec_optimizer, disc_optimizer,
        lang1_loader, lang2_loader,
        lang1_config['vocab_embeddings'], lang2_config['vocab_embeddings'],
        None, None, CLIP
    )

    # Save models
    os.makedirs(os.path.dirname(encoder_save_path), exist_ok=True)
    helper.save_model(encoder_model, enc_dec_optimizer, encoder_save_path)
    helper.save_model(decoder_model, enc_dec_optimizer, decoder_save_path)
    helper.save_model(disc_model, disc_optimizer, discriminator_save_path)

    print()
    print(f"Models saved: \nEncoder: {encoder_save_path}\nDecoder: {decoder_save_path}\nDiscriminator: {discriminator_save_path}")
    print()
