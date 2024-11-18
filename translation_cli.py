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
from nltk.translate.meteor_score import meteor_score
from sacrebleu.metrics import TER

from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics.pairwise import cosine_similarity

import helper
import train

from model import Encoder, Decoder, Discriminator


SIZE_OF_DATASET = 2

# Hyperparameters
BATCH_SIZE = 32
NUM_LSTM_LAYERS = 3
LEARNING_RATE = 0.0003
LANG_DIM = 32
NUM_LANGS = 3
MAX_SEQ_LEN = 50

# Encoder
ENCODER_INPUT_DIM = 768 
ENCODER_HIDDEN_DIM = 300
ENCODER_OUTPUT_DIM = 100 

# Decoder
DECODER_OUTPUT_DIM = ENCODER_INPUT_DIM
DECODER_LATENT_DIM = ENCODER_OUTPUT_DIM
DECODER_HIIDEN_DIM = 300

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


# Function to load
def load_models(encoder_path, decoder_path, device):
    # Instantiate the model classes
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
    
    # Load checkpoints
    encoder_checkpoint = torch.load(encoder_path, map_location=device)
    decoder_checkpoint = torch.load(decoder_path, map_location=device)

    # Load model states
    encoder_model.load_state_dict(encoder_checkpoint['model_state_dict'])
    decoder_model.load_state_dict(decoder_checkpoint['model_state_dict'])

    # Move to device
    encoder_model = encoder_model.to(device)
    decoder_model = decoder_model.to(device)

    # Set to evaluation mode
    encoder_model.eval()
    decoder_model.eval()

    return encoder_model, decoder_model


# Preprocess the dataset
def preprocess_and_prepare_data(source_file, tokenizer_source, nlp_source, bert_model_source, source_idx):
    # Load the datasets
    with open(source_file, 'r', encoding='utf-8') as file:
        source_sentences = [line.strip() for line in file.readlines()]

    # Convert to DataFrames
    source_df = pd.DataFrame(source_sentences, columns=['Sentence'])

    # Preprocess the sentences
    source_df['Sentence'] = source_df['Sentence'].apply(helper.preprocess_text)

    # Tokenize the sentences
    source_df['Tokens'] = source_df['Sentence'].apply(lambda x: helper.tokenize_with_spacy(x, nlp_source))

    # Generate embeddings for both datasets
    source_embeddings = helper.generate_embeddings(source_df, tokenizer_source, bert_model_source)

    # Add padded embeddings
    helper.generate_padded_embeddings(source_df, source_embeddings)

    return source_df


# Preprocess parallel datasets
def preprocess_and_prepare_parallel_data(
    source_file, tokenizer_source, nlp_source, bert_model_source, source_idx,
    target_file, tokenizer_target, nlp_target, bert_model_target, target_idx,
):
    with open(source_file, 'r', encoding='utf-8') as file:
        source_sentences = [line.strip() for line in file.readlines()]
    with open(target_file, 'r', encoding='utf-8') as file:
        target_sentences = [line.strip() for line in file.readlines()]
    
    # Convert to DataFrames
    source_df = pd.DataFrame(source_sentences, columns=['Sentence'])
    target_df = pd.DataFrame(target_sentences, columns=['Sentence'])

    x = 100
    source_df = source_df[:x]
    target_df = target_df[:x]

    # Remove empty sentences if any
    source_df = source_df[source_df['Sentence'].notna() & (source_df['Sentence'] != '')]
    target_df = target_df[target_df['Sentence'].notna() & (target_df['Sentence'] != '')]

    # Preprocess the sentences
    source_df['Sentence'] = source_df['Sentence'].apply(helper.preprocess_text)
    target_df['Sentence'] = target_df['Sentence'].apply(helper.preprocess_text)

    # Tokenize the sentences
    source_df['Tokens'] = source_df['Sentence'].apply(lambda x: helper.tokenize_with_spacy(x, nlp_source))
    target_df['Tokens'] = target_df['Sentence'].apply(lambda x: helper.tokenize_with_spacy(x, nlp_target))

    # Drop sentences longer than 50 words
    source_df = source_df[source_df['Tokens'].apply(len) <= MAX_SEQ_LEN]
    target_df = target_df[target_df['Tokens'].apply(len) <= MAX_SEQ_LEN]

    # Generate embeddings for both datasets
    source_embeddings = helper.generate_embeddings(source_df, tokenizer_source, bert_model_source)
    target_embeddings = helper.generate_embeddings(target_df, tokenizer_target, bert_model_target)

    # Add padded embeddings
    helper.generate_padded_embeddings(source_df, source_embeddings)
    helper.generate_padded_embeddings(target_df, target_embeddings)

    min_length = min(len(source_df), len(target_df))
    source_df = source_df.iloc[:min_length]
    target_df = target_df.iloc[:min_length]

    return source_df, target_df


# Function for taking monolingual datasets and training the model
def train_func():
    # Collect user inputs
    print()
    lang1_name = input("Enter the source language (e.g., 'english (en)', 'french (fr)', 'german (de)'): ").strip().lower()
    lang2_name = input("Enter the target language (e.g., 'english (en)', 'french (fr)', 'german (de)'): ").strip().lower()

    lang1_data_path = input(f"Enter the path to the dataset for {lang1_name}: ").strip()
    if not os.path.exists(lang1_data_path):
        print(f"Error: File not found at {lang1_data_path}")
        return

    lang2_data_path = input(f"Enter the path to the dataset for {lang2_name}: ").strip()
    if not os.path.exists(lang2_data_path):
        print(f"Error: File not found at {lang2_data_path}")
        return

    encoder_save_path = input("Enter the path to save the encoder model: ").strip()
    decoder_save_path = input("Enter the path to save the decoder model: ").strip()
    discriminator_save_path = input("Enter the path to save the discriminator model: ").strip()

    print("\n===== Summary =====")
    print(f"Source Language: {lang1_name}")
    print(f"Target Language: {lang2_name}")
    print(f"Source Dataset Path: {lang1_data_path}")
    print(f"Target Dataset Path: {lang2_data_path}")
    print(f"Encoder Save Path: {encoder_save_path}")
    print(f"Decoder Save Path: {decoder_save_path}")
    print(f"Discriminator Save Path: {discriminator_save_path}")

    print()
    confirm = input("Do you want to proceed with training? (yes/no): ").strip().lower()
    print()

    if confirm != "yes":
        print("Training canceled...")
        return

    # Training function
    try:
        print("Training...")
        train.train_cross_domain_autoencoder(
            lang1_name=lang1_name,
            lang2_name=lang2_name,
            lang1_data_path=lang1_data_path,
            lang2_data_path=lang2_data_path,
            encoder_save_path=encoder_save_path,
            decoder_save_path=decoder_save_path,
            discriminator_save_path=discriminator_save_path
        )
        print("Training completed successfully!")
    except Exception as e:
        print(f"An error occurred during training: {e}")


# Function for translating between multiple languages
def inference():
    """Handles the inference process."""
    # Input paths and parameters
    encoder_path = input("Enter the path to the saved encoder model: ").strip()
    decoder_path = input("Enter the path to the saved decoder model: ").strip()
    source_language = input("Enter the source language code (e.g., 'english (en)', 'french (fr)', 'german (de)'): ").strip()
    target_language = input("Enter the target language code (e.g., 'english (en)', 'french (fr)', 'german (de)'): ").strip()
    source_sentence = input("Enter the source sentence: ").strip()

    # Validate paths
    if not os.path.exists(encoder_path) or not os.path.exists(decoder_path):
        print("Error: Provided model paths do not exist.")
        return

    # Load models
    encoder_model, decoder_model = load_models(encoder_path, decoder_path, device)

    source_config = lang_settings[source_language]
    target_config = lang_settings[target_language]

    # Write the source sentence to a text file (to be used for generating source_df)
    source_file_path = "./source_sentence.txt"
    with open(source_file_path, 'w') as f:
        f.write(source_sentence)

    # Preprocess source sentence using the text file
    source_df = preprocess_and_prepare_data(
        source_file=source_file_path,
        tokenizer_source=source_config['tokenizer'],
        nlp_source=source_config['nlp'],
        bert_model_source=source_config['bert_model'],
        source_idx=source_config['index'],
    )

    # Generate translation
    print("\nTranslating text...")
    translated_sentence = helper.generate_cross_domain_text(
        source_df,
        encoder_model,
        decoder_model,
        target_config['tokenizer'],
        target_config['vocab_embeddings'],
        source_lang_idx=source_config['index'],  
        target_lang_idx=target_config['index'],
    )

    print("\nTranslation:")
    print(" ".join(translated_sentence))
    print()

    os.remove(source_file_path)


# Function to display metrics
def evaluate_translation(df, hypothesis, nlp):
    tokenized_hypotheses = [list(map(str, nlp(sentence))) for sentence in hypothesis]
    references = [[sentence] for sentence in df['Tokens']]  # 2D array
    bleu_score = corpus_bleu(references, tokenized_hypotheses)
   
    ter_scorer = TER()
    references = [sentence for sentence in df['Sentence']]
    ter_score = ter_scorer.corpus_score(hypothesis, references)
    
    return bleu_score, ter_score


# Test model on parallel datasets
def test_model():
    # Input paths and parameters
    encoder_path = input("Enter the path to the saved encoder model: ").strip()
    decoder_path = input("Enter the path to the saved decoder model: ").strip()
    source_language = input("Enter the source language code (e.g., 'english (en)', 'french (fr)', 'german (de)'): ").strip()
    target_language = input("Enter the target language code (e.g., 'english (en)', 'french (fr)', 'german (de)'): ").strip()

    source_language_path = input(f"Enter the path to the parallel dataset for {source_language}: ").strip()
    if not os.path.exists(source_language_path):
        print(f"Error: File not found at {source_language_path}")
        return

    target_language_path = input(f"Enter the path to the parallel dataset for {target_language}: ").strip()
    if not os.path.exists(target_language_path):
        print(f"Error: File not found at {target_language_path}")
        return
    
    # Load models
    encoder_model, decoder_model = load_models(encoder_path, decoder_path, device)

    source_config = lang_settings[source_language]
    target_config = lang_settings[target_language]

    source_df, target_df = preprocess_and_prepare_parallel_data(
        source_file=source_language_path,
        tokenizer_source=source_config['tokenizer'],
        nlp_source=source_config['nlp'],
        bert_model_source=source_config['bert_model'],
        source_idx=source_config['index'],

        target_file=target_language_path,
        tokenizer_target=target_config['tokenizer'],
        nlp_target=target_config['nlp'],
        bert_model_target=target_config['bert_model'],
        target_idx=target_config['index'],
    )

    print('Translating and Evaluating...')
    target_hypothesis = helper.generate_cross_domain_text(
        source_df,
        encoder_model,
        decoder_model,
        target_config['tokenizer'],
        target_config['vocab_embeddings'],
        source_lang_idx=source_config['index'],  
        target_lang_idx=target_config['index'],
    )

    bleu_score, ter_score = evaluate_translation(target_df, target_hypothesis, target_config['nlp'])

    print()
    print(f'---BLEU Score: {bleu_score}---')
    print(f'---TER Score: {ter_score}---')
    print()


# Driver Function
def main():
    while True:
        print()
        print("===== Machine Translation CLI =====")
        print()
        print("1. Train a new translation model")
        print("2. Translate text using a trained model")
        print("3. Test trained model using parallel datasets")
        print("4. Exit")
        choice = input("Enter your choice: ").strip()

        if choice == "1":
            train_func()
        elif choice == "2":
            inference()
        elif choice == "3":
            test_model()
        elif choice == "4":
            print("Exiting the CLI!")
            break
        else:
            print("Invalid choice! Please enter 1, 2, 3 or 4.\n")


if __name__ == "__main__":
    main()
