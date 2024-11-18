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

NUM_LANGS = 3

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, lang_embed_dim=32):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(input_dim + lang_embed_dim, hidden_dim, bidirectional=True, batch_first=True, num_layers=num_layers)
        
        # Fully connected layers for final prediction
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, output_dim)

        self.activation = nn.ReLU()  # Activation function
        self.dropout = nn.Dropout(0.5)  # Dropout for regularization
        self.batch_norm = nn.BatchNorm1d(hidden_dim * 2)

        # Language embedding layer
        self.lang_embedding = nn.Embedding(NUM_LANGS, lang_embed_dim)  

    def forward(self, inputs, lang_idx):
        # Get language embedding
        lang_emb = self.lang_embedding(lang_idx).unsqueeze(1).expand(-1, inputs.size(1), -1)

        # Concatenate the language embedding with the input embeddings
        inputs_with_lang = torch.cat((inputs, lang_emb), dim=-1)

        lstm_out, _ = self.lstm(inputs_with_lang)  # (batch_size, seq_len, hidden_dim * 2)
        
        # Apply dropout and batch normalization
        lstm_out = self.dropout(lstm_out)
        lstm_out = self.batch_norm(lstm_out.transpose(1, 2)).transpose(1, 2)
        
        # Pass through fully connected layers
        fc1_output = self.activation(self.fc1(lstm_out))  # (batch_size, seq_len, hidden_dim)
        fc2_output = self.activation(self.fc2(fc1_output))  # (batch_size, seq_len, hidden_dim // 2)
        latent_output = self.fc3(fc2_output)  # (batch_size, seq_len, output_dim)
        
        # fc1_output has shape 300*1
        return fc1_output, latent_output


class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()

        self.hidden_dim = hidden_dim
        
        # Attention mechanism to compute attention weights
        # Concatenate hidden state and encoder outputs, and project to hidden_dim
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)  # Input dim = 2 * hidden_dim, Output = hidden_dim
        self.v = nn.Parameter(torch.rand(hidden_dim))      # Attention vector for scoring

    def forward(self, hidden, encoder_outputs):
        """
        Compute the attention weights and context vector.
        
        Args:
            hidden: The hidden state of the decoder (batch_size, hidden_dim)
            encoder_outputs: The outputs from the encoder (batch_size, seq_len, hidden_dim)
        
        Returns:
            context: The context vector (batch_size, hidden_dim)
        """
        batch_size = encoder_outputs.size(0)
        seq_len = encoder_outputs.size(1)
        hidden_dim = encoder_outputs.size(2)

        # Expand hidden state to match the sequence length
        hidden_expanded = hidden.unsqueeze(1).expand(batch_size, seq_len, hidden_dim)  # (batch_size, seq_len, hidden_dim)

        # Concatenate hidden state and encoder outputs along the last dimension
        concatenated = torch.cat((hidden_expanded, encoder_outputs), dim=2)  # (batch_size, seq_len, 2 * hidden_dim)

        # Pass through the attention network
        energy = torch.tanh(self.attn(concatenated))  # (batch_size, seq_len, hidden_dim)

        # Transpose energy to prepare for batch matrix multiplication
        energy = energy.transpose(1, 2)  # (batch_size, hidden_dim, seq_len)

        # Repeat v to match the batch size (for each batch, we use the same attention vector)
        v_expanded = self.v.repeat(batch_size, 1).unsqueeze(1)  # (batch_size, 1, hidden_dim)

        # Compute attention scores using batch matrix multiplication
        attn_weights = torch.bmm(v_expanded, energy).squeeze(1)  # (batch_size, seq_len)

        # Normalize attention weights using softmax
        attn_weights = F.softmax(attn_weights, dim=1)  # (batch_size, seq_len)

        # Compute the context vector as the weighted sum of encoder outputs
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)  # (batch_size, hidden_dim)

        return context


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim, num_layers=3, lang_embed_dim=32):
        super(Decoder, self).__init__()
        
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM layer for decoding
        # Takes input as the context vector of hidden_dim and output from the encoder of latent_dim
        # Outputs a hidden state of hidden_dim
        self.lstm = nn.LSTM(latent_dim + hidden_dim + lang_embed_dim, hidden_dim, num_layers=num_layers, batch_first=True)

        # Attention mechanism
        self.attn = Attention(hidden_dim)

        # Fully connected layer to output the prediction
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.5)

        # Language embedding layer
        self.lang_embedding = nn.Embedding(NUM_LANGS, lang_embed_dim)

    def forward(self, latent_vectors, encoder_outputs, hidden, lang_idx):
        """
        Decoder forward pass with attention mechanism.

        Args:
            latent_vectors: The latent vectors from the encoder (batch_size, seq_len, latent_dim), output given by bidirectional LSTM encoder after summarizing (abstracting) the input sentence, that is passing through the fc layers.
            They will be used as input for decoder LSTM, for finding the target word.

            encoder_outputs: The encoder outputs (batch_size, seq_len, latent_dim), output given by bidirectional LSTM encoder of dim 300.
            They will by used for finding the attention weights and context vector.
            
            hidden: The hidden state from the previous time step (num_layers, batch_size, hidden_dim)
            
        Returns:
            outputs: The generated output sequence (batch_size, seq_len, output_dim)
            hidden: The updated hidden state (num_layers, batch_size, hidden_dim)
        """
        batch_size = latent_vectors.size(0)
        seq_len = latent_vectors.size(1)

        # Get language embedding
        lang_emb = self.lang_embedding(lang_idx).unsqueeze(1).expand(-1, latent_vectors.size(1), -1)

        # Initialize the output tensor
        outputs = torch.zeros(batch_size, seq_len, self.fc_out.out_features).to(latent_vectors.device)

        # Iterate through the sequence to generate each word
        for t in range(seq_len):
            # Get attention weights and context vector
            # hidden[0][-1] gives the previous hidden state, need to take hidden[0] as hidden is a tuple (h,c)
            context = self.attn(hidden[0][-1], encoder_outputs)

            # Concatenate the context vector, latent vector, and language embedding
            lstm_input = torch.cat((context, latent_vectors[:, t], lang_emb[:, t]), dim=1).unsqueeze(1)

            # Pass through the LSTM
            lstm_out, hidden = self.lstm(lstm_input, hidden)  # lstm_out: (batch_size, 1, hidden_dim)
            lstm_out = self.dropout(lstm_out)

            # Predict the next word
            outputs[:, t] = self.fc_out(lstm_out.squeeze(1))

        return outputs, hidden

    def init_hidden(self, batch_size):
        """ Initialize hidden state and cell state for LSTM """
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(next(self.parameters()).device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(next(self.parameters()).device)
        return (h0, c0)

# Discriminator Model
class Discriminator(nn.Module):
    def __init__(self, latent_dim):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.3)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.dropout(self.activation(self.fc1(x)))
        x = self.dropout(self.activation(self.fc2(x)))
        x = torch.sigmoid(self.fc3(x))  # Sigmoid output for binary classification
        return x
