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

NUM_EPOCHS = 2
MAX_SEQ_LEN = 50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess_text(sentence):
    sentence = sentence.lower()
    sentence = re.sub(r'[^\w\s]', '', sentence)  # Remove punctuation
    return sentence


def remerge_sent(sent):
    # merges tokens which are not separated by white-space
    # does this recursively until no further changes
    changed = True
    while changed:
        changed = False
        i = 0
        while i < sent.__len__() - 1:
            tok = sent[i]
            if not tok.whitespace_:
                ntok = sent[i + 1]
                # in-place operation.
                with sent.retokenize() as retokenizer:
                    retokenizer.merge(sent[i: i + 2])
                changed = True
            i += 1
    return sent


def tokenize_with_spacy(sentence, nlp):
    doc = nlp(sentence)
    spacy_sentence = remerge_sent(doc)
    return [token.text for token in spacy_sentence]


# Retrieves BERT embeddings for a list of tokens using a fast tokenizer, enabling accurate aggregation of subword embeddings into their original token representations.
def get_bert_embeddings(tokens, tokenizer, bert_model):
    inputs = tokenizer(tokens, return_tensors='pt', is_split_into_words=True, padding=False, truncation=True)

    # Get BERT embeddings from the model
    with torch.no_grad():
        outputs = bert_model(**inputs)

    # Get the embeddings for each subword
    token_embeddings = outputs.last_hidden_state.squeeze(0)  # Shape: (sequence_length, hidden_size)
    # Get word_ids to align subword tokens with the original tokens
    word_ids = inputs.word_ids()

    # Aggregate subword embeddings back to their original tokens
    aggregated_embeddings = []
    current_token_embeddings = []

    for idx, word_id in enumerate(word_ids):
        if word_id is None:
            continue
        if len(current_token_embeddings) > 0 and word_id != word_ids[idx - 1]:
            aggregated_embeddings.append(torch.mean(torch.stack(current_token_embeddings), dim=0))
            current_token_embeddings = []
        current_token_embeddings.append(token_embeddings[idx])
    
    if len(current_token_embeddings) > 0:
        aggregated_embeddings.append(torch.mean(torch.stack(current_token_embeddings), dim=0))

    return torch.stack(aggregated_embeddings)


# Function to generate BERT embeddings for dataFrame
def generate_embeddings(df, tokenizer, bert_model):
    embeddings_list = []
    for _, row in df.iterrows():
        tokenized_sentence = row['Tokens']
        embeddings = get_bert_embeddings(tokenized_sentence, tokenizer, bert_model)
        embeddings_list.append(embeddings)
    return embeddings_list


def generate_padded_embeddings(df, embedding_list, max_len=MAX_SEQ_LEN):
    # Pad the sequences using pad_sequence. It will pad them to the length of the longest sentence.
    padded_embeddings = pad_sequence(embedding_list, batch_first=True, padding_value=0)
    
    # Truncate the sequences if they are longer than max_len
    if padded_embeddings.size(1) > max_len:
        padded_embeddings = padded_embeddings[:, :max_len, :]
    # If sequences are shorter than max_len, pad them manually
    elif padded_embeddings.size(1) < max_len:
        padding_size = max_len - padded_embeddings.size(1)
        padded_embeddings = F.pad(padded_embeddings, (0, 0, 0, padding_size), value=0)
    
    # Assign the padded embeddings to the dataframe
    df['Embeddings'] = [padded_embeddings[i] for i in range(padded_embeddings.shape[0])]


def apply_word_dropout(sentence, pwd=0.1, padding_embedding=None):
    noisy_sentence = []
    drop_count = 0

    for word in sentence:
        if random.random() > pwd:
            noisy_sentence.append(word)
        else:
            drop_count += 1

    if padding_embedding is None:
        padding_embedding = torch.zeros_like(sentence[0]) 

    noisy_sentence.extend([padding_embedding] * drop_count)
    noisy_sentence_tensor = torch.stack(noisy_sentence)

    return noisy_sentence_tensor


def apply_sentence_shuffling(sentence, k=3, alpha=0.5):
    n = sentence.size(0)

    # Generating random permutation vector q
    q = torch.arange(n).float() + torch.rand(n) * alpha  # Slightly perturb the indices
    _, permuted_indices = torch.sort(q)  # Sorting to get the permutation

    # Apply the shuffle, respecting the condition |σ(i) - i| <= k
    for i in range(n):
        if abs(permuted_indices[i] - i) > k:
            permuted_indices[i] = i 
    
    # Shuffle sentence according to the permuted indices
    shuffled_sentence = sentence[permuted_indices]
    
    return shuffled_sentence


def apply_noise(x, pwd=0.1, k=3, alpha=0.5):
    if isinstance(x, list):
        x = torch.tensor(x)

    # Apply word dropout
    x_noisy = apply_word_dropout(x, pwd)
    
    # Apply sentence shuffling
    x_noisy = apply_sentence_shuffling(x_noisy, k, alpha)
    
    return x_noisy


# Loss function (Cosine Loss)
def cosine_similarity_loss(predictions, targets, pad_idx=0): 
    # Flatten the predictions and targets
    predictions = predictions.view(-1, predictions.size(-1))  # (batch_size * seq_len, embedding_dim)
    targets = targets.view(-1, targets.size(-1))  # (batch_size * seq_len, embedding_dim)

    # Create a mask to exclude padded positions
    mask = (targets.sum(dim=-1) != pad_idx).float()  # (batch_size * seq_len,)

    # Compute the cosine similarity between the predictions and targets
    cos_sim = F.cosine_similarity(predictions, targets, dim=-1)  # (batch_size * seq_len,)
    loss = (1 - cos_sim) * mask  # Apply mask to exclude padding positions
    loss = loss.sum() / mask.sum()  # Average the loss over non-padding tokens

    return loss


def plot_loss(losses):
    epochs = np.arange(1, len(losses) + 1)
    fig, ax = plt.subplots(figsize=(6, 4))  

    norm = plt.Normalize(vmin=min(losses), vmax=max(losses))
    colors = plt.cm.cividis(norm(losses)) 

    for i in range(len(epochs) - 1):
        alpha_value = max(0.3, 0.8 - i * 0.02) 
        ax.plot(epochs[i:i+2], losses[i:i+2], color=colors[i], alpha=alpha_value, linewidth=2)

    ax.fill_between(epochs, losses, color='lightblue', alpha=0.3)

    ax.set_title('Training Loss Over Epochs', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epochs', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)

    sm = plt.cm.ScalarMappable(cmap='cividis', norm=norm)
    sm.set_array([]) 
    fig.colorbar(sm, ax=ax, label='Loss Intensity')

    plt.tight_layout()
    plt.show()


# Function for training the model
def train_dae(encoder_model, decoder_model, optimizer, data_loader, clip, pad_idx=0):
    all_losses = []

    for epoch in range(NUM_EPOCHS):
        encoder_model.train()
        decoder_model.train()

        epoch_loss = 0
        for batch in data_loader:
            inputs = batch[0].to(device)  # BERT embeddings
            targets = batch[1].to(device)  # Same because of autoencoding
            lang_idx = batch[2].to(device)  # Get language index for each sentence
            
            # Add noise to inputs and stack them into a tensor
            noisy_inputs = torch.stack([apply_noise(sentence) for sentence in inputs]).to(device)

            optimizer.zero_grad()

            # Forward pass through encoder
            encoder_outputs, latent_vectors = encoder_model(noisy_inputs, lang_idx)
            
            # Initialize decoder hidden state for each batch
            hidden = decoder_model.init_hidden(inputs.size(0))  
            
            # Decoder forward pass with attention
            outputs, hidden = decoder_model(latent_vectors, encoder_outputs, hidden, lang_idx)

            # Compute the loss
            loss = cosine_similarity_loss(outputs, targets, pad_idx)

            # Backward pass and optimization
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(encoder_model.parameters(), clip)
            torch.nn.utils.clip_grad_norm_(decoder_model.parameters(), clip)

            # Optimizer step
            optimizer.step()
            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / len(data_loader)
        all_losses.append(avg_epoch_loss)

        # Print the loss for each epoch
        print(f'Epoch [{epoch + 1}/{NUM_EPOCHS}], Loss: {avg_epoch_loss:.4f}')
    
    plot_loss(all_losses)


def get_closest_token_from_vocab(embedding, vocab_embeddings):
    # Find the closest token from the vocabulary for a given embedding.
    similarities = F.cosine_similarity(embedding, vocab_embeddings, dim=1)
    closest_token_idx = torch.argmax(similarities)
    return closest_token_idx.item()


def embeddings_to_text(output_embeddings, tokenizer, vocab_embeddings):

    batch_size, seq_len, embedding_dim = output_embeddings.size()
    tokens = []
    
    # Iterate over each sequence in the batch
    for i in range(batch_size):
        sequence_tokens = []
        
        # Iterate over each embedding in the sequence
        for j in range(seq_len):
            embedding = output_embeddings[i, j]  # Shape: (embedding_dim,)
            token_idx = get_closest_token_from_vocab(embedding, vocab_embeddings)
            token = tokenizer.convert_ids_to_tokens(token_idx)
            sequence_tokens.append(token)
        
        sentence = tokenizer.convert_tokens_to_string(sequence_tokens)
        tokens.append(sentence)
    
    return tokens


def generate_text_using_dae(df, tokenizer, vocab_embeddings, lang_idx, encoder_model, decoder_model):
    inputs = torch.stack([sentence for sentence in df['Embeddings']]).to(device)
    lang_idx_tensor = torch.tensor([lang_idx] * len(df)).to(device)

    encoder_outputs, latent_vectors = encoder_model(inputs, lang_idx_tensor)

    # Initialize decoder hidden state for each batch
    hidden = decoder_model.init_hidden(len(df['Embeddings']))  

    # Decoder forward pass with attention
    outputs, hidden = decoder_model(latent_vectors, encoder_outputs, hidden, lang_idx_tensor)

    generated_text = embeddings_to_text(outputs, tokenizer, vocab_embeddings)

    def clean_output(text):
        return ' '.join(
            [token for token in text.split() if token not in [
                '[SEP]', '[SEP].', '[MASK]', '[CLS]'
            ]]
        )

    # Clean the generated text
    cleaned_text = [clean_output(sentence) for sentence in generated_text]
    for sentence in cleaned_text:
        print("Generated Sentence:", sentence)


def initialize_word_by_word_model(source_embeddings, target_embeddings, source_vocab, target_vocab, num_iterations=NUM_EPOCHS):    
    # Normalize embeddings
    source_embeddings = source_embeddings / np.linalg.norm(source_embeddings, axis=1, keepdims=True)
    target_embeddings = target_embeddings / np.linalg.norm(target_embeddings, axis=1, keepdims=True)

    # Convert embeddings to torch tensors
    X = torch.tensor(source_embeddings, dtype=torch.float32).to(device)
    Y = torch.tensor(target_embeddings, dtype=torch.float32).to(device)

    # Ensure that the dimensionality of W matches the original embeddings
    embedding_dim = X.shape[2]  # Original dimensionality of the embeddings
    W = torch.nn.Linear(embedding_dim, embedding_dim, bias=False).to(device)  # Keep the embedding size the same
    torch.nn.init.eye_(W.weight)  # Start with identity matrix
    optimizer = torch.optim.Adam(W.parameters(), lr=0.1)
    
    # Discriminator model
    discriminator = torch.nn.Sequential(
        torch.nn.Linear(embedding_dim, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 1),
        torch.nn.Sigmoid()
    ).to(device)
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.1)

    # Adversarial Training Loop
    for epoch in range(num_iterations):
        # Train discriminator
        discriminator_optimizer.zero_grad()
        WX = W(X)
        src_preds = discriminator(WX)
        tgt_preds = discriminator(Y)
        
        # Labels: 1 for source, 0 for target
        src_labels = torch.ones(WX.size(0), WX.size(1), 1).to(device)  # [10, 50, 1] for source
        tgt_labels = torch.zeros(Y.size(0), Y.size(1), 1).to(device)  # [10, 50, 1] for target
    
        disc_loss = (
            torch.nn.functional.binary_cross_entropy(src_preds, src_labels) +
            torch.nn.functional.binary_cross_entropy(tgt_preds, tgt_labels)
        )
        disc_loss.backward()
        discriminator_optimizer.step()
        
        # Train mapping (W)
        optimizer.zero_grad()
        WX = W(X)
        src_preds = discriminator(WX)
        
        # Reverse the labels for adversarial training
        mapping_loss = torch.nn.functional.binary_cross_entropy(src_preds, tgt_labels)
        mapping_loss.backward()
        optimizer.step()
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch}, Discriminator Loss: {disc_loss.item():.4f}, Mapping Loss: {mapping_loss.item():.4f}")

    # Procrustes Refinement
    with torch.no_grad():
        # Compute SVD of Y^T * W * X
        # Ensure that we are doing matrix multiplication between embeddings in compatible shapes
        A = Y.view(-1, Y.size(-1)).T @ W(X).view(-1, W(X).size(-1))  # Flatten the dimensions for multiplication
        U, _, V = torch.svd(A)
        W.weight.data = (U @ V.T).to(device)  # Update W to enforce orthogonality

    
    # Detach WX before converting to numpy
    WX_flat = WX.detach().view(-1, WX.size(-1)).numpy()  # Shape: [500, 768] if there are 500 words
    target_embeddings_flat = target_embeddings.reshape(-1, target_embeddings.shape[-1])  # Shape: [500, 768]

    # Compute cosine similarity
    similarity_matrix = cosine_similarity(WX_flat, target_embeddings_flat)

    # Mapping from source to target
    source_to_target_mapping = {}
    for i, source_word in enumerate(source_vocab):  # Iterate over source_vocab
        if i < similarity_matrix.shape[0]:  # Ensure index is within bounds
            best_match_idx = np.argmax(similarity_matrix[i])
            target_word = target_vocab[best_match_idx]
            source_to_target_mapping[source_word] = target_word

    # Mapping from target to source
    target_to_source_mapping = {}
    for j, target_word in enumerate(target_vocab):  # Iterate over target_vocab
        if j < similarity_matrix.shape[0]:  # Ensure index is within bounds
            best_match_idx = np.argmax(similarity_matrix[:, j])
            source_word = source_vocab[best_match_idx]
            target_to_source_mapping[target_word] = source_word

    return source_to_target_mapping, target_to_source_mapping


def compute_csls(source_embeddings, target_embeddings, k=10):
    """
    Compute CSLS (Cross Domain Similarity Local Scaling) similarity between source and target embeddings.
    Parameters:
    - source_embeddings: Source embeddings matrix (N x D).
    - target_embeddings: Target embeddings matrix (M x D).
    - k: Number of neighbors to consider for local scaling.
    Returns:
    - CSLS similarity matrix (N x M).
    """
    # Compute pairwise cosine similarities
    similarity_matrix = cosine_similarity(source_embeddings, target_embeddings)

    # Compute mean similarity for source and target neighborhoods
    r_source = np.mean(np.sort(similarity_matrix, axis=1)[:, -k:], axis=1)
    r_target = np.mean(np.sort(similarity_matrix, axis=0)[-k:, :], axis=0)

    # Adjust similarities using CSLS
    csls_matrix = 2 * similarity_matrix - r_source[:, np.newaxis] - r_target[np.newaxis, :]

    return csls_matrix


def apply_word_by_word_translation(
    source_inputs,
    source_to_target_mapping,
    device="cpu"
):
    # Extract target embeddings from the mapping (assuming all source embeddings have the same size)
    target_embeddings = torch.stack(list(source_to_target_mapping.values())).to(device)  # Shape: [vocab_size, embedding_size]
    
    # Get the embedding size dynamically
    embedding_size = target_embeddings.size(1)  # Size of the second dimension, the embedding size

    translated_sentences = []
    for sentence in source_inputs:
        translated_sentence = []
        for word in sentence:
            if word in source_to_target_mapping:
                # Exact match in the mapping, use the target embedding
                translated_word = source_to_target_mapping[word].to(device)
            else:
                # Handle missing words by finding the closest key in the source_to_target_mapping
                similarity_scores = {}
                # Get all keys in the dictionary
                for key, target_embedding in source_to_target_mapping.items():
                    # Get the embedding of the current word from the source dictionary
                    word_embedding = word 

                    # Ensure no gradient tracking when converting to numpy
                    word_embedding = word_embedding.detach()

                    # Calculate cosine similarity between the word embedding and the target embeddings
                    similarity_scores[key] = cosine_similarity(
                        word_embedding.unsqueeze(0).cpu().numpy(),  # Word embedding to compare
                        key.unsqueeze(0).detach().cpu().numpy()  # Target word embedding to compare
                    )
                
                # Get the closest word key based on similarity
                closest_key = max(similarity_scores, key=similarity_scores.get)
                translated_word = source_to_target_mapping[closest_key].to(device)
                
            translated_sentence.append(translated_word)

        # Stack tensors for the translated sentence
        translated_sentences.append(torch.stack(translated_sentence))

    # Pad sentences to the same length
    max_len = max(len(sentence) for sentence in translated_sentences)
    padded_sentences = [
        torch.cat([sentence, torch.zeros(max_len - len(sentence), embedding_size).to(device)], dim=0)
        if len(sentence) < max_len else sentence
        for sentence in translated_sentences
    ]

    # Combine all sentences into a batch tensor
    batch_tensor = torch.stack(padded_sentences)  # Shape: [batch_size, seq_len, embedding_size]
    return batch_tensor


def update_word_by_word_model(encoder_model, source_loader, target_loader, source_vocab, target_vocab):
    encoder_model.eval()

    # Step 1: Extract Latent Representations for Source and Target (Word-Level)
    source_embeddings = []
    target_embeddings = []

    with torch.no_grad():
        for source_batch in source_loader:
            source_inputs = source_batch[0].to(device)
            source_lang_idx = source_batch[2].to(device)
            _, source_latents = encoder_model(source_inputs, source_lang_idx)
            
            # Reshape embeddings from [batch_size, seq_len, embedding_dim] to [num_words, embedding_dim]
            # Flatten the batch dimension and sequence length
            source_latents = source_latents.view(-1, source_latents.size(-1))  # shape: [num_source_words, embedding_dim]
            source_embeddings.append(source_latents.cpu().numpy())

        for target_batch in target_loader:
            target_inputs = target_batch[0].to(device)
            target_lang_idx = target_batch[2].to(device)
            _, target_latents = encoder_model(target_inputs, target_lang_idx)
            
            # Reshape embeddings from [batch_size, seq_len, embedding_dim] to [num_words, embedding_dim]
            target_latents = target_latents.view(-1, target_latents.size(-1))  # shape: [num_target_words, embedding_dim]
            target_embeddings.append(target_latents.cpu().numpy())

    # Concatenate embeddings across all batches
    source_embeddings = np.concatenate(source_embeddings, axis=0)  # shape: [num_source_words, embedding_dim]
    target_embeddings = np.concatenate(target_embeddings, axis=0)  # shape: [num_target_words, embedding_dim]

    # Normalize embeddings for cosine similarity
    source_embeddings = source_embeddings / np.linalg.norm(source_embeddings, axis=1, keepdims=True)
    target_embeddings = target_embeddings / np.linalg.norm(target_embeddings, axis=1, keepdims=True)

    # Step 2: Compute Similarity Matrix (Word-Level)
    similarity_matrix = compute_csls(source_embeddings, target_embeddings)

    # Step 3: Find Nearest Neighbors (Source to Target)
    source_to_target_mapping = {}
    for i, src_word in enumerate(source_vocab):
        if i >= similarity_matrix.shape[0]:
            continue
        best_match_idx = np.argmax(similarity_matrix[i])  # Find best match in target
        # Only map if a match exists
        if best_match_idx < len(target_vocab):
            source_to_target_mapping[src_word] = target_vocab[best_match_idx]

    # Step 4: Find Nearest Neighbors (Target to Source)
    target_to_source_mapping = {}
    for j, tgt_word in enumerate(target_vocab):
        if j >= similarity_matrix.shape[1]:
            continue
        best_match_idx = np.argmax(similarity_matrix[:, j])  # Find best match in source
        # Only map if a match exists
        if best_match_idx < len(source_vocab):
            target_to_source_mapping[tgt_word] = source_vocab[best_match_idx]

    return source_to_target_mapping, target_to_source_mapping


def generate_word_by_word_mapping(source_loader, target_loader, source_vocab, target_vocab):
    # Extract embeddings without encoder
    source_embeddings = []
    target_embeddings = []

    for source_batch, target_batch in zip(source_loader, target_loader):
        source_inputs = source_batch[0] 
        target_inputs = target_batch[0]
        
        source_embeddings.append(source_inputs.numpy())
        target_embeddings.append(target_inputs.numpy())

    # Convert lists to numpy arrays for further processing
    source_embeddings = np.vstack(source_embeddings)
    target_embeddings = np.vstack(target_embeddings)

    # Pass embeddings directly to initialize the word-by-word mapping model
    source_to_target_mapping, target_to_source_mapping = initialize_word_by_word_model(
        source_embeddings, target_embeddings, source_vocab, target_vocab
    )

    return source_to_target_mapping, target_to_source_mapping


def train_cross_domain_non_parallel(
        encoder_model, decoder_model, disc_model, enc_dec_optimizer, disc_optimizer,
        source_loader, target_loader, source_vocab, target_vocab,
        source_to_target_mapping, target_to_source_mapping,
        clip, lambda_auto=1.0, lambda_cd=1.0, lambda_adv=1.0, pad_idx=0,
    ):
    
    all_losses = []

    for epoch in range(NUM_EPOCHS):
        encoder_model.train()
        decoder_model.train()
        disc_model.train()

        epoch_loss = 0
        for source_batch, target_batch in zip(source_loader, target_loader):
            enc_dec_optimizer.zero_grad()

            # Step 1: DAE for Source Domain (Auto-Encoding Loss)
            source_inputs = source_batch[0].to(device)
            source_lang_idx = source_batch[2].to(device)
            # Add noise to input sentences
            noisy_source_inputs = torch.stack([apply_noise(sentence) for sentence in source_inputs]).to(device)
            # Encoder decoder for autoencoding
            source_enc_outputs, source_latents = encoder_model(noisy_source_inputs, source_lang_idx)
            hidden = decoder_model.init_hidden(source_inputs.size(0))
            source_auto_encoded, _ = decoder_model(source_latents, source_enc_outputs, hidden, source_lang_idx)
            # Auto-encoding loss for source
            loss_auto_source = cosine_similarity_loss(source_auto_encoded, source_batch[1].to(device), pad_idx)


            # Step 2: DAE for Target Domain (Auto-Encoding Loss)
            target_inputs = target_batch[0].to(device)
            target_lang_idx = target_batch[2].to(device)
            # Add noise to input sentences
            noisy_target_inputs = torch.stack([apply_noise(sentence) for sentence in target_inputs]).to(device)
            # Encoder decoder for autoencoding
            target_enc_outputs, target_latents = encoder_model(noisy_target_inputs, target_lang_idx)
            hidden = decoder_model.init_hidden(target_inputs.size(0))
            target_auto_encoded, _ = decoder_model(target_latents, target_enc_outputs, hidden, target_lang_idx)
            # Auto-encoding loss for target
            loss_auto_target = cosine_similarity_loss(target_auto_encoded, target_batch[1].to(device), pad_idx)


            # # Step 3: Cross-Domain Translation (Source -> Target)
            # translated_to_target_M = apply_word_by_word_translation(source_inputs, source_to_target_mapping)
            # noisy_translation_target = torch.stack([apply_noise(sentence) for sentence in translated_to_target_M]).to(device)
            # target_enc_outputs, target_latents = encoder_model(noisy_translation_target, target_lang_idx)
            # hidden = decoder_model.init_hidden(target_inputs.size(0))
            # translated_back_to_source, _ = decoder_model(target_latents, target_enc_outputs, hidden, lang_idx=source_lang_idx)
            # # Cross-domain reconstruction loss
            # loss_source_target_cd = cosine_similarity_loss(translated_back_to_source, source_batch[1].to(device), pad_idx)
            

            # # Step 4: Cross-Domain Translation (Target -> Source)
            # translated_to_source_M = apply_word_by_word_translation(target_inputs, target_to_source_mapping)
            # noisy_translation_source = torch.stack([apply_noise(sentence) for sentence in translated_to_source_M]).to(device)
            # source_enc_outputs, source_latents = encoder_model(noisy_translation_source, source_lang_idx)
            # hidden = decoder_model.init_hidden(source_inputs.size(0))
            # translated_back_to_target, _ = decoder_model(source_latents, source_enc_outputs, hidden, lang_idx=target_lang_idx)
            # # Cross-domain reconstruction loss
            # loss_target_source_cd = cosine_similarity_loss(translated_back_to_target, target_batch[1].to(device), pad_idx)

            # Translate source to target (without needing target data to be a direct translation)
            hidden = decoder_model.init_hidden(source_inputs.size(0))
            translated_to_target, _ = decoder_model(source_latents, source_enc_outputs, hidden, lang_idx=target_lang_idx)
            # Re-encode the translated target sentence to enforce shared latent space structure
            target_re_enc_outputs, target_re_latents = encoder_model(translated_to_target, target_lang_idx)
            hidden = decoder_model.init_hidden(target_inputs.size(0))
            translated_back_to_source, _ = decoder_model(target_re_latents, target_re_enc_outputs, hidden, lang_idx=source_lang_idx)
            loss_source_target_cd = cosine_similarity_loss(translated_back_to_source, source_batch[1].to(device), pad_idx)

            # Translate target to source (without needing target data to be a direct translation)
            hidden = decoder_model.init_hidden(target_inputs.size(0))
            translated_to_source, _ = decoder_model(target_latents, target_enc_outputs, hidden, lang_idx=source_lang_idx)
            # Re-encode the translated target sentence to enforce shared latent space structure
            source_re_enc_outputs, source_re_latents = encoder_model(translated_to_source, source_lang_idx)
            hidden = decoder_model.init_hidden(source_inputs.size(0))
            translated_back_to_target, _ = decoder_model(source_re_latents, source_re_enc_outputs, hidden, lang_idx=target_lang_idx)
            loss_target_source_cd = cosine_similarity_loss(translated_back_to_target, target_batch[1].to(device), pad_idx)


            # Step 5: Adversarial Loss for Shared Latent Space
            disc_optimizer.zero_grad()
            # Train discriminator to differentiate source and target latent vectors
            source_latents = encoder_model(source_inputs, source_lang_idx)[1]
            target_latents = encoder_model(target_inputs, target_lang_idx)[1]
            
            # Find cross entropy loss, 0 is for source domain and 1 is for target domain
            disc_loss_src = F.binary_cross_entropy_with_logits(disc_model(source_latents), torch.zeros(source_latents.size(0), MAX_SEQ_LEN, 1).to(device))
            disc_loss_tgt = F.binary_cross_entropy_with_logits(disc_model(target_latents), torch.ones(target_latents.size(0), MAX_SEQ_LEN, 1).to(device))
            disc_loss = disc_loss_src + disc_loss_tgt
            # Update discriminator
            disc_loss.backward(retain_graph=True)
            disc_optimizer.step()


            # Step 6: Adversarial loss for encoder to fool the discriminator
            adv_loss_gen = F.binary_cross_entropy_with_logits(disc_model(source_latents), torch.ones(source_latents.size(0), MAX_SEQ_LEN, 1).to(device))


            # Step 7: Combine Losses and Update Encoder-Decoder
            total_loss = (
                lambda_auto * (loss_auto_source + loss_auto_target) +
                lambda_cd * (loss_source_target_cd + loss_target_source_cd) +
                lambda_adv * adv_loss_gen
            )
            total_loss.backward(retain_graph=True)

            # Clip gradients and update encoder-decoder
            torch.nn.utils.clip_grad_norm_(encoder_model.parameters(), clip)
            torch.nn.utils.clip_grad_norm_(decoder_model.parameters(), clip)
            enc_dec_optimizer.step()

            # Accumulate loss
            epoch_loss += total_loss.item()

        # Update M for next iteration
        # source_to_target_mapping, target_to_source_mapping = update_word_by_word_model(encoder_model, source_loader, target_loader, source_vocab, target_vocab)

        avg_epoch_loss = epoch_loss / len(source_loader)
        all_losses.append(avg_epoch_loss)

        print(f'Epoch [{epoch + 1}/{NUM_EPOCHS}], Auto Loss (Source): {loss_auto_source.item():.4f}, '
              f'Auto Loss (Target): {loss_auto_target.item():.4f}, Cross-Domain Loss (Source to Target): {loss_source_target_cd.item():.4f}, '
              f'Cross-Domain Loss (Target to Source): {loss_target_source_cd.item():.4f}, Adv Loss: {disc_loss.item():.4f}, Total Loss: {avg_epoch_loss:.4f}')

    plot_loss(all_losses)



def generate_cross_domain_text(df, encoder_model, decoder_model, tokenizer, vocab_embeddings, source_lang_idx, target_lang_idx):
    # Stack embeddings from DataFrame and set to device
    inputs = torch.stack([sentence for sentence in df['Embeddings']]).to(device)
    source_lang_tensor = torch.tensor([source_lang_idx] * len(df)).to(device)

    # Step 1: Encode input sentences in the source language
    encoder_outputs, latent_vectors = encoder_model(inputs, source_lang_tensor)

    # Initialize decoder hidden state for each batch in target language
    hidden = decoder_model.init_hidden(len(df['Embeddings']))
    target_lang_tensor = torch.tensor([target_lang_idx] * len(df)).to(device)

    # Step 2: Decode latent vectors in the target language
    outputs, hidden = decoder_model(latent_vectors, encoder_outputs, hidden, target_lang_tensor)

    # Step 3: Convert output embeddings to text using vocabulary embeddings
    generated_text = embeddings_to_text(outputs, tokenizer, vocab_embeddings)

    # Clean up special tokens from generated sentences
    def clean_output(text):
        return ' '.join(
            [token for token in text.split() if token not in [
                '[SEP]', '[SEP].', '[MASK]', '[CLS]'
            ]]
        )

    cleaned_text = [clean_output(sentence) for sentence in generated_text]
    return cleaned_text


def save_model(model, optimizer, save_path):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, save_path)
    