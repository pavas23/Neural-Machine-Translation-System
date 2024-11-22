## Neural Machine Translation System

Developed a machine translation system capable of translating between multiple languages without using parallel corpora (datasets where there are source and target sentence pairs) for training. 

The system has been built using the following concepts.

### Denoising Autoencoders (DAEs)

- Denoising Autoencoders are a key component in our approach, designed to prevent
the trivial copying of input data during training. By corrupting input sentences with
noise—such as ```dropping random words``` or ```slightly shuffling their order```—the autoencoder
is forced to reconstruct the original sentence.

###  Encoders and Decoders

- Encoders and decoders form the backbone of our translation system. The encoder, an
```LSTM-based model```, processes input sentences into a sequence of hidden states, which are then mapped into a shared latent space.
- By sharing parameters across both source and target languages, the encoder ensures that the ```latent representations align across languages```.
- The decoder, also an LSTM, generates target sentences one token at a time, using an ```attention mechanism``` to focus on the most relevant encoder states during each
step of decoding.

<br/>
<p align = 'center'>
<img width="550" height='175' alt="Screenshot 2024-11-22 at 7 53 36 PM" src="https://github.com/user-attachments/assets/09101da6-b7f8-424c-a615-268cd3100c6f">
</p>

### Cross-Domain Training

- Cross-domain training is a critical step to ensure that sentences can be translated between languages and reconstructed back into their original form. During this process, a
noisy translation is generated using the current model, and the encoder-decoder pair is
trained to reconstruct the original sentence from this translation.

### Adversarial Training

- Adversarial training introduces a ```discriminator to align the latent spaces``` of the source
and target languages. The discriminator is tasked with predicting the language of a latent
representation, while the encoder is trained to make these representations indistinguishable. This adversarial interaction ensures that the encoder outputs language-invariant
features, enabling the decoder to generate accurate and fluent sentences regardless of the
input language.

<br/>
<p align = 'center'>
<img width="547" alt="Screenshot 2024-11-22 at 7 58 07 PM" src="https://github.com/user-attachments/assets/720a9e7c-b7dc-4405-a71d-6656cadbdaae">
</p>


### Download Dataset

- From this link: https://data.statmt.org/news-commentary/v18.1/ download parallel datasets for english, french and german languages.

- While sourcing the monlingual datasets keep in mind that they should be from different domains for each language, as this website has parallel sentences language wise if they are extracted from the same domain.

### Spacy Installation

- Download Spacy libraries for tokenizing text

```c
python3.9 -m pip install sapcy
python3.9 -m spacy download en_core_web_lg
python3.9 -m spacy download fr_core_news_lg
python3.9 -m spacy download de_core_news_lg
```

### CLI Tool

The CLI tool has three main features described below:

- The CLI tool can train a new translation model using two independent monolingual corpora and save the trained encoder and decoder models.
- It can translate text or a single sentence between the source and target languages using the trained encoder and decoder models as input.
- The tool can evaluate the trained model using parallel datasets and display metrics such as BLEU and TER scores.
