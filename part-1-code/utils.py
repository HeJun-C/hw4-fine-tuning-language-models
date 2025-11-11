import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

random.seed(0)

def example_transform(example):
    example["text"] = example["text"].lower()
    return example

### Rough guidelines --- typos
# For typos, you can try to simulate nearest keys on the QWERTY keyboard for some of the letter (e.g. vowels)
# You can randomly select each word with some fixed probability, and replace random letters in that word with one of the
# nearest keys on the keyboard. You can vary the random probablity or which letters to use to achieve the desired accuracy.

### Rough guidelines --- synonym replacement
# For synonyms, use can rely on wordnet (already imported here). Wordnet (https://www.nltk.org/howto/wordnet.html) includes
# something called synsets (which stands for synonymous words) and for each of them, lemmas() should give you a possible synonym word.
# You can randomly select each word with some fixed probability to replace by a synonym.

def _keyboard_neighbors():
    # Minimal neighbor map; enough to create realistic typos without destroying meaning
    return {
        "a": list("qs"),
        "e": list("wr"),
        "i": list("uo"),
        "o": list("ip"),
        "u": list("yi"),
        "s": list("awed"),
        "t": list("ry"),
        "n": list("bhm"),
        "r": list("etf"),
        "l": list("ko"),
    }

def _random_typo(token, rng):
    neigh = _keyboard_neighbors()
    chars = list(token)
    idxs = [i for i, c in enumerate(chars) if c.lower() in neigh]
    if not idxs:
        return token
    i = rng.choice(idxs)
    c = chars[i]
    rep = rng.choice(neigh[c.lower()])
    chars[i] = rep.upper() if c.isupper() else rep
    return "".join(chars)

def _synonym_or_none(word, rng):
    # Try to get a different lemma from wordnet
    synsets = wordnet.synsets(word)
    lemmas = []
    for s in synsets:
        for l in s.lemmas():
            w = l.name().replace("_", " ")
            if w.lower() != word.lower():
                lemmas.append(w)
    if not lemmas:
        return None
    return rng.choice(lemmas)

def custom_transform(example):
    ################################
    ##### YOUR CODE BEGINGS HERE ###
    
    # Hybrid transformation with higher probabilities for more challenging OOD data:
    #  - 20% probability: replace a content word by a WordNet synonym
    #  - 15% probability: inject a single-keyboard-neighbor typo
    # Total: 35% of words are transformed
    
    # Handle batched input (example["text"] is a list)
    texts = example["text"] if isinstance(example["text"], list) else [example["text"]]
    transformed_texts = []
    
    for text in texts:
        rng = random.Random()  # Different transformations per example
        tokens = word_tokenize(text)
        new_tokens = []
        
        for tok in tokens:
            # Only consider alphabetic tokens; keep punctuation/numbers intact
            if tok.isalpha():
                p = rng.random()
                
                # 20% chance: synonym replacement (increased from 10%)
                if p < 0.20:
                    syn = _synonym_or_none(tok, rng)
                    if syn:
                        new_tokens.append(syn)
                        continue  # Skip to next token
                
                # 15% chance (separate from synonym): introduce a keyboard typo (increased from 5%)
                elif p < 0.35:  # 0.20 to 0.35 = 15% range
                    new_tokens.append(_random_typo(tok, rng))
                    continue
            
            # Keep original token (punctuation, numbers, or words not transformed)
            new_tokens.append(tok)
        
        detok = TreebankWordDetokenizer().detokenize(new_tokens)
        transformed_texts.append(detok)
    
    example["text"] = transformed_texts if isinstance(example["text"], list) else transformed_texts[0]
    
    ##### YOUR CODE ENDS HERE ######
    
    return example