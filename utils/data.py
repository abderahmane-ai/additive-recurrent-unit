"""
Data loading and preprocessing utilities.
"""

import re
import torch
from collections import Counter
from typing import Dict, List, Tuple, Optional


def clean_text(text: str) -> str:
    """Clean and normalize text."""
    text = re.sub(r'<[^>]+>', '', text)  # Remove HTML
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s.,!?]', '', text)
    text = ' '.join(text.split())
    return text


def build_vocab(texts: List[str], max_vocab_size: int = 10000) -> Dict[str, int]:
    """Build vocabulary from texts."""
    word_counts = Counter()
    for text in texts:
        word_counts.update(text.split())
    
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for word, _ in word_counts.most_common(max_vocab_size - 2):
        vocab[word] = len(vocab)
    
    return vocab


def text_to_indices(text: str, vocab: Dict[str, int], max_len: int) -> List[int]:
    """Convert text to padded token indices."""
    words = text.split()
    if max_len:
        words = words[:max_len]
    indices = [vocab.get(w, 1) for w in words]
    if max_len:
        indices += [0] * (max_len - len(indices))
    return indices


def load_imdb_dataset(
    max_vocab_size: int = 10000,
    max_len: int = 200,
    train_samples: Optional[int] = None,
    test_samples: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, int]]:
    """
    Load IMDB dataset from Hugging Face.
    
    Returns:
        X_train, y_train, X_test, y_test, vocab
    """
    from datasets import load_dataset
    
    print("Loading IMDB dataset...")
    dataset = load_dataset('imdb')
    
    train_data = dataset['train'].shuffle(seed=42)
    test_data = dataset['test'].shuffle(seed=42)
    
    if train_samples:
        train_data = train_data.select(range(min(train_samples, len(train_data))))
    if test_samples:
        test_data = test_data.select(range(min(test_samples, len(test_data))))
    
    print(f"✓ Loaded {len(train_data)} train / {len(test_data)} test samples")
    
    # Preprocess
    train_texts = [clean_text(t) for t in train_data['text']]
    test_texts = [clean_text(t) for t in test_data['text']]
    
    # Build vocab
    vocab = build_vocab(train_texts, max_vocab_size)
    print(f"✓ Vocabulary: {len(vocab):,} words")
    
    X_train = torch.tensor([text_to_indices(t, vocab, max_len) for t in train_texts])
    X_test = torch.tensor([text_to_indices(t, vocab, max_len) for t in test_texts])
    y_train = torch.tensor(train_data['label'])
    y_test = torch.tensor(test_data['label'])
    
    return X_train, y_train, X_test, y_test, vocab


def load_sentiment_dataset(
    dataset_name: str,
    max_vocab_size: int = 10000,
    max_len: int = 200,
    seed: int = 42
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, int], int]:
    """
    Load a supported sentiment dataset.
    
    Supported: 'sst2', 'sst5', 'mr', 'cr', 'subj', 'trec'
    
    Returns:
        X_train, y_train, X_test, y_test, vocab, num_classes
    """
    from datasets import load_dataset, concatenate_datasets
    
    print(f"Loading {dataset_name.upper()} dataset...")
    
    dataset_config = {
        'sst2': {'path': 'glue', 'name': 'sst2', 'text': 'sentence', 'label': 'label', 'classes': 2},
        'sst5': {'path': 'SetFit/sst5', 'name': None, 'text': 'text', 'label': 'label', 'classes': 5},
        'mr': {'path': 'rotten_tomatoes', 'name': None, 'text': 'text', 'label': 'label', 'classes': 2},
        'cr': {'path': 'SetFit/CR', 'name': None, 'text': 'text', 'label': 'label', 'classes': 2},
        'subj': {'path': 'SetFit/subj', 'name': None, 'text': 'text', 'label': 'label', 'classes': 2},
        'trec': {'path': 'trec', 'name': None, 'text': 'text', 'label': 'coarse_label', 'classes': 6},
    }
    
    if dataset_name not in dataset_config:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    cfg = dataset_config[dataset_name]
    if cfg['name']:
        dataset = load_dataset(cfg['path'], cfg['name'])
    else:
        dataset = load_dataset(cfg['path'])
        
    # Standardize splits
    if 'validation' in dataset and 'test' not in dataset:
        test_split = dataset['validation']
    elif 'test' in dataset:
        test_split = dataset['test']
    else:
        # If no test set, split train
        full_data = dataset['train'].train_test_split(test_size=0.1, seed=seed)
        dataset = full_data
        test_split = dataset['test']
        
    train_data = dataset['train'].shuffle(seed=seed)
    # Some datasets like SST-2 have hidden test labels (-1), use validation as test if needed
    if dataset_name == 'sst2':
         # Glue SST-2 test set has no labels, use validation
         test_split = dataset['validation']

    print(f"✓ {len(train_data)} train / {len(test_split)} test samples")
    
    # Preprocess
    train_texts = [clean_text(t) for t in train_data[cfg['text']]]
    test_texts = [clean_text(t) for t in test_split[cfg['text']]]
    
    # Build vocab covering both for safety in suite
    full_texts = train_texts + test_texts
    vocab = build_vocab(full_texts, max_vocab_size)
    print(f"✓ Vocabulary: {len(vocab):,} words")
    
    X_train = torch.tensor([text_to_indices(t, vocab, max_len) for t in train_texts])
    X_test = torch.tensor([text_to_indices(t, vocab, max_len) for t in test_texts])
    
    y_train = torch.tensor(train_data[cfg['label']])
    y_test = torch.tensor(test_split[cfg['label']])
    
    return X_train, y_train, X_test, y_test, vocab, cfg['classes']


def load_ag_news_dataset(
    max_vocab_size: int = 10000,
    max_len: int = 128,
    train_samples: Optional[int] = None,
    test_samples: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, int]]:
    """
    Load AG News dataset from Hugging Face.
    
    Returns:
        X_train, y_train, X_test, y_test, vocab
    """
    from datasets import load_dataset
    
    print("Loading AG News dataset...")
    dataset = load_dataset('ag_news')
    
    train_data = dataset['train'].shuffle(seed=42)
    test_data = dataset['test'].shuffle(seed=42)
    
    if train_samples:
        train_data = train_data.select(range(min(train_samples, len(train_data))))
    if test_samples:
        test_data = test_data.select(range(min(test_samples, len(test_data))))
        
    print(f"✓ Loaded {len(train_data)} train / {len(test_data)} test samples")
    
    # Preprocess
    train_texts = [clean_text(t) for t in train_data['text']]
    test_texts = [clean_text(t) for t in test_data['text']]
    
    # Build vocab
    vocab = build_vocab(train_texts, max_vocab_size)
    print(f"✓ Vocabulary: {len(vocab):,} words")
    
    X_train = torch.tensor([text_to_indices(t, vocab, max_len) for t in train_texts])
    X_test = torch.tensor([text_to_indices(t, vocab, max_len) for t in test_texts])
    y_train = torch.tensor(train_data['label'])
    y_test = torch.tensor(test_data['label'])
    
    return X_train, y_train, X_test, y_test, vocab
