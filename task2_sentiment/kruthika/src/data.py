"""Task 2.1 - data analysis and preprocessing for Yelp Polarity.

Pipeline: raw review -> clean (lowercase, fix escaped newlines, expand "n't", drop
punctuation/digits) -> tokenise -> remove stopwords (keeping negations) -> Porter stem
-> integer ids from a vocabulary built on the training split -> fixed-length padded arrays.
"""
import re
from collections import Counter
from functools import lru_cache

import numpy as np

PAD, UNK = 0, 1

# NLTK's English stopword list contains negations; removing them would turn
# "not good" into "good" and flip the sentiment, so they are kept.
NEGATIONS = {"no", "not", "nor", "never", "none", "nothing", "nobody", "nowhere", "neither", "cannot"}
CONTRAST = {"but", "however", "although", "though", "yet"}


def load_stopwords():
    import nltk
    try:
        from nltk.corpus import stopwords
        words = stopwords.words("english")
    except LookupError:
        nltk.download("stopwords", quiet=True)
        from nltk.corpus import stopwords
        words = stopwords.words("english")
    # Contraction fragments ("didn", "wasn", ...) are covered by expanding n't -> not.
    return set(words) - NEGATIONS


_stemmer = None


@lru_cache(maxsize=None)
def stem(word):
    """Porter stem, cached: the vocabulary is small compared to the token count."""
    global _stemmer
    if _stemmer is None:
        from nltk.stem import PorterStemmer
        _stemmer = PorterStemmer()
    return _stemmer.stem(word)


_ESCAPED = re.compile(r'\\n|\\"|\\t')     # literal "\n" etc. stored as text in this dataset
_ENTITY = re.compile(r"&[a-z]+;|&#\d+;")    # HTML entities such as &amp;
_NEGATION = re.compile(r"n['’]t\b")          # didn't -> did not
_NON_ALPHA = re.compile(r"[^a-z\s]")        # punctuation, digits, special characters


def clean_text(text):
    text = text.lower()
    text = _ESCAPED.sub(" ", text)
    text = _ENTITY.sub(" ", text)
    text = text.replace("can't", "can not").replace("won't", "will not")
    text = _NEGATION.sub(" not", text)
    text = text.replace("'", "")   # don't split remaining contractions: "i'm" -> "im"
    return _NON_ALPHA.sub(" ", text)


def preprocess(text, stopwords, use_stemming=True):
    """Raw review -> list of tokens."""
    tokens = [t for t in clean_text(text).split() if (t not in stopwords and len(t) > 1) or t in NEGATIONS]
    return [stem(t) for t in tokens] if use_stemming else tokens


def build_vocab(token_lists, min_freq, max_size):
    """Words seen at least min_freq times in training, most frequent first; 0 = <pad>, 1 = <unk>."""
    counts = Counter(t for toks in token_lists for t in toks)
    words = [w for w, c in counts.most_common(max_size) if c >= min_freq]
    vocab = {"<pad>": PAD, "<unk>": UNK}
    for w in words:
        vocab[w] = len(vocab)
    return vocab


def encode(token_lists, vocab, max_len):
    """Token lists -> (ids [N, max_len] int32 padded with 0, lengths [N]).
    Reviews longer than max_len are truncated; empty reviews become a single <unk>."""
    ids = np.zeros((len(token_lists), max_len), dtype=np.int32)
    lengths = np.zeros(len(token_lists), dtype=np.int32)
    for i, toks in enumerate(token_lists):
        seq = [vocab.get(t, UNK) for t in toks[:max_len]] or [UNK]
        ids[i, :len(seq)] = seq
        lengths[i] = len(seq)
    return ids, lengths


def stratified_split(labels, val_fraction, seed):
    """Indices for train/validation with the same class ratio in both."""
    rng = np.random.default_rng(seed)
    train_idx, val_idx = [], []
    for c in np.unique(labels):
        idx = rng.permutation(np.flatnonzero(labels == c))
        n_val = int(round(len(idx) * val_fraction))
        val_idx.append(idx[:n_val])
        train_idx.append(idx[n_val:])
    return np.sort(np.concatenate(train_idx)), np.sort(np.concatenate(val_idx))


def data_quality(train_texts, test_texts):
    """Counts of missing / malformed / duplicated entries (brief 2.1.2)."""
    train_set = set(train_texts)
    return {
        "train_missing_or_empty": sum(1 for t in train_texts if not isinstance(t, str) or not t.strip()),
        "test_missing_or_empty": sum(1 for t in test_texts if not isinstance(t, str) or not t.strip()),
        "train_duplicate_texts": len(train_texts) - len(train_set),
        "test_texts_also_in_train": sum(1 for t in set(test_texts) if t in train_set),
        "train_with_escaped_newlines": sum(1 for t in train_texts if "\\n" in t),
        "train_with_html_entities": sum(1 for t in train_texts if _ENTITY.search(t.lower())),
    }


def slices(raw_texts):
    """Data slices for robustness analysis: review length, negation, contrast words."""
    words = [t.lower().split() for t in raw_texts]
    n_words = np.array([len(w) for w in words])
    length = np.where(n_words <= 50, "short (<=50 words)",
                      np.where(n_words <= 200, "medium (51-200)", "long (>200)"))
    has_neg = np.array([any(x.strip(".,!?\"'") in NEGATIONS or "n't" in x for x in w) for w in words])
    has_con = np.array([any(x.strip(".,!?\"'") in CONTRAST for x in w) for w in words])
    return {
        "length": length,
        "negation": np.where(has_neg, "has negation", "no negation"),
        "contrast": np.where(has_con, "has contrast word", "no contrast word"),
    }
