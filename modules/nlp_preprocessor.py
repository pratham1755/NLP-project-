"""
Module 2: NLP Preprocessing
Handles text cleaning, lowercasing, tokenization, stopword removal,
and lemmatization using spaCy with resilient fallback.
"""

import re
from typing import List, Dict, Any, Tuple

# Try loading spaCy
_SPACY_NLP = None
_SPACY_LOADED = False

try:
    import spacy
    try:
        _SPACY_NLP = spacy.load("en_core_web_sm")
        _SPACY_LOADED = True
    except OSError:
        # Fallback to blank English model if en_core_web_sm not downloaded
        _SPACY_NLP = spacy.blank("en")
        _SPACY_LOADED = False
except Exception:
    _SPACY_NLP = None
    _SPACY_LOADED = False

# Default English stopwords list for fallback
STANDARD_STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both",
    "but", "by", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't",
    "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn't",
    "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll",
    "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's",
    "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", "once",
    "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same",
    "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there",
    "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those",
    "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we", "we'd",
    "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", "when's", "where",
    "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "won't",
    "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "yourself", "yourselves", "candidate", "responsibilities", "requirements", "experience",
    "working", "looking", "knowledge", "skills", "ability", "years"
}


def clean_raw_text(text: str) -> str:
    """
    Cleans raw text:
    - Removes URLs and hyperlinks
    - Removes email addresses and phone numbers
    - Normalizes special bullet points and punctuation
    - Strips excess whitespace
    """
    if not text:
        return ""

    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    # Remove emails
    text = re.sub(r"\S+@\S+\.\S+", " ", text)
    # Remove phone-like strings
    text = re.sub(r"\(?\+?\d{1,3}\)?[-.\s]?\d{3}[-.\s]?\d{3,4}[-.\s]?\d{3,4}", " ", text)
    # Remove non-alphanumeric punctuation except common tech symbols like C++, C#, .NET
    # We protect C++, C#, .NET first
    text = text.replace("C++", "Cplusplus").replace("c++", "cplusplus")
    text = text.replace("C#", "Csharp").replace("c#", "csharp")
    text = text.replace(".NET", "dotnet").replace(".net", "dotnet")
    # Replace non-letters/digits/spaces with space
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    # Restore tech names
    text = text.replace("Cplusplus", "c++").replace("cplusplus", "c++")
    text = text.replace("Csharp", "c#").replace("csharp", "c#")
    text = text.replace("dotnet", ".net")
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_text(text: str) -> str:
    """
    Full NLP preprocessing pipeline:
    1. Cleaning
    2. Lowercasing
    3. Tokenization
    4. Stopword removal
    5. Lemmatization
    Returns clean preprocessed text string for vectorization.
    """
    if not text:
        return ""

    cleaned = clean_raw_text(text).lower()

    if _SPACY_NLP is not None and _SPACY_LOADED:
        doc = _SPACY_NLP(cleaned)
        lemmas = []
        for token in doc:
            if not token.is_stop and not token.is_punct and len(token.lemma_.strip()) > 1:
                lemmas.append(token.lemma_.strip())
        return " ".join(lemmas)

    # Fallback tokenizer and stop-word filter
    tokens = cleaned.split()
    filtered = [
        token for token in tokens
        if token not in STANDARD_STOPWORDS and len(token) > 1 and not token.isdigit()
    ]
    return " ".join(filtered)


def get_preprocessing_demonstration(text: str) -> Dict[str, Any]:
    """
    Produces step-by-step transformation for college viva demonstration:
    - Step 1: Raw snippet
    - Step 2: Cleaned & lowercased
    - Step 3: Tokens
    - Step 4: After stopword removal
    - Step 5: After lemmatization
    """
    raw_sample = text[:250].strip() if text else "The candidate is working with Python and developing data analysis projects."
    cleaned = clean_raw_text(raw_sample).lower()
    tokens = cleaned.split()

    if _SPACY_NLP is not None and _SPACY_LOADED:
        doc = _SPACY_NLP(cleaned)
        all_tokens = [t.text for t in doc if t.text.strip()]
        without_stops = [t.text for t in doc if not t.is_stop and not t.is_punct and len(t.text.strip()) > 1]
        lemmatized = [t.lemma_ for t in doc if not t.is_stop and not t.is_punct and len(t.lemma_.strip()) > 1]
    else:
        all_tokens = tokens
        without_stops = [t for t in tokens if t not in STANDARD_STOPWORDS]
        lemmatized = without_stops  # approximate fallback

    return {
        "raw_text": raw_sample,
        "cleaned_text": cleaned,
        "all_tokens": all_tokens[:20],
        "without_stopwords": without_stops[:20],
        "lemmatized_tokens": lemmatized[:20],
        "spacy_model_active": _SPACY_LOADED
    }
