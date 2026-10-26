"""
Module 3: Similarity Engine
Implements TF-IDF Vectorization and Cosine Similarity to compare Resume and Job Description.
Also extracts top contributing keywords and missing high-weight terms.
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def compute_similarity(resume_clean_text: str, jd_clean_text: str) -> Dict[str, Any]:
    """
    Computes TF-IDF vectorization and cosine similarity between resume and JD.
    Returns:
    - match_percentage (float, 0-100)
    - top_shared_keywords: list of tuples (word, importance)
    - top_jd_keywords: list of tuples (word, importance in JD)
    - missing_keywords: list of high-value JD keywords missing in resume
    - vectorizer_stats: vocab size, etc.
    """
    if not resume_clean_text.strip() or not jd_clean_text.strip():
        return {
            "similarity_score": 0.0,
            "match_percentage": 0.0,
            "match_level": "No Data",
            "match_color": "#94a3b8",
            "top_shared_keywords": [],
            "top_jd_keywords": [],
            "missing_keywords": [],
            "vector_details": {}
        }

    # Use 1-gram and 2-gram (unigrams + bigrams) for richer NLP phrase capture
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=150,
        stop_words=None  # Stopwords already removed in preprocessor
    )

    documents = [resume_clean_text, jd_clean_text]
    tfidf_matrix = vectorizer.fit_transform(documents)

    # Compute cosine similarity
    cos_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    match_percentage = round(float(cos_sim) * 100, 2)

    # Feature names (words / bi-grams)
    feature_names = np.array(vectorizer.get_feature_names_out())

    resume_vector = tfidf_matrix[0].toarray().flatten()
    jd_vector = tfidf_matrix[1].toarray().flatten()

    # Find shared words (non-zero in both)
    shared_mask = (resume_vector > 0) & (jd_vector > 0)
    shared_indices = np.where(shared_mask)[0]
    # Shared score = product of both weights
    shared_scores = resume_vector[shared_indices] * jd_vector[shared_indices]
    shared_sort = np.argsort(shared_scores)[::-1]

    top_shared_keywords = []
    for idx in shared_sort[:15]:
        kw_idx = shared_indices[idx]
        top_shared_keywords.append({
            "keyword": feature_names[kw_idx],
            "resume_weight": round(float(resume_vector[kw_idx]), 3),
            "jd_weight": round(float(jd_vector[kw_idx]), 3),
            "relevance": round(float(shared_scores[idx]), 4)
        })

    # Missing keywords: words high in JD (weight > 0) but zero or near zero in resume
    missing_mask = (jd_vector > 0) & (resume_vector == 0)
    missing_indices = np.where(missing_mask)[0]
    missing_scores = jd_vector[missing_indices]
    missing_sort = np.argsort(missing_scores)[::-1]

    missing_keywords = []
    for idx in missing_sort[:15]:
        kw_idx = missing_indices[idx]
        missing_keywords.append({
            "keyword": feature_names[kw_idx],
            "jd_importance": round(float(missing_scores[idx]), 3)
        })

    # Top JD keywords overall
    jd_sort = np.argsort(jd_vector)[::-1]
    top_jd_keywords = []
    for kw_idx in jd_sort[:15]:
        if jd_vector[kw_idx] > 0:
            top_jd_keywords.append({
                "keyword": feature_names[kw_idx],
                "jd_importance": round(float(jd_vector[kw_idx]), 3),
                "in_resume": bool(resume_vector[kw_idx] > 0)
            })

    # Match tier classification
    if match_percentage >= 75:
        match_level = "Strong Match"
        match_color = "#10b981"  # Emerald green
    elif match_percentage >= 50:
        match_level = "Moderate Match"
        match_color = "#f59e0b"  # Amber
    elif match_percentage >= 30:
        match_level = "Low-Moderate Match"
        match_color = "#f97316"  # Orange
    else:
        match_level = "Low Match"
        match_color = "#ef4444"  # Red

    return {
        "similarity_score": round(float(cos_sim), 4),
        "match_percentage": match_percentage,
        "match_level": match_level,
        "match_color": match_color,
        "top_shared_keywords": top_shared_keywords,
        "top_jd_keywords": top_jd_keywords,
        "missing_keywords": missing_keywords,
        "vocab_size": len(feature_names)
    }
