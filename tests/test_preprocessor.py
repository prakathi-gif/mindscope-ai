"""
Unit tests for data preprocessing and privacy redaction.
"""

import pytest
from data.loader import redact_sensitive_pii
from data.preprocessor import clean_academic_text, ClassicalPreprocessor

def test_redact_sensitive_pii():
    raw = "Hey /u/anxious_user99 check this out at https://reddit.com/r/depression or email me at user@test.com"
    redacted = redact_sensitive_pii(raw)
    assert "[USER_REDACTED]" in redacted
    assert "[URL_REDACTED]" in redacted
    assert "[EMAIL_REDACTED]" in redacted
    assert "user@test.com" not in redacted
    assert "anxious_user99" not in redacted

def test_clean_academic_text_preserves_negations():
    raw = "I do not want to fail, I cannot sleep and never feel rested... &#39;exhausted&#39;"
    cleaned = clean_academic_text(raw)
    # Ensure negations are preserved
    assert "not" in cleaned
    assert "never" in cleaned
    assert "cannot" in cleaned
    # Ensure HTML entities unescaped
    assert "&#39;" not in cleaned
    assert "'" in cleaned or "exhausted" in cleaned

def test_classical_preprocessor():
    texts = [
        "I feel extreme panic and anxiety today",
        "Depression and lack of motivation are overwhelming",
        "Struggling with attention deficit and focus issues"
    ]
    prep = ClassicalPreprocessor(max_features=100, min_df=1)
    X = prep.fit_transform(texts)
    assert X.shape[0] == 3
    assert X.shape[1] > 0
    # Transform test
    X_test = prep.transform(["New panic and anxiety text"])
    assert X_test.shape[0] == 1
    assert X_test.shape[1] == X.shape[1]

