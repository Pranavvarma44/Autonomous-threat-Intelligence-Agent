import re
import pandas as pd


LABEL_LIST = ["O", "B-SOFTWARE", "I-SOFTWARE", "B-ATTACK", "I-ATTACK"]

# 🔥 Expanded SOFTWARE list
SOFTWARE_TERMS = [
    ("apache",),
    ("mysql",),
    ("nginx",),
    ("openssl",),
    ("linux",),
    ("windows",),
    ("ubuntu",),
    ("cisco",),
    ("router",),
    ("server",),
    ("kernel",),
]

# 🔥 Expanded ATTACK list
ATTACK_TERMS = [
    ("sql", "injection"),
    ("buffer", "overflow"),
    ("cross", "site", "scripting"),
    ("xss",),
    ("remote", "code", "execution"),
    ("rce",),
    ("denial", "of", "service"),
    ("dos",),
    ("authentication", "bypass"),
    ("privilege", "escalation"),
]


def normalize_token(token):
    return re.sub(r"^\W+|\W+$", "", str(token).lower())


def apply_phrase_labels(normalized_words, labels, phrases, entity_name):
    for phrase in phrases:
        phrase_length = len(phrase)

        for start_index in range(len(normalized_words) - phrase_length + 1):
            window = normalized_words[start_index:start_index + phrase_length]

            if tuple(window) != phrase:
                continue

            # Skip if already labeled
            if any(label != "O" for label in labels[start_index:start_index + phrase_length]):
                continue

            labels[start_index] = f"B-{entity_name}"

            for offset in range(1, phrase_length):
                labels[start_index + offset] = f"I-{entity_name}"


def label_text(text):
    words = str(text).split()
    normalized_words = [normalize_token(word) for word in words]
    labels = ["O"] * len(words)

    apply_phrase_labels(normalized_words, labels, SOFTWARE_TERMS, "SOFTWARE")
    apply_phrase_labels(normalized_words, labels, ATTACK_TERMS, "ATTACK")

    return words, labels


def prepare_dataset(csv_path):
    df = pd.read_csv(csv_path)

    # 🔥 Optional: reduce dataset for faster training
    df = df.sample(min(len(df), 8000))

    data = []

    for _, row in df.iterrows():
        words, labels = label_text(row["description"])

        if words:
            data.append({"tokens": words, "ner_tags": labels})

    return data