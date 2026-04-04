import torch
from transformers import AutoTokenizer,AutoModelForTokenClassification
MODEL_PATH="/content/drive/MyDrive/Autonomous-threat-Intelligence-Agent/models/ner-bert"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)
id2label = model.config.id2label
def predict(text):
    inputs = tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    predictions = torch.argmax(outputs.logits, dim=2)

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    predicted_labels = [id2label[p.item()] for p in predictions[0]]

    print("\nDEBUG:")
    print("Tokens:", tokens)
    print("Labels:", predicted_labels)

    results = list(zip(tokens, predicted_labels))

    return results
def extract_entities(results):
    entities = []
    current_entity = ""
    current_label = None

    for token, label in results:

        # Skip special tokens
        if token in ["[CLS]", "[SEP]"]:
            continue

        # Remove ## from subwords
        if token.startswith("##"):
            current_entity += token[2:]
            continue

        if label.startswith("B-"):
            # Save previous entity
            if current_entity:
                entities.append((current_entity, current_label))

            current_entity = token
            current_label = label[2:]

        elif label.startswith("I-") and current_entity:
            current_entity += " " + token

        else:
            if current_entity:
                entities.append((current_entity, current_label))
                current_entity = ""
                current_label = None

    # Last entity
    if current_entity:
        entities.append((current_entity, current_label))

    return entities
if __name__ == "__main__":
    test_cases = [
        "SQL injection vulnerability in MySQL server",
        "Remote code execution in Apache HTTP Server",
        "Cross-site scripting attack in web application",
        "Buffer overflow in OpenSSL library",
        "Denial of service vulnerability in Linux kernel",
        "Authentication bypass in Cisco router firmware"
    ]

    for text in test_cases:
        print("\n==============================")
        print("INPUT:", text)

        results = predict(text)
        entities = extract_entities(results)

        print("\nOUTPUT:")
        if entities:
            for entity, label in entities:
                print(f"{label} → {entity}")
        else:
            print("No entities found")