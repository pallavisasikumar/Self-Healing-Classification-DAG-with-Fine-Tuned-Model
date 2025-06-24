from transformers import pipeline

# Zero-shot backup model
backup_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
import logging

# Load model
model = DistilBertForSequenceClassification.from_pretrained("./trained_model")
tokenizer = DistilBertTokenizerFast.from_pretrained("./trained_model")
model.eval()

# Setup logging
logging.basicConfig(filename="logs.txt", level=logging.INFO, format='%(asctime)s - %(message)s')

# Inference node
def inference_node(text):
    tokens = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        output = model(**tokens)
        probs = torch.softmax(output.logits, dim=1)
        confidence, pred = torch.max(probs, dim=1)

    label = "Positive" if pred.item() == 1 else "Negative"
    confidence_value = confidence.item()
    logging.info(f"[InferenceNode] Input: '{text}' | Prediction: {label} | Confidence: {confidence_value:.2f}")
    return label, confidence_value

# Confidence check node
def confidence_check_node(label, confidence, threshold=0.70):
    if confidence < threshold:
        print("[ConfidenceCheckNode] Confidence too low. Triggering fallback...")
        logging.info(f"[ConfidenceCheckNode] Confidence {confidence:.2f} below threshold. Triggering fallback.")
        return False
    else:
        print(f"[ConfidenceCheckNode] Confidence high ({int(confidence*100)}%). Proceeding.")
        logging.info(f"[ConfidenceCheckNode] Confidence {confidence:.2f} accepted.")
        return True


# Fallback node
def fallback_node(label, confidence):
    print(f"[FallbackNode] Model predicted '{label}' with {int(confidence * 100)}% confidence.")
    response = input("Do you agree with this label? (yes/no): ").strip().lower()
    
    if response == "yes":
        logging.info(f"[FallbackNode] User accepted label '{label}'")
        return label
    else:
        backup = input("Do you want help from a backup model? (yes/no): ").strip().lower()
        if backup == "yes":
            text = input("Re-enter the sentence for backup model: ").strip()
            result = backup_classifier(text, candidate_labels=["Positive", "Negative"])
            backup_label = result["labels"][0]
            print(f"[BackupModel] Suggested label: {backup_label}")
            logging.info(f"[BackupModel] Suggested label: {backup_label} | Scores: {result['scores']}")
            return backup_label
        else:
            corrected = input("Please enter the correct label (Positive/Negative): ").strip()
            logging.info(f"[FallbackNode] User corrected label to '{corrected}'")
            return corrected
