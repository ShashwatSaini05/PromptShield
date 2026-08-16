"""
predict.py
----------
Standalone CLI for classifying a single prompt with the trained
PromptShield model. Loads the already-trained model and vectorizer
from disk — it does NOT retrain anything.

Usage:
    python predict.py
    Enter a prompt: Ignore previous instructions

    Prediction: PROMPT_INJECTION
    Confidence: 97.23%

Press Ctrl+C or type an empty line to exit.
"""

import sys
import joblib

MODEL_PATH = "model/prompt_injection_model.pkl"
VECTORIZER_PATH = "model/tfidf_vectorizer.pkl"


def load_artifacts():
    try:
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
    except FileNotFoundError as e:
        print(f"Could not find saved model artifacts ({e}).")
        print("Run `python train.py` first to train and save the model.")
        sys.exit(1)
    return model, vectorizer


def predict(text, model, vectorizer):
    x = vectorizer.transform([text])
    pred = model.predict(x)[0]
    label = "PROMPT_INJECTION" if pred == 1 else "SAFE"

    if hasattr(model, "predict_proba"):
        confidence = float(model.predict_proba(x)[0][pred])
    else:
        # Fallback for models without predict_proba
        score = model.decision_function(x)[0]
        confidence = float(1 / (1 + abs(score)))

    return label, confidence


def main():
    model, vectorizer = load_artifacts()
    print("PromptShield — prompt injection classifier")
    print("(this is a statistical classifier, not a guarantee of safety)")
    print()

    while True:
        try:
            text = input("Enter a prompt: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if not text:
            print("Exiting.")
            break

        label, confidence = predict(text, model, vectorizer)
        print(f"\nPrediction: {label}")
        print(f"Confidence: {confidence * 100:.2f}%\n")


if __name__ == "__main__":
    main()
