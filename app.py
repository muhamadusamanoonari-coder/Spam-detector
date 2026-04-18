from flask import Flask, request, jsonify, render_template
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

app = Flask(__name__)

    # Load your fine-tuned model and tokenizer
    # Note: Ensure you have run train.py first so the ./spam_model folder exists!
MODEL_PATH = "./spam_model"
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
except OSError:
    print("Error: Model not found. Please run train.py first!")
    exit()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get('text', '')

    if not text:
        return jsonify({'error': 'No text provided'}), 400

        # Tokenize and predict
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        
    with torch.no_grad():
        logits = model(**inputs).logits
            
    predicted_class_id = logits.argmax().item()
        
        # In Enron Spam: 1 is Spam, 0 is Non-Spam (Ham)
    result = "Spam" if predicted_class_id == 1 else "Non-Spam"
        
    return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(debug=True)