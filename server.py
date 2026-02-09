from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

app = Flask(__name__)

print("Loading translation models... (first deploy takes 3-8 minutes)")

# Arabic → English model
ar_en_tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ar-en")
ar_en_model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-ar-en")

# English → Urdu model
en_ur_tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-ur")
en_ur_model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-ur")

print("Models loaded successfully!")

def translate(text, tokenizer, model):
    tokens = tokenizer(text, return_tensors="pt", padding=True)
    output = model.generate(**tokens, max_length=512)
    return tokenizer.decode(output[0], skip_special_tokens=True)

@app.route("/")
def home():
    return "Arabic → Urdu Translator Running"

@app.route("/translate", methods=["POST"])
def t():
    arabic = request.json["text"]
    english = translate(arabic, ar_en_tokenizer, ar_en_model)
    urdu = translate(english, en_ur_tokenizer, en_ur_model)
    return jsonify({"urdu": urdu})

app.run(host="0.0.0.0", port=3000)
