from flask import Flask, request, jsonify
import threading

app = Flask(__name__)

translator = None
loading = False

def load_model():
    global translator, loading
    loading = True

    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

    model_name = "UBC-NLP/araT5-small-arabic-to-urdu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    translator = pipeline("translation", model=model, tokenizer=tokenizer)
    loading = False


@app.route("/")
def home():
    return "Translator running"

@app.route("/translate", methods=["POST"])
def translate():
    global translator, loading

    text = request.json.get("text", "")

    if translator is None:
        if not loading:
            threading.Thread(target=load_model).start()
        return jsonify({"status": "loading"}), 202

    result = translator(text, max_length=512)
    return jsonify({"translated": result[0]["translation_text"]})
