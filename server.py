from flask import Flask, request, jsonify
import subprocess
import sys
import threading

app = Flask(__name__)

translator = None
loading = False
installing = False

def install_ai():
    global installing
    installing = True

    subprocess.check_call([sys.executable, "-m", "pip", "install", "torch", "--index-url", "https://download.pytorch.org/whl/cpu"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers", "sentencepiece"])

    installing = False


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
    return "Translator alive"


@app.route("/translate", methods=["POST"])
def translate():
    global translator, loading, installing

    text = request.json.get("text", "")

    # Install libraries first time
    if translator is None and not installing and not loading:
        threading.Thread(target=install_ai).start()
        return jsonify({"status": "installing_ai"}), 202

    if installing:
        return jsonify({"status": "installing_ai"}), 202

    if translator is None and not loading:
        threading.Thread(target=load_model).start()
        return jsonify({"status": "loading_model"}), 202

    if loading:
        return jsonify({"status": "loading_model"}), 202

    result = translator(text, max_length=512)
    return jsonify({"translated": result[0]["translation_text"]})
