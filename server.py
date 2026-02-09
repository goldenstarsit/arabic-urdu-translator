from flask import Flask, request, jsonify
import subprocess, sys, threading, os

app = Flask(__name__)

translator = None
status = "cold"   # cold | installing | loading | ready

# ---------------- INSTALL ----------------
def install_ai():
    global status
    status = "installing"

    subprocess.call([sys.executable, "-m", "pip", "install",
                     "torch", "--index-url", "https://download.pytorch.org/whl/cpu"])
    subprocess.call([sys.executable, "-m", "pip", "install",
                     "transformers", "sentencepiece"])

    load_model()

# ---------------- LOAD MODEL ----------------
def load_model():
    global translator, status
    status = "loading"

    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

    model_name = "UBC-NLP/araT5-small-arabic-to-urdu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    translator = pipeline("translation", model=model, tokenizer=tokenizer)

    status = "ready"

# ---------------- HOME ----------------
@app.route("/")
def home():
    return f"Translator status: {status}"

# ---------------- TRANSLATE ----------------
@app.route("/translate", methods=["POST"])
def translate():
    global translator, status

    text = request.json.get("text", "")

    # first call ever → start installation in background
    if status == "cold":
        threading.Thread(target=install_ai).start()
        return jsonify({"status": "warming_up"}), 200

    # still preparing
    if status in ["installing", "loading"]:
        return jsonify({"status": status}), 200

    # ready
    result = translator(text, max_length=512)
    return jsonify({"translated": result[0]["translation_text"]}), 200


# ---------------- START ----------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
