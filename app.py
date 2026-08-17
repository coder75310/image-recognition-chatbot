import os, base64, logging
from io import BytesIO
from flask import Flask, render_template, request, jsonify
from PIL import Image, UnidentifiedImageError
from huggingface_hub import InferenceClient

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024
logging.basicConfig(level=logging.INFO)

HF_TOKEN = os.getenv("HF_TOKEN")
VISION_MODEL = os.getenv("HF_VISION_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct")
CAPTION_MODEL = os.getenv("HF_CAPTION_MODEL", "Salesforce/blip-image-captioning-base")

client = InferenceClient(provider="auto", api_key=HF_TOKEN, timeout=90) if HF_TOKEN else None

@app.get("/")
def index():
    return render_template("index.html")

def data_url(data, mime):
    return f"data:{mime if mime and mime.startswith('image/') else 'image/jpeg'};base64,{base64.b64encode(data).decode()}"

def vision(data, mime, question):
    if not client:
        raise RuntimeError("HF_TOKEN is not configured.")
    r = client.chat.completions.create(
        model=VISION_MODEL,
        messages=[
            {"role":"system","content":"You are a helpful image recognition assistant. Answer only from what can reasonably be inferred from the image. If uncertain, say so."},
            {"role":"user","content":[
                {"type":"text","text":question},
                {"type":"image_url","image_url":{"url":data_url(data, mime)}}
            ]}
        ],
        max_tokens=180
    )
    answer = r.choices[0].message.content
    if isinstance(answer, list):
        answer = " ".join(x.get("text","") for x in answer if isinstance(x,dict))
    return str(answer).strip()

def caption(data):
    if not client:
        raise RuntimeError("HF_TOKEN is not configured.")
    r = client.image_to_text(data, model=CAPTION_MODEL)
    return str(getattr(r, "generated_text", r)).strip()

@app.post("/api/analyze")
def analyze():
    if "image" not in request.files:
        return jsonify(error="Please upload an image."), 400
    f = request.files["image"]
    if not f.filename:
        return jsonify(error="Please choose an image."), 400
    data = f.read()
    try:
        with Image.open(BytesIO(data)) as im:
            im.verify()
    except (UnidentifiedImageError, OSError):
        return jsonify(error="Please upload a valid image."), 400

    question = request.form.get("question","").strip() or "Describe this image and identify the main objects or scene."
    try:
        answer = vision(data, f.mimetype, question)
        return jsonify(answer=answer, mode="vision")
    except Exception as e:
        logging.exception("Vision inference failed: %s", e)
        try:
            answer = caption(data)
            return jsonify(answer=answer, mode="caption-fallback")
        except Exception:
            logging.exception("Caption fallback failed")
            return jsonify(error="Unable to analyze the image right now. Check that HF_TOKEN is correctly set in Render."), 502

@app.errorhandler(413)
def too_large(_):
    return jsonify(error="Image is too large. Please upload an image under 8 MB."), 413

@app.get("/health")
def health():
    return jsonify(status="ok", hf_token_configured=bool(HF_TOKEN), vision_model=VISION_MODEL)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT","10000")))
