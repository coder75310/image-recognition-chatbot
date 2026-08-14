import base64
import os
from flask import Flask, render_template, request, jsonify
from openai import OpenAI
from PIL import Image

app = Flask(__name__)

HF_TOKEN = os.environ.get("HF_TOKEN")
MODEL = os.environ.get("HF_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct")

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=HF_TOKEN,
) if HF_TOKEN else None


def image_to_data_url(file_storage):
    image = Image.open(file_storage).convert("RGB")
    image.thumbnail((1024, 1024))
    from io import BytesIO
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=85, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def answer_image_question(image_file, question):
    if not client:
        raise RuntimeError("HF_TOKEN is not configured on the server.")

    image_url = image_to_data_url(image_file)
    prompt = question or "Describe this image clearly and briefly."

    completion = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ],
        max_tokens=200,
    )

    return completion.choices[0].message.content.strip()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/ask", methods=["POST"])
def ask():
    if "image" not in request.files:
        return jsonify({"error": "Missing image in the request."}), 400

    image_file = request.files["image"]
    if not image_file.filename:
        return jsonify({"error": "No image selected."}), 400

    question = request.form.get("question", "").strip()

    try:
        answer = answer_image_question(image_file, question)
        return jsonify({"answer": answer})
    except Exception as exc:
        print(f"Inference error: {exc}")
        return jsonify({"error": "Unable to analyze the image right now. Please try again."}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
