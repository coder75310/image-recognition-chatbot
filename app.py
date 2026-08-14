import os
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

app = Flask(__name__)

# Load the BLIP image-captioning model once when the service starts.
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
model.eval()

UPLOAD_FOLDER = os.path.join("static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def get_caption_from_model(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")

        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=50)

        caption = processor.decode(outputs[0], skip_special_tokens=True)
        return caption.capitalize()

    except Exception as e:
        print(f"Error generating caption: {e}")
        return "Sorry, I encountered an error while analyzing the image."


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

    filename = secure_filename(image_file.filename)
    image_path = os.path.join(UPLOAD_FOLDER, filename)
    image_file.save(image_path)

    caption = get_caption_from_model(image_path)
    return jsonify({"answer": caption})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
