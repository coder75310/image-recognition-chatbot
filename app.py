import os
import base64
import logging
from io import BytesIO

from flask import Flask, render_template, request, jsonify
from PIL import Image, UnidentifiedImageError
from huggingface_hub import InferenceClient

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

HF_TOKEN = os.getenv("HF_TOKEN", "").strip()
VISION_MODEL = os.getenv(
    "HF_VISION_MODEL",
    "Qwen/Qwen2.5-VL-3B-Instruct:featherless-ai"
).strip()
CAPTION_MODEL = os.getenv(
    "HF_CAPTION_MODEL",
    "Salesforce/blip-image-captioning-base"
).strip()

client = (
    InferenceClient(
        provider="featherless-ai",
        api_key=HF_TOKEN,
        timeout=90
    )
    if HF_TOKEN
    else None
)


@app.get("/")
def index():
    return render_template("index.html")


def make_data_url(data: bytes, mime: str) -> str:
    content_type = mime if mime and mime.startswith("image/") else "image/jpeg"
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{content_type};base64,{encoded}"


def extract_answer(response) -> str:
    answer = response.choices[0].message.content

    if isinstance(answer, str):
        return answer.strip()

    if isinstance(answer, list):
        parts = []
        for item in answer:
            if isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(str(text))
            elif isinstance(item, str):
                parts.append(item)
        return " ".join(parts).strip()

    return str(answer).strip()


def analyze_with_vision(data: bytes, mime: str, question: str) -> str:
    if not HF_TOKEN:
        raise RuntimeError(
            "HF_TOKEN is missing from the Render environment."
        )

    if client is None:
        raise RuntimeError(
            "Hugging Face client could not be initialized."
        )

    response = client.chat.completions.create(
        model=VISION_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful image recognition assistant. "
                    "Answer the user's question using only information "
                    "reasonably visible or inferable from the image. "
                    "If something is uncertain, say so."
                ),
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": make_data_url(data, mime)
                        },
                    },
                ],
            },
        ],
        max_tokens=180,
    )

    answer = extract_answer(response)

    if not answer:
        raise RuntimeError("The vision model returned an empty response.")

    return answer


def generate_caption(data: bytes) -> str:
    if not HF_TOKEN:
        raise RuntimeError(
            "HF_TOKEN is missing from the Render environment."
        )

    response = client.image_to_text(
        data,
        model=CAPTION_MODEL
    )

    text = getattr(response, "generated_text", response)
    text = str(text).strip()

    if not text:
        raise RuntimeError("The caption model returned an empty response.")

    return text


@app.post("/api/analyze")
def analyze():
    if "image" not in request.files:
        return jsonify(error="Please upload an image."), 400

    uploaded = request.files["image"]

    if not uploaded.filename:
        return jsonify(error="Please choose an image."), 400

    data = uploaded.read()

    if not data:
        return jsonify(error="The uploaded image is empty."), 400

    # Validate that the uploaded bytes are actually an image.
    try:
        with Image.open(BytesIO(data)) as image:
            image.verify()
    except (UnidentifiedImageError, OSError):
        return jsonify(error="Please upload a valid image file."), 400

    question = (
        request.form.get("question", "").strip()
        or "Describe this image and identify the main objects or scene."
    )

    # Main vision request.
    try:
        answer = analyze_with_vision(
            data,
            uploaded.mimetype,
            question
        )
        return jsonify(
            answer=answer,
            mode="vision"
        ), 200

    except Exception as vision_error:
        logging.exception(
            "Vision inference failed: %s",
            vision_error
        )

        # Caption fallback is useful when the vision provider temporarily
        # fails, but the response clearly identifies that it is a fallback.
        try:
            answer = generate_caption(data)
            return jsonify(
                answer=(
                    f"Image description: {answer}\n\n"
                    "Note: the vision question-answering service "
                    "was temporarily unavailable, so this is a "
                    "caption-only fallback."
                ),
                mode="caption-fallback"
            ), 200

        except Exception as caption_error:
            logging.exception(
                "Caption fallback failed: %s",
                caption_error
            )

            # Return the real safe error to the browser. Never return the
            # token itself or any secret value.
            return jsonify(
                error=(
                    "Image analysis failed. "
                    f"Vision error: {type(vision_error).__name__}: "
                    f"{str(vision_error)[:500]}"
                )
            ), 502


@app.errorhandler(413)
def too_large(_):
    return jsonify(
        error="Image is too large. Please upload an image under 8 MB."
    ), 413


@app.get("/health")
def health():
    return jsonify(
        status="ok",
        hf_token_configured=bool(HF_TOKEN),
        vision_model=VISION_MODEL,
        caption_model=CAPTION_MODEL,
    )


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.getenv("PORT", "10000"))
    )
