import os
import base64
import logging
from io import BytesIO

from flask import Flask, render_template, request, jsonify
from PIL import Image, UnidentifiedImageError
from openai import OpenAI


app = Flask(__name__)

app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024


# ---------------------------------------------------------
# Logging
# ---------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)


# ---------------------------------------------------------
# Hugging Face configuration
# ---------------------------------------------------------

HF_TOKEN = os.getenv("HF_TOKEN", "").strip()

VISION_MODEL = os.getenv(
    "HF_VISION_MODEL",
    "moonshotai/Kimi-K3:featherless-ai"
).strip()


# ---------------------------------------------------------
# Hugging Face OpenAI-compatible client
# ---------------------------------------------------------

client = (
    OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=HF_TOKEN
    )
    if HF_TOKEN
    else None
)


# ---------------------------------------------------------
# Home page
# ---------------------------------------------------------

@app.get("/")
def index():
    return render_template("index.html")


# ---------------------------------------------------------
# Convert image to base64 data URL
# ---------------------------------------------------------

def make_data_url(data: bytes, mime: str) -> str:

    content_type = (
        mime
        if mime and mime.startswith("image/")
        else "image/jpeg"
    )

    encoded = base64.b64encode(data).decode("ascii")

    return f"data:{content_type};base64,{encoded}"


# ---------------------------------------------------------
# Extract model answer
# ---------------------------------------------------------

def extract_answer(response) -> str:

    if not response.choices:
        raise RuntimeError(
            "The model returned no choices."
        )

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


# ---------------------------------------------------------
# Vision analysis
# ---------------------------------------------------------

def analyze_with_vision(
    data: bytes,
    mime: str,
    question: str
) -> str:

    if not HF_TOKEN:

        raise RuntimeError(
            "HF_TOKEN is missing from the Render environment."
        )

    if client is None:

        raise RuntimeError(
            "Hugging Face client could not be initialized."
        )

    image_url = make_data_url(
        data,
        mime
    )

    logging.info(
        "Sending image to model: %s",
        VISION_MODEL
    )

    response = client.chat.completions.create(

        model=VISION_MODEL,

        messages=[
            {
                "role": "user",

                "content": [

                    {
                        "type": "text",

                        "text": (
                            "You are an accurate image recognition "
                            "and visual question-answering assistant.\n\n"

                            "Carefully inspect the uploaded image. "
                            "Answer the user's question using the "
                            "actual contents of the image.\n\n"

                            "Pay special attention to:\n"
                            "- visible text\n"
                            "- names\n"
                            "- dates\n"
                            "- numbers\n"
                            "- certificates\n"
                            "- documents\n"
                            "- objects\n"
                            "- people\n"
                            "- logos\n"
                            "- signs\n"
                            "- colors and shapes\n\n"

                            "Do not invent information. "
                            "If something cannot be read clearly, "
                            "say that it is unclear.\n\n"

                            f"User question: {question}"
                        )
                    },

                    {
                        "type": "image_url",

                        "image_url": {
                            "url": image_url
                        }
                    }

                ]
            }
        ],

        max_tokens=500
    )

    answer = extract_answer(response)

    if not answer:

        raise RuntimeError(
            "The vision model returned an empty response."
        )

    return answer


# ---------------------------------------------------------
# Analyze uploaded image
# ---------------------------------------------------------

@app.post("/api/analyze")
def analyze():

    if "image" not in request.files:

        return jsonify(
            error="Please upload an image."
        ), 400

    uploaded = request.files["image"]

    if not uploaded.filename:

        return jsonify(
            error="Please choose an image."
        ), 400

    data = uploaded.read()

    if not data:

        return jsonify(
            error="The uploaded image is empty."
        ), 400


    # -----------------------------------------------------
    # Validate image
    # -----------------------------------------------------

    try:

        with Image.open(
            BytesIO(data)
        ) as image:

            image.verify()

    except (
        UnidentifiedImageError,
        OSError
    ):

        return jsonify(
            error="Please upload a valid image file."
        ), 400


    # -----------------------------------------------------
    # Question
    # -----------------------------------------------------

    question = (
        request.form.get(
            "question",
            ""
        ).strip()
        or
        "Describe this image and identify the main objects, "
        "text, and important details."
    )


    # -----------------------------------------------------
    # Vision request
    # -----------------------------------------------------

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


    except Exception as error:

        logging.exception(
            "Vision inference failed: %s",
            error
        )

        return jsonify(
            error=(
                "Image analysis failed. "
                f"{type(error).__name__}: "
                f"{str(error)[:800]}"
            )
        ), 502


# ---------------------------------------------------------
# File too large
# ---------------------------------------------------------

@app.errorhandler(413)
def too_large(_):

    return jsonify(
        error=(
            "Image is too large. "
            "Please upload an image under 8 MB."
        )
    ), 413


# ---------------------------------------------------------
# Health check
# ---------------------------------------------------------

@app.get("/health")
def health():

    return jsonify(
        status="ok",
        hf_token_configured=bool(HF_TOKEN),
        vision_model=VISION_MODEL
    )


# ---------------------------------------------------------
# Start application
# ---------------------------------------------------------

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=int(
            os.getenv(
                "PORT",
                "10000"
            )
        )
    )
