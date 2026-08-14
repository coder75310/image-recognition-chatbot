# Conversational Image Chatbot - Render Deployment

This version is optimized for Render Free's low-memory instance. The large BLIP model is NOT loaded inside Render. Instead, the Flask app calls Hugging Face Inference Providers using a Hugging Face token.

## Render settings
Build Command: `pip install -r requirements.txt`
Start Command: `gunicorn app:app`

## Required Environment Variable
Add `HF_TOKEN` in Render Environment Variables. Create a Hugging Face User Access Token with Inference Providers permission. Do NOT commit the token to GitHub.

Optional:
`HF_MODEL=Qwen/Qwen2.5-VL-3B-Instruct`

The app keeps the existing frontend and `/ask` endpoint, and now actually sends the user's question together with the image to the vision-language model.
