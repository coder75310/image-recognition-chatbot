# Image Recognition Chatbot - Render Fixed

This version does NOT load BLIP/PyTorch inside the Render container, avoiding the 512 MB memory problem.

## Render
Add this Environment Variable:
- `HF_TOKEN` = your Hugging Face access token

Optional:
- `HF_VISION_MODEL` = `Qwen/Qwen2.5-VL-3B-Instruct`
- `HF_CAPTION_MODEL` = `Salesforce/blip-image-captioning-base`

Start command:
`gunicorn app:app`

Never put the HF token in GitHub or frontend JavaScript.

Health check:
`https://YOUR-APP.onrender.com/health`
