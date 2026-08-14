# Image Recognition Chatbot — Public Deployment

This is a cleaned deployment copy of the uploaded Flask project. The local `venv/` folder and uploaded sample files were intentionally excluded.

## Render
Build command:
`pip install -r requirements.txt`

Start command:
`gunicorn app:app`

After deployment, Render provides a public HTTPS URL. Use that URL in the resume instead of `127.0.0.1:5000`.

## Project structure
```text
app.py
requirements.txt
Procfile
.python-version
templates/index.html
static/style.css
static/script.js
```
