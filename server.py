# server.py
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import joblib
import os

# ---- App ----
app = FastAPI(title="Sentiment Analysis")

# Use "Templates" folder
templates = Jinja2Templates(directory="templates")

# ---- Load model pipeline (vectorizer + classifier) ----
PIPE_PATH = os.path.join(os.path.dirname(__file__), "sentiment_pipeline.joblib")
pipe = joblib.load(PIPE_PATH)

# Map sentiments  to friendly labels
LABEL_MAP = {1: "Positive 😊", 0: "Neutral 😐", -1: "Negative ☹️"}

# ---- Schemas ----
class ReviewIn(BaseModel):
    text: str

# ---- Routes ----
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    # Renders Templates/index.html
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
def predict(review: ReviewIn):
    text = review.text or ""
    pred = pipe.predict([text])[0]
    label = LABEL_MAP.get(int(pred), str(pred))

    # probabilities (if model supports predict_proba)
    proba = None
    try:
        probs = pipe.predict_proba([text])[0]
        proba = {str(cls): float(p) for cls, p in zip(pipe.classes_, probs)}
    except Exception:
        pass

    return {"label": label, "raw_label": int(pred), "proba": proba}

@app.get("/health")
def health():
    return {"status": "ok"}