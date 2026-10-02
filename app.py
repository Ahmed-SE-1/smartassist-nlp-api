"""
Hugging Face Space (Gradio SDK) that also exposes a clean REST /predict
endpoint for the Flutter app — same request/response shape as our
original FastAPI design, so no Flutter-side changes are needed.
"""

import gradio as gr
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline

MODEL_REPO = "Ahmed-AI-Engineer/smartassist-nlp-intent-model"

print(f"Loading model from: {MODEL_REPO}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_REPO)
classifier = TextClassificationPipeline(model=model, tokenizer=tokenizer)
print("Model loaded.")

CONFIDENCE_THRESHOLD = 0.60


def predict_intent(text: str) -> dict:
    result = classifier(text)[0]
    confidence = float(result["score"])
    return {
        "intent": result["label"],
        "confidence": confidence,
        "needs_confirmation": confidence < CONFIDENCE_THRESHOLD,
        "raw_text": text,
    }


# ---- Simple Gradio UI (satisfies the Space's SDK requirement, also lets
#      you test manually in the browser at the Space's root URL) ----
demo = gr.Interface(
    fn=predict_intent,
    inputs=gr.Textbox(label="Voice command text (English/Urdu/Roman Urdu)"),
    outputs=gr.JSON(label="Prediction"),
    title="SmartAssist NLP API",
    description="Intent classification for the SmartAssist home automation voice commands.",
)

# ---- Custom FastAPI app mounted alongside Gradio — this is what Flutter calls ----
fastapi_app = FastAPI(title="SmartAssist NLP API")


class CommandRequest(BaseModel):
    text: str


class CommandResponse(BaseModel):
    intent: str
    confidence: float
    needs_confirmation: bool
    raw_text: str


@fastapi_app.get("/health")
def health_check():
    return {"status": "ok", "message": "SmartAssist NLP API running on HF Space (Gradio)"}


@fastapi_app.post("/predict", response_model=CommandResponse)
def predict_api(request: CommandRequest):
    return predict_intent(request.text)


# Mount Gradio UI at "/" and keep our custom FastAPI routes (like /predict) active
app = gr.mount_gradio_app(fastapi_app, demo, path="/")