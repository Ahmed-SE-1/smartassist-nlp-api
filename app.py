"""
FastAPI app for Hugging Face Spaces deployment.
Loads the fine-tuned model from your HF Hub model repo (not local files),
so this Space stays lightweight and rebuilds fast.
"""

from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline

# CHANGE THIS to your actual model repo id (from step 2)
MODEL_REPO = "Ahmed-AI-Engineer/smartassist-nlp-intent-model"

print(f"Loading model from Hugging Face Hub: {MODEL_REPO}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_REPO)
classifier = TextClassificationPipeline(model=model, tokenizer=tokenizer)
print("Model loaded. API ready.")

CONFIDENCE_THRESHOLD = 0.60

app = FastAPI(title="SmartAssist NLP API")


class CommandRequest(BaseModel):
    text: str


class CommandResponse(BaseModel):
    intent: str
    confidence: float
    needs_confirmation: bool
    raw_text: str


@app.get("/")
def health_check():
    return {"status": "ok", "message": "SmartAssist NLP API is running on Hugging Face Spaces"}


@app.post("/predict", response_model=CommandResponse)
def predict_intent(request: CommandRequest):
    result = classifier(request.text)[0]
    intent = result["label"]
    confidence = float(result["score"])

    return CommandResponse(
        intent=intent,
        confidence=confidence,
        needs_confirmation=confidence < CONFIDENCE_THRESHOLD,
        raw_text=request.text,
    )
