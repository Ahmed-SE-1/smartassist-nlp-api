"""
Lightweight FastAPI app using raw ONNX Runtime + transformers tokenizer only.
No PyTorch, no optimum dependency — minimal memory footprint for free-tier hosting.
"""

import json
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download

# CHANGE THIS to your ONNX model repo
MODEL_REPO = "Ahmed-AI-Engineer/smartassist-nlp-intent-model-onnx"

print(f"Loading tokenizer and ONNX model from: {MODEL_REPO}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO)

# Download the full repo snapshot so model.onnx.data (external weights file,
# if present) is fetched alongside model.onnx — a single hf_hub_download
# only grabs the one named file and misses this companion file.
local_dir = snapshot_download(repo_id=MODEL_REPO)

onnx_path = f"{local_dir}/model.onnx"
session = ort.InferenceSession(onnx_path)

with open(f"{local_dir}/id2label.json", encoding="utf-8") as f:
    id2label = {int(k): v for k, v in json.load(f).items()}

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


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


@app.get("/")
def health_check():
    return {"status": "ok", "message": "SmartAssist NLP API (manual ONNX) is running"}


@app.post("/predict", response_model=CommandResponse)
def predict_intent(request: CommandRequest):
    inputs = tokenizer(request.text, return_tensors="np")

    onnx_inputs = {
        "input_ids": inputs["input_ids"].astype(np.int64),
        "attention_mask": inputs["attention_mask"].astype(np.int64),
    }
    logits = session.run(["logits"], onnx_inputs)[0][0]
    probs = softmax(logits)

    pred_id = int(np.argmax(probs))
    intent = id2label[pred_id]
    confidence = float(probs[pred_id])

    return CommandResponse(
        intent=intent,
        confidence=confidence,
        needs_confirmation=confidence < CONFIDENCE_THRESHOLD,
        raw_text=request.text,
    )