#!/usr/bin/env python3
"""
SwarmRouter-v1 FastAPI Server
Serves routing decisions via REST API.
"""

import os
import sys
import json
import torch
import uvicorn
from pathlib import Path
from typing import Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

sys.path.append(str(Path(__file__).parent.parent))

from unsloth import FastLanguageModel
from swarmrouter.schema import RouterOutput


# Request/Response models
class RouteRequest(BaseModel):
    message: str = Field(..., description="User message to route", min_length=1)
    max_tokens: Optional[int] = Field(512, description="Max tokens for response")
    temperature: Optional[float] = Field(0.1, description="Sampling temperature")


class RouteResponse(BaseModel):
    routing: RouterOutput
    latency_ms: float
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_path: str
    uptime_seconds: float


# Global model state
class ModelState:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_path = None
        self.loaded = False
        self.start_time = datetime.now()


state = ModelState()
app = FastAPI(
    title="SwarmRouter API",
    description="Intelligent routing for multi-model inference fleet",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def load_model(model_path: Path):
    """Load model into memory."""
    print(f"Loading model from {model_path}...")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(model_path),
        max_seq_length=1024,
        dtype=torch.bfloat16,
        load_in_4bit=True,
        device_map="auto"
    )
    FastLanguageModel.for_inference(model)

    state.model = model
    state.tokenizer = tokenizer
    state.model_path = str(model_path)
    state.loaded = True

    print("✓ Model loaded successfully")


def generate_routing(user_message: str, max_tokens: int = 512, temperature: float = 0.1) -> str:
    """Generate routing decision."""
    if not state.loaded:
        raise RuntimeError("Model not loaded")

    messages = [
        {"role": "system", "content": RouterOutput.system_prompt()},
        {"role": "user", "content": user_message}
    ]

    prompt = state.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = state.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(state.model.device)

    with torch.no_grad():
        outputs = state.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=False if temperature < 0.2 else True,
            pad_token_id=state.tokenizer.pad_token_id,
            eos_token_id=state.tokenizer.eos_token_id
        )

    response = state.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return response.strip()


def extract_json(text: str) -> dict:
    """Extract JSON from response."""
    if "```json" in text:
        start = text.find("```json") + 7
        end = text.find("```", start)
        text = text[start:end].strip()
    elif "```" in text:
        start = text.find("```") + 3
        end = text.find("```", start)
        text = text[start:end].strip()

    try:
        return json.loads(text)
    except:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end])
            except:
                pass
        raise ValueError("Failed to parse JSON from response")


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    model_path = Path(os.getenv("MODEL_PATH", "./models/swarmrouter-v1/final"))
    if not model_path.is_absolute():
        base_dir = Path(__file__).parent.parent
        model_path = base_dir / model_path

    if not model_path.exists():
        print(f"WARNING: Model path does not exist: {model_path}")
        print("Server started but model not loaded. Set MODEL_PATH environment variable.")
    else:
        load_model(model_path)


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    uptime = (datetime.now() - state.start_time).total_seconds()
    return HealthResponse(
        status="healthy" if state.loaded else "model_not_loaded",
        model_loaded=state.loaded,
        model_path=state.model_path or "not_loaded",
        uptime_seconds=uptime
    )


@app.post("/route", response_model=RouteResponse)
async def route(request: RouteRequest):
    """Route a user message to appropriate model."""
    if not state.loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start_time = datetime.now()

    try:
        # Generate routing decision
        response = generate_routing(
            request.message,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )

        # Parse JSON
        routing_json = extract_json(response)

        # Validate with Pydantic
        routing = RouterOutput(**routing_json)

        latency = (datetime.now() - start_time).total_seconds() * 1000

        return RouteResponse(
            routing=routing,
            latency_ms=round(latency, 2),
            model_version="swarmrouter-v1"
        )

    except ValueError as e:
        raise HTTPException(status_code=500, detail=f"JSON parsing failed: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Routing failed: {str(e)}")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "SwarmRouter API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "route": "/route (POST)"
        }
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Serve SwarmRouter API")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--model-path", type=Path, default=None, help="Path to model")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    args = parser.parse_args()

    # Set model path if provided
    if args.model_path:
        os.environ["MODEL_PATH"] = str(args.model_path)

    print("=" * 70)
    print("SwarmRouter API Server")
    print("=" * 70)
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print("=" * 70)

    uvicorn.run(
        "serve_api:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )


if __name__ == "__main__":
    main()
