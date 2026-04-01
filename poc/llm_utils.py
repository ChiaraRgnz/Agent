from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

import pdfplumber
from anthropic import Anthropic


def extract_pdf_text(pdf_path: Path, max_pages: int = 5, max_chars: int = 20000) -> str:
    """Extract text from the first pages of a PDF, capped by characters."""
    text_parts = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages[:max_pages]:
            text_parts.append(page.extract_text() or "")
            if sum(len(t) for t in text_parts) >= max_chars:
                break
    text = "\n".join(text_parts)
    return text[:max_chars]


def extract_paper_insights(pdf_path: Path, api_key: str, model_name: str) -> Dict[str, Any]:
    """Call Anthropic to extract key model details from a PK paper."""
    text = extract_pdf_text(pdf_path)
    if not text.strip():
        return {"error": "no_text_extracted"}

    client = Anthropic(api_key=api_key)
    prompt = (
        "You are reading a PK/PD paper. Extract the model details as JSON.\n"
        "Return compact JSON with keys: model_structure, dosing, parameters, units, "
        "estimation_method, reported_results, notes.\n"
        "If a field is not found, set it to null.\n\n"
        f"PAPER TEXT:\n{text}\n"
    )
    msg = client.messages.create(
        model=model_name,
        max_tokens=800,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    content = msg.content[0].text if msg.content else ""
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {"raw": content}


def extract_paper_insights_gemini(
    pdf_path: Path, api_key: str, model_name: str = "gemini-2.5-flash-lite"
) -> Dict[str, Any]:
    """Call Gemini to extract key model details from a PK paper."""
    try:
        from google import genai
        from google.genai import types
    except Exception as exc:
        return {"error": f"missing_google_genai: {exc}"}

    text = extract_pdf_text(pdf_path)
    if not text.strip():
        return {"error": "no_text_extracted"}

    client = genai.Client(api_key=api_key)
    prompt = (
        "You are reading a PK/PD paper. Extract the model details as JSON.\n"
        "Return compact JSON with keys: model_structure, dosing, parameters, units, "
        "estimation_method, reported_results, notes.\n"
        "If a field is not found, set it to null.\n\n"
        f"PAPER TEXT:\n{text}\n"
    )
    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0, max_output_tokens=800),
    )
    content = (response.text or "").strip()
    content = content.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {"raw": content}


def extract_paper_insights_local(
    pdf_path: Path, model_name: str, max_new_tokens: int = 800
) -> Dict[str, Any]:
    """Call a local HF model to extract key model details from a PK paper."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from transformers import BitsAndBytesConfig
    except Exception as exc:  # pragma: no cover - optional dependency
        return {"error": f"missing_transformers: {exc}"}

    text = extract_pdf_text(pdf_path)
    if not text.strip():
        return {"error": "no_text_extracted"}

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        device_map="auto",
    )
    prompt = (
        "Extract PK model details as JSON with keys: model_structure, dosing, "
        "parameters, units, estimation_method, reported_results, notes. "
        "If unknown, set to null.\n\n"
        f"PAPER TEXT:\n{text}\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    content = tokenizer.decode(outputs[0], skip_special_tokens=True)
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {"raw": content}
