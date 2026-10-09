
import os
import json
import traceback
from io import StringIO
from contextlib import redirect_stdout, redirect_stderr
from typing import List

import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


app = FastAPI()

# Enable CORS for browser-based testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CodeRequest(BaseModel):
    code: str


class ErrorAnalysis(BaseModel):
    error_lines: List[int] = Field(default_factory=list)


def execute_python_code(code: str) -> dict:
    """Execute Python code and capture its output or traceback."""
    stdout = StringIO()
    stderr = StringIO()

    try:
        with redirect_stdout(stdout), redirect_stderr(stderr):
            exec(compile(code, "<submitted_code>", "exec"), {})

        return {
            "success": True,
            "output": stdout.getvalue()
        }

    except Exception:
        return {
            "success": False,
            "output": traceback.format_exc()
        }


def analyze_error_with_ai(code: str, error_traceback: str) -> List[int]:
    """Ask Gemini through AI Pipe to identify the error line numbers."""
    token = os.environ.get("AIPIPE_TOKEN")

    if not token:
        raise RuntimeError("AIPIPE_TOKEN environment variable is missing")

    prompt = f"""
Identify the line number(s) in the submitted Python code that caused
the exception shown in the traceback.

Use the traceback and the submitted code. Return only line numbers
from the submitted code, using 1-based line numbering.
Do not return line numbers belonging to Python library internals.

SUBMITTED CODE:
{code}

TRACEBACK:
{error_traceback}

Return JSON in this format:
{{"error_lines": [3]}}
"""

    response = requests.post(
        "https://aipipe.org/openrouter/v1/chat/completions",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}"
        },
        json={
            "model": os.getenv(
                "AIPIPE_MODEL",
                "google/gemini-2.0-flash-lite-001"
            ),
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "response_format": {"type": "json_object"}
        },
        timeout=60
    )

    response.raise_for_status()
    answer = response.json()["choices"][0]["message"]["content"]

    analysis = ErrorAnalysis.model_validate_json(answer)

    # Keep only valid line numbers from the submitted code.
    return sorted({
        line for line in analysis.error_lines
        if 1 <= line <= len(code.splitlines())
    })


@app.post("/code-interpreter")
def code_interpreter(request: CodeRequest):
    execution = execute_python_code(request.code)

    # If execution succeeds, do not call AI.
    if execution["success"]:
        return {
            "error": [],
            "result": execution["output"]
        }

    # If execution fails, ask AI to identify the error lines.
    try:
        error_lines = analyze_error_with_ai(
            request.code,
            execution["output"]
        )
    except Exception as e:
        print("AI ERROR:", repr(e))
        error_lines = []

    return {
        "error": error_lines,
        "result": execution["output"]
    }