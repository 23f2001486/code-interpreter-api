
import os
import re
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
    """Execute Python code and capture output or traceback."""
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
    """Ask the AI Pipe model to identify error line numbers."""

    token = os.environ.get("AIPIPE_TOKEN")

    if not token:
        raise RuntimeError("AIPIPE_TOKEN environment variable is missing")

    prompt = f"""
Identify the line number(s) in the submitted Python code that caused
the exception shown in the traceback.

Use 1-based line numbering.
Return only line numbers from the submitted code.
Do not return Python library or internal traceback line numbers.

SUBMITTED CODE:
{code}

TRACEBACK:
{error_traceback}

Return JSON in this format:
{{"error_lines": [2]}}
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
                "openai/gpt-4.1-nano"
            ),
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "response_format": {
                "type": "json_object"
            }
        },
        timeout=60
    )

    response.raise_for_status()

    answer = response.json()["choices"][0]["message"]["content"]
    analysis = ErrorAnalysis.model_validate_json(answer)

    # Keep only valid line numbers
    valid_lines = {
        line for line in analysis.error_lines
        if 1 <= line <= len(code.splitlines())
    }

    return sorted(valid_lines)


@app.post("/code-interpreter")
def code_interpreter(request: CodeRequest):

    execution = execute_python_code(request.code)

    # If execution succeeds, do not call AI
    if execution["success"]:
        return {
            "error": [],
            "result": execution["output"]
        }

    error_traceback = execution["output"]

    # Extract line numbers belonging to submitted Python code
    matches = re.findall(
        r'File "<submitted_code>", line (\d+)',
        error_traceback
    )

    if matches:
        # Use the final submitted-code frame in the traceback
        error_lines = [int(matches[-1])]
    else:
        # Use AI when no submitted-code frame is available
        try:
            error_lines = analyze_error_with_ai(
                request.code,
                error_traceback
            )
        except Exception as e:
            print("AI ERROR:", repr(e))
            error_lines = []

    return {
        "error": error_lines,
        "result": error_traceback
    }
