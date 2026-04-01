"""
LLM-driven orchestrator using MCP tools.

Replaces the fixed agent pipeline — the LLM decides which tool to call
next based on the results it observes, until it decides the analysis is done.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from anthropic import Anthropic

from . import mcp_server

SYSTEM_PROMPT = """You are a pharmacokinetics (PK) analysis agent.
You have tools to analyze concentration-time data from a clinical study (Acocella 1984).
The data contains IV infusion observations for multiple subjects.

Your goal is to:
1. Explore the dataset with get_data_summary
2. Run an individual fit to assess per-subject RMSE
3. Identify outlier subjects (RMSE > 2x median) and decide whether to exclude them
4. Run a pooled fit on the active subjects
5. Optionally extract insights from the reference paper
6. Generate the final report

Use your judgment: if the fit quality is already good, do not over-iterate.
If a subject has very high RMSE, consider whether it reflects a data issue or a real biological difference.
Stop when you are satisfied with the analysis."""

# Tool schemas for Claude's tool use API
_TOOLS: List[Dict[str, Any]] = [
    {
        "name": "get_data_summary",
        "description": "Return summary statistics of the PK dataset.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "run_individual_fit",
        "description": (
            "Fit CL and V per subject. Returns RMSE per subject and outlier threshold. "
            "Use this before pooled fit to identify subjects to exclude."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "exclude": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Subject IDs to exclude from the fit.",
                    "default": [],
                }
            },
        },
    },
    {
        "name": "run_pooled_fit",
        "description": "Fit a single CL and V across all active subjects.",
        "input_schema": {
            "type": "object",
            "properties": {
                "exclude": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Subject IDs to exclude.",
                    "default": [],
                }
            },
        },
    },
    {
        "name": "extract_paper",
        "description": "Extract model details from the reference PDF. Call once to compare with fit results.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "generate_report",
        "description": "Write the final CSV and markdown report. Call this last.",
        "input_schema": {"type": "object", "properties": {}},
    },
]

_TOOL_MAP = {
    "get_data_summary": mcp_server.get_data_summary,
    "run_individual_fit": mcp_server.run_individual_fit,
    "run_pooled_fit": mcp_server.run_pooled_fit,
    "extract_paper": mcp_server.extract_paper,
    "generate_report": mcp_server.generate_report,
}


def run_orchestrated(api_key: str, model: str = "claude-3-5-sonnet-latest") -> None:
    """Run the PK analysis driven by Claude's tool use decisions."""
    client = Anthropic(api_key=api_key)
    messages = [{"role": "user", "content": "Analyze the PK dataset and generate a report."}]

    while True:
        response = client.messages.create(
            model=model,
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            tools=_TOOLS,
            messages=messages,
        )

        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            break

        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    fn = _TOOL_MAP.get(block.name)
                    result = fn(**block.input) if fn else {"error": f"unknown tool: {block.name}"}
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result, default=str),
                    })
            messages.append({"role": "user", "content": tool_results})
