"""
Evaluate fine-tuned LLM responses using a local Llama3 judge via Ollama.

Pipeline:
  1. Load test data (JSON with 'instruction', 'input', 'output', 'model_response')
  2. For each entry, prompt Llama3 to score the model response 0–100
  3. Compute mean score across all entries
  4. Print per-entry breakdown and overall result

Requirements:
  - Ollama installed and running: https://ollama.ai
  - Llama3 pulled: `ollama pull llama3`
  - pip install psutil tqdm

Usage:
  python scripts/evaluate_with_llama.py --file test_data_with_responses.json
"""

import argparse
import json
import re
import urllib.request
import psutil
from tqdm import tqdm

from src.finetune_instruct import format_input


# ---------------------------------------------------------------------------
# Ollama helpers
# ---------------------------------------------------------------------------

def check_ollama_running() -> bool:
    for proc in psutil.process_iter(["name"]):
        if "ollama" in proc.info["name"]:
            return True
    return False


def query_model(
    prompt: str,
    model: str = "llama3",
    url: str = "http://localhost:11434/api/chat",
) -> str:
    """Send a prompt to a local Ollama model and return the text response."""
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "options": {"temperature": 0.0, "seed": 123},
        "stream": False,
    }
    payload = json.dumps(data).encode("utf-8")
    request = urllib.request.Request(url, data=payload, method="POST")
    request.add_header("Content-Type", "application/json")

    with urllib.request.urlopen(request, timeout=60) as response:
        result = json.loads(response.read().decode("utf-8"))
    return result["message"]["content"]


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def build_score_prompt(entry: dict) -> str:
    return (
        f"Given the input `{format_input(entry)}` "
        f"and correct output `{entry['output']}`, "
        f"score the model response `{entry['model_response']}` "
        f"on a scale from 0 to 100, where 100 is the best score. "
        f"Respond with the number only."
    )


def extract_score(text: str) -> float | None:
    """Parse the first integer found in the judge's response."""
    match = re.search(r"\b(\d{1,3})\b", text)
    if match:
        score = int(match.group(1))
        return float(score) if 0 <= score <= 100 else None
    return None


def generate_model_scores(
    json_data: list,
    judge_model: str = "llama3",
) -> list[float]:
    scores = []
    for entry in tqdm(json_data, desc="Scoring"):
        prompt   = build_score_prompt(entry)
        response = query_model(prompt, model=judge_model)
        score    = extract_score(response)
        if score is not None:
            scores.append(score)
        else:
            print(f"  ⚠ Could not parse score from: {response!r}")
    return scores


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Score LLM responses with Llama3")
    parser.add_argument("--file",  required=True, help="JSON file with model responses")
    parser.add_argument("--model", default="llama3", help="Ollama model to use as judge")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only first N entries")
    args = parser.parse_args()

    if not check_ollama_running():
        raise RuntimeError(
            "Ollama is not running. Start it with: ollama serve\n"
            "Then pull a model: ollama pull llama3"
        )

    with open(args.file, "r") as f:
        data = json.load(f)

    if args.limit:
        data = data[:args.limit]

    print(f"\nEvaluating {len(data)} entries with judge model: {args.model}\n")

    scores = generate_model_scores(data, judge_model=args.model)

    if scores:
        mean_score = sum(scores) / len(scores)
        print(f"\n{'─' * 40}")
        print(f"  Entries scored : {len(scores)}")
        print(f"  Mean score     : {mean_score:.1f} / 100")
        print(f"  Min            : {min(scores):.0f}")
        print(f"  Max            : {max(scores):.0f}")
        print(f"{'─' * 40}\n")
    else:
        print("No scores were successfully parsed.")


if __name__ == "__main__":
    main()
