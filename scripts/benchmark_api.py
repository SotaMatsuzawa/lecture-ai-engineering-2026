"""Compare response time and token usage across Gemini models."""

import argparse
import math
import statistics
import time

from google import genai
from google.genai import errors

from app.config import api_error_message, get_api_key


def summarize(label: str, values: list[float]) -> None:
    if not values:
        print(f"  {label}: no successful measurements")
        return

    ordered = sorted(values)
    p95 = ordered[math.ceil(0.95 * len(ordered)) - 1]
    deviation = statistics.stdev(values) if len(values) > 1 else 0.0
    print(
        f"  {label}: mean={statistics.mean(values):.2f}, "
        f"median={statistics.median(values):.2f}, sd={deviation:.2f}, "
        f"min={ordered[0]:.2f}, max={ordered[-1]:.2f}, p95={p95:.2f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Gemini API response-time and token benchmark")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["gemini-3.5-flash-lite", "gemini-3.1-flash-lite"],
    )
    parser.add_argument("--runs-per-model", type=int, default=25)
    parser.add_argument("--interval", type=float, default=0.1)
    parser.add_argument("--question", default="LoRAとは何ですか？")
    args = parser.parse_args()

    if args.runs_per_model < 1 or args.interval < 0:
        parser.error("--runs-per-model must be positive and --interval cannot be negative")

    client = genai.Client(api_key=get_api_key())
    for model in args.models:
        elapsed_values: list[float] = []
        input_tokens: list[float] = []
        output_tokens: list[float] = []
        failures = 0
        print(f"[model] {model} ({args.runs_per_model} runs)")

        for run_number in range(1, args.runs_per_model + 1):
            started_at = time.perf_counter()
            try:
                response = client.models.generate_content(model=model, contents=args.question)
            except errors.APIError as error:
                failures += 1
                print(f"  run {run_number}: API error {error.code}: {api_error_message(error.code, error.message)}")
            else:
                elapsed = time.perf_counter() - started_at
                elapsed_values.append(elapsed)
                usage = response.usage_metadata
                if usage is not None:
                    if usage.prompt_token_count is not None:
                        input_tokens.append(float(usage.prompt_token_count))
                    if usage.candidates_token_count is not None:
                        output_tokens.append(float(usage.candidates_token_count))
                print(
                    f"  run {run_number}: {elapsed:.2f}s, "
                    f"input={usage.prompt_token_count if usage else 'n/a'}, "
                    f"output={usage.candidates_token_count if usage else 'n/a'}"
                )
            time.sleep(args.interval)

        print(f"[summary] {model}: success={len(elapsed_values)}, failure={failures}")
        summarize("seconds", elapsed_values)
        summarize("input tokens", input_tokens)
        summarize("output tokens", output_tokens)


if __name__ == "__main__":
    main()