#!/usr/bin/env python3
"""LLM regression model selection agent — CLI entry point."""

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import anthropic

import tools as tool_module
from prompts import SYSTEM_PROMPT, TOOL_DEFINITIONS

TOOL_DISPATCH = {
    "load_data": tool_module.load_data,
    "train_and_evaluate": tool_module.train_and_evaluate,
    "generate_report": tool_module.generate_report,
}

_USE_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def _c(code: str, text: str) -> str:
    if not _USE_COLOR:
        return text
    return f"\033[{code}m{text}\033[0m"


def _print_tool_call(name: str, inputs: dict) -> None:
    label = _c("36;1", f"→ {name}")
    args_str = ", ".join(f"{k}={v!r}" for k, v in inputs.items())
    print(f"\n{label}({args_str})", flush=True)


def _print_tool_result(name: str, result: dict | str) -> None:
    label = _c("32", f"← {name}")
    if isinstance(result, str):
        preview = result.split("\n")[0]
        print(f"{label}: {preview} [report generated]", flush=True)
        return

    if "error" in result:
        print(f"{label}: {_c('31', result['error'])}", flush=True)
        return

    interesting = {}
    for key in ("total_rows", "train_rows", "test_rows", "features",
                 "model", "cv_rmse", "cv_rmse_std", "r2", "r2_std", "fit_time_ms"):
        if key in result:
            interesting[key] = result[key]

    summary = "  ".join(f"{k}={v!r}" for k, v in interesting.items())
    print(f"{label}: {summary}", flush=True)


def run_agent(
    csv_path: str,
    target_col: str,
    verbose: bool = False,
    report_path: str | None = None,
) -> None:
    client = anthropic.Anthropic()

    messages: list[dict] = [
        {
            "role": "user",
            "content": (
                f"Analyse the regression dataset at `{csv_path}` "
                f"and recommend the best model for predicting `{target_col}`. "
                "Follow your four-step workflow."
            ),
        }
    ]

    print(_c("33;1", "\n=== Regression Model Selection Agent ==="), flush=True)
    print(f"Dataset : {csv_path}", flush=True)
    print(f"Target  : {target_col}", flush=True)
    print(flush=True)

    report_text: str | None = None
    agent_winner: dict | None = None

    while True:
        if verbose:
            with client.messages.stream(
                model="claude-opus-4-7",
                max_tokens=8096,
                system=SYSTEM_PROMPT,
                tools=TOOL_DEFINITIONS,
                messages=messages,
                thinking={"type": "adaptive"},
            ) as stream:
                printed_any_text = False
                for event in stream:
                    if hasattr(event, "type") and event.type == "content_block_delta":
                        delta = event.delta
                        if delta.type == "text_delta":
                            print(delta.text, end="", flush=True)
                            printed_any_text = True
                if printed_any_text:
                    print(flush=True)
                response = stream.get_final_message()
        else:
            response = client.messages.create(
                model="claude-opus-4-7",
                max_tokens=8096,
                system=SYSTEM_PROMPT,
                tools=TOOL_DEFINITIONS,
                messages=messages,
                thinking={"type": "adaptive"},
            )

        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            break

        if response.stop_reason != "tool_use":
            print(_c("31", f"Unexpected stop_reason: {response.stop_reason}"), flush=True)
            break

        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue

            name = block.name
            inputs = block.input or {}

            _print_tool_call(name, inputs)

            fn = TOOL_DISPATCH.get(name)
            if fn is None:
                result = {"error": f"Unknown tool: {name}"}
            else:
                # Occasionally the model embeds later params inside an earlier
                # string param using XML tags — strip anything after a closing tag
                for k, v in inputs.items():
                    if isinstance(v, str):
                        inputs[k] = re.sub(r"</\w+>.*", "", v, flags=re.DOTALL).strip()
                result = fn(**inputs)

            _print_tool_result(name, result)

            if name == "generate_report" and isinstance(result, str):
                report_text = result
                agent_winner = {
                    "model": inputs.get("winner_model"),
                    "params": inputs.get("winner_params", {}),
                    "cv_rmse": inputs.get("winner_cv_rmse"),
                    "r2": inputs.get("winner_r2"),
                }

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": json.dumps(result) if not isinstance(result, str) else result,
            })

        messages.append({"role": "user", "content": tool_results})

    if report_text and agent_winner:
        print(_c("33", "\nRunning GridSearchCV baseline for post-hoc comparison..."), flush=True)
        tool_module.run_baseline()

        print(_c("33", "Running Optuna (TPE, 50 trials) for post-hoc comparison..."), flush=True)
        tool_module.run_optuna_baseline(n_trials=50)

        print(_c("33", "Evaluating final models on held-out test set..."), flush=True)
        agent_test = tool_module.evaluate_on_test(
            agent_winner["model"], agent_winner["params"]
        )
        baseline_section = tool_module.build_baseline_section(
            agent_winner_model=agent_winner["model"],
            agent_winner_cv_rmse=agent_winner["cv_rmse"],
            agent_winner_r2=agent_winner["r2"],
            agent_test_rmse=agent_test["test_rmse"],
            agent_test_r2=agent_test["test_r2"],
        )
        report_text = report_text + "\n\n" + baseline_section

        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        auto_path = results_dir / f"run_{ts}.md"
        auto_path.write_text(report_text, encoding="utf-8")
        print(_c("32;1", f"Report saved → {auto_path}"), flush=True)

        if report_path:
            Path(report_path).write_text(report_text, encoding="utf-8")
            print(_c("32;1", f"Report saved → {report_path}"), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LLM-powered regression model selection agent"
    )
    parser.add_argument("--data", required=True, help="Path to input CSV file")
    parser.add_argument("--target", required=True, help="Name of the target column")
    parser.add_argument(
        "--verbose", action="store_true", help="Stream agent reasoning token-by-token"
    )
    parser.add_argument(
        "--report", default=None, help="Optional path to save the markdown report"
    )
    args = parser.parse_args()

    run_agent(
        csv_path=args.data,
        target_col=args.target,
        verbose=args.verbose,
        report_path=args.report,
    )


if __name__ == "__main__":
    main()
