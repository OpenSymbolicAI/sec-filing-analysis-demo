"""CLI harness: run the same queries through both agents, compare metrics."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from evaluate import evaluate_answer
from models import AgentResult, DemoQuery, EvalScore, QueryLog
from react_agent import create_react_agent
from retriever import Retriever
from behavior_agent import create_behavior_agent
from tui import run_side_by_side

console = Console()

_LOGS_DIR = Path(__file__).parent / "logs"


def _create_log_file() -> Path:
    """Create a timestamped log file and return its path."""
    _LOGS_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return _LOGS_DIR / f"run_{ts}.json"

_QUERIES_PATH = Path(__file__).parent / "queries.json"
DEMO_QUERIES: list[DemoQuery] = [
    DemoQuery(**q) for q in json.loads(_QUERIES_PATH.read_text())
]


def _print_result(result: AgentResult) -> None:
    """Pretty-print a single agent result."""
    color = "green" if result.success else "red"
    console.print(
        Panel(
            f"[bold]{result.answer}[/bold]",
            title=f"[{color}]{result.agent_name}[/{color}]",
            subtitle=(
                f"LLM calls: {result.llm_calls} | "
                f"Tokens: {result.total_tokens:,} | "
                f"Latency: {result.latency_seconds}s"
            ),
            border_style=color,
        )
    )
    if result.plan and result.plan != "(step-by-step tool calling — no upfront plan)":
        console.print(f"  [dim]Plan:[/dim]\n{result.plan}\n")


def _print_eval(label: str, score: EvalScore, color: str) -> None:
    """Print eval scores as a compact row."""
    dims = [
        ("Correct", score.correctness, score.correctness_reason),
        ("Complete", score.completeness, score.completeness_reason),
        ("Relevant", score.relevance, score.relevance_reason),
        ("Faithful", score.faithfulness, score.faithfulness_reason),
        ("Concise", score.conciseness, score.conciseness_reason),
    ]
    scores_str = "  ".join(f"{name}: {val}/5" for name, val, _ in dims)
    avg = sum(val for _, val, _ in dims) / len(dims)
    console.print(
        f"  [{color}]{label} Eval[/{color}]  {scores_str}  (avg {avg:.1f}/5)"
    )
    for name, _, reason in dims:
        console.print(f"    [dim]{name}: {reason}[/dim]")


def _print_comparison(react_result: AgentResult, behavior_result: AgentResult) -> None:
    """Print a side-by-side metrics comparison table."""
    table = Table(title="Metrics Comparison", show_header=True)
    table.add_column("Metric", style="bold")
    table.add_column("ReAct", justify="right")
    table.add_column("Behaviour Programming", justify="right")
    table.add_column("Ratio", justify="right", style="yellow")

    def _ratio(react_val: float, sym_val: float) -> str:
        if sym_val == 0:
            return "—"
        return f"{react_val / sym_val:.1f}x"

    table.add_row(
        "LLM Calls",
        str(react_result.llm_calls),
        str(behavior_result.llm_calls),
        _ratio(react_result.llm_calls, behavior_result.llm_calls),
    )
    table.add_row(
        "Total Tokens",
        f"{react_result.total_tokens:,}",
        f"{behavior_result.total_tokens:,}",
        _ratio(react_result.total_tokens, behavior_result.total_tokens),
    )
    table.add_row(
        "Latency",
        f"{react_result.latency_seconds}s",
        f"{behavior_result.latency_seconds}s",
        _ratio(react_result.latency_seconds, behavior_result.latency_seconds),
    )
    table.add_row(
        "Success",
        "[green]Yes[/green]" if react_result.success else "[red]No[/red]",
        "[green]Yes[/green]" if behavior_result.success else "[red]No[/red]",
        "",
    )
    console.print(table)
    if react_result.failure_reason:
        console.print(f"  [red]ReAct failure:[/red] {react_result.failure_reason}")
    if behavior_result.failure_reason:
        console.print(f"  [red]Behaviour Programming failure:[/red] {behavior_result.failure_reason}")
    console.print()


def run_demo_side_by_side(query_indices: list[int] | None = None) -> None:
    """Run the demo with a live side-by-side Textual TUI."""
    console.print("[bold]Connecting to LanceDB index...[/bold]")
    retriever = Retriever()
    console.print(f"  {retriever.chunk_count} chunks ready\n")

    queries = DEMO_QUERIES
    if query_indices:
        queries = [DEMO_QUERIES[i - 1] for i in query_indices if 1 <= i <= len(DEMO_QUERIES)]

    # Textual app handles all display; returns results when user quits
    results = run_side_by_side(queries, retriever)

    # Write log file
    log_file = _create_log_file()
    run_log: list[QueryLog] = []
    for q, react_result, behavior_result, react_eval, behavior_eval in results:
        run_log.append(QueryLog(
            label=q.label,
            query=q.query,
            ground_truth=q.ground_truth,
            react_result=react_result,
            behavior_result=behavior_result,
            react_eval=react_eval,
            behavior_eval=behavior_eval,
        ))
    log_file.write_text(json.dumps([entry.model_dump() for entry in run_log], indent=2))
    console.print(f"[dim]Results logged to {log_file}[/dim]")


def run_demo(query_indices: list[int] | None = None, *, bp_only: bool = False) -> None:
    """Run the demo. With bp_only=True, skip ReAct and evaluation for fast BP-only runs."""
    console.print("\n[bold blue]SEC 10-K Agent Comparison Demo[/bold blue]")
    if bp_only:
        console.print("[dim]Behaviour Programming only[/dim]\n")
    else:
        console.print("[dim]Behaviour Programming vs Tool Calling[/dim]\n")

    # Connect to pre-built LanceDB index (run scripts/ingest.py first)
    console.print("[bold]Connecting to LanceDB index...[/bold]")
    retriever = Retriever()
    console.print(f"  {retriever.chunk_count} chunks ready\n")

    # Create agents
    react_agent = None if bp_only else create_react_agent(retriever)
    behavior_agent = create_behavior_agent(retriever)

    # Select queries
    queries = DEMO_QUERIES
    if query_indices:
        queries = [DEMO_QUERIES[i - 1] for i in query_indices if 1 <= i <= len(DEMO_QUERIES)]

    # Run each query through both agents
    log_file = _create_log_file()
    run_log: list[QueryLog] = []

    for q in queries:
        console.rule(f"[bold cyan]{q.label}[/bold cyan]")
        console.print(f"  [italic]\"{q.query}\"[/italic]\n")

        # Run ReAct (skip in bp_only mode)
        react_result = None
        if not bp_only:
            console.print("[bold red]Running ReAct agent...[/bold red]")
            try:
                react_result = react_agent.run(q.query)
            except Exception as e:
                react_result = AgentResult(
                    agent_name="ReAct (Groq Tool Calling)",
                    query=q.query,
                    answer="[CRASHED]",
                    llm_calls=0,
                    total_tokens=0,
                    latency_seconds=0,
                    success=False,
                    failure_reason=f"{type(e).__name__}: {e}",
                )
            _print_result(react_result)

        # Run Behaviour Programming
        console.print("[bold green]Running Behaviour Programming agent...[/bold green]")
        try:
            behavior_agent.reset_metrics()
            start = time.time()
            response = behavior_agent.run(q.query)
            latency = round(time.time() - start, 2)
            answer = response.result if isinstance(response.result, str) else str(response.result)
            # +1 for the plan generation LLM call by Behaviour Programming itself
            behavior_result = AgentResult(
                agent_name="Behaviour Programming",
                query=q.query,
                answer=answer or "",
                plan=response.plan or "",
                llm_calls=behavior_agent.llm_calls + 1,
                total_tokens=behavior_agent.total_tokens,
                latency_seconds=latency,
                success=True,
            )
        except Exception as e:
            behavior_result = AgentResult(
                agent_name="Behaviour Programming",
                query=q.query,
                answer="[CRASHED]",
                llm_calls=0,
                total_tokens=0,
                latency_seconds=0,
                success=False,
                failure_reason=f"{type(e).__name__}: {e}",
            )
        _print_result(behavior_result)

        # Evaluate answers against ground truth
        react_eval = None
        behavior_eval = None
        console.print("[bold]Evaluating answers against ground truth...[/bold]")
        if react_result and react_result.success and react_result.answer != "[CRASHED]":
            react_eval = evaluate_answer(q.query, q.ground_truth, react_result.answer)
            _print_eval("ReAct", react_eval, "red")
        if behavior_result.success and behavior_result.answer != "[CRASHED]":
            behavior_eval = evaluate_answer(q.query, q.ground_truth, behavior_result.answer)
            _print_eval("BP", behavior_eval, "green")
        console.print()

        # Side-by-side comparison (skip in bp_only mode)
        if not bp_only:
            _print_comparison(react_result, behavior_result)

        # Collect log entry
        run_log.append(QueryLog(
            label=q.label,
            query=q.query,
            ground_truth=q.ground_truth,
            react_result=react_result,
            behavior_result=behavior_result,
            react_eval=react_eval,
            behavior_eval=behavior_eval,
        ))

    # Write log file
    log_file.write_text(json.dumps([entry.model_dump() for entry in run_log], indent=2))
    console.print(f"[dim]Results logged to {log_file}[/dim]")


# Hand-picked queries that best showcase ReAct vs BP differences:
#   1  – Simple extraction (both correct, BP 4.7x fewer tokens)
#   8  – Cross-company comparison (both correct, BP 34.6x fewer tokens)
#   11 – Aggregation across all 10 (BP has dedicated decomposition, ReAct hits limits)
DEMO_SHOWCASE = [1, 8, 11]


def main() -> None:
    parser = argparse.ArgumentParser(description="SEC 10-K Agent Comparison Demo")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "-q", "--queries",
        type=int,
        nargs="*",
        help="Which demo queries to run (1-15). Default: all.",
    )
    group.add_argument(
        "--demo",
        action="store_true",
        help="Run the 3 showcase queries (simple, comparison, aggregation).",
    )
    parser.add_argument(
        "--bp-only",
        action="store_true",
        help="Run only the Behaviour Programming agent (skip ReAct and evaluation).",
    )
    parser.add_argument(
        "--side-by-side",
        action="store_true",
        help="Run both agents side-by-side in a live split-pane TUI.",
    )
    args = parser.parse_args()
    indices = DEMO_SHOWCASE if args.demo else args.queries
    if args.side_by_side:
        run_demo_side_by_side(query_indices=indices)
    else:
        run_demo(query_indices=indices, bp_only=args.bp_only)


if __name__ == "__main__":
    main()
