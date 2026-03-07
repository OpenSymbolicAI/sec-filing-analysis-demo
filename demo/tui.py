"""Side-by-side TUI: ReAct vs Behaviour Programming, powered by Textual."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

from rich.markup import escape
from rich.syntax import Syntax
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.widgets import RichLog, Static

from behavior_agent import SECAnalyst, create_behavior_agent
from evaluate import evaluate_answer
from models import AgentResult, DemoQuery, EvalScore
from react_agent import ReactAgent, create_react_agent
from retriever import Retriever


# ── Shared pane state (written by agent threads, read by TUI) ──


@dataclass
class PaneState:
    """Mutable state for one agent's display pane."""

    name: str
    color: str
    steps: list[str] = field(default_factory=list)
    llm_calls: int = 0
    total_tokens: int = 0
    start_time: float = 0.0
    elapsed: float = 0.0
    answer: str = ""
    plan: str = ""
    done: bool = False
    error: str = ""
    eval_lines: list[str] = field(default_factory=list)
    _last_rendered: int = 0
    _answer_written: bool = False
    _eval_written: bool = False
    _plan_written: bool = False

    def add_step(self, text: str) -> None:
        self.steps.append(text)
        self.elapsed = time.time() - self.start_time


# ── Agent instrumentation (same as before) ─────────────────────


def _summarize_tool_args(name: str, args: dict) -> str:
    if name in ("retrieve", "retrieve_for_company"):
        company = args.get("company", "")
        query = args.get("query", "")[:50]
        return f"{company}: {query}" if company else query
    if name in ("extract_number", "extract_count"):
        return args.get("metric", "")[:40]
    if name == "answer_from_context":
        return args.get("question", "")[:40]
    if name == "extract_metrics":
        return ", ".join(args.get("metrics", []))[:40]
    if name == "compute_ratio":
        n = str(args.get("numerator", "?"))[:15]
        d = str(args.get("denominator", "?"))[:15]
        return f"{n} / {d}"
    if name == "combine_texts":
        count = len(args.get("docs", []))
        return f"{count} docs"
    return str(args)[:40]


def _instrument_react(agent: ReactAgent, pane: PaneState) -> None:
    original_execute = agent._execute_tool

    def patched_execute(name: str, args: dict) -> str:
        summary = _summarize_tool_args(name, args)
        pane.add_step(f"[yellow]{name}[/yellow]({summary})")
        result = original_execute(name, args)
        pane.llm_calls = agent._llm_calls
        pane.total_tokens = agent._total_tokens
        pane.elapsed = time.time() - pane.start_time
        return result

    agent._execute_tool = patched_execute


def _instrument_bp(agent: SECAnalyst, pane: PaneState) -> None:
    def _sync_metrics() -> None:
        """Read metrics directly from the agent (works even if _ask patch is bypassed)."""
        pane.llm_calls = agent.llm_calls
        pane.total_tokens = agent.total_tokens
        pane.elapsed = time.time() - pane.start_time

    # Intercept execute() to capture the plan before primitives run
    original_execute = agent.execute

    def patched_execute(plan_code: str):
        pane.plan = plan_code
        pane.add_step("[bold]Plan generated — executing…[/bold]")
        _sync_metrics()
        return original_execute(plan_code)

    agent.execute = patched_execute

    for method_name in [
        "retrieve_for_company", "combine_texts", "extract_number",
        "extract_metrics", "extract_count", "answer_from_context", "compute_ratio",
    ]:
        original = getattr(agent, method_name, None)
        if original is None:
            continue

        def _make_wrapper(orig, name):
            def wrapper(*args, **kwargs):
                if name == "retrieve_for_company":
                    company = kwargs.get("company", args[1] if len(args) > 1 else "?")
                    query = str(kwargs.get("query", args[0] if args else ""))[:40]
                    summary = f"{company}: {query}"
                elif name in ("extract_number", "extract_count"):
                    summary = str(kwargs.get("metric", args[1] if len(args) > 1 else ""))[:40]
                elif name == "answer_from_context":
                    summary = str(kwargs.get("question", args[1] if len(args) > 1 else ""))[:40]
                elif name == "extract_metrics":
                    metrics = kwargs.get("metrics", args[1] if len(args) > 1 else [])
                    summary = ", ".join(metrics)[:40]
                elif name == "compute_ratio":
                    n = str(kwargs.get("numerator", args[0] if args else "?"))[:15]
                    d = str(kwargs.get("denominator", args[1] if len(args) > 1 else "?"))[:15]
                    summary = f"{n} / {d}"
                elif name == "combine_texts":
                    docs = kwargs.get("docs", args[0] if args else [])
                    summary = f"{len(docs)} docs"
                else:
                    summary = ""
                pane.add_step(f"[yellow]{name}[/yellow]({summary})")
                result = orig(*args, **kwargs)
                _sync_metrics()
                return result

            # Copy primitive decorator attrs so the framework still discovers this wrapper
            for attr in ("__method_type__", "__primitive_read_only__", "__primitive_deterministic__",
                         "__wrapped__", "__name__", "__qualname__", "__doc__"):
                if hasattr(orig, attr):
                    setattr(wrapper, attr, getattr(orig, attr))

            return wrapper

        setattr(agent, method_name, _make_wrapper(original, method_name))


# ── Eval formatting ────────────────────────────────────────────


def _format_eval_lines(score: EvalScore, color: str) -> list[str]:
    dims = [
        ("Correct", score.correctness),
        ("Complete", score.completeness),
        ("Relevant", score.relevance),
        ("Faithful", score.faithfulness),
        ("Concise", score.conciseness),
    ]
    avg = sum(v for _, v in dims) / len(dims)
    lines = [f"[bold {color}]Eval[/bold {color}] (avg {avg:.1f}/5)"]
    for name, val in dims:
        bar = "█" * val + "░" * (5 - val)
        lines.append(f"  [{color}]{bar}[/{color}] {name}: {val}/5")
    return lines


# ── Textual App ────────────────────────────────────────────────

PANE_CSS = """
RichLog {
    border: solid $accent;
    scrollbar-size: 1 1;
}
"""

APP_CSS = """
Screen {
    layout: grid;
    grid-size: 2 3;
    grid-rows: 1fr 1fr 3;
    grid-gutter: 0 1;
}

#react-log {
    column-span: 1;
    row-span: 2;
    border: solid red;
    border-title-color: red;
    border-title-style: bold;
    overflow-y: scroll;
    scrollbar-size-vertical: 1;
}

#bp-log {
    column-span: 1;
    row-span: 2;
    border: solid green;
    border-title-color: green;
    border-title-style: bold;
    overflow-y: scroll;
    scrollbar-size-vertical: 1;
}

#footer-bar {
    column-span: 2;
    height: 3;
    content-align: center middle;
    text-align: center;
    border: solid $accent;
}
"""


class SideBySideApp(App):
    CSS = APP_CSS
    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("n", "next_query", "Next query"),
    ]

    def __init__(
        self,
        queries: list[DemoQuery],
        retriever: Retriever,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._queries = queries
        self._retriever = retriever
        self._query_idx = 0
        self._react_pane = PaneState(name="ReAct (Tool Calling)", color="red")
        self._bp_pane = PaneState(name="Behaviour Programming", color="green")
        self._react_result: AgentResult | None = None
        self._bp_result: AgentResult | None = None
        self._react_eval: EvalScore | None = None
        self._bp_eval: EvalScore | None = None
        self._running = False
        self._all_done = False
        self._poll_timer = None
        # Collect results across queries
        self.all_results: list[
            tuple[DemoQuery, AgentResult | None, AgentResult | None, EvalScore | None, EvalScore | None]
        ] = []

    def compose(self) -> ComposeResult:
        yield RichLog(id="react-log", wrap=True, markup=True, auto_scroll=True)
        yield RichLog(id="bp-log", wrap=True, markup=True, auto_scroll=True)
        yield Static("", id="footer-bar")

    def on_mount(self) -> None:
        react_log = self.query_one("#react-log", RichLog)
        bp_log = self.query_one("#bp-log", RichLog)
        react_log.border_title = "ReAct (Tool Calling)"
        bp_log.border_title = "Behaviour Programming"
        self._start_query()

    def _start_query(self) -> None:
        if self._query_idx >= len(self._queries):
            self._update_footer("[bold]All queries complete. Press [reverse] q [/reverse] to quit.[/bold]")
            self._all_done = True
            return

        q = self._queries[self._query_idx]

        # Reset pane state
        self._react_pane = PaneState(name="ReAct (Tool Calling)", color="red")
        self._bp_pane = PaneState(name="Behaviour Programming", color="green")
        self._react_result = None
        self._bp_result = None
        self._react_eval = None
        self._bp_eval = None
        self._running = True

        # Clear logs and re-enable auto-scroll for new query
        react_log = self.query_one("#react-log", RichLog)
        bp_log = self.query_one("#bp-log", RichLog)
        react_log.clear()
        bp_log.clear()
        react_log.auto_scroll = True
        bp_log.auto_scroll = True

        # Show query
        label = f"[bold cyan]Query {self._query_idx + 1}/{len(self._queries)}: {q.label}[/bold cyan]"
        react_log.write(label)
        react_log.write(f"[italic]{q.query}[/italic]\n")
        bp_log.write(label)
        bp_log.write(f"[italic]{q.query}[/italic]\n")

        self._update_footer("[dim]Both agents running…[/dim]")

        # Create and instrument agents
        react_agent = create_react_agent(self._retriever)
        bp_agent = create_behavior_agent(self._retriever)
        _instrument_react(react_agent, self._react_pane)
        _instrument_bp(bp_agent, self._bp_pane)

        # Launch agent threads
        def run_react():
            self._react_pane.start_time = time.time()
            try:
                result = react_agent.run(q.query)
                self._react_pane.answer = result.answer
                self._react_pane.llm_calls = result.llm_calls
                self._react_pane.total_tokens = result.total_tokens
                self._react_pane.elapsed = time.time() - self._react_pane.start_time
                self._react_result = result
                self._react_pane.done = True
            except Exception as exc:
                self._react_pane.error = f"{type(exc).__name__}: {exc}"
                self._react_pane.elapsed = time.time() - self._react_pane.start_time
                self._react_result = AgentResult(
                    agent_name="ReAct (Groq Tool Calling)",
                    query=q.query,
                    answer="[CRASHED]",
                    success=False,
                    failure_reason=self._react_pane.error,
                )
                self._react_pane.done = True

        def run_bp():
            self._bp_pane.start_time = time.time()
            try:
                bp_agent.reset_metrics()
                response = bp_agent.run(q.query)
                answer = response.result if isinstance(response.result, str) else str(response.result)
                self._bp_pane.plan = response.plan or ""
                self._bp_pane.answer = answer or ""
                self._bp_pane.llm_calls = bp_agent.llm_calls + 1
                self._bp_pane.total_tokens = bp_agent.total_tokens
                self._bp_pane.elapsed = time.time() - self._bp_pane.start_time
                self._bp_result = AgentResult(
                    agent_name="Behaviour Programming",
                    query=q.query,
                    answer=answer or "",
                    plan=response.plan or "",
                    llm_calls=bp_agent.llm_calls + 1,
                    total_tokens=bp_agent.total_tokens,
                    latency_seconds=round(self._bp_pane.elapsed, 2),
                    success=True,
                )
                self._bp_pane.done = True
            except Exception as exc:
                self._bp_pane.error = f"{type(exc).__name__}: {exc}"
                self._bp_pane.elapsed = time.time() - self._bp_pane.start_time
                self._bp_result = AgentResult(
                    agent_name="Behaviour Programming",
                    query=q.query,
                    answer="[CRASHED]",
                    success=False,
                    failure_reason=self._bp_pane.error,
                )
                self._bp_pane.done = True

        threading.Thread(target=run_react, daemon=True).start()
        threading.Thread(target=run_bp, daemon=True).start()

        # Start polling timer (cancel any previous one first)
        if self._poll_timer is not None:
            self._poll_timer.stop()
        self._poll_timer = self.set_interval(0.25, self._poll_agents)

    def _poll_agents(self) -> None:
        """Called every 250ms to sync pane state into the RichLog widgets."""
        react_log = self.query_one("#react-log", RichLog)
        bp_log = self.query_one("#bp-log", RichLog)
        rp, bp = self._react_pane, self._bp_pane
        q = self._queries[self._query_idx]

        # Show plan as soon as it's generated (before execution finishes)
        if bp.plan and not bp._plan_written:
            bp._plan_written = True
            bp_log.write("")
            bp_log.write("[bold]Plan:[/bold]")
            bp_log.write(Syntax(bp.plan.strip(), "python", theme="monokai", line_numbers=False))
            bp_log.write("")

        # Flush new steps
        self._flush_steps(rp, react_log)
        self._flush_steps(bp, bp_log)

        # Update border subtitles with live metrics
        react_log.border_subtitle = (
            f"LLM calls: {rp.llm_calls} | "
            f"Tokens: {rp.total_tokens:,} | "
            f"{rp.elapsed:.1f}s"
        )
        bp_log.border_subtitle = (
            f"LLM calls: {bp.llm_calls} | "
            f"Tokens: {bp.total_tokens:,} | "
            f"{bp.elapsed:.1f}s"
        )

        # Write answer as soon as each agent finishes (independently)
        if rp.done and not rp._answer_written:
            rp._answer_written = True
            self._flush_steps(rp, react_log)
            self._write_answer(rp, react_log)
            react_log.auto_scroll = False
            react_log.focus()
            self._start_eval_for("react", q, rp, self._react_result)

        if bp.done and not bp._answer_written:
            bp._answer_written = True
            self._flush_steps(bp, bp_log)
            self._write_answer(bp, bp_log)
            bp_log.auto_scroll = False
            if not rp.done:
                bp_log.focus()
            self._start_eval_for("bp", q, bp, self._bp_result)

        # Update footer based on state
        if bp.done and rp.done and bp._eval_written and rp._eval_written:
            pass  # final footer already set by _write_eval_to_pane
        elif bp.done and not rp.done:
            self._update_footer(
                "[bold green]BP finished![/bold green]  [dim]ReAct still running…[/dim]"
            )
        elif rp.done and not bp.done:
            self._update_footer(
                "[bold red]ReAct finished![/bold red]  [dim]BP still running…[/dim]"
            )

        # Both done and both evals written — mark complete but keep timer alive
        if rp._eval_written and bp._eval_written and self._running:
            self._running = False

        # Keep a RichLog focused so scrolling always works
        if not isinstance(self.focused, RichLog):
            self.query_one("#react-log", RichLog).focus()

    def _start_eval_for(
        self, agent_id: str, q: DemoQuery, pane: PaneState, result: AgentResult | None,
    ) -> None:
        """Kick off eval for one agent in a background thread."""
        color = "red" if agent_id == "react" else "green"

        def _do_eval():
            eval_score = None
            try:
                if result and result.success and result.answer != "[CRASHED]":
                    eval_score = evaluate_answer(q.query, q.ground_truth, result.answer)
                    pane.eval_lines = _format_eval_lines(eval_score, color)
            except Exception as exc:
                pane.eval_lines = [f"[bold red]Eval failed: {type(exc).__name__}: {exc}[/bold red]"]
            if agent_id == "react":
                self._react_eval = eval_score
            else:
                self._bp_eval = eval_score
            self.call_from_thread(self._write_eval_to_pane, agent_id)

        threading.Thread(target=_do_eval, daemon=True).start()

    def _write_eval_to_pane(self, agent_id: str) -> None:
        """Called from main thread after one agent's eval completes."""
        if agent_id == "react":
            pane = self._react_pane
            log = self.query_one("#react-log", RichLog)
        else:
            pane = self._bp_pane
            log = self.query_one("#bp-log", RichLog)

        if pane.eval_lines:
            log.write("")
            for line in pane.eval_lines:
                log.write(line)

        # Print metrics summary below eval
        log.write("")
        log.write(f"[bold {pane.color}]Metrics[/bold {pane.color}]")
        log.write(f"  LLM calls:  {pane.llm_calls}")
        log.write(f"  Tokens:     {pane.total_tokens:,}")
        log.write(f"  Latency:    {pane.elapsed:.1f}s")
        log.write(f"  Steps:      {len(pane.steps)}")

        pane._eval_written = True
        log.scroll_end(animate=False)

        # If both evals are now done, show comparison footer and store results
        rp, bp = self._react_pane, self._bp_pane
        if rp._eval_written and bp._eval_written:
            parts = []
            if bp.total_tokens > 0:
                parts.append(f"Tokens: {rp.total_tokens / bp.total_tokens:.1f}x")
            if bp.llm_calls > 0:
                parts.append(f"LLM calls: {rp.llm_calls / bp.llm_calls:.1f}x")
            if bp.elapsed > 0:
                parts.append(f"Speed: {rp.elapsed / bp.elapsed:.1f}x")
            ratio_text = " │ ".join(parts)

            more = self._query_idx + 1 < len(self._queries)
            nav = "  Press [reverse] n [/reverse] for next query." if more else ""
            self._update_footer(f"[bold]ReAct vs BP → {ratio_text}[/bold]{nav}")

            q = self._queries[self._query_idx]
            self.all_results.append(
                (q, self._react_result, self._bp_result, self._react_eval, self._bp_eval)
            )

            # Focus a RichLog pane so scroll (mouse + keyboard) works after completion
            self.query_one("#react-log", RichLog).focus()

    def _flush_steps(self, pane: PaneState, log: RichLog) -> None:
        new_steps = pane.steps[pane._last_rendered:]
        for i, step in enumerate(new_steps, pane._last_rendered + 1):
            log.write(f"[cyan]{i:>3}.[/cyan] {step}")
        pane._last_rendered = len(pane.steps)

    def _write_answer(self, pane: PaneState, log: RichLog) -> None:
        log.write("")
        if pane.error:
            log.write(f"[bold red]✗ ERROR: {pane.error}[/bold red]")
            return
        log.write("[bold green]✓ DONE[/bold green]")
        if pane.answer:
            log.write(f"\n[bold]Answer:[/bold] {pane.answer}")

    def _update_footer(self, text: str) -> None:
        footer = self.query_one("#footer-bar", Static)
        footer.update(text)

    def on_key(self, event) -> None:
        """Handle scroll and focus keys directly, bypassing binding system."""
        target = self.focused if isinstance(self.focused, RichLog) else self.query_one("#react-log", RichLog)
        key = event.key
        if key == "up":
            target.scroll_y = max(0, target.scroll_y - 2)
            event.prevent_default()
        elif key == "down":
            target.scroll_y = min(target.max_scroll_y, target.scroll_y + 2)
            event.prevent_default()
        elif key == "pageup":
            target.scroll_page_up(animate=False)
            event.prevent_default()
        elif key == "pagedown":
            target.scroll_page_down(animate=False)
            event.prevent_default()
        elif key == "left":
            self.query_one("#react-log", RichLog).focus()
            event.prevent_default()
        elif key == "right":
            self.query_one("#bp-log", RichLog).focus()
            event.prevent_default()

    def on_click(self, event) -> None:
        """Focus whichever RichLog pane was clicked."""
        for log_id in ("#react-log", "#bp-log"):
            log = self.query_one(log_id, RichLog)
            if log.region.contains(event.screen_x, event.screen_y):
                log.focus()
                return

    def action_next_query(self) -> None:
        if self._running:
            return
        self._query_idx += 1
        self._start_query()


# ── Public entry point ─────────────────────────────────────────


def run_side_by_side(
    queries: list[DemoQuery],
    retriever: Retriever,
) -> list[tuple[DemoQuery, AgentResult | None, AgentResult | None, EvalScore | None, EvalScore | None]]:
    """Launch the Textual side-by-side TUI.

    Returns list of (query, react_result, bp_result, react_eval, bp_eval).
    """
    app = SideBySideApp(queries=queries, retriever=retriever)
    app.run()
    return app.all_results
