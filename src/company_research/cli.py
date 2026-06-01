from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from .config import load_config, load_dotenv
from .kb import KnowledgeBase
from .qa import answer_question, answer_to_json
from .researcher import load_companies, research_companies, research_company

console = Console()
logger = logging.getLogger(__name__)

main_app = typer.Typer(help="Bounded public-web company research assistant.")
research_app = typer.Typer(help="Research one company or a batch of companies.")
ask_app = typer.Typer(help="Ask a question against a persisted knowledge base.")
chat_app = typer.Typer(help="Interactive chat against a persisted knowledge base.")


def _configure_logging(env_file: Path | None = Path(".env")) -> None:
    """Configure process logging from LOG_LEVEL and optional LOG_FILE."""
    if env_file:
        path = Path(env_file)

        if path.exists():
            load_dotenv(path, override=False)

    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    log_file = os.getenv("LOG_FILE", "").strip()

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        filename=log_file or None,
        filemode="a",
    )

    logger.debug("Logging configured with level=%s file=%s", logging.getLevelName(level), log_file or "stderr")


@research_app.callback(invoke_without_command=True)
def research_command(
    company: Optional[str] = typer.Option(None, "--company", help="Company name to research."),
    companies: Optional[Path] = typer.Option(None, "--companies", help="Text file with one company per line."),
    kb_path: Path = typer.Option(..., "--kb-path", help="Persistent knowledge-base directory."),
    env_file: Path = typer.Option(Path(".env"), "--env-file", help="Optional .env path."),
) -> None:
    """Run a bounded company research job and print resulting run ids as JSON."""
    _configure_logging(env_file)

    logger.info("Research command started company=%r companies_file=%s kb_path=%s", company, companies, kb_path)

    if not company and not companies:
        logger.error("Research command missing both --company and --companies")
        raise typer.BadParameter("Provide --company or --companies")

    config = load_config(env_file)
    kb = KnowledgeBase(kb_path)

    try:
        if company:
            run_ids = [research_company(kb, company, config)]
        else:
            run_ids = research_companies(kb, load_companies(companies), config)

        logger.info("Research command completed run_ids=%s", run_ids)
        print(json.dumps({"status": "completed", "run_ids": run_ids}, indent=2))

    finally:
        kb.close()


@ask_app.callback(invoke_without_command=True)
def ask_command(
    kb_path: Path = typer.Option(..., "--kb-path", help="Persistent knowledge-base directory."),
    question: str = typer.Option(..., "--question", help="Question to answer from stored evidence."),
    env_file: Path = typer.Option(Path(".env"), "--env-file", help="Optional .env path."),
    output: Optional[Path] = typer.Option(None, "--output", help="Optional path for answer JSON."),
) -> None:
    """Answer one question from a persisted knowledge base and print JSON."""
    _configure_logging(env_file)

    logger.info("Ask command started kb_path=%s output=%s question=%r", kb_path, output, question)

    config = load_config(env_file)
    kb = KnowledgeBase(kb_path)

    try:
        answer = answer_question(kb, question, config)
        payload = answer_to_json(answer)

        if output:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(payload + "\n", encoding="utf-8")
            logger.info("Wrote answer JSON to %s", output)

        print(payload)

    finally:
        kb.close()


@chat_app.callback(invoke_without_command=True)
def chat_command(
    kb_path: Path = typer.Option(..., "--kb-path", help="Persistent knowledge-base directory."),
    env_file: Path = typer.Option(Path(".env"), "--env-file", help="Optional .env path."),
) -> None:
    """Start an interactive question loop against a persisted knowledge base."""
    _configure_logging(env_file)

    logger.info("Chat command started kb_path=%s", kb_path)

    config = load_config(env_file)
    kb = KnowledgeBase(kb_path)

    console.print("Ask questions against the persisted KB. Press Ctrl-D or enter an empty line to exit.")

    try:
        while True:
            try:
                question = typer.prompt("question", default="", show_default=False)
            except (EOFError, KeyboardInterrupt):
                console.print()
                break

            if not question.strip():
                logger.info("Chat command exiting after empty question")
                break

            answer = answer_question(kb, question, config)
            print(answer_to_json(answer))

    finally:
        kb.close()


main_app.add_typer(research_app, name="research")
main_app.add_typer(ask_app, name="ask")
main_app.add_typer(chat_app, name="chat")


def run_research_app() -> None:
    """Entrypoint for the standalone research console script."""
    research_app()


def run_ask_app() -> None:
    """Entrypoint for the standalone ask console script."""
    ask_app()


def run_chat_app() -> None:
    """Entrypoint for the standalone chat console script."""
    chat_app()


def run_main_app() -> None:
    """Entrypoint for the grouped company-research console script."""
    main_app()
