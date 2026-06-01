# Company Research Assistant

Python CLI implementation for NLB AI engineer task. The system performs bounded public-web company research, stores collected evidence in a local SQLite knowledge base, and answers later questions from that persisted evidence with structured JSON and citations.

These instructions are for linux based machines. Windows and MacOS wasn't tested, but are assumed to be working if commands are changed accordingly (for example `source .venv/bin/activate` to `.venv\Scripts\activate`)

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python -m playwright install chromium
cp .env.example .env
```

If your home cache is read-only, install Playwright browsers into the project directory instead:

```bash
PLAYWRIGHT_BROWSERS_PATH=.playwright-browsers python -m playwright install chromium
```

Configure `.env` with a lawful search provider key. Brave Search is the default adapter:

```bash
LOG_LEVEL=INFO
LOG_FILE=data/company-research.log

SEARCH_PROVIDER=brave
BRAVE_SEARCH_API_KEY=...
SEARCH_FIXTURE_PATH=
MAX_SEARCH_RESULTS=100
MAX_VISITED_PAGES=50
MAX_DOWNLOADS=16
MAX_DEPTH=10
MAX_RUNTIME_SECONDS=600
CRAWL_DELAY_SECONDS=1
MAX_FILE_MB=50
HTTP_TIMEOUT_SECONDS=30

QA_FACT_EVIDENCE_LIMIT=25
QA_CHUNK_EVIDENCE_LIMIT=25

LLM_PROVIDER=openai-compatible
LLM_ENDPOINT=
LLM_API_KEY=
LLM_MODEL_NAME=
LLM_TIMEOUT_SECONDS=60
LLM_MAX_RETRIES=2
LLM_MODE=provider
```

`LLM_PROVIDER` currently supports `openai`, `openai-compatible`, or an empty value for
OpenAI-compatible chat-completions APIs. Other values are skipped with a warning and
the deterministic fallback answer is used.

Secrets are read from environment variables and are redacted from audit logs.

## Commands

```bash
research --company "NLB Group" --kb-path data/kb
research --companies data/companies.txt --kb-path data/kb
ask --kb-path data/kb --question "According to NLB Group's publicly available 2024 annual report or FY2024 financial report, what were the Group's total net operating income, profit after tax, total assets, and CET1 ratio?"
chat --kb-path data/kb
```

The same commands are also available through the grouped CLI:

```bash
company-research research --company "NLB Group" --kb-path data/kb
company-research ask --kb-path data/kb --question "What evidence is stored?"
```

## Offline Fixture Mode

For deterministic tests or demos without a search API, set:

```bash
SEARCH_FIXTURE_PATH=tests/fixtures/search_results.json
LLM_MODE=mock
```

Fixture URLs may be `file://` URLs. They are treated as public test fixtures and copied into the raw-source cache.

## Knowledge Base

`--kb-path` points to a directory containing:

- `knowledge_base.sqlite3`: persistent runs, companies, search queries, source decisions, documents, chunks, facts, answers, and audit events.
- `raw/`: downloaded public source snapshots.

Reviewers can research once, stop the process, restart it, and run `ask` against the same `--kb-path` without repeating web research.

## Budgets and Safety Defaults

Default budgets are configured in `.env.example`:

- 100 search results per query
- 50 visited pages per company
- 16 downloads per company
- depth 10
- 10 minute runtime per company
- 1 second crawl delay
- 50 MB max file size
- 30 second HTTP timeout
- 25 fact evidence items and 25 chunk evidence items for Q&A
- 60 second LLM timeout with 2 retries

The crawler skips unsupported schemes, social platforms, login/account/paywall-like URLs, and non-public sources. It checks `robots.txt` for HTTP(S) URLs and logs source accept/reject decisions.

## Benchmark Run

```bash
research --companies data/companies.txt --kb-path data/kb
ask --kb-path data/kb --question "According to NLB Group's publicly available 2024 annual report or FY2024 financial report, what were the Group's total net operating income, profit after tax, total assets, and CET1 ratio? Include the reporting period, units, report title/version/date, entity level, and cited sources." --output outputs/benchmark/nlb_group.json
ask --kb-path data/kb --question "According to Petrol Group's publicly available 2024 annual report or FY2024 financial report, what were the Group's sales revenue, adjusted gross profit, EBITDA, and net profit or loss? Include the reporting period, units, report date/version, and cited sources." --output outputs/benchmark/petrol_group.json
ask --kb-path data/kb --question "According to GEN-I's publicly available 2024 annual report or FY2024 financial report, what were the company's or group's revenue, net profit or loss, and reported electricity/energy sales or trading volume, if available? Include the reporting period, units, entity level, and cited sources." --output outputs/benchmark/gen_i.json
```

## Tests

```bash
pytest
```

The tests cover configuration, redaction, source ranking, extraction, SQLite persistence, restart Q&A, JSON answer shape, etc.
