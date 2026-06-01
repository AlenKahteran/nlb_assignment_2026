from __future__ import annotations

import json
import logging
import re
import sqlite3
import uuid
from pathlib import Path
from typing import Any, Iterable

from .models import (
    Answer,
    ExtractedDocument,
    ExtractedFact,
    FetchResult,
    SearchResult,
    SourceDecision,
    utc_now,
)


logger = logging.getLogger(__name__)

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS runs (
  id TEXT PRIMARY KEY,
  company_name TEXT,
  started_at TEXT NOT NULL,
  completed_at TEXT,
  config_json TEXT NOT NULL,
  status TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS companies (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL UNIQUE,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS search_queries (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id TEXT NOT NULL,
  company_id INTEGER NOT NULL,
  query TEXT NOT NULL,
  created_at TEXT NOT NULL,
  FOREIGN KEY(run_id) REFERENCES runs(id),
  FOREIGN KEY(company_id) REFERENCES companies(id)
);

CREATE TABLE IF NOT EXISTS search_results (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  query_id INTEGER NOT NULL,
  url TEXT NOT NULL,
  title TEXT,
  snippet TEXT,
  rank INTEGER,
  publisher TEXT,
  age TEXT,
  considered_at TEXT NOT NULL,
  UNIQUE(query_id, url),
  FOREIGN KEY(query_id) REFERENCES search_queries(id)
);

CREATE TABLE IF NOT EXISTS sources (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id TEXT NOT NULL,
  company_id INTEGER NOT NULL,
  url TEXT NOT NULL,
  final_url TEXT,
  title TEXT,
  publisher TEXT,
  source_type TEXT NOT NULL,
  publication_date TEXT,
  retrieved_at TEXT,
  decision TEXT NOT NULL,
  decision_reason TEXT NOT NULL,
  score INTEGER NOT NULL,
  status_code INTEGER,
  content_type TEXT,
  metadata_json TEXT NOT NULL DEFAULT '{}',
  UNIQUE(company_id, url),
  FOREIGN KEY(run_id) REFERENCES runs(id),
  FOREIGN KEY(company_id) REFERENCES companies(id)
);

CREATE TABLE IF NOT EXISTS documents (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  source_id INTEGER NOT NULL,
  local_path TEXT,
  title TEXT,
  content_type TEXT,
  content_sha256 TEXT,
  extractor TEXT NOT NULL,
  metadata_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL,
  FOREIGN KEY(source_id) REFERENCES sources(id)
);

CREATE TABLE IF NOT EXISTS chunks (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  document_id INTEGER NOT NULL,
  source_id INTEGER NOT NULL,
  ordinal INTEGER NOT NULL,
  text TEXT NOT NULL,
  FOREIGN KEY(document_id) REFERENCES documents(id),
  FOREIGN KEY(source_id) REFERENCES sources(id)
);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
  text,
  content='chunks',
  content_rowid='id'
);

CREATE TABLE IF NOT EXISTS extracted_facts (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  source_id INTEGER NOT NULL,
  document_id INTEGER NOT NULL,
  metric TEXT NOT NULL,
  value TEXT NOT NULL,
  unit TEXT,
  period TEXT,
  entity_level TEXT,
  evidence TEXT NOT NULL,
  confidence TEXT NOT NULL,
  created_at TEXT NOT NULL,
  FOREIGN KEY(source_id) REFERENCES sources(id),
  FOREIGN KEY(document_id) REFERENCES documents(id)
);

CREATE TABLE IF NOT EXISTS answers (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id TEXT NOT NULL,
  question TEXT NOT NULL,
  answer_json TEXT NOT NULL,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS audit_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id TEXT NOT NULL,
  event_type TEXT NOT NULL,
  message TEXT NOT NULL,
  metadata_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL
);
"""


def _fts_query_from_question(question: str) -> str:
    """Build a safe SQLite FTS5 query from user question text."""
    tokens = [token for token in re.findall(r"[A-Za-z0-9]+", question) if len(token) > 2]

    return " OR ".join(f'"{token}"' for token in tokens)


def _tokens(text: str) -> set[str]:
    """Return normalized word tokens for matching company names in questions."""
    return {token.lower() for token in re.findall(r"[A-Za-z0-9]+", text)}


def _years_from_text(text: str) -> set[str]:
    """Return four-digit years mentioned in text."""
    return set(re.findall(r"20\d{2}", text or ""))


def _fact_year_alignment_score(row: sqlite3.Row, requested_years: set[str]) -> int:
    """Score whether a fact source/evidence aligns with requested reporting years."""
    if not requested_years:
        return 0

    period_text = str(row["period"] or "")
    source_text = " ".join(
        str(row[key] or "") for key in ("title", "url", "publication_date") if key in row.keys()
    )
    evidence_text = str(row["evidence"] or "")

    period_years = _years_from_text(period_text)
    source_years = _years_from_text(source_text)
    evidence_years = _years_from_text(evidence_text)

    score = 0

    if period_years & requested_years:
        score += 6

    if source_years & requested_years:
        score += 5

    if evidence_years & requested_years:
        score += 2

    if source_years and not (source_years & requested_years):
        score -= 5

    if period_years and not (period_years & requested_years):
        score -= 3

    return score


def _company_match_score(company_name: str, question: str) -> int:
    """Score how strongly a question appears to mention a company."""
    generic_tokens = {"group", "company", "inc", "ltd", "dd", "d", "plc"}
    question_tokens = _tokens(question)
    company_tokens = _tokens(company_name)
    significant_tokens = {token for token in company_tokens if token not in generic_tokens and len(token) > 1}

    normalized_company = " ".join(company_name.lower().replace("-", " ").split())
    normalized_question = " ".join(question.lower().replace("-", " ").split())

    if normalized_company in normalized_question:
        return 100 + len(company_tokens)

    if significant_tokens and significant_tokens <= question_tokens:
        return 50 + len(significant_tokens)

    return 0


class KnowledgeBase:
    """SQLite-backed persistence layer for runs, sources, documents, facts, and answers."""

    def __init__(self, kb_path: str | Path):
        """Open or initialize the persistent knowledge base directory and SQLite schema."""
        self.root = Path(kb_path)
        self.root.mkdir(parents=True, exist_ok=True)

        self.raw_dir = self.root / "raw"
        self.raw_dir.mkdir(parents=True, exist_ok=True)

        self.db_path = self.root / "knowledge_base.sqlite3"
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self._ensure_schema_migrations()

        logger.info("Opened knowledge base at %s", self.db_path)

    def _ensure_schema_migrations(self) -> None:
        """Apply lightweight migrations for knowledge bases created by older versions."""
        document_columns = {
            row["name"] for row in self.conn.execute("PRAGMA table_info(documents)").fetchall()
        }

        if "content_sha256" not in document_columns:
            logger.info("Adding documents.content_sha256 column to existing knowledge base")
            self.conn.execute("ALTER TABLE documents ADD COLUMN content_sha256 TEXT")

        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_documents_content_sha256 ON documents(content_sha256)"
        )
        self.conn.commit()

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        logger.debug("Closing knowledge base connection for %s", self.db_path)
        self.conn.close()

    def start_run(self, company_name: str | None, config: dict[str, Any]) -> str:
        """Create a run record for a research or QA operation."""
        run_id = str(uuid.uuid4())

        self.conn.execute(
            "INSERT INTO runs(id, company_name, started_at, config_json, status) VALUES (?, ?, ?, ?, ?)",
            (run_id, company_name, utc_now(), json.dumps(config, sort_keys=True), "running"),
        )
        self.conn.commit()

        logger.info("Started run_id=%s company=%r", run_id, company_name)

        return run_id

    def finish_run(self, run_id: str, status: str = "completed") -> None:
        """Mark a run as completed or failed."""
        self.conn.execute(
            "UPDATE runs SET completed_at = ?, status = ? WHERE id = ?",
            (utc_now(), status, run_id),
        )
        self.conn.commit()

        logger.info("Finished run_id=%s status=%s", run_id, status)

    def upsert_company(self, name: str) -> int:
        """Insert a company if missing and return its database id."""
        self.conn.execute(
            "INSERT OR IGNORE INTO companies(name, created_at) VALUES (?, ?)",
            (name, utc_now()),
        )

        row = self.conn.execute("SELECT id FROM companies WHERE name = ?", (name,)).fetchone()
        self.conn.commit()

        company_id = int(row["id"])
        logger.info("Upserted company id=%s name=%r", company_id, name)

        return company_id

    def log_event(self, run_id: str, event_type: str, message: str, metadata: dict[str, Any] | None = None) -> None:
        """Persist an auditable event for a run."""
        self.conn.execute(
            "INSERT INTO audit_events(run_id, event_type, message, metadata_json, created_at) VALUES (?, ?, ?, ?, ?)",
            (run_id, event_type, message, json.dumps(metadata or {}, sort_keys=True), utc_now()),
        )
        self.conn.commit()

        logger.info("Audit event run_id=%s type=%s message=%s", run_id, event_type, message)

        if metadata:
            logger.debug("Audit metadata run_id=%s type=%s metadata=%s", run_id, event_type, metadata)

    def log_query(self, run_id: str, company_id: int, query: str) -> int:
        """Persist a search query and return its database id."""
        cur = self.conn.execute(
            "INSERT INTO search_queries(run_id, company_id, query, created_at) VALUES (?, ?, ?, ?)",
            (run_id, company_id, query, utc_now()),
        )
        self.conn.commit()

        query_id = int(cur.lastrowid)
        logger.info("Logged query_id=%s run_id=%s query=%r", query_id, run_id, query)

        return query_id

    def log_search_results(self, query_id: int, results: Iterable[SearchResult]) -> None:
        """Persist raw search results for a query."""
        result_list = list(results)

        for result in result_list:
            self.conn.execute(
                """
                INSERT OR IGNORE INTO search_results(
                  query_id, url, title, snippet, rank, publisher, age, considered_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    query_id,
                    result.url,
                    result.title,
                    result.snippet,
                    result.rank,
                    result.publisher,
                    result.age,
                    utc_now(),
                ),
            )

        self.conn.commit()

        logger.info("Logged %s search results for query_id=%s", len(result_list), query_id)

    def upsert_source(
        self,
        run_id: str,
        company_id: int,
        result: SearchResult,
        decision: SourceDecision,
        fetch: FetchResult | None = None,
        doc: ExtractedDocument | None = None,
    ) -> int:
        """Insert or update source ranking, fetch, and extraction metadata."""
        self.conn.execute(
            """
            INSERT INTO sources(
              run_id, company_id, url, final_url, title, publisher, source_type, publication_date,
              retrieved_at, decision, decision_reason, score, status_code, content_type, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(company_id, url) DO UPDATE SET
              final_url=CASE
                WHEN excluded.retrieved_at IS NULL THEN sources.final_url
                ELSE excluded.final_url
              END,
              title=CASE
                WHEN excluded.retrieved_at IS NULL THEN COALESCE(sources.title, excluded.title)
                ELSE excluded.title
              END,
              publisher=CASE
                WHEN excluded.retrieved_at IS NULL THEN COALESCE(sources.publisher, excluded.publisher)
                ELSE excluded.publisher
              END,
              source_type=excluded.source_type,
              publication_date=CASE
                WHEN excluded.retrieved_at IS NULL THEN sources.publication_date
                ELSE excluded.publication_date
              END,
              retrieved_at=COALESCE(excluded.retrieved_at, sources.retrieved_at),
              decision=excluded.decision,
              decision_reason=excluded.decision_reason,
              score=excluded.score,
              status_code=COALESCE(excluded.status_code, sources.status_code),
              content_type=COALESCE(excluded.content_type, sources.content_type),
              metadata_json=CASE
                WHEN excluded.retrieved_at IS NULL THEN sources.metadata_json
                ELSE excluded.metadata_json
              END
            """,
            (
                run_id,
                company_id,
                result.url,
                fetch.final_url if fetch else result.url,
                doc.title if doc and doc.title else result.title,
                doc.publisher if doc else result.publisher,
                doc.source_type if doc else decision.source_type,
                doc.publication_date if doc else None,
                fetch.retrieved_at if fetch else None,
                "accepted" if decision.accepted else "rejected",
                decision.reason,
                decision.score,
                fetch.status_code if fetch else None,
                fetch.content_type if fetch else None,
                json.dumps(doc.metadata if doc else {}, sort_keys=True),
            ),
        )
        row = self.conn.execute(
            "SELECT id FROM sources WHERE company_id = ? AND url = ?",
            (company_id, result.url),
        ).fetchone()
        self.conn.commit()

        source_id = int(row["id"])

        logger.info(
            "Upserted source_id=%s decision=%s score=%s url=%s",
            source_id,
            "accepted" if decision.accepted else "rejected",
            decision.score,
            result.url,
        )

        return source_id

    def parsed_source_for_url(self, company_id: int, url: str) -> sqlite3.Row | None:
        """Return an existing parsed source/document for a URL, if present."""
        row = self.conn.execute(
            """
            SELECT
              sources.id AS source_id,
              sources.url,
              sources.final_url,
              documents.id AS document_id,
              documents.content_sha256,
              documents.created_at
            FROM sources
            JOIN documents ON documents.source_id = sources.id
            WHERE sources.company_id = ?
              AND (sources.url = ? OR sources.final_url = ?)
            ORDER BY documents.id DESC
            LIMIT 1
            """,
            (company_id, url, url),
        ).fetchone()

        if row:
            logger.info(
                "Found parsed source for company_id=%s url=%s source_id=%s document_id=%s",
                company_id,
                url,
                row["source_id"],
                row["document_id"],
            )

        return row

    def parsed_document_for_hash(self, company_id: int, content_sha256: str) -> sqlite3.Row | None:
        """Return an existing parsed document with the same content hash, if present."""
        row = self.conn.execute(
            """
            SELECT
              sources.id AS source_id,
              sources.url,
              sources.final_url,
              documents.id AS document_id,
              documents.content_sha256,
              documents.created_at
            FROM documents
            JOIN sources ON sources.id = documents.source_id
            WHERE sources.company_id = ?
              AND documents.content_sha256 = ?
            ORDER BY documents.id DESC
            LIMIT 1
            """,
            (company_id, content_sha256),
        ).fetchone()

        if row:
            logger.info(
                "Found parsed document for company_id=%s hash=%s source_id=%s document_id=%s",
                company_id,
                content_sha256[:12],
                row["source_id"],
                row["document_id"],
            )

        return row

    def insert_document(
        self,
        source_id: int,
        fetch: FetchResult,
        doc: ExtractedDocument,
        extractor: str,
        content_sha256: str | None = None,
    ) -> int:
        """Persist one extracted document and its searchable chunks."""
        cur = self.conn.execute(
            """
            INSERT INTO documents(
              source_id, local_path, title, content_type, content_sha256, extractor, metadata_json, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                source_id,
                str(fetch.file_path) if fetch.file_path else None,
                doc.title,
                fetch.content_type,
                content_sha256,
                extractor,
                json.dumps(doc.metadata, sort_keys=True),
                utc_now(),
            ),
        )
        document_id = int(cur.lastrowid)

        for ordinal, chunk in enumerate(doc.chunks):
            chunk_cur = self.conn.execute(
                "INSERT INTO chunks(document_id, source_id, ordinal, text) VALUES (?, ?, ?, ?)",
                (document_id, source_id, ordinal, chunk),
            )

            self.conn.execute(
                "INSERT INTO chunks_fts(rowid, text) VALUES (?, ?)",
                (int(chunk_cur.lastrowid), chunk),
            )

        self.conn.commit()

        logger.info(
            "Inserted document_id=%s source_id=%s extractor=%s chunks=%s",
            document_id,
            source_id,
            extractor,
            len(doc.chunks),
        )

        return document_id

    def insert_facts(self, facts: Iterable[ExtractedFact]) -> None:
        """Persist extracted structured facts."""
        fact_list = list(facts)

        for fact in fact_list:
            self.conn.execute(
                """
                INSERT INTO extracted_facts(
                  source_id, document_id, metric, value, unit, period, entity_level,
                  evidence, confidence, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    fact.source_id,
                    fact.document_id,
                    fact.metric,
                    fact.value,
                    fact.unit,
                    fact.period,
                    fact.entity_level,
                    fact.evidence,
                    fact.confidence,
                    utc_now(),
                ),
            )

        self.conn.commit()

        logger.info("Inserted %s extracted facts", len(fact_list))

    def company_for_question(self, question: str) -> sqlite3.Row | None:
        """Return the best matching company for a question, if one is mentioned."""
        companies = self.conn.execute("SELECT id, name FROM companies ORDER BY length(name) DESC").fetchall()
        scored: list[tuple[int, sqlite3.Row]] = []

        for company in companies:
            score = _company_match_score(company["name"], question)

            if score:
                scored.append((score, company))

        if not scored:
            logger.info("No company filter detected for question=%r", question)
            return None

        scored.sort(key=lambda item: item[0], reverse=True)
        score, company = scored[0]

        logger.info("Detected company filter id=%s name=%r score=%s", company["id"], company["name"], score)

        return company

    def search_evidence(self, question: str, limit: int = 12, company_id: int | None = None) -> list[sqlite3.Row]:
        """Search document chunks for evidence relevant to a question."""
        fts_query = _fts_query_from_question(question)

        if not fts_query:
            fts_query = question.replace('"', " ")

        try:
            logger.debug("Searching FTS evidence query=%r limit=%s", fts_query, limit)
            if company_id is None:
                rows = self.conn.execute(
                    """
                    SELECT chunks.id, chunks.text, sources.url, sources.title, sources.publisher,
                           sources.source_type, sources.publication_date, sources.retrieved_at,
                           documents.id AS document_id, sources.id AS source_id
                    FROM chunks_fts
                    JOIN chunks ON chunks_fts.rowid = chunks.id
                    JOIN sources ON sources.id = chunks.source_id
                    JOIN documents ON documents.id = chunks.document_id
                    WHERE chunks_fts MATCH ?
                    LIMIT ?
                    """,
                    (fts_query, limit),
                ).fetchall()
            else:
                rows = self.conn.execute(
                    """
                    SELECT chunks.id, chunks.text, sources.url, sources.title, sources.publisher,
                           sources.source_type, sources.publication_date, sources.retrieved_at,
                           documents.id AS document_id, sources.id AS source_id
                    FROM chunks_fts
                    JOIN chunks ON chunks_fts.rowid = chunks.id
                    JOIN sources ON sources.id = chunks.source_id
                    JOIN documents ON documents.id = chunks.document_id
                    WHERE chunks_fts MATCH ? AND sources.company_id = ?
                    LIMIT ?
                    """,
                    (fts_query, company_id, limit),
                ).fetchall()

        except sqlite3.OperationalError as exc:
            logger.warning("FTS evidence search failed; falling back to LIKE: %s", exc)
            if company_id is None:
                rows = self.conn.execute(
                    """
                    SELECT chunks.id, chunks.text, sources.url, sources.title, sources.publisher,
                           sources.source_type, sources.publication_date, sources.retrieved_at,
                           documents.id AS document_id, sources.id AS source_id
                    FROM chunks
                    JOIN sources ON sources.id = chunks.source_id
                    JOIN documents ON documents.id = chunks.document_id
                    WHERE chunks.text LIKE ?
                    LIMIT ?
                    """,
                    (f"%{question[:40]}%", limit),
                ).fetchall()
            else:
                rows = self.conn.execute(
                    """
                    SELECT chunks.id, chunks.text, sources.url, sources.title, sources.publisher,
                           sources.source_type, sources.publication_date, sources.retrieved_at,
                           documents.id AS document_id, sources.id AS source_id
                    FROM chunks
                    JOIN sources ON sources.id = chunks.source_id
                    JOIN documents ON documents.id = chunks.document_id
                    WHERE chunks.text LIKE ? AND sources.company_id = ?
                    LIMIT ?
                    """,
                    (f"%{question[:40]}%", company_id, limit),
                ).fetchall()

        result = list(rows)
        logger.info(
            "Evidence search returned %s rows for question=%r company_id=%s",
            len(result),
            question,
            company_id,
        )

        return result

    def facts_for_question(self, question: str, limit: int = 20, company_id: int | None = None) -> list[sqlite3.Row]:
        """Return extracted facts whose metric/evidence text matches question keywords."""
        keywords = [token.lower() for token in question.replace("/", " ").split() if len(token) > 2]
        requested_years = _years_from_text(question)

        logger.debug("Looking up facts for keywords=%s limit=%s company_id=%s", keywords, limit, company_id)

        if company_id is None:
            rows = self.conn.execute(
                """
                SELECT extracted_facts.*, sources.url, sources.title, sources.publisher,
                       sources.source_type, sources.publication_date, sources.retrieved_at
                FROM extracted_facts
                JOIN sources ON sources.id = extracted_facts.source_id
                ORDER BY extracted_facts.id DESC
                LIMIT 200
                """
            ).fetchall()
        else:
            rows = self.conn.execute(
                """
                SELECT extracted_facts.*, sources.url, sources.title, sources.publisher,
                       sources.source_type, sources.publication_date, sources.retrieved_at
                FROM extracted_facts
                JOIN sources ON sources.id = extracted_facts.source_id
                WHERE sources.company_id = ?
                ORDER BY extracted_facts.id DESC
                LIMIT 200
                """,
                (company_id,),
            ).fetchall()

        scored: list[tuple[tuple[int, int, int], sqlite3.Row]] = []

        for row in rows:
            text = f"{row['metric']} {row['evidence']} {row['title'] or ''}".lower()
            score = sum(1 for keyword in keywords if keyword in text)

            if score:
                year_score = _fact_year_alignment_score(row, requested_years)
                scored.append(((year_score, score, int(row["id"])), row))

        scored.sort(key=lambda item: item[0], reverse=True)

        result = [row for _, row in scored[:limit]]
        logger.info("Fact lookup returned %s rows for question=%r company_id=%s", len(result), question, company_id)

        return result

    def store_answer(self, run_id: str, question: str, answer: Answer) -> None:
        """Persist a generated answer JSON for audit and later inspection."""
        self.conn.execute(
            "INSERT INTO answers(run_id, question, answer_json, created_at) VALUES (?, ?, ?, ?)",
            (run_id, question, json.dumps(answer.to_json_dict(), sort_keys=True), utc_now()),
        )
        self.conn.commit()

        logger.info("Stored answer for run_id=%s confidence=%s", run_id, answer.confidence)

    def dump_audit(self, output_path: str | Path) -> None:
        """Write all audit events to a JSON file."""
        rows = self.conn.execute("SELECT * FROM audit_events ORDER BY id").fetchall()
        payload = [dict(row) for row in rows]

        Path(output_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

        logger.info("Dumped %s audit events to %s", len(payload), output_path)
