from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app import database, services
from app.main import app
from app.models import TicketClassification


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Every test gets a fresh temporary DB, never your real tickets.db."""
    monkeypatch.setattr(database, "DATABASE_NAME", str(tmp_path / "test.db"))
    database.init_db()
    return TestClient(app)


@pytest.fixture
def fake_ai(monkeypatch):
    """Replace Groq and Chroma. Returns configure(...) and a call counter."""
    calls = {"generate": 0}

    def configure(category="Billing", priority="Medium",
                  confidence=0.95, coverage=1.0):
        monkeypatch.setattr(
            services, "classify_ticket",
            lambda subject, message: TicketClassification(
                category=category, priority=priority,
                sentiment="Neutral", confidence=confidence,
            ),
        )

        chunk = SimpleNamespace(
            document="billing.pdf", section="Refunds", page=1, text="Sample text"
        )
        monkeypatch.setattr(
            services.retriever, "search",
            lambda text, k=4: ([(chunk, 0.9)], coverage),
        )

        def fake_generate(**kwargs):
            calls["generate"] += 1
            return {"answerable": True, "answer": "Mock answer",
                    "evidence": "Sample text", "sources": [1]}


        monkeypatch.setattr(services, "generate_response", fake_generate)

    configure()
    configure.calls = calls
    return configure