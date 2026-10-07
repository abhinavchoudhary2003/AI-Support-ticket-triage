<div align="center">

# 🎫 AI Support Ticket Triage & Resolution Assistant

**Reads a support ticket, classifies it, checks the policy knowledge base, then replies automatically or routes it to a human.**

![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-003B57?style=for-the-badge&logo=sqlite&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-FF6446?style=for-the-badge)
![Groq](https://img.shields.io/badge/Groq-F55036?style=for-the-badge)
![pytest](https://img.shields.io/badge/pytest-0A9EDC?style=for-the-badge&logo=pytest&logoColor=white)
![Tests](https://img.shields.io/badge/tests-35%20passing-brightgreen?style=for-the-badge)

[Architecture](#-architecture-and-how-a-ticket-is-processed) •
[Escalation rules](#-escalation-rules) •
[Safety](#-safety-measures) •
[Setup](#-setup) •
[API](#-api) •
[Tests](#-tests)

</div>

---

A **FastAPI** service that reads a customer support ticket, classifies it, looks up the answer in a policy knowledge base, and then either replies automatically or routes the ticket to a human. A **Streamlit** console is included for demos.

> [!IMPORTANT]
> **Design principle:** when anything is uncertain, risky or broken, the ticket goes to a human. The AI only answers when the rules allow it and the answer can be checked against the knowledge base.

## 📑 Table of contents

- [Architecture and how a ticket is processed](#-architecture-and-how-a-ticket-is-processed)
- [Escalation rules](#-escalation-rules)
- [Safety measures](#-safety-measures)
- [Tech stack](#-tech-stack)
- [Project structure](#-project-structure)
- [Setup](#-setup)
- [Run](#-run)
- [API](#-api)
- [Adding documents](#-adding-documents)
- [Tests](#-tests)
- [Known limitations](#-known-limitations)

---

## 🔄 Architecture and how a ticket is processed

```mermaid
flowchart TD
    A(["Customer ticket"]) --> B("Streamlit UI")
    B --> C("FastAPI<br/>POST /tickets")
    C --> D{"Prompt injection<br/>check"}
    D -- safe --> E("Classification, priority, sentiment<br/>(Groq gpt-oss-20b)")
    D -- detected --> J("Escalated to human")
    E --> F("Semantic retrieval<br/>top-k, cosine similarity")
    F --> G("Grounded generation<br/>answer from retrieved context only")
    G --> H{"Escalation<br/>rules"}
    H -- no rule fires --> I("Resolved by AI")
    H -- any rule fires --> J
    I --> K[("SQLite")]
    J --> K
    K --> L("GET /tickets<br/>GET /tickets/&#123;id&#125;")

    subgraph KB ["Knowledge base ingestion (at startup)"]
        direction TB
        P("KB PDFs<br/>account, billing, refund, technical") --> Q("Chunk + embed<br/>(all-MiniLM-L6-v2)")
        Q --> R[("ChromaDB")]
    end

    R -.-> F
    E -. "priority, legal threat,<br/>refund amount, confidence" .-> H
    F -. "similarity /<br/>missing KB info" .-> H

    classDef step fill:#eef1f8,stroke:#5b677d,color:#1f2937;
    classDef decision fill:#fdf2dc,stroke:#5b677d,color:#1f2937;
    classDef store fill:#e6eaff,stroke:#5b677d,color:#1f2937;
    classDef ok fill:#e3f4ef,stroke:#0f8a74,color:#0b4f43;
    classDef bad fill:#fbe7e3,stroke:#c0392b,color:#7a2118;
    class B,C,E,F,G,L,P,Q,A step;
    class D,H decision;
    class R,K store;
    class I ok;
    class J bad;
    linkStyle 4 stroke:#c0392b,color:#c0392b;
    linkStyle 9 stroke:#c0392b,color:#c0392b;
    linkStyle 8 stroke:#0f8a74,color:#0f8a74;
```

<details>
<summary>📝 Text version of the flow</summary>

```text
Ticket
  |
  v
1. Prompt-injection scan (rules, no LLM)  --flagged--> Human
  |
  v
2. Classify: category, priority, sentiment, confidence (Groq LLM) --error--> Human
  |
  v
3. Retrieve top 4 policy chunks (MiniLM embeddings + Chroma, cosine)  --error--> Human
  |
  v
4. Escalation rules (deterministic code)  --any rule hit--> Human (holding message)
  |
  v
5. Generate answer as JSON: answerable, answer, evidence, sources (Groq LLM) --error--> Human
  |
  v
6. Grounding check in code  --not verified--> Human
  |
  v
AI reply, saved with the sources that were retrieved
```

</details>

> [!NOTE]
> Escalated tickets always get a **fixed holding message**, never an AI-written reply.

---

## 🚦 Escalation rules

Checked in this order, **first match wins** (`app/rules.py`):

| # | Rule | Result |
|:-:|------|:------:|
| 1 | Possible prompt injection | 👤 Human |
| 2 | Priority is Critical | 👤 Human |
| 3 | Classification confidence below 0.70 | 👤 Human |
| 4 | Legal threat (lawyer, sue, court, legal action, ...) | 👤 Human |
| 5 | Knowledge-base coverage below 0.50 | 👤 Human |
| 6 | Money request above $200, or above 100 in another or unstated currency | 👤 Human |
| 7 | None of the above | 🤖 AI |

**Notes:**

- Legal terms are matched as **whole words**, so "issue" and "courtesy" do not trigger the rule.
- The money rule runs on the **ticket text**, not only on the LLM's category. A $1,200 "please reverse this charge" ticket labelled Billing is still escalated.

---

## 🛡️ Safety measures

| | Measure | Details |
|:-:|---------|---------|
| 💉 | **Prompt injection** | The ticket text is normalised (case, unicode, zero-width characters, spacing) and matched against injection patterns. Flagged tickets never reach the LLM. In both prompts the ticket is wrapped in `<ticket>` tags and treated as untrusted data. |
| 📚 | **Grounded answers** | The model must return `answerable`, the answer, a word-for-word `evidence` quote, and the numbers of the excerpts it used. The code escalates unless `answerable` is true, every cited excerpt exists, and the quote really appears in a cited excerpt. |
| 🧯 | **Failures fail safe** | If the LLM or the retrieval step raises an error, the ticket is saved as Human with a reason ("AI service unavailable" or "Knowledge base unavailable") instead of returning a 500. |
| 📏 | **Input limits** | Subject 3 to 200 characters, message 5 to 5,000 characters, whitespace trimmed. Invalid input returns `422`. |

---

## 🧰 Tech stack

| Layer | Technology |
|-------|------------|
| API | FastAPI |
| Storage | SQLite |
| Vector search | ChromaDB with sentence-transformers MiniLM embeddings (cosine similarity) |
| LLM | Groq via the OpenAI-compatible client (`openai/gpt-oss-20b`) |
| Demo UI | Streamlit |
| Testing | pytest |

---

## 🗂️ Project structure

```text
app/
  main.py        API endpoints
  models.py      Pydantic models and enums
  database.py    SQLite storage
  rag.py         Knowledge-base loading, embedding and retrieval
  rules.py       Escalation rules and prompt-injection detection
  services.py    Classification, answer generation, grounding check, process_ticket
knowledge_base/  Policy PDFs (account_policy, billing_policy, refund_policy, technical_support)
scripts/         read_pdf.py prints the text of the knowledge-base PDFs, for inspection
tests/           pytest suite (temporary DB, mocked LLM and retrieval)
streamlit_app.py Demo console
requirements.txt
.env.example
```

---

## ⚙️ Setup

```bash
python -m venv myenv
myenv\Scripts\activate          # Windows
# source myenv/bin/activate     # macOS / Linux
pip install -r requirements.txt
```

> [!NOTE]
> There is no separate indexing step. The vector index is rebuilt automatically from `knowledge_base/` every time the app starts. The first run downloads the `all-MiniLM-L6-v2` embedding model, so an internet connection is needed once.

Create a `.env` file (see `.env.example`):

```env
GROQ_API_KEY=your_key_here
```

> [!WARNING]
> Never commit `.env`.

---

## ▶️ Run

**API:**

```bash
uvicorn app.main:app --reload
```

Interactive docs are at <http://127.0.0.1:8000/docs>.

**Demo console** (starts the API automatically if it is not running):

```bash
streamlit run streamlit_app.py
```

---

## 🔌 API

| Method | Path | Description |
|:------:|------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/tickets` | Process and save a ticket |
| `GET` | `/tickets` | List tickets, newest first (without sources) |
| `GET` | `/tickets/{id}` | One ticket, including the sources that were retrieved |

**Example request:**

```json
{"subject": "Password reset", "message": "How do I reset my password?"}
```

**Example response** (retrieved text shortened):

```json
{
  "id": 140,
  "subject": "Password reset",
  "message": "How do I reset my password?",
  "category": "Account",
  "priority": "Low",
  "sentiment": "Neutral",
  "confidence": 0.95,
  "prompt_injection_detected": false,
  "retrieved_context": [
    {
      "source": "account_policy.pdf",
      "section": "1.1 Password reset",
      "page": 1,
      "score": 0.553,
      "text": "Choose Forgot password on the login page ..."
    }
  ],
  "coverage": 1,
  "response": "To reset your password, click \"Forgot password\" on the login page ...",
  "resolution": "AI",
  "escalation_reason": "Ticket can be resolved using the available knowledge base."
}
```

`resolution` is `AI` or `Human`. `escalation_reason` explains the decision either way.

---

## 📥 Adding documents

Put PDFs in `knowledge_base/`. Each document is split into one chunk per numbered section heading (for example `1.1 Password reset`), so documents need headings in that format. Restart the app to re-index.

---

## 🧪 Tests

```bash
python -m pytest tests -v
```

The tests use a temporary database and mock the LLM and retrieval, so they make no Groq calls and never touch `tickets.db`.

> [!TIP]
> Keep a `.env` file with any placeholder `GROQ_API_KEY` value in place, because the app creates its client when it is imported.

They cover the escalation rules, prompt-injection detection, failure fallbacks, the grounding check and input validation.

---

## ⚠️ Known limitations

- Injection detection is pattern-based, so a new phrasing can slip past the first layer. The prompts also treat ticket text as data, and escalation never depends on what the model says about itself.
- The grounding check verifies that the cited quote exists in a cited excerpt. It cannot prove the excerpt fully supports every sentence of the answer.
- Amounts in non-USD currencies are not converted. Above the review threshold they go to a human.
- No authentication or rate limiting; this is a demo service.
