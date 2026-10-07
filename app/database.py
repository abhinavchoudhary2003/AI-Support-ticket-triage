

import json
import sqlite3


DATABASE_NAME = "tickets.db"


def get_connection():
    connection = sqlite3.connect(DATABASE_NAME)
    connection.row_factory = sqlite3.Row
    return connection


def init_db():
    connection = get_connection()

    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS tickets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            subject TEXT NOT NULL,
            message TEXT NOT NULL,
            category TEXT NOT NULL,
            priority TEXT NOT NULL,
            sentiment TEXT NOT NULL,
            confidence REAL NOT NULL,
            prompt_injection_detected INTEGER NOT NULL,
            coverage REAL NOT NULL,
            response TEXT NOT NULL,
            resolution TEXT NOT NULL,
            escalation_reason TEXT NOT NULL,
            retrieved_context TEXT NOT NULL DEFAULT '[]'
        )
        """
    )

    # upgrade an old tickets.db that lacks the new column
    columns = [row["name"] for row in connection.execute("PRAGMA table_info(tickets)")]
    if "retrieved_context" not in columns:
        connection.execute(
            "ALTER TABLE tickets ADD COLUMN retrieved_context TEXT NOT NULL DEFAULT '[]'"
        )

    connection.commit()
    connection.close()


# Takes the processed ticket, inserts it into SQLite, and returns the new ticket ID.
def save_ticket(
    subject,
    message,
    result,
):
    connection = get_connection()

    cursor = connection.execute(
        """
        INSERT INTO tickets (
            subject,
            message,
            category,
            priority,
            sentiment,
            confidence,
            prompt_injection_detected,
            coverage,
            response,
            resolution,
            escalation_reason,
            retrieved_context
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            subject,
            message,
            result["category"],
            result["priority"],
            result["sentiment"],
            result["confidence"],
            int(result["prompt_injection_detected"]),
            result["coverage"],
            result["response"],
            result["resolution"],
            result["escalation_reason"],
            json.dumps(result.get("retrieved_context", [])),
        ),
    )

    connection.commit()

    ticket_id = cursor.lastrowid

    connection.close()

    return ticket_id


def _row_to_ticket(row, include_context=True):
    ticket = dict(row)
    ticket["prompt_injection_detected"] = bool(ticket["prompt_injection_detected"])

    context = json.loads(ticket.get("retrieved_context") or "[]")
    if include_context:
        ticket["retrieved_context"] = context
    else:
        ticket.pop("retrieved_context", None)

    return ticket


def get_ticket(ticket_id):
    connection = get_connection()

    row = connection.execute(
        """
        SELECT *
        FROM tickets
        WHERE id = ?
        """,
        (ticket_id,),
    ).fetchone()

    connection.close()

    if row is None:
        return None

    return _row_to_ticket(row)


# Retrieves all tickets (without the bulky retrieved_context).
def get_all_tickets():
    connection = get_connection()

    rows = connection.execute(
        """
        SELECT *
        FROM tickets
        ORDER BY id DESC
        """
    ).fetchall()

    connection.close()

    return [_row_to_ticket(row, include_context=False) for row in rows]
