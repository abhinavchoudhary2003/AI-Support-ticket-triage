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
            escalation_reason TEXT NOT NULL
        )
        """
    )
# This function will: Take the processed ticket.Insert it into SQLite. Get the newly created ticket ID. Return that ID.    
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
            escalation_reason
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
        ),
    )

    connection.commit()

    ticket_id = cursor.lastrowid

    connection.close()

    return ticket_id    

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

    ticket = dict(row)

    ticket["prompt_injection_detected"] = bool(
        ticket["prompt_injection_detected"]
    )

    return ticket
#Add a function to retrieve all tickets
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
    tickets = []

    for row in rows:
        ticket = dict(row)

        ticket["prompt_injection_detected"] = bool(
            ticket["prompt_injection_detected"]
        )

        tickets.append(ticket)

    return tickets


