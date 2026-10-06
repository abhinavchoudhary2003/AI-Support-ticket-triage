from fastapi import FastAPI, HTTPException

from app.database import get_all_tickets, get_ticket, init_db, save_ticket
from app.models import TicketCreate
from app.services import process_ticket

app = FastAPI(
    title="AI Support Ticket Triage & Resolution Assistant",
    version="1.0.0"
)
init_db()

@app.get("/")
def root():
    return {
        "message": "AI Support Ticket Triage & Resolution Assistant API",
        "docs": "/docs",
        "health": "/health"
    }
@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/tickets")
def create_ticket(ticket: TicketCreate):
    result = process_ticket(
        subject=ticket.subject,
        message=ticket.message
    )

    ticket_id = save_ticket(
        subject=ticket.subject,
        message=ticket.message,
        result=result,
    )

    return {
        "id": ticket_id,
        "subject": ticket.subject,
        "message": ticket.message,
        **result
    }
@app.get("/tickets/{ticket_id}")
def read_ticket(ticket_id: int):
    ticket = get_ticket(ticket_id)

    if ticket is None:
        raise HTTPException(
            status_code=404,
            detail="Ticket not found"
        )

    return ticket    

@app.get("/tickets")
def read_tickets():
    return get_all_tickets()
    
    
    
  
  
  
    
    
# main.py
#    ↓
# API / receives requests

# models.py
#    ↓
# Defines the shape of our data

# services.py
#    ↓
# Does the actual work
    
    
    
    
# @app.post("/tickets")

# This creates our first real API endpoint.

# When a customer sends:

# {
#   "subject": "I was charged twice",
#   "message": "Please help me with this duplicate charge."
# }

# FastAPI:

# Receives the request.
# Uses TicketCreate to validate it.
# Puts the data into the ticket variable.
# Returns a JSON response.