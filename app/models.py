# Pydantic will make sure subject and message are provided as strings.
from enum import Enum

from pydantic import BaseModel


class TicketCategory(str, Enum):
    BILLING = "Billing"
    TECHNICAL = "Technical"
    ACCOUNT = "Account"
    REFUND = "Refund"
    OTHER = "Other"


class TicketPriority(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CRITICAL = "Critical"


class TicketSentiment(str, Enum):
    POSITIVE = "Positive"
    NEUTRAL = "Neutral"
    NEGATIVE = "Negative"


class TicketCreate(BaseModel):
    subject: str
    message: str
    
# TicketClassification represents what our AI produces:category priority sentiment confidence
class TicketClassification(BaseModel):
    category: TicketCategory
    priority: TicketPriority
    sentiment: TicketSentiment
    confidence: float    