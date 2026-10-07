from app.models import TicketCategory, TicketPriority
from app.rules import decide_resolution


def test_low_confidence_escalation():
    resolution, reason = decide_resolution(
        category=TicketCategory.BILLING,
        priority=TicketPriority.MEDIUM,
        confidence=0.50,
        message="I have a billing question.",
        coverage=1.0,
    )

    assert resolution == "Human"
    assert reason == "Classification confidence is too low."    
    
def test_critical_ticket_escalation():
    resolution, reason = decide_resolution(
        category=TicketCategory.TECHNICAL,
        priority=TicketPriority.CRITICAL,
        confidence=0.95,
        message="Our entire service is down.",
        coverage=1.0,
    )

    assert resolution == "Human"
    assert reason == "Critical issue requires human review."    
    
def test_legal_threat_escalation():
    resolution, reason = decide_resolution(
        category=TicketCategory.BILLING,
        priority=TicketPriority.HIGH,
        confidence=0.95,
        message="I will contact my lawyer and take legal action.",
        coverage=1.0,
    )

    assert resolution == "Human"
    assert reason == "Legal threat requires human review."    
    
    
def test_high_value_refund_escalation():
    resolution, reason = decide_resolution(
        category=TicketCategory.REFUND,
        priority=TicketPriority.HIGH,
        confidence=0.95,
        message="I want a refund of $500.",
        coverage=1.0,
    )

    assert resolution == "Human"
    assert reason == "Refund above $200 requires Finance approval."    
    
def test_missing_knowledge_base_escalation():
    resolution, reason = decide_resolution(
        category=TicketCategory.OTHER,
        priority=TicketPriority.MEDIUM,
        confidence=0.95,
        message="I have a question that is not covered.",
        coverage=0.20,
    )

    assert resolution == "Human"
    assert reason == "Knowledge base does not contain enough information."    
    
def test_normal_ticket_ai_resolution():
    resolution, reason = decide_resolution(
        category=TicketCategory.BILLING,
        priority=TicketPriority.MEDIUM,
        confidence=0.95,
        message="I have a question about my invoice.",
        coverage=1.0,
    )

    assert resolution == "AI"
    assert reason == "Ticket can be resolved using the available knowledge base."    
    
def test_high_value_refund_750():
    resolution, reason = decide_resolution(
        category=TicketCategory.REFUND,
        priority=TicketPriority.HIGH,
        confidence=0.95,
        message="I want a refund of $750.",
        coverage=1.0,
    )

    assert resolution == "Human"
    assert reason == "Refund above $200 requires Finance approval."    