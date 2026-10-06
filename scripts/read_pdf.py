from pypdf import PdfReader

pdf_files = [
    "knowledge_base/account_policy.pdf",
    "knowledge_base/billing_policy.pdf",
    "knowledge_base/refund_policy.pdf",
    "knowledge_base/technical_support.pdf",
]

for pdf_file in pdf_files:
    print(f"\n===== {pdf_file} =====\n")

    reader = PdfReader(pdf_file)

    for page in reader.pages:
        text = page.extract_text() or ""
        print(text)