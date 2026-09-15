from __future__ import annotations

import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parent


def styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "BenchmarkTitle",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=18,
            leading=22,
            spaceAfter=12,
        ),
        "h1": ParagraphStyle(
            "BenchmarkHeading",
            parent=base["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=13,
            leading=16,
            spaceBefore=10,
            spaceAfter=6,
        ),
        "body": ParagraphStyle(
            "BenchmarkBody",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=10,
            leading=14,
            spaceAfter=7,
        ),
        "small": ParagraphStyle(
            "BenchmarkSmall",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=8,
            leading=10,
        ),
    }


def paragraph(text: str, style):
    return Paragraph(text, style)


def table(data):
    t = Table(data, repeatRows=1, hAlign="LEFT")
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8EEF7")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#111827")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("LEADING", (0, 0), (-1, -1), 10),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#9CA3AF")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    return t


def build_pdf(filename: str, title: str, sections: list[tuple[str, list[object]]]):
    s = styles()
    story = [paragraph(title, s["title"])]
    story.append(
        paragraph(
            "Controlled Enterprise RAG Benchmark v1. Text-based single-column source document.",
            s["small"],
        )
    )
    story.append(Spacer(1, 0.1 * inch))

    for heading, items in sections:
        story.append(paragraph(heading, s["h1"]))
        for item in items:
            if isinstance(item, str):
                story.append(paragraph(item, s["body"]))
            else:
                story.append(table(item))
                story.append(Spacer(1, 0.08 * inch))

    doc = SimpleDocTemplate(
        str(ROOT / filename),
        pagesize=letter,
        rightMargin=0.7 * inch,
        leftMargin=0.7 * inch,
        topMargin=0.65 * inch,
        bottomMargin=0.65 * inch,
        title=title,
        author="Enterprise RAG Platform Benchmark",
    )
    doc.build(story)


def travel_sections():
    return [
        (
            "1. Purpose and Scope",
            [
                "This Employee Travel and Expense Policy defines how Meridian Retail Group employees request business travel, book transportation, select lodging, claim meals, and submit reimbursable expenses. The policy applies to all full-time employees, part-time employees, contractors with written travel authorization, and interns traveling on company business.",
                "The policy covers domestic travel inside India and international travel outside India. It does not cover relocation benefits, client-funded event budgets, or personal vacation expenses attached to a business trip.",
            ],
        ),
        (
            "2. Travel Approval Rules",
            [
                "All travel must be requested in the Meridian Travel Portal before booking. Domestic trips require manager approval when the estimated trip cost is INR 20,000 or below and director approval above INR 20,000. International trips always require director approval, and trips above INR 250,000 require an additional finance controller review.",
                "Requests must include business purpose, destination, expected travel dates, customer or project name, estimated airfare, hotel estimate, and expected daily meal allowance. Emergency travel may be approved after departure only when the traveler documents the operational incident or customer escalation.",
                [
                    ["Trip Type", "Cost Threshold", "Required Approval", "Advance Notice"],
                    ["Domestic", "Up to INR 20,000", "Manager", "3 business days"],
                    ["Domestic", "Above INR 20,000", "Director", "5 business days"],
                    ["International", "Up to INR 250,000", "Director", "10 business days"],
                    ["International", "Above INR 250,000", "Director + Finance Controller", "15 business days"],
                ],
            ],
        ),
        (
            "3. Domestic Travel Policy",
            [
                "Employees must select economy class for domestic flights under five hours. Premium economy is permitted for domestic routes of five hours or more when the fare difference is less than INR 4,000. Business class is limited to vice presidents and above, or to medical accommodation approved by Human Resources.",
                "Rail travel may be booked in AC Chair Car, AC 2-tier, or Executive Chair Car when it is more practical than air travel. Personal vehicle reimbursement is allowed only for trips below 300 kilometers and is paid at INR 14 per kilometer.",
            ],
        ),
        (
            "4. International Travel Policy",
            [
                "International flights under six hours must be booked in economy class. Business class may be booked for international flight segments of six hours or more by employees at grade M4 and above. Employees below grade M4 need vice president approval for business class.",
                "Business visas, mandatory vaccinations, and destination entry fees are reimbursable when supported by receipts. Passport renewal, personal travel insurance upgrades, and tourist visa extensions are not reimbursable.",
            ],
        ),
        (
            "5. Hotel Limits",
            [
                "Employees should book hotels through the Meridian Travel Portal unless a client-designated hotel is required. Hotel rates include room charge and taxes but exclude laundry, minibar, movies, and personal services.",
                [
                    ["Location Category", "Employee Grade", "Nightly Limit", "Currency"],
                    ["India Tier 1 cities", "E1-E5", "INR 8,500", "INR"],
                    ["India Tier 1 cities", "M1 and above", "INR 12,000", "INR"],
                    ["India Tier 2 or Tier 3 cities", "All grades", "INR 6,500", "INR"],
                    ["Standard international cities", "E1-E5", "USD 180", "USD"],
                    ["Standard international cities", "M1 and above", "USD 260", "USD"],
                    ["High-cost cities", "All grades", "Standard limit plus 35 percent", "USD"],
                ],
            ],
        ),
        (
            "6. Meal and Per-Diem Limits",
            [
                "Meal reimbursement is based on actual expenses up to the published cap. Alcohol is not reimbursable unless part of a client entertainment event approved by a director. When breakfast is included in the hotel rate, the breakfast allowance may not be claimed separately.",
                [
                    ["Meal Category", "Domestic Cap", "International Cap", "Receipt Required"],
                    ["Breakfast", "INR 350", "USD 12", "Above INR 250 or USD 10"],
                    ["Lunch", "INR 600", "USD 22", "Above INR 500 or USD 15"],
                    ["Dinner", "INR 900", "USD 35", "Above INR 500 or USD 15"],
                    ["Daily total", "INR 1,850", "USD 75", "Always for international"],
                    ["Client entertainment", "INR 7,500 per event", "USD 150 per attendee", "Always"],
                ],
            ],
        ),
        (
            "7. Transportation Rules",
            [
                "Airport transfers, taxis, rideshare, metro fares, and parking are reimbursable when directly connected to approved business travel. Travelers must use standard rideshare categories unless safety, accessibility, or client schedule requirements justify a higher category.",
                "Rental cars require manager approval before booking. Fuel and tolls for rental cars are reimbursable with receipts. Traffic fines, personal detours, and optional vehicle upgrades are not reimbursable.",
            ],
        ),
        (
            "8. Expense Submission and Reimbursement",
            [
                "Expense reports must be submitted in the Meridian Expense Portal within 30 calendar days after trip completion. Claims submitted between 31 and 60 days require manager approval and a late-submission reason. Claims submitted after 60 days are rejected unless a vice president approves an exception.",
                "Approved reimbursements are paid in the next payroll cycle. Finance normally completes review within 7 business days when receipts, approvals, and cost center details are complete.",
            ],
        ),
        (
            "9. Exceptions",
            [
                "Policy exceptions must be requested before booking whenever possible. The request must include the policy limit, actual expected cost, business reason, and approving leader. Finance may audit any exception after reimbursement.",
                "Repeated non-compliant booking outside the Travel Portal may result in loss of self-service travel privileges for 90 days.",
            ],
        ),
    ]


def security_sections():
    return [
        (
            "1. Purpose and Scope",
            [
                "This Information Security and Access Management Policy defines baseline controls for Meridian Retail Group systems, user accounts, privileged access, authentication, endpoint devices, incident reporting, and data classification. The policy applies to employees, contractors, service accounts, vendors, and administrators with access to company systems.",
                "The policy covers production systems, corporate applications, cloud consoles, source code repositories, analytics workspaces, customer support tools, and laptops used to access company data.",
            ],
        ),
        (
            "2. Password and MFA Requirements",
            [
                "All user accounts must use unique passwords that are not reused across personal or external services. Password managers approved by Corporate IT may be used to generate and store credentials. Passwords must never be sent through chat, email, ticket comments, or shared documents.",
                "Multi-factor authentication is required for email, VPN, source code systems, production dashboards, cloud administration consoles, and finance systems. Hardware security keys are required for privileged administrators and recommended for engineering leads.",
                [
                    ["Account Type", "Minimum Length", "Rotation", "MFA Requirement", "Lockout Rule"],
                    ["Standard user", "14 characters", "180 days", "Authenticator app or security key", "10 failed attempts"],
                    ["Privileged admin", "18 characters", "90 days", "Hardware security key", "5 failed attempts"],
                    ["Service account", "32 characters", "365 days", "Vault-managed secret", "Alert on first failed use"],
                    ["Break-glass account", "20 characters", "After every use", "Hardware key + CISO approval", "3 failed attempts"],
                ],
            ],
        ),
        (
            "3. Account Lockout and Recovery",
            [
                "Accounts that hit the lockout threshold remain locked for 30 minutes or until Corporate IT verifies the user's identity. Privileged administrator lockouts must be reviewed by the Security Operations Center before access is restored.",
                "Suspected credential compromise requires immediate password reset, active session revocation, token rotation, and manager notification.",
            ],
        ),
        (
            "4. Access Request and Approval Process",
            [
                "Access requests must be submitted through the Access Portal with business justification, system name, requested role, data classification, duration, and approving manager. Access to confidential or restricted systems requires system owner approval in addition to manager approval.",
                "Temporary access must include an expiration date. The maximum duration for temporary elevated access is 14 calendar days unless the Chief Information Security Officer approves an extension.",
                [
                    ["Access Category", "Approver", "Maximum Duration", "Review Evidence"],
                    ["Standard application access", "Manager", "Indefinite while role remains valid", "Access Portal request"],
                    ["Confidential data access", "Manager + System Owner", "180 days", "Business justification"],
                    ["Production support access", "Engineering Manager + System Owner", "30 days", "Incident or support ticket"],
                    ["Privileged administrator", "Director + Security", "90 days", "Admin access review record"],
                    ["Emergency break-glass", "CISO or delegate", "8 hours", "Incident ticket and post-use review"],
                ],
            ],
        ),
        (
            "5. Role-Based and Privileged Access",
            [
                "Role-based access control must grant the minimum permissions necessary for the employee's job function. Shared accounts are prohibited except for approved service accounts managed through the company vault.",
                "Privileged sessions must be logged. Administrator access to production databases requires a linked change ticket or incident ticket. Direct database changes outside approved maintenance windows require director approval.",
            ],
        ),
        (
            "6. Device and Endpoint Security Requirements",
            [
                "Devices used to access company systems must run supported operating systems, full-disk encryption, endpoint detection and response, automatic screen lock, and current security patches. Laptops missing critical patches for more than 7 days are blocked from VPN access.",
                "Personal devices may access email only through managed mobile application controls. Personal laptops may not access source code, production systems, restricted data, or administrative cloud consoles.",
                [
                    ["Control", "Required Setting", "Compliance Deadline", "Escalation"],
                    ["Full-disk encryption", "Enabled on all laptops", "Before device assignment", "IT manager"],
                    ["Critical patching", "Installed within 7 days", "7 days after release", "Security Operations"],
                    ["Screen lock", "10 minutes inactivity", "Immediate", "Corporate IT"],
                    ["Endpoint detection", "Agent healthy and reporting", "24 hours after enrollment", "Security Operations"],
                ],
            ],
        ),
        (
            "7. Incident Reporting",
            [
                "Employees must report suspected phishing, lost devices, unauthorized access, accidental data sharing, malware alerts, and suspicious authentication prompts to Security Operations within 1 hour of discovery.",
                "Incidents involving restricted data, production credentials, or cross-tenant exposure are treated as high severity until triage proves otherwise. Evidence such as audit logs, suspicious emails, screenshots, and ticket history must be preserved.",
            ],
        ),
        (
            "8. Access Reviews and Data Classification",
            [
                "Managers review standard application access quarterly. System owners review confidential data access every 90 days. Security reviews privileged administrator access monthly and break-glass account usage after every use.",
                "Company data is classified as Public, Internal, Confidential, or Restricted. Restricted data includes government identifiers, payment card data, authentication secrets, production credentials, and customer personal data requiring regulatory protection.",
            ],
        ),
        (
            "9. Exceptions",
            [
                "Security exceptions must document the control gap, affected system, compensating control, business reason, risk owner, and expiration date. Exceptions for restricted data systems may not exceed 30 days without CISO approval.",
                "Expired exceptions automatically return to non-compliant status and must be remediated or reapproved before access continues.",
            ],
        ),
    ]


def dataset():
    cases = [
        {
            "id": "controlled-v1-q01",
            "category": "exact_keyword",
            "question": "What is the advance notice required for a domestic trip above INR 20,000?",
            "expected_document": "employee_travel_expense_policy.pdf",
            "expected_document_id": None,
            "expected_chunk_ids": [],
            "expected_evidence": ["Domestic", "Above INR 20,000", "5 business days"],
            "expected_topic": "Travel approval rules table",
            "relevant_keywords": ["domestic", "above INR 20,000", "advance notice", "5 business days"],
            "answerable": True,
            "expected_answer": "A domestic trip above INR 20,000 requires 5 business days advance notice and director approval.",
        },
        {
            "id": "controlled-v1-q02",
            "category": "semantic_paraphrase",
            "question": "When can an employee get reimbursed for using their own car on a short business trip?",
            "expected_document": "employee_travel_expense_policy.pdf",
            "expected_document_id": None,
            "expected_chunk_ids": [],
            "expected_evidence": ["Personal vehicle reimbursement is allowed only for trips below 300 kilometers", "INR 14 per kilometer"],
            "expected_topic": "Domestic travel policy personal vehicle reimbursement",
            "relevant_keywords": ["personal vehicle", "300 kilometers", "INR 14 per kilometer"],
            "answerable": True,
            "expected_answer": "Personal vehicle reimbursement is allowed only for trips below 300 kilometers and is paid at INR 14 per kilometer.",
        },
        {
            "id": "controlled-v1-q03",
            "category": "numerical_table",
            "question": "What is the nightly hotel limit for E1-E5 employees in India Tier 1 cities?",
            "expected_document": "employee_travel_expense_policy.pdf",
            "expected_document_id": None,
            "expected_chunk_ids": [],
            "expected_evidence": ["India Tier 1 cities", "E1-E5", "INR 8,500"],
            "expected_topic": "Hotel limits table",
            "relevant_keywords": ["hotel", "E1-E5", "India Tier 1 cities", "INR 8,500"],
            "answerable": True,
            "expected_answer": "The nightly hotel limit is INR 8,500 for E1-E5 employees in India Tier 1 cities.",
        },
        {
            "id": "controlled-v1-q04",
            "category": "numerical_table",
            "question": "How long can emergency break-glass access last?",
            "expected_document": "information_security_access_management_policy.pdf",
            "expected_document_id": None,
            "expected_chunk_ids": [],
            "expected_evidence": ["Emergency break-glass", "8 hours", "Incident ticket and post-use review"],
            "expected_topic": "Access request and approval process table",
            "relevant_keywords": ["Emergency break-glass", "8 hours", "CISO", "post-use review"],
            "answerable": True,
            "expected_answer": "Emergency break-glass access can last up to 8 hours and requires an incident ticket and post-use review.",
        },
        {
            "id": "controlled-v1-q05",
            "category": "specific_section",
            "question": "According to the incident reporting section, how quickly must suspected phishing be reported?",
            "expected_document": "information_security_access_management_policy.pdf",
            "expected_document_id": None,
            "expected_chunk_ids": [],
            "expected_evidence": ["Employees must report suspected phishing", "within 1 hour of discovery"],
            "expected_topic": "Incident reporting",
            "relevant_keywords": ["incident reporting", "suspected phishing", "1 hour"],
            "answerable": True,
            "expected_answer": "Suspected phishing must be reported to Security Operations within 1 hour of discovery.",
        },
        {
            "id": "controlled-v1-q06",
            "category": "multi_chunk",
            "question": "What approvals and documentation are needed for a travel policy exception?",
            "expected_document": "employee_travel_expense_policy.pdf",
            "expected_document_id": None,
            "expected_chunk_ids": [],
            "expected_evidence": ["Policy exceptions must be requested before booking whenever possible", "policy limit, actual expected cost, business reason, and approving leader"],
            "expected_topic": "Travel exceptions",
            "relevant_keywords": ["policy exceptions", "before booking", "policy limit", "approving leader"],
            "answerable": True,
            "expected_answer": "A travel exception should be requested before booking and include the policy limit, actual expected cost, business reason, and approving leader.",
        },
        {
            "id": "controlled-v1-q07",
            "category": "cross_section",
            "question": "Compare the normal review schedule for confidential data access with privileged administrator access.",
            "expected_document": "information_security_access_management_policy.pdf",
            "expected_document_id": None,
            "expected_chunk_ids": [],
            "expected_evidence": ["System owners review confidential data access every 90 days", "Security reviews privileged administrator access monthly"],
            "expected_topic": "Access reviews and data classification",
            "relevant_keywords": ["confidential data access", "90 days", "privileged administrator", "monthly"],
            "answerable": True,
            "expected_answer": "Confidential data access is reviewed every 90 days by system owners; privileged administrator access is reviewed monthly by Security.",
        },
        {
            "id": "controlled-v1-q08",
            "category": "plausible_distractor",
            "question": "Is Redis cache availability relevant to employee travel reimbursement approvals?",
            "expected_document": "employee_travel_expense_policy.pdf",
            "expected_document_id": None,
            "expected_chunk_ids": [],
            "expected_evidence": ["All travel must be requested in the Meridian Travel Portal before booking", "Approved reimbursements are paid in the next payroll cycle"],
            "expected_topic": "Travel approvals and reimbursement; distractor term Redis is absent",
            "relevant_keywords": ["Travel Portal", "reimbursement", "approval", "Redis"],
            "answerable": True,
            "expected_answer": "No. Travel reimbursement approvals are governed by the Travel Portal and Expense Portal workflow; Redis cache availability is not part of this policy.",
        },
        {
            "id": "controlled-v1-q09",
            "category": "semantic_different_wording",
            "question": "What happens if a company laptop is missing urgent security updates for more than a week?",
            "expected_document": "information_security_access_management_policy.pdf",
            "expected_document_id": None,
            "expected_chunk_ids": [],
            "expected_evidence": ["Laptops missing critical patches for more than 7 days are blocked from VPN access"],
            "expected_topic": "Device and endpoint security requirements",
            "relevant_keywords": ["critical patches", "7 days", "blocked from VPN access"],
            "answerable": True,
            "expected_answer": "Laptops missing critical patches for more than 7 days are blocked from VPN access.",
        },
        {
            "id": "controlled-v1-q10",
            "category": "unanswerable",
            "question": "What is the company's approved reimbursement rate for pet boarding during international travel?",
            "expected_document": None,
            "expected_document_id": None,
            "expected_chunk_ids": [],
            "expected_evidence": [],
            "expected_topic": "Not present in the benchmark documents",
            "relevant_keywords": ["pet boarding", "international travel", "reimbursement rate"],
            "answerable": False,
            "expected_answer": "The information is not present in the benchmark documents.",
        },
    ]
    return {"benchmark_id": "controlled_enterprise_v1", "cases": cases}


def write_readme():
    text = """# Controlled Enterprise RAG Benchmark v1

This directory contains two clean, text-based enterprise PDF documents and a 10-question evaluation dataset for retrieval benchmarking.

## Documents

- `employee_travel_expense_policy.pdf`: travel approval rules, domestic and international travel, hotels, meals, transportation, reimbursement, and exceptions.
- `information_security_access_management_policy.pdf`: passwords, MFA, account lockout, access approvals, privileged access, endpoint security, incident reporting, access reviews, classification, and exceptions.

The PDFs are single-column, text-based documents with headings, paragraphs, and simple tables. They intentionally avoid scanned pages, images, complex layout, and multi-column formatting.

## Dataset

- `evaluation_dataset.json` contains exactly 10 cases.
- Each case includes a question, category, expected document filename, semantic evidence/topic, relevant keywords, answerability, and expected answer summary.
- `expected_chunk_ids` is intentionally empty because chunk IDs must be captured after the documents are uploaded and indexed.
- `expected_document_id` is intentionally `null` until the upload endpoint returns actual document IDs.

## Question Categories

1. Exact keyword question
2. Semantic/paraphrased question
3. Numerical/table question
4. Another numerical/table question
5. Specific-section evidence question
6. Multi-chunk question
7. Cross-section question
8. Plausible distractor question
9. Semantic question using different wording from the source
10. Unanswerable question

## Upload The PDFs

Start the backend:

```powershell
cd C:\\Users\\91738\\enterprise-rag-platform\\backend
venv\\Scripts\\uvicorn.exe app.main:app --reload
```

Upload each PDF through the existing API:

```powershell
curl.exe -X POST http://127.0.0.1:8000/upload -F \"file=@benchmarks/controlled_enterprise_v1/employee_travel_expense_policy.pdf\"
curl.exe -X POST http://127.0.0.1:8000/upload -F \"file=@benchmarks/controlled_enterprise_v1/information_security_access_management_policy.pdf\"
```

Record the returned `document_id` for each file.

## Run The Existing Evaluator

After upload, update a working copy of `evaluation_dataset.json` with the returned `expected_document_id` values. Then run retrieval-only evaluation:

```powershell
cd C:\\Users\\91738\\enterprise-rag-platform\\backend
venv\\Scripts\\python.exe -m app.evaluation.runner --dataset benchmarks/controlled_enterprise_v1/evaluation_dataset.json --skip-generation --top-k 10
```

Omit `--skip-generation` only when Gemini generation is configured and you want answer-generation latency included.

## Establish Final Chunk-Level Ground Truth

Do not invent chunk IDs. After indexing:

1. Query PostgreSQL for chunks grouped by the returned document IDs.
2. Inspect chunk content and metadata for each question's expected evidence.
3. Copy the matching `chunk_id` values into a separate working dataset or a committed v2 dataset.
4. Keep the semantic fields (`expected_evidence`, `expected_topic`, `relevant_keywords`) so the benchmark remains understandable even if chunking changes later.

Useful verification query:

```sql
select document_id, chunk_id, chunk_index, metadata->>'element_type' as element_type, left(content, 240) as preview
from chunks
where document_id in ('TRAVEL_DOCUMENT_ID', 'SECURITY_DOCUMENT_ID')
order by document_id, chunk_index;
```
"""
    (ROOT / "README.md").write_text(text, encoding="utf-8")


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    build_pdf(
        "employee_travel_expense_policy.pdf",
        "Employee Travel and Expense Policy",
        travel_sections(),
    )
    build_pdf(
        "information_security_access_management_policy.pdf",
        "Information Security and Access Management Policy",
        security_sections(),
    )
    (ROOT / "evaluation_dataset.json").write_text(
        json.dumps(dataset(), indent=2), encoding="utf-8"
    )
    write_readme()


if __name__ == "__main__":
    main()
