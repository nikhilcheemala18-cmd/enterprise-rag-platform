# Controlled Enterprise v1 Baseline: Blocked

Status: blocked

The two controlled benchmark PDFs were not found in the user-provided Supabase `chunks` table, so actual chunk-level ground truth could not be established and the baseline evaluation was not run.

## Database Verification

- Total chunks currently indexed: 407
- Database checked: user-provided Supabase PostgreSQL URL, password redacted
- Documents present:
  - `50dfa43e-356f-4807-8153-bd896fd2a9e5`: 161 chunks; detected as the older Distributed Systems Incident Response / Northstar PDF.
  - `6fbf747e-3d33-41af-ab68-30aa4115f26e`: 161 chunks; detected as another older Distributed Systems Incident Response / Northstar PDF.
  - `34000ebd-2089-4bce-a4aa-7150b7cf3cc4`: 85 chunks; detected as an older Meridian Systems Information Security Policy.

## Controlled v1 Signature Search

No chunks matched the controlled benchmark document signatures:

- `Employee Travel`: 0
- `Travel & Expense`: 0
- `Expense Policy`: 0
- `hotel limit`: 0
- `per-diem`: 0
- `Access Management Policy`: 0
- `Meridian Travel Portal`: 0
- `Expense Portal`: 0
- `INR 8,500`: 0
- `confidential data access`: 0
- `critical patches`: 0
- `pet boarding`: 0

The only broad security terms that matched belonged to the older security handbook, not the controlled v1 Information Security & Access Management Policy.

## Dataset And Evaluation

- `benchmarks/controlled_enterprise_v1/evaluation_dataset.json` was not updated.
- No `expected_document_id` or `expected_chunk_ids` were filled in.
- The evaluation pipeline was not run, because doing so would produce invalid baseline metrics against the wrong corpus.

## Required Next Action

Upload and index these two files into the user-provided Supabase environment:

- `backend/benchmarks/controlled_enterprise_v1/employee_travel_expense_policy.pdf`
- `backend/benchmarks/controlled_enterprise_v1/information_security_access_management_policy.pdf`

Use the application's normal upload endpoint/flow. The API route is:

```text
POST /upload
multipart/form-data field: file
```

After the upload returns document IDs for both PDFs, rerun the chunk ID mapping and baseline evaluation.
