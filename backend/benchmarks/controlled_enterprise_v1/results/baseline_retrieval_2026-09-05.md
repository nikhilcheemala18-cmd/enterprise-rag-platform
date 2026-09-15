# Controlled Enterprise v1 Baseline Retrieval Report

Run type: baseline retrieval only; generation was not evaluated.

## Uploaded Documents
- `employee_travel_expense_policy.pdf`: document_id `2aa42cf5-4b17-4205-8f26-9825eacf97b3`, upload chunks 23, indexed chunks 23, verified DB chunks 23
- `information_security_access_management_policy.pdf`: document_id `b48e36fb-da1c-4542-b080-d86d11ee5f0a`, upload chunks 25, indexed chunks 25, verified DB chunks 25

## Aggregate Metrics
Questions evaluated: 10
Recall@5: 89%
Precision@5: 24%
Recall@10: 89%
Precision@10: 14%
MRR: 0.83
Source grounding: 89%
Answer accuracy: not automatically evaluated
Average query embedding latency: 84.2 ms
Average lexical retrieval latency: 520.7 ms
Average vector retrieval latency: 409.2 ms
Average RRF latency: 0.2 ms
Average hybrid/RRF retrieval latency: 930.1 ms
Average total /ask latency: 1.01 s

## Per-Question Results
- `controlled-v1-q01`: first relevant rank 1; expected chunks: `3a3123d9-cd42-4b2d-bc4c-2278821f668d` rank 1; evidence fully retrieved: True; grounded: True
- `controlled-v1-q02`: first relevant rank 2; expected chunks: `f4f7f509-f1ce-462f-96ee-4638b74c50c8` rank 2; evidence fully retrieved: True; grounded: True
- `controlled-v1-q03`: first relevant rank 1; expected chunks: `067eda37-28bc-4a16-912b-b03fd00efdc3` rank 1; evidence fully retrieved: True; grounded: True
- `controlled-v1-q04`: first relevant rank 1; expected chunks: `eeab82fe-6516-4ef9-a4ef-b05dd22e0923` rank 1; evidence fully retrieved: True; grounded: True
- `controlled-v1-q05`: first relevant rank 1; expected chunks: `a15f250b-2f21-4f23-a746-a81a1eaa6bdb` rank 1; evidence fully retrieved: True; grounded: True
- `controlled-v1-q06`: first relevant rank 1; expected chunks: `5e0fe999-c5f2-40e4-a721-30ea7f97b338` rank 1; evidence fully retrieved: True; grounded: True
- `controlled-v1-q07`: first relevant rank 1; expected chunks: `9a989288-72b2-4709-9767-25bf90dd7f8d` rank 1; evidence fully retrieved: True; grounded: True
- `controlled-v1-q08`: first relevant rank not retrieved; expected chunks: `535ea9ce-0d13-45a7-a971-567a0de6face` rank not retrieved, `a4577cc7-6034-4785-bb2c-ada210359101` rank not retrieved; evidence fully retrieved: False; grounded: False
- `controlled-v1-q09`: first relevant rank 1; expected chunks: `70353e8c-5aac-4254-b6d6-1e9bcd63f087` rank 1; evidence fully retrieved: True; grounded: True
- `controlled-v1-q10`: first relevant rank not retrieved; expected chunks: no expected chunk IDs; unanswerable/control question; evidence fully retrieved: None; grounded: None

## Notes
- Metrics are produced by the existing evaluator without changing ingestion, chunking, embeddings, retrieval, RRF, or evaluator implementation.
- Average total /ask latency is retrieval-only for this run because generation was skipped.
