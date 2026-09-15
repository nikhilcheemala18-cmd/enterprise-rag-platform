# Controlled Enterprise RAG Benchmark v1

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
cd C:\Users\91738\enterprise-rag-platform\backend
venv\Scripts\uvicorn.exe app.main:app --reload
```

Upload each PDF through the existing API:

```powershell
curl.exe -X POST http://127.0.0.1:8000/upload -F "file=@benchmarks/controlled_enterprise_v1/employee_travel_expense_policy.pdf"
curl.exe -X POST http://127.0.0.1:8000/upload -F "file=@benchmarks/controlled_enterprise_v1/information_security_access_management_policy.pdf"
```

Record the returned `document_id` for each file.

## Run The Existing Evaluator

After upload, update a working copy of `evaluation_dataset.json` with the returned `expected_document_id` values. Then run retrieval-only evaluation:

```powershell
cd C:\Users\91738\enterprise-rag-platform\backend
venv\Scripts\python.exe -m app.evaluation.runner --dataset benchmarks/controlled_enterprise_v1/evaluation_dataset.json --skip-generation --top-k 10
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
