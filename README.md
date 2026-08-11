# Enterprise RAG Platform

A production-oriented, organization-level document intelligence platform built using Retrieval-Augmented Generation (RAG).

## Objective

Build a secure, scalable RAG platform capable of ingesting heterogeneous enterprise documents and providing grounded answers with accurate source citations.

Unlike a basic PDF chatbot, this system is designed around:

- Multi-format document ingestion
- Unified document processing
- Structure-aware chunking
- Dense vector retrieval
- BM25+ lexical retrieval
- Hybrid search
- Reciprocal Rank Fusion (RRF)
- Neural reranking
- Metadata-based filtering and access control
- RAG security guardrails
- Citation-aware generation
- Retrieval and generation evaluation
- Production-oriented architecture

## Planned Document Sources

- PDF
- DOCX
- XLSX
- CSV
- TXT
- PPTX

## High-Level Architecture

```text
                    User
                     |
                     v
              Authentication
                     |
                     v
               User Query
                     |
                     v
             Query Processing
                     |
            +--------+--------+
            |                 |
            v                 v
      Dense Retrieval      BM25+
            |                 |
            +--------+--------+
                     |
                     v
              Hybrid Fusion
                     |
                     v
                 Reranker
                     |
                     v
              Relevant Context
                     |
                     v
              Security Layer
                     |
                     v
                    LLM
                     |
                     v
          Grounded Answer + Citations