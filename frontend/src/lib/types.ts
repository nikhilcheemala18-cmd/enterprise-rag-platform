export type UploadResponse = {
  status: string;
  document_id: string;
  filename: string;
  chunk_count: number;
  indexed_chunk_count: number;
};

export type QueryRequest = {
  query: string;
  top_k?: number;
  document_id?: string;
};

export type RetrievalSource = {
  chunk_id: string;
  document_id: string;
  content: string;
  score: number;
  metadata: Record<string, unknown>;
};

export type QueryResponse = {
  query: string;
  result_count: number;
  results: RetrievalSource[];
};

export type AskRequest = {
  question: string;
  top_k?: number;
  document_id?: string;
};

export type AskResponse = {
  question: string;
  answer: string;
  retrieved_count: number;
  sources: RetrievalSource[];
};

export type HealthResponse = {
  status: "healthy" | string;
};
