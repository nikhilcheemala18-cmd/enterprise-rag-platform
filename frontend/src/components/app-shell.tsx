import {
  Activity,
  AlertCircle,
  Bot,
  CheckCircle2,
  Database,
  FileText,
  Loader2,
  type LucideIcon,
  MessageSquareText,
  PanelLeft,
  Search,
  Send,
  ShieldCheck,
  Upload,
} from "lucide-react";
import { useEffect, useRef, useState, type FormEvent, type ReactNode } from "react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { askQuestion, getHealth, uploadDocument } from "@/lib/api";
import { getErrorMessage } from "@/lib/errors";
import type { AskResponse, RetrievalSource, UploadResponse } from "@/lib/types";

const supportedExtensions = [".pdf", ".docx", ".csv", ".xlsx", ".xls"];

type HealthState = "checking" | "online" | "offline";
type UploadState = "idle" | "uploading" | "success" | "error";
type AskState = "idle" | "asking" | "success" | "error";

export function AppShell() {
  const [health, setHealth] = useState<HealthState>("checking");
  const [document, setDocument] = useState<UploadResponse | null>(null);
  const [uploadState, setUploadState] = useState<UploadState>("idle");
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [askState, setAskState] = useState<AskState>("idle");
  const [askError, setAskError] = useState<string | null>(null);
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState<AskResponse | null>(null);

  useEffect(() => {
    let ignore = false;

    async function checkHealth() {
      setHealth("checking");
      try {
        const response = await getHealth();
        if (!ignore) {
          setHealth(response.status === "healthy" ? "online" : "offline");
        }
      } catch {
        if (!ignore) {
          setHealth("offline");
        }
      }
    }

    void checkHealth();
    const interval = window.setInterval(checkHealth, 30000);

    return () => {
      ignore = true;
      window.clearInterval(interval);
    };
  }, []);

  async function handleUpload(file: File) {
    setUploadError(null);
    setAnswer(null);
    setAskError(null);

    if (!isSupportedFile(file)) {
      setUploadState("error");
      setUploadError("Unsupported file type. Choose a PDF, DOCX, CSV, XLSX, or XLS file.");
      return;
    }

    setUploadState("uploading");
    try {
      const uploadedDocument = await uploadDocument(file);
      setDocument(uploadedDocument);
      setUploadState("success");
    } catch (error) {
      setUploadState("error");
      setUploadError(getErrorMessage(error, "Upload failed. Check the file and try again."));
    }
  }

  async function handleAsk(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setAskError(null);

    const trimmedQuestion = question.trim();
    if (!document) {
      setAskState("error");
      setAskError("Upload a document before asking a question.");
      return;
    }

    if (!trimmedQuestion) {
      setAskState("error");
      setAskError("Enter a question about the uploaded document.");
      return;
    }

    setAskState("asking");
    try {
      const response = await askQuestion({
        question: trimmedQuestion,
        top_k: 10,
        document_id: document.document_id,
      });
      setAnswer(response);
      setAskState("success");
    } catch (error) {
      setAskState("error");
      setAskError(
        getErrorMessage(
          error,
          "Answer generation failed. Try a narrower question or confirm Gemini is configured.",
        ),
      );
    }
  }

  return (
    <div className="min-h-screen text-foreground">
      <div className="mx-auto flex min-h-screen w-full max-w-[1440px] flex-col border-border/70 bg-background/65 lg:flex-row lg:border-x">
        <Sidebar document={document} health={health} />
        <main className="flex min-w-0 flex-1 flex-col">
          <WorkspaceHeader document={document} health={health} />
          <DocumentWorkspace
            answer={answer}
            askError={askError}
            askState={askState}
            document={document}
            onAsk={handleAsk}
            onQuestionChange={setQuestion}
            onUpload={handleUpload}
            question={question}
            uploadError={uploadError}
            uploadState={uploadState}
          />
        </main>
      </div>
    </div>
  );
}

function Sidebar({ health, document }: { health: HealthState; document: UploadResponse | null }) {
  return (
    <aside className="flex w-full shrink-0 flex-col border-b border-border bg-card/70 px-4 py-4 lg:min-h-screen lg:w-72 lg:border-b-0 lg:border-r">
      <div className="flex items-center justify-between gap-3">
        <div className="flex min-w-0 items-center gap-3">
          <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg border border-primary/30 bg-primary/10 text-primary">
            <Database className="h-5 w-5" aria-hidden="true" />
          </div>
          <div className="min-w-0">
            <p className="truncate text-sm font-semibold">Enterprise RAG</p>
            <p className="truncate text-xs text-muted-foreground">Document intelligence</p>
          </div>
        </div>
        <Button variant="ghost" size="icon" aria-label="Toggle sidebar" className="lg:hidden">
          <PanelLeft className="h-4 w-4" aria-hidden="true" />
        </Button>
      </div>

      <Separator className="my-4" />

      <nav className="grid gap-1" aria-label="Workspace navigation">
        <SidebarItem icon={FileText} label="Documents" active />
        <SidebarItem icon={MessageSquareText} label="Ask AI" />
        <SidebarItem icon={Search} label="Sources" />
      </nav>

      <div className="mt-5 hidden flex-1 flex-col gap-3 lg:flex">
        <SectionLabel>Documents</SectionLabel>
        {document ? (
          <button className="group min-w-0 rounded-lg border border-primary/30 bg-primary/10 p-3 text-left transition-colors hover:border-primary/50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring">
            <div className="flex items-center justify-between gap-2">
              <span className="min-w-0 truncate text-sm font-medium">{document.filename}</span>
              <Badge variant="outline">{fileExtension(document.filename).toUpperCase()}</Badge>
            </div>
            <div className="mt-2 flex items-center justify-between gap-2 text-xs text-muted-foreground">
              <span>Indexed</span>
              <span className="tabular-nums">{document.indexed_chunk_count} chunks</span>
            </div>
          </button>
        ) : (
          <div className="rounded-lg border border-border bg-muted/25 p-3 text-sm text-muted-foreground">
            No document uploaded yet.
          </div>
        )}
      </div>

      <HealthIndicator health={health} />
    </aside>
  );
}

function SidebarItem({
  icon: Icon,
  label,
  active = false,
}: {
  icon: LucideIcon;
  label: string;
  active?: boolean;
}) {
  return (
    <button
      className={[
        "flex min-h-10 items-center gap-3 rounded-md px-3 text-sm transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
        active
          ? "bg-primary/10 text-foreground"
          : "text-muted-foreground hover:bg-muted/60 hover:text-foreground",
      ].join(" ")}
      aria-current={active ? "page" : undefined}
    >
      <Icon className="h-4 w-4" aria-hidden="true" />
      <span>{label}</span>
    </button>
  );
}

function SectionLabel({ children }: { children: ReactNode }) {
  return (
    <p className="px-1 text-xs font-medium uppercase tracking-[0.16em] text-muted-foreground">
      {children}
    </p>
  );
}

function HealthIndicator({ health }: { health: HealthState }) {
  const content = {
    checking: {
      label: "Checking backend",
      description: "Health check in progress.",
      dot: "bg-amber-300",
      icon: <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" aria-hidden="true" />,
    },
    online: {
      label: "Backend online",
      description: "FastAPI is responding.",
      dot: "bg-emerald-400",
      icon: <Activity className="h-4 w-4 text-muted-foreground" aria-hidden="true" />,
    },
    offline: {
      label: "Backend unavailable",
      description: "Start FastAPI on port 8000.",
      dot: "bg-red-400",
      icon: <AlertCircle className="h-4 w-4 text-muted-foreground" aria-hidden="true" />,
    },
  }[health];

  return (
    <div className="mt-4 rounded-lg border border-border bg-muted/25 p-3">
      <div className="flex items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <span className={`h-2.5 w-2.5 rounded-full ${content.dot}`} aria-hidden="true" />
          <span className="text-sm font-medium">{content.label}</span>
        </div>
        {content.icon}
      </div>
      <p className="mt-1 text-xs text-muted-foreground">{content.description}</p>
    </div>
  );
}

function WorkspaceHeader({
  document,
  health,
}: {
  document: UploadResponse | null;
  health: HealthState;
}) {
  return (
    <header className="border-b border-border bg-background/80 px-4 py-4 backdrop-blur sm:px-6">
      <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
        <div className="min-w-0">
          <div className="flex flex-wrap items-center gap-2">
            <h1 className="text-xl font-semibold tracking-tight sm:text-2xl">
              Document Intelligence Workspace
            </h1>
            <Badge variant="outline">Phase 2</Badge>
          </div>
          <p className="mt-1 max-w-2xl text-sm text-muted-foreground">
            Upload a source document, ask a grounded question, and inspect the retrieved citations.
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <Badge variant={health === "online" ? "outline" : "muted"}>
            {health === "online" ? "API connected" : "API pending"}
          </Badge>
          <Button variant="outline" disabled>
            <ShieldCheck className="h-4 w-4" aria-hidden="true" />
            Auth deferred
          </Button>
          <Badge variant={document ? "secondary" : "muted"}>
            {document ? "Document selected" : "No document"}
          </Badge>
        </div>
      </div>
    </header>
  );
}

function DocumentWorkspace({
  answer,
  askError,
  askState,
  document,
  onAsk,
  onQuestionChange,
  onUpload,
  question,
  uploadError,
  uploadState,
}: {
  answer: AskResponse | null;
  askError: string | null;
  askState: AskState;
  document: UploadResponse | null;
  onAsk: (event: FormEvent<HTMLFormElement>) => void;
  onQuestionChange: (question: string) => void;
  onUpload: (file: File) => void;
  question: string;
  uploadError: string | null;
  uploadState: UploadState;
}) {
  return (
    <section className="grid flex-1 gap-4 p-4 sm:p-6 xl:grid-cols-[minmax(0,1fr)_420px]">
      <div className="grid min-w-0 content-start gap-4">
        <UploadPanel onUpload={onUpload} uploadError={uploadError} uploadState={uploadState} />
        <UploadedDocumentState document={document} uploadState={uploadState} />
      </div>
      <AskAiPanel
        answer={answer}
        askError={askError}
        askState={askState}
        document={document}
        onAsk={onAsk}
        onQuestionChange={onQuestionChange}
        question={question}
      />
    </section>
  );
}

export function UploadPanel({
  onUpload,
  uploadError,
  uploadState,
}: {
  onUpload: (file: File) => void;
  uploadError: string | null;
  uploadState: UploadState;
}) {
  const inputRef = useRef<HTMLInputElement | null>(null);
  const isUploading = uploadState === "uploading";

  function handleSelectedFile(fileList: FileList | null) {
    const file = fileList?.[0];
    if (file) {
      onUpload(file);
    }
  }

  return (
    <section className="rounded-xl border border-border bg-card/80">
      <div className="flex flex-col gap-4 p-4 sm:flex-row sm:items-start sm:justify-between sm:p-5">
        <div className="min-w-0">
          <div className="flex items-center gap-2">
            <Upload className="h-4 w-4 text-primary" aria-hidden="true" />
            <h2 className="text-base font-medium">Upload document</h2>
          </div>
          <p className="mt-1 text-sm text-muted-foreground">
            PDF, DOCX, XLSX, XLS, and CSV are supported.
          </p>
        </div>
        <Badge variant={uploadState === "success" ? "secondary" : "muted"}>
          {isUploading ? "Indexing" : uploadState === "success" ? "Ready" : "Local file"}
        </Badge>
      </div>
      <div className="border-t border-border p-4 sm:p-5">
        <div
          className={[
            "flex min-h-52 flex-col items-center justify-center rounded-lg border border-dashed bg-muted/20 p-6 text-center transition-colors",
            uploadError ? "border-destructive/60" : "border-border hover:border-primary/40",
          ].join(" ")}
          onDragOver={(event) => event.preventDefault()}
          onDrop={(event) => {
            event.preventDefault();
            if (!isUploading) {
              handleSelectedFile(event.dataTransfer.files);
            }
          }}
        >
          <div className="flex h-12 w-12 items-center justify-center rounded-lg border border-border bg-background text-primary">
            {isUploading ? (
              <Loader2 className="h-5 w-5 animate-spin" aria-hidden="true" />
            ) : (
              <FileText className="h-5 w-5" aria-hidden="true" />
            )}
          </div>
          <p className="mt-4 text-sm font-medium">
            {isUploading ? "Uploading and indexing document" : "Drop a source document here"}
          </p>
          <p className="mt-1 max-w-md text-sm text-muted-foreground">
            {isUploading
              ? "The backend is parsing, chunking, embedding, and indexing the file."
              : "The returned document id will be used automatically when asking questions."}
          </p>
          <input
            ref={inputRef}
            type="file"
            className="sr-only"
            accept={supportedExtensions.join(",")}
            disabled={isUploading}
            onChange={(event) => handleSelectedFile(event.target.files)}
          />
          <Button
            className="mt-4"
            disabled={isUploading}
            onClick={() => inputRef.current?.click()}
            type="button"
          >
            {isUploading && <Loader2 className="h-4 w-4 animate-spin" aria-hidden="true" />}
            Select file
          </Button>
          {uploadError && (
            <p className="mt-4 max-w-md text-sm text-destructive" role="alert">
              {uploadError}
            </p>
          )}
        </div>
      </div>
    </section>
  );
}

export function UploadedDocumentState({
  document,
  uploadState,
}: {
  document: UploadResponse | null;
  uploadState: UploadState;
}) {
  return (
    <section className="rounded-xl border border-border bg-card/80">
      <div className="flex flex-col gap-3 border-b border-border p-4 sm:flex-row sm:items-center sm:justify-between sm:p-5">
        <div>
          <h2 className="text-base font-medium">Current document</h2>
          <p className="mt-1 text-sm text-muted-foreground">
            The active document is scoped into each Ask AI request.
          </p>
        </div>
        <Badge variant={document ? "secondary" : "muted"}>{document ? "Selected" : "Empty"}</Badge>
      </div>
      {document ? (
        <div className="grid gap-3 p-4 sm:grid-cols-[minmax(0,1fr)_auto] sm:items-center sm:p-5">
          <div className="min-w-0">
            <div className="flex min-w-0 flex-wrap items-center gap-2">
              <p className="min-w-0 truncate text-sm font-medium">{document.filename}</p>
              <Badge variant="outline">{fileExtension(document.filename).toUpperCase()}</Badge>
            </div>
            <p className="mt-1 break-all text-xs text-muted-foreground">{document.document_id}</p>
          </div>
          <div className="grid gap-2 text-sm sm:text-right">
            <div className="flex items-center gap-2 text-emerald-300 sm:justify-end">
              <CheckCircle2 className="h-4 w-4" aria-hidden="true" />
              <span>{document.status}</span>
            </div>
            <p className="text-xs text-muted-foreground">
              {document.indexed_chunk_count} of {document.chunk_count} chunks indexed
            </p>
          </div>
        </div>
      ) : (
        <div className="p-4 sm:p-5">
          <div className="rounded-lg border border-border bg-muted/20 p-4">
            <p className="text-sm font-medium">
              {uploadState === "uploading" ? "Document processing" : "No document uploaded"}
            </p>
            <p className="mt-1 text-sm text-muted-foreground">
              {uploadState === "uploading"
                ? "The indexed document summary will appear here when upload completes."
                : "Upload a supported file to enable grounded questions."}
            </p>
          </div>
        </div>
      )}
    </section>
  );
}

export function AskAiPanel({
  answer,
  askError,
  askState,
  document,
  onAsk,
  onQuestionChange,
  question,
}: {
  answer: AskResponse | null;
  askError: string | null;
  askState: AskState;
  document: UploadResponse | null;
  onAsk: (event: FormEvent<HTMLFormElement>) => void;
  onQuestionChange: (question: string) => void;
  question: string;
}) {
  const isAsking = askState === "asking";
  const canAsk = Boolean(document) && question.trim().length > 0 && !isAsking;

  return (
    <aside className="flex min-h-[34rem] flex-col rounded-xl border border-border bg-card/80 xl:sticky xl:top-6 xl:max-h-[calc(100vh-3rem)]">
      <div className="border-b border-border p-4 sm:p-5">
        <div className="flex items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <Bot className="h-4 w-4 text-primary" aria-hidden="true" />
            <h2 className="text-base font-medium">Ask AI</h2>
          </div>
          <Badge variant={document ? "secondary" : "muted"}>
            {document ? "Document scoped" : "Upload first"}
          </Badge>
        </div>
        <p className="mt-1 text-sm text-muted-foreground">
          Questions are grounded to the currently uploaded document.
        </p>
      </div>

      <div className="min-h-0 flex-1 space-y-4 overflow-y-auto p-4 sm:p-5">
        {answer ? (
          <AnswerView answer={answer} />
        ) : (
          <div className="rounded-lg border border-border bg-muted/20 p-4">
            <p className="text-sm font-medium">
              {isAsking ? "Generating answer" : "Ready for a grounded question"}
            </p>
            <p className="mt-2 text-sm leading-6 text-muted-foreground">
              {isAsking
                ? "Gemini is generating an answer from retrieved chunks."
                : document
                  ? "Ask about the uploaded file to see the answer and citations here."
                  : "Upload a document to activate the Ask AI flow."}
            </p>
          </div>
        )}

        {isAsking && (
          <div className="rounded-lg border border-border bg-background/50 p-4">
            <div className="flex items-center gap-2 text-sm text-muted-foreground">
              <Loader2 className="h-4 w-4 animate-spin" aria-hidden="true" />
              Retrieving chunks and generating response
            </div>
          </div>
        )}

        {askError && (
          <div className="rounded-lg border border-destructive/50 bg-destructive/10 p-4" role="alert">
            <div className="flex items-start gap-2">
              <AlertCircle className="mt-0.5 h-4 w-4 shrink-0 text-destructive" aria-hidden="true" />
              <div>
                <p className="text-sm font-medium text-destructive">Ask failed</p>
                <p className="mt-1 text-sm text-muted-foreground">{askError}</p>
              </div>
            </div>
          </div>
        )}
      </div>

      <form className="border-t border-border p-4 sm:p-5" onSubmit={onAsk}>
        <label htmlFor="question" className="text-sm font-medium">
          Ask a question
        </label>
        <div className="mt-2 flex gap-2">
          <textarea
            id="question"
            disabled={!document || isAsking}
            value={question}
            onChange={(event) => onQuestionChange(event.target.value)}
            placeholder={document ? "What changed in revenue growth?" : "Upload a document first"}
            className="min-h-20 min-w-0 flex-1 resize-none rounded-md border border-input bg-muted/20 px-3 py-2 text-sm text-foreground placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring disabled:cursor-not-allowed disabled:opacity-60"
          />
          <Button size="icon" disabled={!canAsk} aria-label="Submit question" type="submit">
            {isAsking ? (
              <Loader2 className="h-4 w-4 animate-spin" aria-hidden="true" />
            ) : (
              <Send className="h-4 w-4" aria-hidden="true" />
            )}
          </Button>
        </div>
      </form>
    </aside>
  );
}

export function AnswerView({ answer }: { answer: AskResponse }) {
  return (
    <>
      <div className="rounded-lg border border-border bg-background/60 p-4">
        <p className="text-sm font-medium">Question</p>
        <p className="mt-2 text-sm text-muted-foreground">{answer.question}</p>
      </div>
      <div className="rounded-lg border border-border bg-muted/20 p-4">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <p className="text-sm font-medium">Grounded answer</p>
          <Badge variant="outline">{answer.retrieved_count} sources</Badge>
        </div>
        <p className="mt-3 whitespace-pre-wrap text-sm leading-6 text-foreground">{answer.answer}</p>
      </div>
      <SourceList sources={answer.sources} />
    </>
  );
}

export function SourceList({ sources }: { sources: RetrievalSource[] }) {
  if (sources.length === 0) {
    return (
      <div className="rounded-lg border border-border bg-background/50 p-4">
        <p className="text-sm font-medium">No sources returned</p>
        <p className="mt-1 text-sm text-muted-foreground">The backend returned an answer without citations.</p>
      </div>
    );
  }

  return (
    <div>
      <p className="mb-2 text-sm font-medium">Sources</p>
      <div className="grid gap-2">
        {sources.map((source, index) => (
          <article key={source.chunk_id} className="rounded-lg border border-border bg-background/50 p-3">
            <div className="flex items-start justify-between gap-3">
              <div className="min-w-0">
                <p className="truncate text-sm font-medium">Source {index + 1}</p>
                <p className="mt-1 truncate text-xs text-muted-foreground">{sourceLabel(source)}</p>
              </div>
              <Badge variant="outline">score {formatScore(source.score)}</Badge>
            </div>
            <p className="mt-3 line-clamp-6 text-sm leading-6 text-muted-foreground">
              {source.content}
            </p>
          </article>
        ))}
      </div>
    </div>
  );
}

function isSupportedFile(file: File): boolean {
  return supportedExtensions.includes(fileExtension(file.name));
}

function fileExtension(filename: string): string {
  const index = filename.lastIndexOf(".");
  return index >= 0 ? filename.slice(index).toLowerCase() : "";
}

function formatScore(score: number): string {
  return Number.isFinite(score) ? score.toFixed(3) : "n/a";
}

function sourceLabel(source: RetrievalSource): string {
  const page = metadataValue(source.metadata, ["page", "page_number", "page_index"]);
  const filename = metadataValue(source.metadata, ["filename", "source", "file_name"]);
  const pageLabel = page ? `page ${page}` : null;
  return [filename, pageLabel, source.chunk_id].filter(Boolean).join(" · ");
}

function metadataValue(metadata: Record<string, unknown>, keys: string[]): string | null {
  const entries = Object.entries(metadata);
  for (const key of keys) {
    const match = entries.find(([entryKey]) => entryKey.toLowerCase() === key);
    if (match && (typeof match[1] === "string" || typeof match[1] === "number")) {
      return String(match[1]);
    }
  }

  return null;
}
