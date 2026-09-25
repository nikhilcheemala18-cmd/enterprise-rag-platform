import {
  ArrowRight,
  BarChart3,
  CheckCircle2,
  FileArchive,
  FileCheck2,
  FileText,
  Github,
  KeyRound,
  Layers3,
  Search,
  Send,
  ShieldCheck,
  SlidersHorizontal,
  Sparkles,
} from "lucide-react";
import { useEffect, useState, type CSSProperties, type ReactNode } from "react";

import { Button } from "@/components/ui/button";

const githubUrl = "https://github.com/nikhilcheemala18-cmd/enterprise-rag-platform";

const metaItems = [
  { label: "6+ formats", icon: FileArchive },
  { label: "Hybrid retrieval", icon: Search },
  { label: "Neural reranking", icon: SlidersHorizontal },
  { label: "Source citations", icon: FileCheck2 },
];

const formatCards = [
  { label: "PDF", tone: "coral" },
  { label: "DOCX", tone: "blue" },
  { label: "XLSX", tone: "green" },
  { label: "PPTX", tone: "orange" },
];

const capabilities = [
  ["Enterprise security", "Access control and audit", ShieldCheck],
  ["Multi-format ingestion", "PDF, DOCX, XLSX, CSV, PPTX", FileArchive],
  ["Hybrid retrieval", "Dense + BM25+ with RRF", BarChart3],
  ["Neural reranking", "Better evidence before generation", SlidersHorizontal],
  ["Evaluation", "Track quality and observability", FileCheck2],
] as const;

export function LandingPage() {
  const [activeCitation, setActiveCitation] = useState<"approval" | "receipt">("approval");

  useEffect(() => {
    const reveals = document.querySelectorAll<HTMLElement>(".landing-reveal");
    const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    if (reducedMotion) {
      reveals.forEach((element) => element.classList.add("is-visible"));
      return;
    }

    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.classList.add("is-visible");
            observer.unobserve(entry.target);
          }
        });
      },
      { threshold: 0.16, rootMargin: "0px 0px -8% 0px" },
    );

    reveals.forEach((element) => observer.observe(element));
    return () => observer.disconnect();
  }, []);

  return (
    <div className="landing-page min-h-screen overflow-x-hidden text-[var(--text-primary)]">
      <LandingNav />
      <main>
        <HeroSection />
        <ProductDemo />
        <HowItWorks />
        <CitationExperience activeCitation={activeCitation} onCitationChange={setActiveCitation} />
        <TechnicalCredibility />
        <FinalCta />
      </main>
      <LandingFooter />
    </div>
  );
}

function LandingNav() {
  return (
    <header className="landing-nav">
      <nav className="landing-container flex h-16 items-center justify-between" aria-label="Primary navigation">
        <a href="/" className="brand-mark focus-ring">
          <span className="brand-icon" aria-hidden="true">
            <Layers3 className="h-5 w-5" />
          </span>
          <span>Enterprise RAG</span>
        </a>

        <div className="hidden items-center gap-8 text-sm text-[var(--text-secondary)] lg:flex">
          <a className="nav-link" href="#product">Product</a>
          <a className="nav-link" href="#how-it-works">How it works</a>
          <a className="nav-link" href="#architecture">Architecture</a>
          <a className="nav-link" href="#security">Security</a>
          <a className="nav-link" href={githubUrl} target="_blank" rel="noreferrer">GitHub</a>
        </div>

        <div className="flex items-center gap-4">
          <span className="nav-sun hidden h-9 w-9 items-center justify-center rounded-full text-[var(--text-secondary)] md:flex" aria-hidden="true">
            <Sparkles className="h-4 w-4" />
          </span>
          <Button asChild className="landing-button-primary">
            <a href="/workspace">
              Open Workspace
              <ArrowRight className="h-4 w-4" aria-hidden="true" />
            </a>
          </Button>
        </div>
      </nav>
    </header>
  );
}

function HeroSection() {
  return (
    <section className="hero-section">
      <div className="hero-atmosphere" aria-hidden="true" />
      <div className="landing-container hero-grid">
        <div className="landing-reveal hero-copy">
          <p className="eyebrow">Enterprise Document Intelligence</p>
          <h1>
            From your files
            <br />
            to trusted <span>insights.</span>
          </h1>
          <p className="hero-subcopy">
            Upload, search, and get accurate answers from your organization's documents — with
            hybrid retrieval, reranking, and source citations.
          </p>
          <div className="hero-actions">
            <Button asChild className="landing-button-primary h-12 px-6">
              <a href="/workspace">
                Open Workspace
                <ArrowRight className="h-4 w-4" aria-hidden="true" />
              </a>
            </Button>
            <Button asChild variant="outline" className="landing-button-secondary h-12 px-6">
              <a href={githubUrl} target="_blank" rel="noreferrer">
                <Github className="h-4 w-4" aria-hidden="true" />
                View on GitHub
              </a>
            </Button>
          </div>
          <div className="hero-meta">
            {metaItems.map(({ label, icon: Icon }) => (
              <span key={label}>
                <Icon className="h-4 w-4" aria-hidden="true" />
                {label}
              </span>
            ))}
          </div>
        </div>

        <HeroSpatialFlow />
      </div>
    </section>
  );
}

function HeroSpatialFlow() {
  return (
    <div className="landing-reveal hero-visual" aria-label="Files to cited answer visualization">
      <div className="hero-visual-grid" aria-hidden="true" />
      <div className="hero-file-stack" aria-hidden="true">
        {formatCards.map((file, index) => (
          <div
            className={`hero-file-plane ${file.tone}`}
            key={file.label}
            style={
              {
                "--file-index": index,
                "--file-left": `${index * 16}px`,
                "--file-top": `${index * 46}px`,
                "--file-z": `${index * 35}px`,
                "--file-z-active": `${index * 47}px`,
                "--file-drift": `${index * 3}px`,
              } as CSSProperties
            }
          >
            <span className="file-glyph">
              <FileText className="h-4 w-4" />
            </span>
            <span>{file.label}</span>
          </div>
        ))}
      </div>

      <div className="chunk-cluster" aria-hidden="true">
        {Array.from({ length: 12 }).map((_, index) => {
          const column = index % 3;
          const row = Math.floor(index / 3);
          return (
            <span
              key={index}
              style={
                {
                  "--chunk-index": index,
                  "--chunk-left": `${column * 52}px`,
                  "--chunk-top": `${row * 38}px`,
                  "--chunk-z": `${(index % 4) * 14}px`,
                  "--chunk-z-active": `${(index % 4) * 20}px`,
                } as CSSProperties
              }
            />
          );
        })}
        <em>Chunking</em>
      </div>

      <svg className="hero-flow-lines" viewBox="0 0 760 420" role="presentation" aria-hidden="true">
        <path className="flow-path semantic" d="M250 178 C350 88 420 76 505 126 C572 166 598 166 642 143" />
        <path className="flow-path lexical" d="M250 248 C350 332 434 325 502 274 C558 232 600 235 642 252" />
        <path className="flow-path neutral" d="M524 198 C570 190 595 194 630 204 C665 216 690 206 726 184" />
        <circle className="path-particle semantic-particle" r="3" />
        <circle className="path-particle lexical-particle" r="3" />
      </svg>

      <div className="retrieval-card semantic-card">
        <p>Semantic Search</p>
        <span>(Dense)</span>
        <div className="neural-orbit" aria-hidden="true">
          {Array.from({ length: 10 }).map((_, index) => (
            <i key={index} style={{ "--dot-index": index } as CSSProperties} />
          ))}
        </div>
      </div>

      <div className="retrieval-card lexical-card">
        <p>Lexical Search</p>
        <span>(BM25+)</span>
        <div className="bar-signal" aria-hidden="true">
          <i />
          <i />
          <i />
          <i />
        </div>
      </div>

      <div className="fusion-node">
        <strong>RRF</strong>
        <span>Fusion</span>
      </div>

      <div className="rerank-node">
        <span>Rerank</span>
        <i />
      </div>

      <div className="answer-proof-card">
        <div className="proof-header">
          <span>Grounded Answer</span>
          <CheckCircle2 className="h-4 w-4" aria-hidden="true" />
        </div>
        <p>International travel expenses require prior approval before reimbursement.</p>
        <ul>
          <li><span>1</span> employee_policy.pdf</li>
          <li><span>2</span> travel_guidelines.docx</li>
          <li><span>3</span> finance_handbook.xlsx</li>
        </ul>
      </div>

      <p className="hero-caption">Document → Retrieve → Verify</p>
    </div>
  );
}

function ProductDemo() {
  return (
    <section id="product" className="landing-section product-section">
      <div className="landing-container product-grid">
        <div className="landing-reveal section-copy">
          <p className="eyebrow">Product Demo</p>
          <h2>Ask. Verify. Move forward.</h2>
          <p>See how Enterprise RAG turns your documents into accurate, citation-backed answers.</p>
        </div>

        <div className="landing-reveal workspace-preview" aria-label="Workspace product preview">
          <div className="preview-topbar"><span /><span /><span /></div>
          <div className="preview-shell">
            <aside className="preview-sidebar">
              <div className="preview-brand">
                <Layers3 className="h-4 w-4" aria-hidden="true" />
                Enterprise RAG
              </div>
              {["Documents", "Ask AI", "Sources", "Settings"].map((item, index) => (
                <div key={item} className={index === 1 ? "active" : ""}>
                  {index === 0 && <FileText className="h-4 w-4" aria-hidden="true" />}
                  {index === 1 && <Sparkles className="h-4 w-4" aria-hidden="true" />}
                  {index === 2 && <Search className="h-4 w-4" aria-hidden="true" />}
                  {index === 3 && <KeyRound className="h-4 w-4" aria-hidden="true" />}
                  {item}
                </div>
              ))}
            </aside>
            <div className="preview-main">
              <div className="question-bar">
                <span>What is the travel expense policy for international trips?</span>
                <button aria-label="Preview send question">
                  <Send className="h-4 w-4" aria-hidden="true" />
                </button>
              </div>
              <div className="preview-tabs">
                <span className="active">Answer</span>
                <span>Sources (3)</span>
                <span>Retrieved Chunks (6)</span>
              </div>
              <article className="preview-answer">
                <div className="flex items-center gap-3">
                  <CheckCircle2 className="h-6 w-6 text-[var(--evidence-bright)]" aria-hidden="true" />
                  <strong>Grounded Answer</strong>
                </div>
                <p>
                  International travel expenses require prior approval from the department head and
                  must be submitted within 30 days with valid receipts. <button>[1]</button>{" "}
                  <button>[2]</button>
                </p>
              </article>
              <div className="source-strip">
                <article>
                  <FileText className="h-5 w-5 text-red-300" aria-hidden="true" />
                  <span>employee_policy.pdf</span>
                  <small>Section 3.2 · Page 5</small>
                  <b>0.92</b>
                </article>
                <article>
                  <FileText className="h-5 w-5 text-blue-300" aria-hidden="true" />
                  <span>travel_guidelines.docx</span>
                  <small>Section 4.1 · Page 12</small>
                  <b>0.87</b>
                </article>
                <button>View all sources →</button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

function HowItWorks() {
  return (
    <section id="how-it-works" className="landing-section architecture-section">
      <div className="landing-container architecture-grid">
        <div className="landing-reveal section-copy">
          <p className="eyebrow">How It Works</p>
          <h2>A retrieval system built for real documents.</h2>
          <p>Multiple retrieval methods. A unified pipeline. Higher-quality context. Verifiable answers.</p>
          <a className="text-link" href="#architecture">
            Explore the architecture <ArrowRight className="h-4 w-4" aria-hidden="true" />
          </a>
        </div>

        <div className="landing-reveal architecture-flow" aria-label="Retrieval architecture visualization">
          {[
            ["01", "Ingest", "Parse and structure your documents"],
            ["02", "Retrieve", "Dense + BM25+ in parallel"],
            ["03", "RRF", "Combine and rank results"],
            ["04", "Rerank", "Neural reranker for better context"],
            ["05", "Secure", "Access control and guardrails"],
            ["06", "Generate", "Grounded answer with citations"],
          ].map(([number, title, body]) => (
            <div className="arch-step" key={number}>
              <span>{number}</span>
              <strong>{title}</strong>
              <p>{body}</p>
            </div>
          ))}
          <div className="arch-stage documents"><i /><i /><i /></div>
          <div className="arch-stage retrieval dense">Dense Search</div>
          <div className="arch-stage retrieval lexical">BM25+ Search</div>
          <div className="arch-stage rank-stack before">
            {Array.from({ length: 8 }).map((_, index) => (
              <i
                key={index}
                style={
                  {
                    "--stack-index": index,
                    "--stack-left": `${(index % 2) * 30}px`,
                    "--stack-top": `${index * 10}px`,
                    "--stack-opacity": `${1 - index * 0.07}`,
                    "--stack-nudge": `${index * -1}px`,
                    "--stack-opacity-active": `${0.95 - index * 0.09}`,
                  } as CSSProperties
                }
              />
            ))}
          </div>
          <div className="arch-stage rank-stack after">
            {Array.from({ length: 5 }).map((_, index) => (
              <i
                key={index}
                style={
                  {
                    "--stack-index": index,
                    "--stack-left": `${(index % 2) * 30}px`,
                    "--stack-top": `${index * 10}px`,
                    "--stack-opacity": `${1 - index * 0.07}`,
                    "--stack-nudge": `${index * -1}px`,
                    "--stack-opacity-active": `${0.95 - index * 0.09}`,
                  } as CSSProperties
                }
              />
            ))}
          </div>
          <div className="arch-stage security-gate"><ShieldCheck className="h-10 w-10" aria-hidden="true" /></div>
          <div className="arch-stage final-doc" />
          <svg className="arch-lines" viewBox="0 0 920 250" aria-hidden="true">
            <path d="M105 147 C180 118 230 110 305 126" />
            <path d="M105 147 C182 196 228 206 305 178" />
            <path d="M388 150 C448 150 473 150 530 150" />
            <path d="M610 150 C660 150 695 150 748 150" />
            <path d="M805 150 C840 150 866 150 895 150" />
          </svg>
        </div>
      </div>
    </section>
  );
}

function CitationExperience({
  activeCitation,
  onCitationChange,
}: {
  activeCitation: "approval" | "receipt";
  onCitationChange: (citation: "approval" | "receipt") => void;
}) {
  const source = activeCitation === "approval"
    ? { section: "3.2", page: "5", score: "0.92" }
    : { section: "4.1", page: "12", score: "0.87" };

  return (
    <section className="landing-section evidence-section">
      <div className="landing-container evidence-grid">
        <div className="landing-reveal section-copy">
          <p className="eyebrow">See The Evidence</p>
          <h2>Every answer links to the source.</h2>
          <p>Click a citation to view the exact passage in the original document.</p>
          <div className="evidence-steps" aria-label="Evidence workflow">
            <span className="active">1 Ask</span>
            <span>2 Get answer</span>
            <span>3 Inspect source</span>
          </div>
        </div>

        <article className="landing-reveal evidence-answer">
          <div className="mini-heading"><Sparkles className="h-4 w-4" aria-hidden="true" /> AI Answer</div>
          <p>
            International travel expenses require prior approval from the department head and must
            be submitted within 30 days with valid receipts.{" "}
            <CitationButton active={activeCitation === "approval"} onClick={() => onCitationChange("approval")}>
              [1]
            </CitationButton>
          </p>
          <button className="citation-alt" type="button" onClick={() => onCitationChange("receipt")}>
            Inspect receipt deadline source
          </button>
          <div className="evidence-source-card">
            <span>1</span>
            <div>
              <strong>employee_policy.pdf</strong>
              <small>Section {source.section} · Page {source.page}</small>
            </div>
            <b>{source.score}</b>
          </div>
        </article>

        <article className="landing-reveal evidence-document">
          <div className="doc-window-bar">
            <span>employee_policy.pdf</span>
            <small>{source.page} / 32</small>
          </div>
          <div className="doc-window-body">
            <aside><i /><i /><i /></aside>
            <div className="doc-paper">
              <strong>3.2 International Travel</strong>
              <p className="paper-line w-full" />
              <p className="paper-line w-[88%]" />
              <p className="paper-highlight active">
                International travel expenses require prior approval from the department head and
                must be submitted within 30 days of travel with valid receipts.
              </p>
              <p className="paper-line w-[72%]" />
              <p className="paper-line w-[58%]" />
            </div>
          </div>
        </article>
      </div>
    </section>
  );
}

function CitationButton({ active, children, onClick }: { active: boolean; children: ReactNode; onClick: () => void }) {
  return (
    <button
      type="button"
      aria-pressed={active}
      className={active ? "citation-token active" : "citation-token"}
      onClick={onClick}
      onMouseEnter={onClick}
    >
      {children}
    </button>
  );
}

function TechnicalCredibility() {
  return (
    <section id="architecture" className="landing-section credibility-section">
      <div className="landing-container">
        <div className="landing-reveal">
          <p className="eyebrow">Built For Real Organizations</p>
          <h2>Engineered for trust and control.</h2>
        </div>
        <div id="security" className="landing-reveal capability-row">
          {capabilities.map(([title, body, Icon]) => (
            <article key={title}>
              <Icon className="h-6 w-6" aria-hidden="true" />
              <div>
                <strong>{title}</strong>
                <p>{body}</p>
              </div>
            </article>
          ))}
        </div>
      </div>
    </section>
  );
}

function FinalCta() {
  return (
    <section className="final-cta">
      <div className="cta-geometry" aria-hidden="true"><i /><i /><i /></div>
      <div className="landing-container final-cta-grid">
        <div className="landing-reveal">
          <p className="eyebrow">Ready To Explore?</p>
          <h2>Bring your documents. Get real answers.</h2>
        </div>
        <div className="landing-reveal">
          <p>Start using Enterprise RAG or explore the source code.</p>
          <div className="mt-8 flex flex-col gap-3 sm:flex-row">
            <Button asChild className="landing-button-primary h-12 px-6">
              <a href="/workspace">
                Open Workspace
                <ArrowRight className="h-4 w-4" aria-hidden="true" />
              </a>
            </Button>
            <Button asChild variant="outline" className="landing-button-secondary h-12 px-6">
              <a href={githubUrl} target="_blank" rel="noreferrer">
                <Github className="h-4 w-4" aria-hidden="true" />
                View on GitHub
              </a>
            </Button>
          </div>
        </div>
      </div>
    </section>
  );
}

function LandingFooter() {
  return (
    <footer className="landing-footer">
      <div className="landing-container flex flex-col gap-8 py-9 sm:flex-row sm:items-center sm:justify-between">
        <div className="brand-mark">
          <span className="brand-icon" aria-hidden="true"><Layers3 className="h-5 w-5" /></span>
          <div><span>Enterprise RAG</span><small>Built with purpose.</small></div>
        </div>
        <div className="flex flex-wrap gap-6 text-sm text-[var(--text-secondary)]">
          <a className="nav-link" href="#product">Product</a>
          <a className="nav-link" href="#how-it-works">How it works</a>
          <a className="nav-link" href="#architecture">Architecture</a>
          <a className="nav-link" href="#security">Security</a>
          <a className="nav-link" href={githubUrl} target="_blank" rel="noreferrer">GitHub</a>
        </div>
      </div>
    </footer>
  );
}
