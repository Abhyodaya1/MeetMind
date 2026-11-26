# MeetMind — Stage-Wise Implementation Bible

> **Purpose:** Living engineering reference for MeetMind. Documents every stage of the system — **what** was built, **why** it was built that way, **what alternatives were avoided** (and why), and **interview-ready Q&A** grounded in the actual codebase.

---

## Hardware & Design Constraints

- **Local Machine:** 8 GB RAM, 512 GB SSD, NVIDIA RTX 3050 Laptop GPU (4 GB VRAM), PyTorch + CUDA.
- **Architectural Consequence:**
  - Heavy LLM reasoning (`openai/gpt-oss-120b` generation, `openai/gpt-oss-20b` evaluation judge) runs via `GroqProvider` API rather than local inference.
  - Local ASR (`openai/whisper-small.en`) uses `float16` on CUDA when available, `float32` on CPU.
  - Local embeddings (`all-MiniLM-L6-v2`) are pinned to `device="cpu"` in `HFEmbeddingProvider` so they never compete with Whisper for the 4 GB VRAM.

---

## Stage 0 (V1 Core): Evidence-Grounded Audio-to-Minutes Pipeline

### What We Built
- **Provider Abstractions** (`app/providers/`): `ASRProvider` (`whisper.py`), `LLMProvider` (`groq.py`), `TokenizerProvider` (`hf_tokenizer.py`).
- **Transcript Normalization** (`app/transcript.py` → `normalize_transcript`): Filters empty/malformed segments, rejects `start is None` or `end < start`, and permits `end=None` on the final Whisper segment.
- **Token-Aware Chunking** (`app/chunking.py` → `chunk_transcript`): Groups `TranscriptSegment`s into overlapping `TranscriptChunk`s bounded by token count rather than character count.
- **Two-Stage LLM Pipeline** (`app/meeting.py` → `process_meeting`):
  1. Per-chunk structured extraction (`ChunkAnalysis`)
  2. Meeting-level synthesis (`MeetingMinutes`)
- **Strict Deterministic Validation** (`app/validation.py` → `validate_evidence`): Verifies non-negative timestamps, valid range, existence of segment timestamps within `TIMESTAMP_TOLERANCE = 0.05`, and normalized substring containment in the referenced transcript range.

### Why We Built It This Way
- **Typed Pydantic Contracts (`app/schemas.py`)**: Turning raw LLM JSON into `ChunkAnalysis` / `MeetingMinutes` immediately catches malformed outputs before downstream stages consume them.
- **Provider Injection**: Orchestration in `app/meeting.py` depends on abstract interfaces (`ASRProvider`, `LLMProvider`, `TokenizerProvider`), so swapping Whisper or Groq requires zero changes to chunking, prompting, or validation.
- **Deterministic Evidence Validation**: LLMs can hallucinate quotes or timestamps; validating against the source transcript guarantees the system never blindly trusts model-generated evidence.

### What We Avoided & Why
- **Avoided character-based chunking:** LLM context limits are token-based; character counts vary wildly by vocabulary and formatting.
- **Avoided single-shot whole-meeting prompting:** Long meetings exceed context budgets and degrade extraction recall; chunk + synthesis scales to arbitrary meeting lengths.
- **Avoided using an LLM to validate evidence at runtime:** Adds latency/cost and can still hallucinate; deterministic timestamp + text checks are fast, free, and auditable.

### Interviewer Q&A
- **Q: Why two stages (chunk analysis → synthesis) instead of summarizing each chunk and concatenating?**
  - *A:* Concatenating chunk summaries duplicates decisions/action items across overlapping segments and loses meeting-wide coherence. Synthesis merges duplicates across chunks while preserving the most direct evidence.
- **Q: Why allow `end=None` on `TranscriptSegment`?**
  - *A:* Whisper's timestamped pipeline can omit the final segment's `end` timestamp. Rejecting `end=None` would drop the end of the meeting, so `normalize_transcript` and `validation.py` explicitly handle it.

---

## Stage 1: Offline Evaluation Subsystem & Hybrid Evaluator

### What We Built
- **Metrics** (`app/evaluation/metrics.py`): `precision`, `recall`, `f1_score` with safe `0.0` handling on zero denominators.
- **Independent Evaluators**:
  - Exact matching (`app/evaluation/matcher.py` → `evaluate_items`)
  - Embedding similarity (`app/evaluation/semantic_matcher.py` → `SemanticMatcher` with `all-MiniLM-L6-v2`)
  - LLM-as-a-Judge (`app/evaluation/llm_judge.py` → `LLMJudge`, `app/evaluation/llm_evaluator.py` → `llm_evaluate_items`)
  - **Hybrid Evaluator** (`app/evaluation/hybrid_evaluator.py` → `hybrid_evaluate_items`): Semantic candidate retrieval (top-K) → `LLMJudge`
- **Unified Meeting Report** (`app/evaluation/evaluator.py` → `evaluate_meeting`, `compute_evidence_validity`) returning `EvaluationReport`.
- **Unit Tests** (`tests/test_evaluation.py`, `tests/test_validation.py`) using fake `SemanticMatcher` and `LLMProvider` implementations.

### Why We Built It This Way
- **Hybrid Retrieval → Judge:** Running `LLMJudge` on every `(prediction, ground_truth)` pair costs $N \times M$ LLM calls. Ranking unmatched ground-truth items by cheap local embeddings and judging only the `top_k` candidates cuts cost to $\le N \times K$ while keeping semantic accuracy.
- **Judge Separation:** `LLMJudge` uses `LLMProvider` (intended judge model `openai/gpt-oss-20b`) rather than the runtime generation model (`openai/gpt-oss-120b`) to reduce self-preference bias.

### What We Avoided & Why
- **Avoided making evaluation part of the live runtime path:** Evaluation is offline against ground truth; mixing it into runtime would add latency and couple runtime to test labels.
- **Avoided relying solely on exact match or solely on embeddings:** Exact match fails on valid paraphrases (e.g., `"Develop the working design..."` vs `"David will create..."`), while raw cosine similarity lacks reasoning about missing owners or subtle contradictions.

### Interviewer Q&A
- **Q: Why enforce 1-to-1 greedy matching (`matched_ground_truth`) in evaluation?**
  - *A:* Without 1-to-1 matching, a model that outputs 5 paraphrases of the same action item would get 5 true positives and inflate precision/recall.
- **Q: How do you test an LLM evaluator without burning API credits in CI?**
  - *A:* Because `LLMJudge` depends on `LLMProvider`, tests inject a `FakeLLMProvider` and `FakeMatcher` to verify candidate ranking, `top_k` call limits, and F1 math deterministically offline.

---

## Stage 2: Labeled Benchmark Dataset & Threshold Calibration

### What We Built
- **Labeled Benchmark Dataset** (`data/benchmark/meetings.json`): Versioned `BenchmarkCase` records containing `chunks`, `minutes`, and `ground_truth` (`GroundTruthMeeting`), covering exact matches, paraphrases, extra items, and invalid evidence.
- **Benchmark Runner & Calibration** (`app/evaluation/benchmark.py`):
  - `load_benchmark`, `run_benchmark` (macro-averaged `avg_action_f1`, `avg_decision_f1`, `avg_evidence_validity`)
  - `calibrate_semantic_threshold` (sweeps thresholds empirically)
- **CLI** (`run_benchmark.py`): Supports `--mode exact | semantic | calibrate | llm | hybrid`.
- **Tests** (`tests/test_benchmark.py`).

### Why We Built It This Way
- Thresholds like `0.75` for cosine similarity should be calibrated on labeled examples rather than guessed. A fixed JSON benchmark lets us measure regressions whenever prompts, chunking, or models change.

### What We Avoided & Why
- **Avoided storing benchmark data in a database:** At this stage, a version-controlled JSON file is diffable in Git, portable, and requires zero infrastructure.

### Interviewer Q&A
- **Q: What did your exact-match baseline show on the benchmark?**
  - *A:* `avg_action_f1: 0.0` and `avg_decision_f1: 0.5`, proving that exact string matching severely undercounts valid paraphrased extractions and justifying semantic/hybrid evaluation.

---

## Stage 3: Prompt-Aware Token Budgeting & Deterministic Evidence Repair

### What We Built
- **Prompt-Aware Token Budgeting**:
  - `format_timestamped_segment` and `estimate_chunk_prompt_overhead` in `app/prompting.py`.
  - `prompt_aware` and `prompt_overhead_tokens` options in `app/chunking.py` (`chunk_transcript`), wired into `app/meeting.py` against `MAX_CONTEXT_TOKENS`.
- **Deterministic Evidence Repair** (`app/validation.py` → `repair_evidence`):
  - Keeps `validate_evidence` 100% strict as the deterministic baseline.
  - Searches 1–3 adjacent `TranscriptSegment` windows for punctuation-normalized matches or token containment `>= min_overlap_ratio` and snaps only to real segment timestamps.
- **Tests** (`tests/test_chunking.py`, expanded `tests/test_validation.py`).

### Why We Built It This Way
- `create_chunk_prompt` adds `[start → end]` metadata per segment plus ~350+ template tokens; budgeting only raw `segment.text` underestimates actual prompt tokens.
- Models sometimes quote text accurately but drift on timestamps or punctuation; `repair_evidence` recovers valid grounding without ever inventing timestamps.

### What We Avoided & Why
- **Avoided silently replacing `validate_evidence`:** Retaining strict validation as the baseline lets us measure how often repair is needed and prevents hiding grounding regressions.

### Interviewer Q&A
- **Q: Why search sliding windows of 1–3 adjacent segments in `repair_evidence`?**
  - *A:* A single decision or action item often spans two consecutive Whisper segments. Limiting to small windows preserves the "smallest supporting range" rule and avoids matching the entire meeting.

---

## Stage 4: Speaker Diarization Foundation

### What We Built
- **Contracts** (`app/schemas.py`): `SpeakerTurn(speaker, start, end)` and optional `speaker: Optional[str] = None` on `TranscriptSegment`.
- **Provider Abstraction** (`app/providers/diarization.py` → `DiarizationProvider`).
- **Speaker Alignment** (`app/diarization.py` → `align_speakers`): Assigns each segment the `SpeakerTurn` with maximum timestamp overlap (handling `end=None`).
- **Speaker-Aware Prompting & Wiring**: `format_timestamped_segment` formats `[Speaker | start → end]` when present; `process_meeting(audio_file, diarizer=None)` optionally runs diarization.
- **Tests** (`tests/test_diarization.py`).

### Why We Built It This Way
- Decoupling `DiarizationProvider` from `align_speakers` means we can swap local vs. API diarization later without touching alignment or prompting, and `diarizer=None` keeps existing callers working.

### What We Avoided & Why
- **Avoided loading `pyannote.audio` alongside Whisper in default runtime:** On a 4 GB VRAM GPU, running two heavy audio neural models concurrently risks OOM.

### Interviewer Q&A
- **Q: How do you align Whisper segments with speaker diarization turns when boundaries don't match exactly?**
  - *A:* `align_speakers` computes timestamp overlap duration between each `TranscriptSegment` and every `SpeakerTurn`, assigning the speaker with the largest overlap (or `None` if there is zero overlap).

---

## Stage 5: Cross-Meeting RAG Foundation

### What We Built
- **Contracts** (`app/schemas.py`): `IndexedChunk`, `RetrievedChunk`, `GroundedAnswer`.
- **Embedding Abstraction**: `EmbeddingProvider` (`app/providers/embeddings.py`) and `HFEmbeddingProvider` (`app/providers/hf_embeddings.py`, pinned to `device="cpu"`).
- **In-Memory Cross-Meeting Index & QA** (`app/rag.py`):
  - `MeetingIndex.add_meeting(meeting_id, chunks)`
  - `MeetingIndex.search(query, top_k=3, meeting_id=None, speaker=None)`
  - `answer_question(...) -> GroundedAnswer` (skips LLM call when no chunks match).
- **Tests** (`tests/test_rag.py`).

### Why We Built It This Way
- Builds the retrieval and citation contract first in pure Python before introducing PostgreSQL/pgvector in V2.

### What We Avoided & Why
- **Avoided adding PostgreSQL/pgvector/Chroma/Qdrant in Stage 5:** Premature database infrastructure slows down core AI iteration. The `MeetingIndex` interface can be backed by pgvector in V2 without changing `answer_question` callers.
- **Avoided GPU embeddings:** Pinning `HFEmbeddingProvider` to CPU leaves the 4 GB VRAM free for Whisper.

### Interviewer Q&A
- **Q: What happens in `answer_question` if retrieval returns nothing?**
  - *A:* It short-circuits and returns `"No relevant meeting records found."` with `citations=[]` without calling the LLM, preventing hallucinated answers and saving API cost.
