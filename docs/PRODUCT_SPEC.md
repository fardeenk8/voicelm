# VoiceLM — Product Specification

Last updated: 2026-09-20

## What VoiceLM is

A personal knowledge workspace that runs entirely on the user's own machine. The user
imports their documents; VoiceLM indexes them; the user then asks questions — by typing
or by speaking — and receives answers grounded in those documents, with citations
pointing back to the exact source and location.

## What makes it different

Cloud knowledge assistants require you to upload your documents to someone else's
servers. That is disqualifying for medical records, legal documents, unpublished
research, financial statements, and private journals — precisely the material where a
knowledge assistant would be most valuable.

VoiceLM's differentiator is not a better model. It is that the model runs on your
hardware, so the privacy question never arises.

## Core principles

These are commitments, not preferences. A feature that violates one of them does not
ship.

1. **Local by default.** No user document content leaves the device. See ADR-0009.
2. **Always grounded.** Every factual claim in an answer traces to a specific location
   in a specific source. An answer VoiceLM cannot cite is an answer it should not give.
3. **Honest about uncertainty.** "Your documents do not cover this" is a correct and
   valuable answer. Confident fabrication is the primary failure mode we are designing
   against.
4. **Voice as a first-class input,** not a speech-to-text box bolted onto a chat UI.
5. **The user owns their data.** Plain files and open formats on disk. No lock-in.

## Phases

Each phase is only started once the previous one works end to end.

### Phase 1 — Local knowledge engine
Ingest PDF, TXT, and Markdown. Extract text, clean it, split it into chunks, embed the
chunks, store them, retrieve the relevant ones for a question, and have a local LLM
answer using only those chunks — with citations.

*Done when:* a user imports a PDF, asks a question about it, and receives a correct
answer citing the right page.

### Phase 2 — Multimodal knowledge
Widen the set of sources: DOCX, PPTX, web pages, YouTube, audio, video, images, and
GitHub repositories. The retrieval engine from Phase 1 does not change; only the
ingestion front end grows.

*Done when:* a single question can be answered from a mix of a PDF, a web page, and a
video transcript, each cited correctly.

### Phase 3 — Voice
Local speech-to-text, local text-to-speech, streaming responses, barge-in
(interrupting the assistant mid-sentence), voice activity detection, and conversational
memory across turns.

*Done when:* a user can hold a multi-turn spoken conversation about their documents and
interrupt naturally.

### Phase 4 — Agentic knowledge workspace
Research agents that plan multi-step investigations, tool calling, document generation,
saved workflows, knowledge actions, and long-term memory.

*Done when:* a user can ask for a comparison across ten sources and receive a
structured, cited document.

## Non-goals

Stated explicitly so we do not drift into them:

- Not a general-purpose chatbot. Without sources, VoiceLM has little to say.
- Not a cloud service, and not multi-tenant. Single user, single machine.
- Not a document editor. We read and cite documents; we do not replace Word.
- Not real-time collaboration.
- Not a mobile app in Phases 1–3 — though the backend is architected so one becomes
  possible without a rewrite.

## Primary user scenarios

- **The student.** Imports a semester of lecture slides and readings, then quizzes
  themselves out loud while walking, with every answer cited to a specific lecture.
- **The researcher.** Imports forty papers and asks where their methodologies disagree.
- **The professional.** Imports contracts or medical records — material that legally or
  ethically cannot be uploaded to a third party — and asks questions about them.
