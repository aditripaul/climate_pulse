---
name: climate-pulse-patent-generation
description: Manage, review, and compile the consolidated patent specification and patent PDF for Climate Pulse.
---

# Climate Pulse Patent Generation & Legal Alignment

This skill guides the management, statutory alignment, and compilation of the consolidated patent application draft for the ClimatePulse platform.

## Key Document

- **Specification File:** `docs/patent/patent_application.md`
- **Output PDF:** `docs/patent/climatepulse_patent_application.pdf`
- **Target Office:** IPINDIA (Controller General of Patents, Designs and Trade Marks) / PCT

## Workflow

### 1. Sync Evaluation Artifacts to Patent Directory
```bash
make sync-patent-artifacts
```
Copies evaluation plots and metric summaries from `runtime/eval_artifacts/` to `docs/patent/artifacts/`.

### 2. Compile Patent Specification PDF
```bash
make gen-patent-application-pdf
```
Runs Pandoc + XeLaTeX on `docs/patent/patent_application.md` to produce `climatepulse_patent_application.pdf`.

## Statutory Compliance Rules (IPINDIA / International)

1. **Section 3(k) Protection (CRI Guidelines):** Indian patent law excludes "computer programs per se or algorithms". Claims must be framed as physical hardware systems, atmospheric telemetry processing apparatuses, or hardware-accelerated methods (`repeat_interleave`).
2. **Section 10(5) Unity of Invention:** All claim areas must maintain a **Single Inventive Concept** centered around the cross-domain ENSO conditioning pathway connecting long-horizon ocean forecasting to latent spherical GNN weather message passing.
3. **Claims Structure:** Maintain 14 total claims (independent system, independent method, independent framework, dependent structural/mathematical claims).
