# AI Engineering Platform — Target Architecture

**Status:** Architecture baseline  
**Repository:** `ai-engineering-platform`  
**Goal:** Build an integrated AI platform with our own models, model-training systems, agent runtime, content-generation capabilities, and commercial product.

## 1. Vision

Build a platform that combines conversational AI, reasoning workflows, browser-based tasks, software development, document generation, image generation, audio and music generation, and eventually video generation.

The long-term objective is to own the model weights, training pipelines, model-serving interfaces, orchestration logic, application code, and user experience.

The platform will grow incrementally inside this repository. Each capability is a module in one integrated system, not a separate demo or disposable project.

## 2. Non-Negotiable Principles

1. **Own the AI models.** Our language and multimodal models are intended to be trained from random initialization. We will not use pretrained model weights as the foundation of our proprietary models.
2. **No hosted AI inference dependency.** Core inference and training must not depend on external commercial LLM, image, audio, or video APIs.
3. **Use engineering libraries intelligently.** PyTorch and other suitable libraries may provide tensor operations, automatic differentiation, numerical kernels, file-format support, databases, web frameworks, testing, and mobile development. Using these libraries does not mean using someone else's pretrained AI model.
4. **One cumulative repository.** Continue building on the existing `ai-engineering-platform` repository. Preserve functioning components and extend them through defined interfaces.
5. **Measure real capability.** Every model has documented training data, configuration, evaluation results, limitations, and reproducible checkpoints. A successful test does not imply production-level intelligence.
6. **Local-first development.** The initial development environment should work locally wherever hardware permits. Deployment can later support self-hosted infrastructure and controlled scaling.
7. **Security by design.** Protect user data, model artifacts, secrets, generated code, browser sessions, and payment operations.
8. **No unlicensed training data.** Dataset ingestion must record provenance, usage rights, processing history, and filtering rules. Example student datasets in the repository are not the foundation of the platform's language model.
9. **Build in vertical slices.** Each phase must produce a working, tested addition to the integrated platform.

## 3. Existing Foundation

The repository already contains useful foundations:

- FastAPI application and request/response schemas.
- SQLite database schema and repositories.
- Project, conversation, and message persistence.
- Model registry and model artifact storage.
- Model inference runtime and active model-version selection.
- General model training, evaluation, hyperparameter tuning, and training-run tracking.
- Tabular preprocessing and an integrated training/inference test workflow.
- API tests and a Git-based development workflow.

These capabilities are infrastructure, not a completed conversational AI engine.

The following major capabilities still need to be designed and implemented: our own language-model architecture and weights, tokenizer and language-model corpus pipeline, language-model trainer, text generation, chat orchestration, agent execution, and multimodal model-training systems.

## 4. Target System Architecture

```text
Web Application / Mobile Application
                 |
         FastAPI Application
                 |
   Authentication / Quotas / Projects
                 |
        Application Services
                 |
   +-------------+------------------+
   |             |                  |
Chat Engine   Agent Runtime    Artifact Workflows
   |             |                  |
   |         Tool Registry       File Builders
   |         Task Planner        DOCX / PPTX / PDF
   |         Task State          XLSX / ZIP / Code
   |
Model Serving Interfaces
   |
   +-- Our Language Model
   +-- Our Image Model
   +-- Our Audio / Music Models
   +-- Our Video Model (later)
   |
Model Lab: Data -> Tokenizer -> Architecture -> Training
          -> Evaluation -> Checkpoints -> Model Registry

Shared Platform Services
   +-- SQLite / Future Production Database
   +-- Conversation and Message History
   +-- Model Registry and Artifacts
   +-- Training Runs and Evaluation Results
   +-- Authentication, Usage and Billing
   +-- Logging, Metrics, Testing and Deployment


This is the target architecture, not a claim that every component already exists.
5. Model Lab — Our Training System
The first major development objective is a language model trained by us from random initialization.
5.1 Corpus Pipeline
Responsibilities:
- Ingest text from explicitly approved, documented sources.
- Preserve source provenance and license metadata.
- Normalize text, remove unwanted material, and identify duplicates.
- Filter sensitive or unsuitable content where appropriate.
- Train and maintain our tokenizer vocabulary.
- Split data into training, validation, and test sets without leakage.
- Save reproducible dataset manifests and processing configurations.
Raw corpora and large intermediate files must not be committed to Git. Store them under a configured local data directory or approved storage location.
5.2 Tokenizer
Implement and evaluate a tokenizer pipeline that converts text into token IDs and decodes model output back to text.

Track vocabulary size, special tokens, unknown-input handling, serialization format, and compatibility between tokenizer and checkpoint versions.

The first implementation should favor correctness and reproducibility over sophistication.
5.3 Language-Model Architecture
Start with a small decoder-only Transformer implemented using PyTorch.

The initial implementation should include:
- Token embeddings and positional information.
- Causal self-attention.
- Feed-forward layers.
- Residual connections and normalization.
- Language-model output projection.
- Configurable model dimensions and sequence length.
- Autoregressive generation with explicit stopping rules.
Every trainable parameter must begin from a new initialization for the from-scratch training path. No pretrained LLM weights are to be silently imported.
5.4 Training Engine
Implement:
- Next-token prediction and cross-entropy loss.
- Batching and sequence construction.
- Optimizer and learning-rate schedule.
- Gradient clipping and reproducible random seeds.
- Training and validation metrics.
- Periodic checkpoints and recovery after interruption.
- Training-run metadata, configuration, and error reporting.
- CPU/GPU selection based on detected capabilities.
The first model is an architecture and training validation milestone. Its actual quality must be measured; a small corpus and model cannot be assumed to produce competitive general-purpose answers.
5.5 Evaluation
At minimum, record training loss, validation loss, perplexity when mathematically applicable, training duration, tokens processed, checkpoint configuration, and representative generated samples.

Keep test data separate from training and model-selection data.
6. Model Serving and Conversation Engine
Once a trained checkpoint supports reliable generation, expose it through a stable internal model interface.

Responsibilities:
- Load a compatible tokenizer and checkpoint.
- Validate inference parameters.
- Build input sequences and maintain context limits.
- Generate tokens and decode responses.
- Support controlled sampling parameters and stopping.
- Track latency, token counts, failures, and model version.
- Release model resources cleanly.
The conversation service will connect the existing conversation and message repositories to this inference interface.

A chat operation will eventually:
1. Authenticate and authorize the user.
2. Resolve the user's conversation.
3. Store the incoming message.
4. Assemble authorized conversation history and context.
5. Invoke the configured internal model.
6. Store the model response.
7. Return the result, with streaming support added as a distinct capability.
Keep model execution separate from database repositories and HTTP route functions.
7. Agent Runtime
The agent runtime will later allow models to execute multi-step work using registered tools.

Core components:
- Task and step state machine.
- Planner and execution loop.
- Tool registry with explicit input/output schemas.
- Execution limits, timeouts, and cancellation.
- Task history and intermediate artifacts.
- Permission checks for filesystem, browser, and network access.
- Human confirmation for sensitive or destructive actions.
- Sandboxed code execution.
The agent runtime must never assume that model-generated instructions are safe. Tool permissions are enforced by application code, not by model promises.
8. File and Software Generation
Use suitable format libraries to create real, valid output files.

Planned capabilities include:
- Word documents: DOCX.
- Presentations: PPTX.
- Spreadsheets: XLSX.
- PDF export and processing.
- Images and downloadable generated assets.
- Source-code projects, directory trees, and ZIP archives.
A shared artifact service should manage artifact IDs, ownership, filenames, MIME types, storage paths, size limits, integrity, download permissions, and retention.

Generated software projects should be written into isolated workspace directories. File creation must prevent path traversal and unintended overwrites.

Creating DOCX or PPTX with a library is acceptable; training our own generative model is a separate AI task.
9. Multimodal Model Roadmap
Each modality has its own data pipeline, architecture, training objective, evaluation, serving requirements, and compute profile.
Image generation
Develop a training path for our own image-generation model, with documented data rights, preprocessing, model architecture, checkpointing, and image-quality evaluation.
Audio and music
Develop independent speech/audio and music-generation research paths. Audio transcription, speech synthesis, audio understanding, and music generation are different capabilities and must not be treated as one model task.
Video
Video generation comes later because it introduces substantial temporal modeling, video-data handling, memory, and compute requirements. Establish measurable image and sequence-modeling foundations first.

The same internal job and artifact interfaces should be reusable across modalities without pretending their model implementations are identical.
10. Browser and Task Execution
Build browser access as a controlled tool capability of the agent runtime.

Requirements:
- Explicit per-task browser permissions.
- Isolation of browser profiles and session data.
- Download and upload controls.
- Domain and network policies where possible.
- Protection against prompt injection from web pages.
- User confirmation before sensitive actions such as purchasing, publishing, or submitting important forms.
- Audit trails for actions and artifacts.
The product must distinguish between planning a task, executing it, and confirming the external result.
11. Web and Mobile Applications
The user-facing applications should share the same backend APIs and authorization model.

Planned experiences:
- Authentication and account settings.
- Conversation history and chat workspace.
- Project and file workspace.
- Long-running task status and cancellation.
- Document, presentation, code, image, audio, and video outputs.
- Artifact preview and download.
- Usage history, subscription state, and billing portal.
- Responsive web UI and mobile application.
The interface should expose only capabilities that the configured backend can actually perform.
12. Plans, Ads, and Payments
The commercial platform will include:
- Free plan and paid plans, including planned Plus and Pro tiers.
- Server-enforced quotas and usage measurement.
- Subscription lifecycle and entitlement checks.
- Billing history and usage transparency.
- Ad placements governed by account eligibility and applicable policies.
- Payment integration through a qualified provider supporting the required Mastercard transactions and target market.
Never store raw card numbers or card security codes in the platform database. Use provider-hosted or provider-tokenized payment flows, signed webhook verification, idempotent event handling, and auditable subscription transitions.

A payment provider is infrastructure, not the source of our AI intelligence. Its regional availability and commercial requirements must be verified before choosing one.
13. Security, Reliability, and Operations
Security and reliability requirements include:
- User and project-level authorization.
- Secret management outside source control.
- API authentication and rate limiting.
- Validation of all external input.
- File-size and upload restrictions.
- Sandboxed execution for untrusted code.
- Safe model-artifact paths and ownership.
- Structured application logs and traceable task IDs.
- Database backups and schema migrations.
- Dependency pinning and reproducible environments.
- Unit, API, integration, and end-to-end tests.
- Model, dataset, and artifact version metadata.
- Monitoring for failures, latency, compute usage, and resource exhaustion.
A feature is not production-ready merely because its happy-path test passes.
14. Development Phases and Acceptance Criteria
Phase 0 — Architecture and repository baseline
- Document the architecture and target boundaries.
- Preserve functioning modules and tests.
- Record current interfaces and missing capabilities.
- Establish consistent conventions for modules, configuration, artifacts, and tests.
Exit condition: The architectural baseline is committed and the repository's existing tests remain green.
Phase 1 — First language model from scratch
- Create the corpus manifest and tokenizer implementation.
- Implement a configurable decoder-only Transformer.
- Initialize new model weights.
- Add a small, reproducible next-token training loop.
- Save and reload checkpoints.
- Measure training and validation loss.
- Generate text from a saved checkpoint.
Exit condition: A reproducible test proves that a newly initialized model can train, checkpoint, reload, and generate tokens. Results and limitations are documented.
Phase 2 — Internal model inference
- Introduce stable model and tokenizer loading interfaces.
- Add input validation and generation configuration.
- Test inference using saved project checkpoints.
- Track model version and inference timing.
Exit condition: A locally trained checkpoint can be loaded and queried through an internal Python interface without an external AI API.
Phase 3 — Conversation integration
- Build conversation orchestration above existing repositories.
- Persist user and assistant messages.
- Assemble bounded conversation context.
- Add an authenticated chat API.
- Add streaming, cancellation, and error handling incrementally.
Exit condition: A local model can answer through the platform API, with conversation messages persisted and recoverable.
Phase 4 — Model Lab expansion and agents
- Expand training configurations and evaluation reports.
- Establish safe task execution and tool interfaces.
- Add workflow state, retries, cancellation, and artifacts.
Exit condition: A tool-enabled task runs through a controlled workflow and leaves an auditable execution record.
Phase 5 — Content and software workflows
- Integrate document, presentation, spreadsheet, PDF, and ZIP builders.
- Add project workspaces and artifact downloads.
- Introduce code-generation and execution workflows with sandboxing.
Exit condition: Generated artifacts are valid, owned by the correct user, and downloadable through authenticated APIs.
Phase 6 — Image, audio, music, and video models
- Develop one modality at a time.
- Establish separate datasets, training configurations, checkpoints, and evaluations.
- Connect completed models through shared model and artifact interfaces.
Exit condition: Each capability has a real trained or training-ready model path, measurable evaluation, and documented limits.
Phase 7 — Web and mobile product
- Deliver account, chat, project, task, and artifact experiences.
- Add responsive web and mobile clients.
- Validate authorization and usability against the same API.
Exit condition: A user can complete an end-to-end workflow using the product interface.
Phase 8 — Commercial and production readiness
- Add plan entitlements, usage metering, ads, subscriptions, and payment workflows.
- Verify provider, regional, security, and compliance requirements.
- Add deployment automation, monitoring, recovery, and operational documentation.
Exit condition: The product can enforce plans consistently and operate with monitored, secure deployments.
15. Hardware-Aware Model Development
Model architecture and training settings must reflect actual hardware.

Before selecting the first training configuration, record CPU, RAM, GPU model, available GPU memory, and storage. Use this information to select a model size and sequence length that can actually run.

Start small enough to verify the complete training loop. Scale model size, dataset size, and batch throughput only after measuring memory use and training speed.

Record configuration, random seed, dataset manifest, software versions, and checkpoints so that experiments can be reproduced.
16. Completion Definition
A module is complete only when:
- Its responsibility and interfaces are documented.
- Valid and invalid input behavior is tested.
- Relevant existing integration points work.
- Artifacts and metadata are stored consistently.
- Security assumptions are documented.
- Its limitations are explicit.
- Existing project tests remain green.
- The result is committed to the same repository.
This document is the target architecture. Implementation status must be updated as the project evolves; planned capabilities must never be represented as already implemented.
