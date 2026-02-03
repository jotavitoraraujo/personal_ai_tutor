# Linguist-Engine: High-Precision Local Multimodal AI Tutor

**Linguist-Engine** is a specialized local-first AI system designed for advanced language acquisition through high-fidelity speech-to-text and large language model orchestration. Unlike consumer-grade applications that prioritize response speed, this system is architected on the **Latency-Precision Paradigm**, utilizing high-parameter models and intensive quantization to ensure near-absolute linguistic accuracy on consumer-grade hardware.

---

## Architectural Defense

### The Latency-Precision Paradigm
In the domain of language learning, the cost of an incorrect correction is higher than the cost of delay. This project intentionally sacrifices real-time response (accepting latencies of 10-60 seconds) to enable the deployment of the **Whisper Large-v3** and **Llama 3.1 8B (Q6/Q8)** models within a constrained **8GB VRAM** environment (NVIDIA RTX 2060 Super). By implementing an asynchronous, message-based communication flow—similar to a voice messaging exchange—the system ensures that every phoneme and grammatical nuance is processed with maximum depth.

### Hardware Resource Management
To maintain system stability on a single-GPU setup, the engine implements a serialized resource allocation strategy:
* **VRAM Offloading:** Strict management of model weights between the STT (Speech-to-Text) and LLM (Large Language Model) stages to prevent CUDA out-of-memory (OOM) errors.
* **Quantization Optimization:** Use of GGUF/K-Quantization formats to preserve the logical reasoning of the Llama model while fitting within the physical memory limits.

---

## Technical Stack

| Component | Technology | Specification |
| :--- | :--- | :--- |
| **STT Engine** | Faster-Whisper | Large-v3 Model (High-Fidelity) |
| **Inference Core** | Llama 3.1 | 8B Parameters (Q6_K / Q8 Quantization) |
| **TTS Engine** | Piper / XTTSv2 | Local neural synthesis |
| **Data Persistence** | SQLite | Relational audit trail and telemetry |
| **Orchestration** | Python 3.11+ | Asynchronous (Asyncio) and Multiprocessing |

---

## Data Pipeline and System Flow

1.  **Signal Ingestion:** Captures raw audio via the hardware interface, utilizing a Voice Activity Detection (VAD) layer to isolate speech segments.
2.  **Transcription:** The **Whisper Large-v3** model performs deep spectral analysis to convert audio into text, ensuring that even subtle grammatical markers (e.g., third-person 's' or past-tense 'ed') are preserved.
3.  **Linguistic Reasoning:** The LLM acts as a senior pedagogical auditor. It evaluates the user's input against formal English standards before generating a contextual response.
4.  **Telemetry and Logging:** All interactions, identified grammatical failures, and system performance metrics are persisted in a relational database for long-term progress analysis.
5.  **Aural Synthesis:** The response is synthesized into a natural voice stream, providing the user with a correct phonetic model.

---

## Key Engineering Features

* **Grammar Audit Trail:** Every interaction is cataloged and categorized by error type (Syntax, Morphology, Vocabulary), providing a data-driven view of the user's evolution.
* **Zero-Cloud Dependency:** 100% offline operation, ensuring total data privacy and independence from external API availability or costs.
* **Asynchronous Orchestration:** Robust management of concurrent tasks to ensure the system remains responsive even during heavy inference loads.

---

## Requirements

* **GPU:** NVIDIA RTX 2060 Super (8GB VRAM) or higher.
* **Processor:** High-frequency multicore CPU (e.g., Ryzen 7 3800X).
* **Environment:** Python 3.11+ with isolated virtual environment management.