# Linguist-Engine: High-Precision Local Multimodal AI Tutor

**Linguist-Engine** is a specialized local-first AI system designed for advanced language acquisition through high-fidelity speech-to-text and large language model orchestration. Unlike consumer-grade applications that prioritize response speed, this system is architected on the **Latency-Precision Paradigm**, utilizing high-parameter models and intensive quantization to ensure near-absolute linguistic accuracy on consumer-grade hardware.

---

## Architectural Defense

### The Latency-Precision Paradigm
In the domain of language learning, the cost of an incorrect correction is higher than the cost of delay. This project intentionally sacrifices real-time response (accepting latencies of 10-60 seconds) to enable the deployment of the **Whisper Large-v3** and **Llama 3.1 8B (Q6/Q8)** models within a constrained **8GB VRAM** environment (NVIDIA RTX 2060 Super).

### Manual Trigger (Half-Duplex) Ingestion
To ensure hardware predictability and semantic completeness, the system utilizes an atomic ingestion model.
* **User-Driven Capture**: Audio ingestion is triggered manually (Push-to-Talk style), avoiding VAD instability in noisy environments.
* **Batch Processing**: Audio is captured in a dedicated thread and processed in a single batch to ensure the STT engine receives the full context of the utterance.
* **System Dormancy**: The audio engine enters a state of total dormancy during GPU inference to release physical resources for the LLM.

---

## Technical Stack

| Component | Technology | Specification |
| :--- | :--- | :--- |
| **STT Engine** | Faster-Whisper | Large-v3 Model (High-Fidelity) |
| **Inference Core** | Llama 3.1 | 8B Parameters (Q6_K / Q8 Quantization) |
| **Audio Ingestion** | PyAudio / NumPy | 16kHz, Mono, 16-bit -> Float32 Batch |
| **Data Persistence** | SQLite | Relational audit trail and telemetry |
| **Orchestration** | Python 3.11+ | Asynchronous (Asyncio) and Threading |

---

## Data Pipeline and System Flow

1.  **Manual Ingestion**: Captures raw audio via the hardware interface in a dedicated thread to maintain a stable physical clock.
2.  **Atomic Normalization**: Converts the raw 16-bit PCM buffer into a normalized Float32 `np.ndarray` in a single vectorized operation.
3.  **Transcription**: The **Whisper Large-v3** model performs deep spectral analysis, ensuring that even subtle grammatical markers are preserved.
4.  **Linguistic Reasoning**: The LLM acts as a senior pedagogical auditor, evaluating the user's input before generating a contextual response.
5.  **Aural Synthesis**: The response is synthesized into a natural voice stream, providing the user with a correct phonetic model.

---

## Key Engineering Features

* **Grammar Audit Trail**: Every interaction is cataloged and categorized by error type (Syntax, Morphology, Vocabulary), providing a data-driven view of evolution.
* **Zero-Cloud Dependency**: 100% offline operation, ensuring total data privacy and independence from external API costs.
* **Asynchronous Orchestration**: Robust management of concurrent tasks to ensure the system remains responsive during heavy inference loads.

---

## Requirements

* **GPU**: NVIDIA RTX 2060 Super (8GB VRAM) or higher.
* **Processor**: High-frequency multicore CPU (e.g., Ryzen 7 3800X).
* **Environment**: Python 3.11+ with strict type checking (Strict Mode).