# ADR 0001: Pivot to .NET Ecosystem and Cloud-Based Inference

## 1. Status
Accepted (March 2026)

## 2. Context
The project's initial prototype was developed in Python, which served its purpose for rapid iteration and local AI inference testing. However, a critical misalignment has emerged between the current tech stack and my long-term strategic career goals. In the current industry landscape, Python is frequently subjected to a perception bias, often associated with entry-level scripting or "vibe coding." This stigma overshadows deep technical discussions regarding memory management, CPU/IO bound operations, and low-level hardware constraints. Furthermore, my analysis of the local market demonstrates a robust, mature demand for authoritative engineering roles within the .NET ecosystem. Continuing with a dynamically typed, highly abstracted language limits my exposure to the rigorous architectural patterns expected in enterprise environments.

## 3. Decision
I am executing a complete architectural rewrite, migrating the core system from Python to C# (.NET). Simultaneously, I am deprecating local hardware inference in favor of cloud-based inference (Groq API) to achieve near-zero latency. I actively choose to embrace the inherent structural complexity of a compiled, strongly typed ecosystem. Instead of reinventing the wheel in an environment plagued by perception issues, I am aligning this project—and my underlying engineering discipline—with proven, enterprise-grade industry standards.

## 4. Consequences
* **Positive:** Immediate strategic alignment with high-value enterprise market demands. Establishment of an authoritative technical posture by enforcing strict object-oriented programming (OOP), explicit memory allocation (Stack/Heap), and native asynchronous concurrency management.
* **Negative:** Complete invalidation and necessary rewrite of the existing Python legacy codebase. A significantly steeper learning curve and increased cognitive load required to master C# syntax, strong typing boundaries, and enterprise architecture design patterns.