# Introduction

This document outlines the overall project architecture for **QuantDesk**, including backend systems, shared services, and non-UI specific concerns. Its primary goal is to serve as the guiding architectural blueprint for AI-driven development, ensuring consistency and adherence to chosen patterns and technologies.

**Relationship to Frontend Architecture:**
If the project includes a significant user interface, a separate Frontend Architecture Document will detail the frontend-specific design and MUST be used in conjunction with this document. Core technology stack choices documented herein (see "Tech Stack") are definitive for the entire project, including any frontend components.

## Starter Template or Existing Project

Based on my review of the PRD and project structure, QuantDesk appears to be an **existing production-ready project** rather than a starter template. The project shows:

- **Multi-service architecture** already implemented (Backend, Frontend, MIKEY-AI, Data Ingestion)
- **Production deployment** on Vercel with enterprise-grade security
- **Comprehensive codebase** with smart contracts, database schemas, and full infrastructure
- **Established technology stack** with specific versions and configurations

**Decision:** This is a **brownfield architecture** project - an existing production system requiring architectural documentation and potential optimization rather than greenfield design.

## Change Log

| Date | Version | Description | Author |
|------|---------|-------------|---------|
| 2025-10-19 | v1.0 | Initial architecture documentation for production QuantDesk system | Winston (Architect) |
