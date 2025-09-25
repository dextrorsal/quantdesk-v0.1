Guide for Preparing Codebase for Public GitHub Repository
This document provides a concise, step-by-step process for the Cursor agent to prepare your codebase for upload to a public GitHub repository. The focus is on identifying and removing sensitive data, addressing security vulnerabilities, and ensuring overall code quality. Use the available tools (SonarQube, Endor Labs Socket, Semgrep, and Snyk) where applicable. Perform all actions locally to avoid exposing code prematurely.
Prerequisites

Ensure the codebase is in a local Git repository (initialize with git init if needed).
Install and configure the tools:

SonarQube: Set up a local server or use SonarCloud (free tier) with anonymized data.
Endor Labs Socket: Install via CLI and authenticate.
Semgrep: Install via pip or Homebrew; use open-source rules.
Snyk: Install CLI and create a free account.


Backup the entire codebase before making changes.

Step 1: Scan for Sensitive Data
Goal: Detect and remove API keys, passwords, tokens, credentials, PII (e.g., emails, IP addresses), or hardcoded secrets.

Tools to Use:

Semgrep: Run semgrep scan --config=p/r2c-security-audit to detect common secrets patterns.
Snyk: Run snyk code test for code scanning, focusing on secrets.
Endor Labs Socket: Run socket scan to identify exposed secrets.


Actions:

Execute scans on the entire codebase.
Review results: For each detected item, refactor code to use environment variables (e.g., .env files) or secrets management (e.g., GitHub Secrets for CI/CD).
Add .gitignore entries for sensitive files (e.g., .env, *.key, node_modules/).
Commit changes with message: "Remove sensitive data and update .gitignore".
Re-scan to confirm no secrets remain.



Step 2: Identify and Fix Security Vulnerabilities
Goal: Detect vulnerabilities in code, dependencies, and configurations.

Tools to Use:

SonarQube: Run a full analysis with sonar-scanner CLI to check for bugs, vulnerabilities, and code smells.
Semgrep: Run semgrep scan --config=auto for security rulesets (e.g., OWASP Top 10).
Snyk: Run snyk test for dependency vulnerabilities and snyk code test for code issues.
Endor Labs Socket: Run socket scan --security for supply chain risks.


Actions:

Scan the codebase and dependencies.
Prioritize high/critical issues (e.g., SQL injection, XSS, outdated libs).
Fix issues:

Update vulnerable dependencies (e.g., via npm update or pip install --upgrade).
Refactor insecure code patterns (e.g., use prepared statements for DB queries).
Apply patches or mitigations from tool recommendations.


Re-scan after fixes to verify resolution.
Commit changes with message: "Address security vulnerabilities from scans".



Step 3: Ensure Code Quality and Best Practices
Goal: Improve readability, maintainability, and compliance without introducing risks.

Tools to Use:

SonarQube: Review quality gates for duplication, coverage, and maintainability.
Semgrep: Use quality rulesets like semgrep scan --config=p/python-best-practices.


Actions:

Run quality scans.
Address issues: Refactor duplicated code, add comments, enforce style (e.g., via Prettier or Black).
Add LICENSE file (e.g., MIT) and README.md with project overview, setup instructions, and contribution guidelines.
Remove unnecessary files (e.g., logs, temp files).
Commit changes with message: "Improve code quality and add documentation".



Step 4: Final Review and Preparation

Manually review:

Search codebase for strings like "password", "key", "secret", "token".
Check for embedded configs or endpoints that could leak info.


Run all scans again to confirm zero high-risk issues.
Create a new branch: git checkout -b public-release.
Squash commits if needed for a clean history: git rebase -i HEAD~n.
Push to a private repo first for testing, then make public.

Post-Upload Recommendations

Enable GitHub security features: Dependabot, Secret Scanning.
Monitor for issues post-release.

Follow these steps sequentially. If issues arise, log them and seek clarification. This ensures a secure, professional public release.