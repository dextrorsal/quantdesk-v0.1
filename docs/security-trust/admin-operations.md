# Admin Operations

QuantDesk includes an admin console for support teams and system operators. This page captures the guardrails and workflows before the console goes back into the spotlight.

## Access Model

- **Dedicated admin frontend** served on a separate route (Port 5173 at `/admin`)
- **Authentication**: SIWS (Solana In-App Web3 Signing) with wallet signature verification
- **Session management**: HTTP-only cookies with 7-day expiration for secure authentication
- **Role-based access** with clear tiers: read-only support, account operations, system admin, and super admin for emergency controls
- **Hardening roadmap**: MFA and VPN/IP allow-listing planned for production

## Controls Available Today

- Monitor user accounts, balances, and recent activity for support troubleshooting.
- Manage trading permissions (freeze/unfreeze accounts, reset sessions, reissue invites).
- View platform metrics and health dashboards without touching customer capital.
- Review audit logs that capture every admin action with timestamps and operator identity.

## Hardening Roadmap

- **Security Enhancements**: MFA enforcement, hardware key support, VPN requirement for production access.
- **Session Rules**: Short-lived sessions with continuous logging and anomaly detection for suspicious behavior.
- **Infrastructure Separation**: Segregated hosting and dedicated TLS certificates for admin surfaces.

## Emergency Playbooks

- **Emergency Stop** – Freeze trading, alert operators, and coordinate incident response when systemic issues arise.
- **Security Incident** – Disable admin accounts, rotate secrets, and audit logs to scope impact before reactivating access.

Admins operate under the principle of least privilege. Pair this reference with the broader [Security & Trust](./security-and-trust.md) and [Security Posture](./security-posture.md) docs when outlining operational readiness to auditors or partners.
