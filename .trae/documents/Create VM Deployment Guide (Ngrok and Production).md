## Goal
Create a deployment guide folder with a single, structured markdown that provides two complete paths for deploying on a secured Ubuntu VM:
- Version A: Ngrok tunnel (no firewall/NAT changes required)
- Version B: Production‑grade reverse proxy with Caddy (TLS, firewall, Docker Compose)
Include base setup, port checks, environment file guidance (without secrets), WhatsApp webhook configuration, and reverse‑proxy troubleshooting.

## Deliverables
- New folder: `deploy/`
- New document: `deploy/VM_DEPLOYMENT_GUIDE.md` containing:
  - Prerequisites and security notes
  - Port visibility checks (`ss`, `ufw`, `iptables`, `lsof`)
  - Common base setup: Docker, Docker Compose plugin, folder layout
  - Version A (Ngrok): install/configure ngrok, run stack, expose webhook via ngrok URL, test
  - Version B (Production): firewall/DNS/TLS, compose, Caddy reverse proxy, health, metrics, webhook
  - Reverse proxy and access troubleshooting
  - Appendix: commands and verification steps

## Key Choices
- Use existing `docker-compose.yml` and `Caddyfile` patterns; recommend removing external DB port exposure for production.
- Keep secrets in `.env.hlas` on the VM; do not embed any secret values in the document.
- Provide exact command blocks suitable for Ubuntu SSH sessions.

## Implementation Steps
1. Add `deploy/VM_DEPLOYMENT_GUIDE.md` with detailed, copy‑pasteable commands for both paths.
2. Ensure the guide references repository files (`Dockerfile.hlas`, `docker-compose.yml`, `Caddyfile`) and instructs safe environment handling.
3. Include WhatsApp Cloud API webhook configuration instructions for both ngrok and production URLs.
4. Add troubleshooting guidance focused on closed ports, NAT, and binding addresses.

## Validation
- Commands verified for Ubuntu compatibility (apt, systemctl, ufw, ss).
- Health/ready checks via `curl` to confirm stack readiness.
- Meta webhook verification path `/meta-whatsapp` used consistently.

## Next
Upon approval, I will create the folder and add the guide file with all steps and commands. 