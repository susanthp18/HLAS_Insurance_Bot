# HLAS Insurance Bot — Ubuntu VM Deployment Guide

## Overview
- Two deployment options:
  - Version A: Ngrok tunnel for secure external access without opening VM ports
  - Version B: Production-grade reverse proxy (Caddy), Docker Compose, firewall rules, optional TLS
- Assumptions:
  - You access a secured Ubuntu VM via VPN and SSH (username/password)
  - Repository files: `Dockerfile.hlas`, `docker-compose.yml`, `Caddyfile`, `hlas/src/hlas/*`
  - Secrets live only in `.env.hlas` on the VM (do not commit)

## Prerequisites
- VPN connected to the client network
- SSH access to the Ubuntu VM
- A non-root user with `sudo` privileges
- Domain name (production path) or Ngrok account (ngrok path)

## Step 0 — Port and Firewall Visibility Checks
- On the VM, check which services listen locally:
  - `sudo ss -tulpen`
  - `sudo lsof -i -P -n | grep LISTEN`
- Check Ubuntu firewall:
  - `sudo ufw status verbose`
- If `ufw` is inactive, check iptables:
  - `sudo iptables -L -n -v`
- From an external machine (if allowed), test ingress with `nmap`:
  - `nmap -Pn -p 80,443 <VM_PUBLIC_IP>`

Notes:
- If ports 80/443 are closed externally, a reverse proxy alone won’t make your app reachable; open the ports and ensure upstream network/NAT allows ingress, or use Ngrok (Version A).

## Step 1 — Base System Setup (Common for A and B)
- Update and install dependencies:
  - `sudo apt-get update && sudo apt-get upgrade -y`
  - `sudo apt-get install -y ca-certificates curl gnupg lsb-release unzip`
- Install Docker Engine:
  - `curl -fsSL https://get.docker.com | sudo sh`
  - `sudo systemctl enable docker && sudo systemctl start docker`
- Install Docker Compose plugin:
  - `sudo apt-get install -y docker-compose-plugin`
  - `docker compose version`
- Add your user to `docker` group (optional):
  - `sudo usermod -aG docker $USER && newgrp docker`

## Step 2 — Project Layout on the VM
- Create working directory:
  - `mkdir -p ~/hlas && cd ~/hlas`
- Copy the repository contents into `~/hlas` (via `scp`, `sftp`, or Git clone if allowed).
- Verify key files:
  - `ls Dockerfile.hlas docker-compose.yml Caddyfile`

## Step 3 — Environment File (Never commit secrets)
- Create `.env.hlas` in `~/hlas` with required variables. Do not paste secrets here in the document.
- Use the values your client provided; examples (placeholders only):
  - LLM provider selection (`LLM_PROVIDER=gpt` or Grok/Azure as needed)
  - Redis URL (`REDIS_URL=redis://redis:6379/0` for compose)
  - Mongo URI internal (`MONGO_URI=mongodb://<user>:<password>@mongo:27017/?authSource=admin`)
  - Weaviate URL (`WEAVIATE_URL=http://weaviate:8080`)
  - WhatsApp Cloud API (`META_ACCESS_TOKEN`, `META_PHONE_NUMBER_ID`, `META_VERIFY_TOKEN`)
  - Set `DEBUG=false` for production

Command:
- `nano .env.hlas` (or `vim .env.hlas`)

## Version A — Ngrok Tunnel (No firewall/NAT changes)
### A1. Install and Configure Ngrok
- Download and install:
  - `curl -sSL https://ngrok-agent.s3.amazonaws.com/ngrok.asc | sudo tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null`
  - `echo "deb https://ngrok-agent.s3.amazonaws.com buster main" | sudo tee /etc/apt/sources.list.d/ngrok.list`
  - `sudo apt-get update && sudo apt-get install -y ngrok`
- Add your auth token:
  - `ngrok config add-authtoken <YOUR_NGROK_AUTHTOKEN>`

### A2. Run the Stack Locally
- From `~/hlas`:
  - `docker compose up -d`
- Verify health from the VM:
  - `curl -fsS http://localhost/health`
  - `curl -fsS http://localhost/ready`

### A3. Start Ngrok Tunnel
- Tunnel HTTP port 80 (Caddy):
  - `ngrok http 80`
- You will get a URL like `https://<random>.ngrok.io`.

### A4. Configure WhatsApp Cloud API Webhook
- Webhook URL: `https://<ngrok-url>/meta-whatsapp`
- Verification token: set to your `META_VERIFY_TOKEN`
- Test verification: WhatsApp will call `GET /meta-whatsapp` and expect your token.

### A5. External Testing
- Share `https://<ngrok-url>/ready` and `/meta-whatsapp` with external testers.
- If access works via Ngrok but not via public IP, your VM/network blocks inbound traffic; proceed with Version B when you can open ports 80/443.

### A6. Optional Persistent Ngrok Tunnel
- Use reserved domains (paid) or run as a systemd service for stability.

## Version B — Production‑Grade (Caddy + TLS + Firewall)
### B1. DNS and Certificates
- Point your domain’s A record to the VM’s public IP.
- Ensure ports 80 and 443 are open in:
  - VM firewall (`ufw`) and
  - Upstream network/security groups (VPN gateway, cloud provider, etc.)

Commands:
- `sudo ufw allow 80/tcp`
- `sudo ufw allow 443/tcp`
- `sudo ufw reload`

### B2. Harden docker-compose.yml
- Use internal networking for DBs (no external port exposure). In your compose file:
  - Remove/comment the `ports:` entries for `mongo` (`27017:27017`) and `weaviate` (`8080:8080`) to keep them internal-only.
- Ensure `hlas-api` uses internal service hosts:
  - `REDIS_URL=redis://redis:6379/0`
- Ensure `WEAVIATE_URL` uses internal service host:
  - `WEAVIATE_URL=http://weaviate:8080`
- Ensure `MONGO_URI` uses internal service host:
  - `MONGO_URI=mongodb://<user>:<password>@mongo:27017/?authSource=admin`

### B3. Caddyfile for TLS
- Minimal HTTPS Caddyfile example:
```
https://your-domain.com {
  encode gzip zstd
  log
  handle {
    reverse_proxy http://hlas-api:8000
  }
}
```
- Replace `Caddyfile` content accordingly and ensure DNS resolves.

### B4. Start the Stack
- `docker compose up -d`
- Health checks:
  - `curl -fsS http://localhost/health`
  - `curl -fsS http://localhost/ready`
- External checks:
  - `curl -fsS https://your-domain.com/health`
  - `curl -fsS https://your-domain.com/ready`

### B5. Configure WhatsApp Webhook
- Webhook URL: `https://your-domain.com/meta-whatsapp`
- Verification token: `META_VERIFY_TOKEN`
- Test endpoints: `GET /meta-whatsapp` (verification) and `POST /meta-whatsapp` (message events)

### B6. Observability and Logs
- App logs: `docker logs -f hlas-api`
- Caddy logs: inside container or configure a volume
- Prometheus metrics: `https://your-domain.com/metrics`

### B7. Security Recommendations
- Set `DEBUG=false`
- Keep `.env.hlas` readable only by required users (`chmod 600 .env.hlas`)
- Do not expose Mongo/Weaviate ports externally
- Consider Weaviate auth if it must be exposed (`AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=false`)
- Rotate API keys periodically

## Reverse Proxy & Access Troubleshooting
- Friend cannot access but you can:
  - Likely cause: closed inbound ports (80/443) or NAT rules blocking external access
  - Actions:
    - Confirm service binds to `0.0.0.0` (Caddy listens globally inside container)
    - Open `ufw` ports and upstream network rules
    - Validate with `curl` externally and `nmap -Pn -p 80,443 <VM_PUBLIC_IP>`
- 502/504 from Caddy:
  - Check `docker ps` (containers running)
  - Check `docker logs -f hlas-api` and `docker logs -f hlas-caddy`
  - Verify `reverse_proxy http://hlas-api:8000` matches service name and port
- WhatsApp webhook failing verification:
  - Ensure correct URL and token
  - Confirm inbound reachability (Ngrok or open ports)
  - Check app logs for `/meta-whatsapp` handlers

## Verification Checklist
- Base: Docker and Compose installed, repo present, `.env.hlas` configured
- Version A: Ngrok up, external URL returns `/ready`, WhatsApp webhook verified
- Version B: DNS resolves, TLS enabled, ports open, external `/ready` OK, webhook verified

## Useful Commands
- Compose lifecycle:
  - `docker compose up -d`
  - `docker compose ps`
  - `docker compose restart`
- Logs:
  - `docker logs -f hlas-api`
- Health:
  - `curl -fsS http://localhost/ready`
  - `curl -fsS https://your-domain.com/ready`

## Notes on Environment Keys
- Paste your actual keys into `.env.hlas` only on the VM
- Avoid putting secrets in terminal history; use editors (`nano`) or `read -s` inputs
- Set `DEBUG=false` before production rollout