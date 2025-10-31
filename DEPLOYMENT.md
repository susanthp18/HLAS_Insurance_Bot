HLAS Insurance Chatbot – Docker Deployment (HTTP via Caddy, no TLS)

Overview
- HLAS API behind Caddy on port 80 (HTTP only)
- Redis, MongoDB, Weaviate as dependencies
- Deliver prebuilt Docker images to avoid sharing source code
- WhatsApp webhook routes proxied by Caddy to HLAS API

Repo files you need
- Dockerfile.hlas
- docker-compose.yml
- Caddyfile
- deploy/env.hlas (copy to VM as .env.hlas)

Build images on your dev machine (offline-friendly)
```bash
# from repo root
docker build -f Dockerfile.hlas -t hlas-api:1.0.0 .

mkdir -p dist
docker save -o dist/hlas-api_1.0.0.tar hlas-api:1.0.0
docker pull redis:7.2-alpine && docker save -o dist/redis_7.2-alpine.tar redis:7.2-alpine
docker pull mongo:7 && docker save -o dist/mongo_7.tar mongo:7
docker pull semitechnologies/weaviate:1.25.5 && docker save -o dist/weaviate_1.25.5.tar semitechnologies/weaviate:1.25.5
docker pull caddy:2.8 && docker save -o dist/caddy_2.8.tar caddy:2.8
```

Transfer to the client VM
- Copy files to the VM: docker-compose.yml, Caddyfile, deploy/env.hlas, and all dist/*.tar
- Use scp, sftp, or a secure file transfer method approved by client

Prepare the VM
```bash
sudo mkdir -p /opt/hlas/images
cd /opt/hlas

# Place files here:
#  - docker-compose.yml
#  - Caddyfile
#  - env.hlas (rename to .env.hlas)
#  - images/*.tar

sudo mv env.hlas .env.hlas
for f in images/*.tar; do sudo docker load -i "$f"; done

docker compose version
sudo docker compose up -d

sudo docker compose ps
sudo docker logs -f hlas-api

curl -fsS http://127.0.0.1/health
curl -fsS http://127.0.0.1/ready
```

Configure WhatsApp Cloud API (Meta)
- Webhook URL: http://<VM_PUBLIC_IP>/meta-whatsapp
- Verification token: kikibiki (GET /meta-whatsapp)
- Incoming messages: POST /meta-whatsapp

Operations
```bash
# Restart all services
sudo docker compose restart

# Tail application logs
sudo docker logs -f hlas-api

# Update only the app image in-place
sudo docker load -i images/hlas-api_1.0.1.tar
sudo docker compose up -d hlas-api
```

Security notes
- This deployment is HTTP-only per request. If later needed, enable HTTPS in Caddy.
- Mongo and Weaviate ports are exposed in compose for convenience; remove the ports mappings to restrict to internal access.
- Keep .env.hlas secret; it includes API keys.


