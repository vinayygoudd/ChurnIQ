#!/usr/bin/env bash
set -euo pipefail

APP_DIR="/opt/ChurnIQ"
REPO_URL="${REPO_URL:-https://github.com/vinayygoudd/ChurnIQ.git}"

sudo apt-get update
sudo apt-get install -y python3 python3-venv python3-pip git nginx

if [ ! -d "$APP_DIR/.git" ]; then
    sudo git clone "$REPO_URL" "$APP_DIR"
else
    cd "$APP_DIR"
    sudo git pull --ff-only
fi

sudo chown -R ubuntu:ubuntu "$APP_DIR"
cd "$APP_DIR"

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

sudo cp deploy/aws/churniq.service /etc/systemd/system/churniq.service
sudo cp deploy/aws/nginx.conf /etc/nginx/sites-available/churniq
sudo ln -sf /etc/nginx/sites-available/churniq /etc/nginx/sites-enabled/churniq
sudo rm -f /etc/nginx/sites-enabled/default

sudo nginx -t
sudo systemctl daemon-reload
sudo systemctl enable --now churniq
sudo systemctl enable --now nginx

echo "ChurnIQ API is starting on port 80 via nginx."
