#!/bin/bash
set -e

APP_DIR="/opt/momentum-scanner"
SERVICE="momentum-scanner"
NGINX_PORT=83
NODE_PORT=3000

echo "======================================================"
echo " Momentum Scanner — Deployment Setup"
echo "======================================================"

# ── 1. Create app directory and extract tarball ───────────────────────────────
echo "[1/7] Extracting application..."
mkdir -p "$APP_DIR"
tar -xzf /root/momentum-scanner.tar.gz -C "$APP_DIR"
echo "      Extracted to $APP_DIR"

# ── 2. Node.js — install if missing ──────────────────────────────────────────
echo "[2/7] Checking Node.js..."
if ! command -v node &>/dev/null; then
  echo "      Installing Node.js 20..."
  curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
  apt-get install -y nodejs
fi
echo "      Node $(node -v)"

# ── 3. Python venv + dependencies ────────────────────────────────────────────
echo "[3/7] Setting up Python environment..."
apt-get install -y python3-venv python3-pip --quiet
python3 -m venv "$APP_DIR/venv"
"$APP_DIR/venv/bin/pip" install --upgrade pip --quiet
"$APP_DIR/venv/bin/pip" install -r "$APP_DIR/requirements.txt" --quiet
echo "      Python packages installed"

# Set PYTHON_CMD in .env to use venv
if [ -f "$APP_DIR/.env" ]; then
  sed -i 's|^PYTHON_CMD=.*|PYTHON_CMD='"$APP_DIR"'/venv/bin/python3|' "$APP_DIR/.env"
else
  # Create .env from .env.example if it doesn't exist
  cp "$APP_DIR/.env.example" "$APP_DIR/.env"
  sed -i 's|^PYTHON_CMD=.*|PYTHON_CMD='"$APP_DIR"'/venv/bin/python3|' "$APP_DIR/.env"
  echo "      .env created from .env.example — update TELEGRAM credentials!"
fi

# ── 4. npm install + React build ─────────────────────────────────────────────
echo "[4/7] Building React frontend..."
cd "$APP_DIR"
npm install --production=false --silent
npm run build
echo "      Build complete: dist/"

# ── 5. Create data directories ────────────────────────────────────────────────
echo "[5/7] Creating data directories..."
mkdir -p "$APP_DIR/scan_cache"
mkdir -p "$APP_DIR/resources"
# Note: resources/universe_master.csv must be uploaded manually once:
#   scp resources/universe_master.csv root@<server>:/opt/momentum-scanner/resources/
if [ ! -f "$APP_DIR/resources/universe_master.csv" ]; then
  echo "      ⚠️  resources/universe_master.csv not found — upload it before running Quality Universe scans"
fi

# ── 6. Systemd service ───────────────────────────────────────────────────────
echo "[6/7] Configuring systemd service..."
cat > /etc/systemd/system/${SERVICE}.service << EOF
[Unit]
Description=Momentum Scanner (Node.js + Express)
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=${APP_DIR}
EnvironmentFile=${APP_DIR}/.env
ExecStart=/usr/bin/node ${APP_DIR}/server.js
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=${SERVICE}

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable "$SERVICE"
systemctl restart "$SERVICE"
sleep 2
systemctl status "$SERVICE" --no-pager -l | head -20
echo "      Service started"

# ── 7. Nginx reverse proxy ────────────────────────────────────────────────────
echo "[7/7] Configuring Nginx..."
apt-get install -y nginx --quiet

cat > /etc/nginx/sites-available/${SERVICE} << EOF
server {
    listen ${NGINX_PORT};
    server_name _;

    # Serve React SPA
    location / {
        proxy_pass http://localhost:${NODE_PORT};
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host \$host;
        proxy_cache_bypass \$http_upgrade;
        proxy_read_timeout 600s;
    }
}
EOF

ln -sf /etc/nginx/sites-available/${SERVICE} /etc/nginx/sites-enabled/${SERVICE}
nginx -t && systemctl reload nginx
ufw allow "${NGINX_PORT}/tcp" &>/dev/null || true

echo ""
echo "======================================================"
echo " Deployment complete!"
echo " App:     http://$(hostname -I | awk '{print $1}'):${NGINX_PORT}"
echo " Logs:    journalctl -u ${SERVICE} -f"
echo " DB:      ${APP_DIR}/scan_history.db"
echo " Cron:    Daily scan fires at 4:30 PM IST (Mon-Fri)"
echo "======================================================"
