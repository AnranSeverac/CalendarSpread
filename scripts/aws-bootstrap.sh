#!/usr/bin/env bash
# CalendarSpread EC2 bootstrap. Works on Ubuntu 22.04/24.04 (apt) and
# Amazon Linux 2023 (dnf). Auto-detects user (ubuntu / ec2-user).
#
# Idempotent — safe to re-run after code updates.

set -euo pipefail

# Auto-detect target user + project dir
if [ -d /home/ec2-user ]; then
    DEFAULT_USER="ec2-user"
elif [ -d /home/ubuntu ]; then
    DEFAULT_USER="ubuntu"
else
    DEFAULT_USER="$(whoami)"
fi
APP_USER="${APP_USER:-$DEFAULT_USER}"
APP_DIR="/home/${APP_USER}/CalendarSpread"
SERVICE_NAME="calendarspread"

cd "$APP_DIR" || { echo "ERROR: expected repo at $APP_DIR"; exit 1; }

echo "── 1/5  install base packages (auto-detect distro) ──"
if command -v dnf >/dev/null 2>&1; then
    # Amazon Linux 2023 / RHEL family
    sudo dnf install -y python3.11 python3.11-pip git
    PY="python3.11"
elif command -v apt-get >/dev/null 2>&1; then
    sudo DEBIAN_FRONTEND=noninteractive apt-get update -qq
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
        python3 python3-pip python3-venv git curl ca-certificates
    PY="python3"
else
    echo "ERROR: unsupported package manager"
    exit 1
fi

echo "── 2/5  Python venv ──"
if [ ! -d ".venv" ]; then
    "$PY" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt
# Install the v2 client (not on pypi as part of requirements.txt by default)
pip install --quiet py_clob_client_v2

echo "── 3/5  config/.env check ──"
if [ ! -f "config/.env" ]; then
    echo "ERROR: config/.env missing. Must be deployed alongside the code."
    exit 1
fi
if ! grep -q "^POLYMARKET_PRIVATE_KEY=0x" config/.env; then
    echo "ERROR: POLYMARKET_PRIVATE_KEY not populated in config/.env"
    exit 1
fi
if ! grep -q "^POLYMARKET_FUNDER_ADDRESS=0x" config/.env; then
    echo "ERROR: POLYMARKET_FUNDER_ADDRESS not populated in config/.env"
    exit 1
fi
chmod 600 config/.env
echo "  config/.env present and populated ✓"

echo "── 4/5  systemd service (user=$APP_USER, dir=$APP_DIR) ──"
sed -e "s|/home/ubuntu|/home/${APP_USER}|g" \
    -e "s|User=ubuntu|User=${APP_USER}|" \
    scripts/calendarspread.service > /tmp/${SERVICE_NAME}.service
sudo mv /tmp/${SERVICE_NAME}.service /etc/systemd/system/${SERVICE_NAME}.service
sudo systemctl daemon-reload
sudo systemctl enable ${SERVICE_NAME}.service > /dev/null 2>&1 || true
echo "  service installed (enabled, not started)"

echo "── 5/5  dry-run smoke test ──"
python3 live_execution.py 2>&1 | tail -12 || true

cat <<EOF

Bootstrap complete.

To start the live trading loop:
    sudo systemctl start ${SERVICE_NAME}
To watch logs:
    sudo journalctl -u ${SERVICE_NAME} -f
To stop:
    sudo systemctl stop ${SERVICE_NAME}
To check status:
    sudo systemctl status ${SERVICE_NAME}

Local state lives in: ${APP_DIR}/logs/
EOF
