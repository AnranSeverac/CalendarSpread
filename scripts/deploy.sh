#!/usr/bin/env bash
# Deploy CalendarSpread to a freshly-provisioned Ubuntu EC2 box.
#
# Usage:
#   bash scripts/deploy.sh <ec2-public-ip> </path/to/key.pem>
#
# Pushes the working copy (excluding .cache/, logs/, .git/, etc.), then runs
# the bootstrap on the remote box. Re-run any time after code edits.

set -euo pipefail

if [ $# -lt 2 ]; then
    echo "Usage: bash scripts/deploy.sh <ec2-public-ip> </path/to/key.pem>"
    exit 1
fi

EC2_IP="$1"
KEY_PATH="$2"
# Detect default user from the AMI: ec2-user (Amazon Linux), ubuntu (Ubuntu)
EC2_USER="${EC2_USER:-ec2-user}"
REMOTE_DIR="/home/${EC2_USER}/CalendarSpread"

if [ ! -f "$KEY_PATH" ]; then
    echo "ERROR: key file not found: $KEY_PATH"
    exit 1
fi
chmod 600 "$KEY_PATH"

# Ensure repo root
cd "$(dirname "$0")/.." || exit 1
ROOT="$(pwd)"
echo "Deploying from: $ROOT"
echo "          to:   ${EC2_USER}@${EC2_IP}:${REMOTE_DIR}"

SSH="ssh -i $KEY_PATH -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10"
RSYNC="rsync -avz --delete -e \"$SSH\""

# Make sure remote dir exists
$SSH "${EC2_USER}@${EC2_IP}" "mkdir -p ${REMOTE_DIR}"

# rsync everything except generated / local-only stuff
eval "$RSYNC \
    --exclude='.cache/' \
    --exclude='logs/' \
    --exclude='.git/' \
    --exclude='__pycache__/' \
    --exclude='.claude/' \
    --exclude='*.pyc' \
    --exclude='analytics/spread_output/' \
    \"$ROOT/\" \"${EC2_USER}@${EC2_IP}:${REMOTE_DIR}/\""

echo
echo "── Running bootstrap on remote ──"
$SSH "${EC2_USER}@${EC2_IP}" "cd ${REMOTE_DIR} && bash scripts/aws-bootstrap.sh"

cat <<EOF

Deploy complete.

Next steps (manual):
  ssh -i $KEY_PATH ${EC2_USER}@${EC2_IP}
  sudo systemctl start calendarspread
  sudo journalctl -u calendarspread -f
EOF
