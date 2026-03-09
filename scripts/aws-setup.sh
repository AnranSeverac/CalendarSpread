#!/bin/bash
# Run this on a fresh Amazon Linux 2 or Ubuntu EC2 instance (after SSH in)
# to install the app from GitHub and prepare for running the live loop.

set -e
REPO="${REPO:-https://github.com/AnranSeverac/polymarketTermStructure.git}"
DIR="${DIR:-polymarketTermStructure}"

echo "Installing system deps..."
if command -v yum &>/dev/null; then
  sudo yum update -y
  sudo yum install -y python3 python3-pip git
else
  sudo apt-get update -y
  sudo apt-get install -y python3 python3-pip python3-venv git
fi

echo "Cloning repo..."
cd "$HOME"
if [ -d "$DIR" ]; then
  cd "$DIR" && git pull && cd ..
else
  git clone "$REPO" "$DIR"
fi
cd "$DIR"

echo "Creating venv and installing Python deps..."
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo "Checking config/.env..."
if [ ! -f config/.env ]; then
  echo "  WARNING: config/.env not found. Copy config/.env.example and add your keys:"
  echo "    cp config/.env.example config/.env"
  echo "    nano config/.env"
else
  echo "  config/.env exists."
fi

echo "Done. To run the live loop (after editing config/.env):"
echo "  cd $HOME/$DIR"
echo "  source .venv/bin/activate"
echo "  nohup python3 live_execution.py --execute-live --loop-seconds 300 > live.log 2>&1 &"
echo "  tail -f live.log"
