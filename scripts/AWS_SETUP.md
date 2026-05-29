# AWS deployment — one-time setup

Provisions a tiny Ubuntu box in an unblocked region, installs CalendarSpread,
and starts the live trading loop as a systemd service that restarts on crash.

Total cost: ~$4/month on `t3.nano` reserved, ~$8/mo on-demand.

## What you do once, in the AWS console

### Step 1 — Pick a region in the top-right region selector

Use a region where Polymarket allows trading. Verified working:
- **`eu-west-1` (Ireland)** — recommended
- `eu-central-1` (Frankfurt)
- `ap-southeast-1` (Singapore)

Avoid `us-*` regions — the geofilter will block orders.

### Step 2 — Create an SSH key pair (if you don't have one yet)

- EC2 → Key Pairs → Create key pair → name it `calendarspread`, type `RSA`, format `.pem`.
- Download the `.pem` file. Move it to `~/.ssh/calendarspread.pem` and run:
  ```bash
  chmod 600 ~/.ssh/calendarspread.pem
  ```

### Step 3 — Launch an EC2 instance

EC2 → Launch instance:

| Setting | Value |
|---|---|
| Name | `calendarspread` |
| AMI | **Ubuntu Server 24.04 LTS** (x86-64) |
| Instance type | `t3.nano` or `t3.micro` |
| Key pair | `calendarspread` (from step 2) |
| Network → Security group | Create new, allow `SSH (22)` from **my IP** |
| Storage | 8 GB gp3 (default) |

Launch. Wait for status to be "Running". Copy the **public IPv4 address** —
you'll need it.

### Step 4 — From your Mac, deploy

```bash
cd /Users/AnranSeverac/CalendarSpread
bash scripts/deploy.sh <PUBLIC_IP> ~/.ssh/calendarspread.pem
```

This rsyncs the repo to the box (skipping caches/logs/.git) and runs
`scripts/aws-bootstrap.sh` remotely. The bootstrap installs Python deps,
creates a `.venv`, validates `config/.env`, installs the systemd service.

If your local `config/.env` has placeholder values, the bootstrap will refuse;
fix it and re-deploy.

### Step 5 — Start the service

SSH in once to start it (so you can confirm it's working before walking away):

```bash
ssh -i ~/.ssh/calendarspread.pem ubuntu@<PUBLIC_IP>
sudo systemctl start calendarspread
sudo journalctl -u calendarspread -f
```

Press `Ctrl-C` to stop following logs. The service keeps running.

## Operations

| | command |
|---|---|
| Watch logs live | `sudo journalctl -u calendarspread -f` |
| Last 100 lines | `sudo journalctl -u calendarspread -n 100 --no-pager` |
| Stop trading | `sudo systemctl stop calendarspread` |
| Restart | `sudo systemctl restart calendarspread` |
| Disable autostart on reboot | `sudo systemctl disable calendarspread` |
| Check open positions | `cat ~/CalendarSpread/logs/positions.json` |

## Update the code later

After editing on your Mac:

```bash
# from your Mac
bash scripts/deploy.sh <PUBLIC_IP> ~/.ssh/calendarspread.pem
ssh -i ~/.ssh/calendarspread.pem ubuntu@<PUBLIC_IP> "sudo systemctl restart calendarspread"
```

The deploy is idempotent.

## Cost & safety

- Stop the instance when you're not using it (you keep the EBS volume, no compute charges):
  EC2 → Instances → select → Instance state → Stop.
- Terminate it to fully delete (loses the data on the box):
  Instance state → Terminate.
- Set up an AWS billing alert at $20/mo if you want a safety net.

## What you need to give me (none — but checklist)

- [x] `config/.env` already populated with your private key + funder address on your Mac.
- [x] An AWS account with billing enabled.
- [x] An SSH key pair generated and downloaded.
- [x] An EC2 instance running in `eu-west-1` (or similar).

The deploy command does the rest.
