#!/bin/bash
###############################################################################
# wexac_disconnect.sh — Clean up WEXAC GPU jobs, tunnels, and temp files
###############################################################################
set -uo pipefail

SSH_HOST="wexac"

cat << 'LOGO'

 __        _______ __  __    _     ____
 \ \      / / ____|\ \/ /   / \   / ___|
  \ \ /\ / /|  _|   \  /   / _ \ | |
   \ V  V / | |___  /  \  / ___ \| |___
    \_/\_/  |_____|/_/\_\/_/   \_\\____|
          Disconnect & Cleanup

LOGO

# ── Kill local SSH tunnels ──────────────────────────────────────────────────
echo "[1/3] Killing local SSH tunnels..."
pkill -f "ssh -f -N.*${SSH_HOST}" 2>/dev/null && echo "  Killed local tunnel(s)." || echo "  No local tunnels found."
pkill -f "ssh -N -L.*${SSH_HOST}" 2>/dev/null || true
# Free up local port
lsof -ti:8899 2>/dev/null | xargs kill 2>/dev/null || true

# ── Kill remote jobs and tunnels ────────────────────────────────────────────
echo "[2/3] Killing remote jobs and processes..."
ssh "$SSH_HOST" bash -lc '
    # Kill all LSF jobs
    bkill 0 2>/dev/null && echo "  Killed LSF jobs." || echo "  No LSF jobs to kill."

    # Kill any stray Jupyter processes
    pkill -u $USER jupyter 2>/dev/null && echo "  Killed Jupyter processes." || echo "  No Jupyter processes found."

    # Kill SSH tunnels on login2
    pkill -u $USER -f "ssh -N -L" 2>/dev/null && echo "  Killed login2 tunnels." || echo "  No login2 tunnels found."
    pkill -u $USER -f "ssh -f -N -L" 2>/dev/null || true
' 2>/dev/null || echo "  Could not reach $SSH_HOST (VPN disconnected?)."

# ── Clean up temp files ─────────────────────────────────────────────────────
echo "[3/3] Cleaning temp files..."
ssh "$SSH_HOST" "rm -f ~/wexac_jupyter_info.txt ~/wexac_jupyter_log.txt ~/wexac_gpu_jupyter.sh ~/wexac_bsub_out.txt ~/wexac_bsub_err.txt ~/wexac_gpu_mode.txt" 2>/dev/null || true
echo "  Done."

echo ""
echo "All clean. Reconnect with: ./wexac_connect.sh"
