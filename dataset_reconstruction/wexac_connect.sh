#!/bin/bash
###############################################################################
# wexac_connect.sh — WEXAC GPU + Jupyter + SSH tunnel automation
#
# Follows the documented WEXAC GPU Jupyter workflow:
#   1. SSH to wexac (login2), clean previous jobs
#   2. Request GPU node via bsub -q interactive-gpu
#   3. On GPU node: activate conda, launch JupyterLab
#   4. Tunnel login2 → GPU node (ssh -N -L ... lgnXX)
#   5. Tunnel Mac → login2   (ssh -N -L ... wexac)
#   6. Open in VS Code
#
# Usage:  ./wexac_connect.sh            # Jupyter mode (default)
#         ./wexac_connect.sh shell      # Interactive GPU shell for experiments
# Stop:   ./wexac_disconnect.sh
###############################################################################
set -euo pipefail

# ── Parse arguments ──────────────────────────────────────────────────────────
MODE="${1:-jupyter}"   # "jupyter" (default) or "shell"
if [[ "$MODE" != "jupyter" && "$MODE" != "shell" ]]; then
    echo "Usage: ./wexac_connect.sh [jupyter|shell]"
    echo "  jupyter  — launch JupyterLab on GPU (default)"
    echo "  shell    — interactive GPU shell for running experiments"
    exit 1
fi

# ── Configuration ────────────────────────────────────────────────────────────
SSH_HOST="wexac"                       # must match ~/.ssh/config
LOCAL_PORT=8899                        # port on your Mac
CONDA_ENV="/home/projects/galvardi/yoado/.conda/envs/rec"
GPU_QUEUE="interactive-gpu"
GPU_MEM="8GB"
GPU_GMEM="8GB"
POLL_INTERVAL=5                        # seconds between checks
MAX_WAIT=300                           # max seconds to wait for Jupyter
JUPYTER_TOKEN="abcdef2026abcdef"       # fixed token for easy reconnect
INFO_FILE="~/wexac_jupyter_info.txt"   # coordination file on shared filesystem
LOG_FILE="~/wexac_jupyter_log.txt"     # Jupyter + debug log
GPU_SCRIPT="~/wexac_gpu_jupyter.sh"    # script that runs on the GPU node
# ─────────────────────────────────────────────────────────────────────────────

cleanup_on_exit() {
    echo ""
    echo "Interrupted. Run ./wexac_disconnect.sh to clean up tunnels and jobs."
}
trap cleanup_on_exit INT TERM

cat << 'LOGO'

 __        _______ __  __    _     ____
 \ \      / / ____|\ \/ /   / \   / ___|
  \ \ /\ / /|  _|   \  /   / _ \ | |
   \ V  V / | |___  /  \  / ___ \| |___
    \_/\_/  |_____|/_/\_\/_/   \_\\____|
       GPU Jupyter Connector v2.0

LOGO

# ── VPN Check ────────────────────────────────────────────────────────────────
echo "*** REMINDER: Make sure you are connected to the Weizmann VPN! ***"
echo ""
if ! ssh -o ConnectTimeout=5 -o BatchMode=yes "$SSH_HOST" "echo ok" &>/dev/null; then
    echo "ERROR: Cannot reach WEXAC. Are you connected to the Weizmann VPN?"
    exit 1
fi
echo "VPN check passed (WEXAC reachable)."
echo ""

# ── Step 1: Clean previous jobs (PDF Step 0 — MANDATORY) ────────────────────
echo "[1/6] Cleaning previous jobs and stale processes..."

# Kill local SSH tunnels FIRST (before touching remote, to avoid breaking our own SSH)
pkill -f "ssh -f -N.*${SSH_HOST}" 2>/dev/null || true
pkill -f "ssh -N -L.*${SSH_HOST}" 2>/dev/null || true

# Free up local port if something is lingering
if lsof -nP -iTCP:${LOCAL_PORT} -sTCP:LISTEN >/dev/null 2>&1; then
    echo "  Freeing local port ${LOCAL_PORT}..."
    lsof -ti:${LOCAL_PORT} | xargs kill 2>/dev/null || true
    sleep 1
fi

# Kill stale SSH tunnels on login2 (MUST be separate SSH call — pkill can kill our session)
ssh "$SSH_HOST" "pkill -u \$USER -f 'ssh.*-N.*-L.*127.0.0.1' 2>/dev/null; true" 2>/dev/null || true
sleep 1

# Now clean jobs and Jupyter (separate SSH call since previous one may have died)
ssh "$SSH_HOST" bash -lc '
    echo "  Current jobs:"
    bjobs -u $USER 2>&1 || echo "  (no jobs)"
    bkill 0 2>/dev/null || true
    pkill -u $USER jupyter 2>/dev/null || true
' 2>/dev/null || true

# Clean coordination files from previous run
ssh "$SSH_HOST" "rm -f $INFO_FILE $LOG_FILE $GPU_SCRIPT" 2>/dev/null || true

echo "  Done."
echo ""

# ── Step 2: Upload GPU node script (PDF Steps 2-3 / 4-5) ────────────────────
echo "[2/6] Uploading GPU helper script..."

# This script runs ON THE GPU NODE after bsub allocates it.
# Uses #!/bin/bash -l (login shell) so the module system is available.
ssh "$SSH_HOST" "cat > $GPU_SCRIPT" << 'GPU_EOF'
#!/bin/bash -l
# ── Runs on the GPU node via bsub ──

# Fallback: if module system not available from login shell, source it explicitly
if ! type module &>/dev/null; then
    for f in /etc/profile.d/modules.sh /etc/profile.d/lmod.sh /etc/profile; do
        [ -f "$f" ] && source "$f" 2>/dev/null && break
    done
fi

# Record GPU hostname
GPU_NODE=$(hostname)
echo "GPU_HOST=$GPU_NODE" > ~/wexac_jupyter_info.txt
echo "--- GPU node: $GPU_NODE ---" > ~/wexac_jupyter_log.txt

# Verify GPU is accessible (critical check)
echo "--- nvidia-smi ---" >> ~/wexac_jupyter_log.txt
if nvidia-smi >> ~/wexac_jupyter_log.txt 2>&1; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    echo "GPU_NAME=$GPU_NAME" >> ~/wexac_jupyter_info.txt
    echo "GPU_OK=yes" >> ~/wexac_jupyter_info.txt
else
    echo "GPU_OK=no" >> ~/wexac_jupyter_info.txt
    echo "ERROR: nvidia-smi failed on $GPU_NODE" >> ~/wexac_jupyter_log.txt
fi

# Activate conda environment (PDF Step 2 / Step 4)
echo "--- Activating conda ---" >> ~/wexac_jupyter_log.txt
module load miniconda >> ~/wexac_jupyter_log.txt 2>&1
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec >> ~/wexac_jupyter_log.txt 2>&1

# Log which python and environment we're using
echo "Python: $(which python)" >> ~/wexac_jupyter_log.txt
echo "Conda env: $(conda info --envs 2>/dev/null | grep '*' || echo 'unknown')" >> ~/wexac_jupyter_log.txt
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')" >> ~/wexac_jupyter_log.txt 2>&1 || true

# Check which mode was requested
RUN_MODE=$(cat ~/wexac_gpu_mode.txt 2>/dev/null || echo "jupyter")

if [ "$RUN_MODE" = "shell" ]; then
    # ── Shell mode: no Jupyter, just mark ready and keep alive ──
    echo "READY=true" >> ~/wexac_jupyter_info.txt
    echo "Shell mode — GPU session ready on $GPU_NODE" >> ~/wexac_jupyter_log.txt
    # Keep the bsub job alive until killed
    sleep infinity
else
    # ── Jupyter mode: original behavior (unchanged) ──

    # Find a free port starting from 8899 (PDF Step 3 / Step 5)
    PORT=8899
    while lsof -nP -iTCP:$PORT -sTCP:LISTEN >/dev/null 2>&1; do
        PORT=$((PORT+1))
    done
    echo "PORT=$PORT" >> ~/wexac_jupyter_info.txt
    echo "Using port $PORT on $GPU_NODE" >> ~/wexac_jupyter_log.txt

    # Launch JupyterLab (PDF Step 3 / Step 5)
    echo "--- Starting Jupyter ---" >> ~/wexac_jupyter_log.txt
    jupyter lab \
        --no-browser \
        --ip=127.0.0.1 \
        --port=$PORT \
        --ServerApp.port_retries=0 \
        --ServerApp.token=abcdef2026abcdef \
        >> ~/wexac_jupyter_log.txt 2>&1 &
    JPID=$!

    # Wait for Jupyter to become ready (up to 120s)
    for i in $(seq 1 60); do
        if curl -s -o /dev/null "http://127.0.0.1:$PORT/api" 2>/dev/null; then
            echo "READY=true" >> ~/wexac_jupyter_info.txt
            echo "Jupyter is ready on $GPU_NODE:$PORT" >> ~/wexac_jupyter_log.txt
            break
        fi
        if ! kill -0 $JPID 2>/dev/null; then
            echo "READY=failed" >> ~/wexac_jupyter_info.txt
            echo "ERROR: Jupyter process exited unexpectedly" >> ~/wexac_jupyter_log.txt
            exit 1
        fi
        sleep 2
    done

    # Keep the batch job alive as long as Jupyter is running
    wait $JPID
fi
GPU_EOF

ssh "$SSH_HOST" "chmod +x $GPU_SCRIPT"
echo "  Done."
echo ""

# ── Step 2.5: Write mode file so the GPU script knows what to do ─────────────
ssh "$SSH_HOST" "echo '$MODE' > ~/wexac_gpu_mode.txt"

# ── Step 3: Request GPU node (PDF Step 1 / Step 3) ──────────────────────────
echo "[3/6] Requesting GPU node (queue: $GPU_QUEUE)..."
JOB_OUTPUT=$(ssh "$SSH_HOST" bash -lc "
    bsub -q $GPU_QUEUE \
         -R 'rusage[mem=$GPU_MEM]' \
         -gpu 'num=1:j_exclusive=no:gmem=$GPU_GMEM' \
         -o ~/wexac_bsub_out.txt -e ~/wexac_bsub_err.txt \
         $GPU_SCRIPT
" 2>&1)
echo "  $JOB_OUTPUT"

JOB_ID=$(echo "$JOB_OUTPUT" | grep -o 'Job <[0-9]*>' | grep -o '[0-9]*' || echo "")
if [ -z "$JOB_ID" ]; then
    echo "ERROR: Failed to submit GPU job. Output was:"
    echo "  $JOB_OUTPUT"
    exit 1
fi
echo "  Job ID: $JOB_ID"
echo ""

# ── Step 4: Wait for GPU allocation and startup ─────────────────────────────
if [ "$MODE" = "shell" ]; then
    echo "[4/6] Waiting for GPU allocation..."
else
    echo "[4/6] Waiting for GPU allocation and Jupyter startup..."
fi
ELAPSED=0
GPU_HOST=""
while [ "$ELAPSED" -lt "$MAX_WAIT" ]; do
    INFO=$(ssh "$SSH_HOST" "cat $INFO_FILE 2>/dev/null" 2>/dev/null || echo "")

    # Check if Jupyter is ready
    if echo "$INFO" | grep -q "^READY=true"; then
        break
    fi

    # Check if Jupyter failed
    if echo "$INFO" | grep -q "^READY=failed"; then
        echo ""
        echo "ERROR: Jupyter failed to start on the GPU node."
        echo "  Debug log:"
        ssh "$SSH_HOST" "cat $LOG_FILE" 2>/dev/null || true
        exit 1
    fi

    # Show GPU allocation status
    if echo "$INFO" | grep -q "^GPU_HOST=" && [ -z "$GPU_HOST" ]; then
        GPU_HOST=$(echo "$INFO" | grep "^GPU_HOST=" | cut -d= -f2)
        GPU_OK=$(echo "$INFO" | grep "^GPU_OK=" | cut -d= -f2 || echo "")
        echo ""
        echo "  GPU node allocated: $GPU_HOST"
        if [ "$GPU_OK" = "yes" ]; then
            GPU_NAME=$(echo "$INFO" | grep "^GPU_NAME=" | cut -d= -f2 || echo "unknown")
            echo "  GPU verified: $GPU_NAME"
        elif [ "$GPU_OK" = "no" ]; then
            echo "  WARNING: nvidia-smi failed on $GPU_HOST!"
            echo "  The GPU may not be accessible. Check the log after connection."
        fi
        if [ "$MODE" = "shell" ]; then
            echo "  Waiting for GPU session to be ready..."
        else
            echo "  Waiting for Jupyter to start..."
        fi
    else
        printf "\r  Waiting for GPU allocation... (%ds / %ds)" "$ELAPSED" "$MAX_WAIT"
    fi

    sleep "$POLL_INTERVAL"
    ELAPSED=$((ELAPSED + POLL_INTERVAL))
done
echo ""

if [ "$ELAPSED" -ge "$MAX_WAIT" ]; then
    echo "ERROR: Timed out after ${MAX_WAIT}s."
    echo "  Check jobs:  ssh wexac bjobs"
    echo "  Check log:   ssh wexac 'cat $LOG_FILE'"
    echo "  bsub stderr: ssh wexac 'cat ~/wexac_bsub_err.txt'"
    exit 1
fi

# Parse connection info
GPU_HOST=$(echo "$INFO" | grep "^GPU_HOST=" | cut -d= -f2)

if [ "$MODE" = "shell" ]; then
    # ── Shell mode: skip Jupyter tunnels, open interactive SSH to GPU node ────
    echo "  GPU node : $GPU_HOST"
    echo ""
    echo "================================================================"
    echo "  Interactive GPU shell on $GPU_HOST"
    echo "  Conda env 'rec' will be pre-activated."
    echo "  When done: type 'exit', then run ./wexac_disconnect.sh"
    echo "================================================================"
    echo ""
    echo "[5/6] Connecting to GPU node..."
    echo "[6/6] Opening interactive shell..."
    echo ""
    # SSH through login2 to GPU node, activate conda, drop into bash
    ssh -J "$SSH_HOST" "$GPU_HOST" -t "bash -lc '
        module load miniconda 2>/dev/null
        source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
        conda activate /home/projects/galvardi/yoado/.conda/envs/rec
        echo \"\"
        echo \"GPU shell ready. conda env: rec\"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
        echo \"\"
        exec bash
    '"
else
    # ── Jupyter mode: original behavior (unchanged) ───────────────────────────
    REMOTE_PORT=$(echo "$INFO" | grep "^PORT=" | head -1 | cut -d= -f2)

    echo "  GPU node : $GPU_HOST"
    echo "  Jupyter  : 127.0.0.1:$REMOTE_PORT on $GPU_HOST"
    echo ""

    # ── Step 5: Set up SSH tunnels (PDF Steps 4-5 / Steps 7-8) ──────────────────
    echo "[5/6] Setting up SSH tunnels..."

    # Tunnel 1: login2 → GPU node (PDF Step 4 / Step 7)
    # Equivalent to: ssh -N -L 127.0.0.1:8899:127.0.0.1:8899 lgnXX
    # (run on wexac login2, forwarding to the GPU node)
    echo "  Tunnel 1: wexac(login2):${REMOTE_PORT} -> ${GPU_HOST}:${REMOTE_PORT}"
    ssh "$SSH_HOST" "ssh -f -N -L 127.0.0.1:${REMOTE_PORT}:127.0.0.1:${REMOTE_PORT} ${GPU_HOST}" 2>/dev/null
    echo "    established."

    # Tunnel 2: Mac → login2 (PDF Step 5 / Step 8)
    # Equivalent to: ssh -N -L 8899:127.0.0.1:8899 wexac
    echo "  Tunnel 2: localhost:${LOCAL_PORT} -> wexac(login2):${REMOTE_PORT}"
    ssh -f -N -L "${LOCAL_PORT}:127.0.0.1:${REMOTE_PORT}" "$SSH_HOST" 2>/dev/null
    echo "    established."

    # Verify the full tunnel chain works
    sleep 3
    if curl -s -o /dev/null --connect-timeout 5 "http://127.0.0.1:${LOCAL_PORT}/api" 2>/dev/null; then
        echo "  Tunnel chain verified — Jupyter is reachable from localhost."
    else
        echo "  WARNING: Could not reach Jupyter through tunnels yet."
        echo "  It may need a few more seconds. Try: curl http://127.0.0.1:${LOCAL_PORT}/api"
    fi
    echo ""

    # ── Step 6: Open JupyterLab in browser + VS Code ────────────────────────────
    URL="http://127.0.0.1:${LOCAL_PORT}/lab?token=${JUPYTER_TOKEN}"

    # Copy URL to clipboard
    echo "$URL" | pbcopy 2>/dev/null || true

    echo "[6/6] Opening JupyterLab in browser + VS Code..."

    # Open JupyterLab in default browser
    open "$URL" 2>/dev/null || true

    # Open VS Code with the project and notebook
    THESIS_DIR="$HOME/Documents/Weizmann/Thesis"
    NOTEBOOK="$THESIS_DIR/dataset_reconstruction/reconstruction_mnist.ipynb"
    (unset CLAUDECODE; code "$THESIS_DIR"; sleep 2; code "$NOTEBOOK") 2>/dev/null || true

    echo ""
    echo "================================================================"
    echo "  GPU Jupyter is ready!"
    echo "================================================================"
    echo ""
    echo "  GPU node:  $GPU_HOST"
    echo "  URL:       $URL"
    echo "  (URL copied to clipboard)"
    echo ""
    echo "  JupyterLab opened in browser."
    echo "  VS Code opened with your notebook."
    echo ""
    echo "  In JupyterLab: select 'Python (rec)' kernel"
    echo "  Verify GPU:    import torch; print(torch.cuda.is_available())"
    echo ""
    echo "  To stop: ./wexac_disconnect.sh"
    echo "================================================================"
    echo ""
    echo "Press Ctrl-C to exit (tunnels stay open until you run disconnect)."
    echo ""

    # Keep script alive so user can see the output
    wait 2>/dev/null || true
fi
