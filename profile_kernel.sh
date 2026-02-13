#!/bin/bash

# --- Variables ---
PROJECT_DIR="/home/moonboyknm/Documents/Projects/Pragna1B_Custom_Kernel"
LOG_FILE="profiling_results_$(date +%Y%m%d_%H%M%S).txt"
PYTHON_SCRIPT="profile_triton.py" # Your Triton kernel script

cd "$PROJECT_DIR" || exit
source venv/bin/activate

echo "Starting Profiling Session: $(date)" | tee "$LOG_FILE"
echo "------------------------------------" | tee -a "$LOG_FILE"

# Run Nsight Compute (ncu)
# -f: Overwrite existing reports
# --set full: Capture all detailed hardware metrics
# --csv: Export to CSV format within the text file for easier parsing later
ncu --set full --csv python "$PYTHON_SCRIPT" >>"$LOG_FILE" 2>&1

echo "------------------------------------" | tee -a "$LOG_FILE"
echo "Profiling Complete. Results saved to: $LOG_FILE"
