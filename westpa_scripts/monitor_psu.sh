#!/usr/bin/env bash
# monitor_psu.sh — live tail of gpu_util.log; alert/halt on PSU stress events.
#
# Watches the GPU telemetry stream that run_local.sh writes to
# $WEST_SIM_ROOT/gpu_util.log and surfaces any event that indicates
# the PSU is struggling:
#
#   hw_power_brake_slowdown = Active
#       The external power source asserted the brake line back to the
#       GPU — i.e. the PSU told the GPU to back off because it couldn't
#       deliver requested power. THIS IS THE DEFINITIVE PSU-TROUBLE
#       SIGNAL. One event during a long run = PSU was caught stressed
#       at least once. Repeated events = PSU is failing under your load.
#
#   hw_thermal_slowdown = Active
#       GPU thermal limit hit. Not a PSU issue per se, but a sign of
#       cooling problems that often precede PSU trouble.
#
#   sustained temperature.gpu above WATCH_TEMP_C
#       Approaching thermal trip. Default threshold 83 C, well above
#       normal operating temps but below the hardware throttle point.
#
# Usage:
#   ./monitor_psu.sh                # tail forever, print alerts to stdout
#   ./monitor_psu.sh --halt-on-brake # also send SIGINT to a running
#                                      run_local.sh on first brake event
#                                      (saves the PSU at the cost of
#                                       at most one iteration of work)
#
# Run alongside production. The script tails the existing gpu_util.log
# (no extra GPU polling, no overhead).

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
WEST_SIM_ROOT="$PWD"
LOG="$WEST_SIM_ROOT/gpu_util.log"
ALERT_LOG="$WEST_SIM_ROOT/psu_alerts.log"
WATCH_TEMP_C=${WATCH_TEMP_C:-83}

HALT=0
[ "${1:-}" = "--halt-on-brake" ] && HALT=1

if [ ! -f "$LOG" ]; then
    echo "[psu-watch] $LOG missing — is run_local.sh running?" >&2
    echo "[psu-watch] will wait for it to appear..."
    while [ ! -f "$LOG" ]; do sleep 1; done
fi

echo "[psu-watch] tailing $LOG (halt-on-brake=$HALT, temp threshold=${WATCH_TEMP_C}C)"
echo "[psu-watch] alert events also appended to $ALERT_LOG"

alert() {  # $1 severity  $2 message
    local line="[psu-watch $(date -Iseconds)] $1: $2"
    echo "$line"
    echo "$line" >> "$ALERT_LOG"
}

halt_run() {
    # Find the run_local.sh PID and send SIGINT so WESTPA drains cleanly
    local pids
    pids=$(pgrep -f "$WEST_SIM_ROOT/run_local.sh" || true)
    if [ -n "$pids" ]; then
        alert "HALT" "sending SIGINT to run_local.sh PIDs: $pids"
        echo "$pids" | xargs -r kill -INT
    else
        alert "HALT" "no run_local.sh PID found to signal"
    fi
}

# Parse new rows. Column indexes (0-based) for the gpu_util.log format
# run_local.sh writes (after Tier-1 patch):
#   0 timestamp, 1 index, 2 name, 3 util, 4 mem, 5 temp,
#   6 power.draw, 7 hw_power_brake, 8 hw_thermal_slowdown
HAVE_BRAKED=0
tail -F -n 0 "$LOG" 2>/dev/null | while IFS=, read -r ts idx name util mem temp pwr brake therm; do
    # Skip CSV header rows
    [[ "$ts" == "timestamp" ]] && continue
    # Trim whitespace + strip units
    brake="${brake## }"; brake="${brake%% }"
    therm="${therm## }"; therm="${therm%% }"
    temp_n="${temp// /}"; temp_n="${temp_n%%C*}"  # "65 C" -> "65"

    if [[ "$brake" == "Active" ]]; then
        alert "BRAKE" "GPU $idx hw_power_brake_slowdown=Active (PSU asserted brake; power.draw=$pwr, temp=$temp)"
        if [ "$HALT" -eq 1 ] && [ "$HAVE_BRAKED" -eq 0 ]; then
            HAVE_BRAKED=1
            halt_run
        fi
    fi
    if [[ "$therm" == "Active" ]]; then
        alert "THERMAL" "GPU $idx hw_thermal_slowdown=Active (cooling issue; temp=$temp)"
    fi
    if [ -n "$temp_n" ] && [ "$temp_n" -ge "$WATCH_TEMP_C" ] 2>/dev/null; then
        alert "HOT" "GPU $idx temp=${temp_n}C >= ${WATCH_TEMP_C}C threshold"
    fi
done
