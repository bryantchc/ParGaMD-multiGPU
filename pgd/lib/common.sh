# shellcheck shell=bash
# Shared helpers for the pgd dispatcher and its .sh subcommands.

pgd_log()   { printf '[pgd] %s\n' "$*" >&2; }
pgd_warn()  { printf '[pgd] WARNING: %s\n' "$*" >&2; }
pgd_die()   { printf '[pgd] %s\n' "$*" >&2; exit 1; }
pgd_debug() { [[ -n "${PGD_VERBOSE:-}" ]] && printf '[pgd] %s\n' "$*" >&2; return 0; }

# Resolve the run directory. Precedence:
#   --root DIR  >  $WEST_SIM_ROOT  >  dir of the run-dir symlink  >  $PWD
# We deliberately use the symlink's own dirname (not readlink -f), because the
# dispatcher is symlinked from the repo INTO each run dir -- the same trick
# westpa_scripts/strip_iter.sh uses.
pgd_resolve_root() {
    local cand="${1:-}"
    if [[ -z "$cand" ]]; then cand="${WEST_SIM_ROOT:-}"; fi
    if [[ -z "$cand" ]]; then cand="$PGD_INVOKED_DIR"; fi
    if [[ -z "$cand" ]]; then cand="$PWD"; fi
    cand="$(cd "$cand" 2>/dev/null && pwd)" || pgd_die "cannot cd to '$1'"
    [[ -f "$cand/west.cfg" ]] || pgd_die \
        "$cand is not a ParGaMD run dir (no west.cfg). cd to a run dir, or pass --root DIR."
    [[ -f "$cand/env.sh" ]] || pgd_die \
        "$cand has no env.sh -- pgd needs it for SYSTEM_NAME / STRIP_MASK / the conda env."
    printf '%s\n' "$cand"
}

# Source the run's env.sh exactly once per pgd invocation; every child process
# then inherits the activated environment. No subcommand ever calls conda.
pgd_activate_env() {
    local root="$1"
    if [[ "${PGD_ENV_READY:-}" == "$root" ]]; then return 0; fi
    # env.sh sources ~/.bashrc, which is not `set -u` clean.
    set +u
    # shellcheck disable=SC1091
    if [[ -n "${PGD_VERBOSE:-}" ]]; then
        source "$root/env.sh"
    else
        source "$root/env.sh" >/dev/null
    fi
    set -u
    export PGD_ENV_READY="$root"

    [[ -n "${CONDA_PREFIX:-}" ]] || pgd_die \
        "env.sh did not activate a conda env (CONDA_PREFIX unset)."
    if [[ "$(basename "$CONDA_PREFIX")" != "pargamd" ]]; then
        pgd_die "active conda env is '$(basename "$CONDA_PREFIX")', expected 'pargamd'.
The base env has none of h5py/MDAnalysis/westpa/cpptraj. Check 'conda env list'."
    fi
}

pgd_require_bin() {  # $1 label, $2 path
    [[ -x "$2" ]] || pgd_die "$1 not found or not executable at: $2"
}

# y/N confirmation that still works when stdin is a pipe (falls back to /dev/tty).
# Auto-yes under --yes; auto-NO when there is no tty at all, so a batch script
# can never be silently answered for.
pgd_confirm() {
    local prompt="$1"
    [[ -n "${PGD_YES:-}" ]] && return 0
    if [[ -t 0 ]]; then
        read -r -p "$prompt [y/N] " ans </dev/tty >/dev/tty 2>&1 || return 1
    elif [[ -e /dev/tty ]]; then
        read -r -p "$prompt [y/N] " ans </dev/tty >/dev/tty 2>&1 || return 1
    else
        pgd_warn "no tty available; refusing (pass --yes to proceed non-interactively)"
        return 1
    fi
    [[ "$ans" =~ ^[Yy] ]]
}
