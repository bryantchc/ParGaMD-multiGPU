"""Size estimation and the load gate.

Two limits, not one.  The requested 95 GiB disk gate is correct but rarely
fires for this system: 95 GiB / 297,560 B is ~343,000 frames, about 30 full
lineages.  The limit that actually bites on this machine is RAM -- VMD holds
the whole trajectory as float32 xyz (12 B/atom/frame) and MemAvailable is
~63 GB with the live simulation resident.  So check both.
"""

from __future__ import annotations

import math
import os

from . import dcdio

VMD_BYTES_PER_ATOM_FRAME = 12       # float32 x/y/z
DEFAULT_WARN_GB = 95.0
DEFAULT_RAM_FRACTION = 0.5


def estimate_disk_bytes(natoms: int, nframes: int) -> int:
    return dcdio.dcd_bytes(natoms, nframes)


def estimate_vmd_bytes(natoms: int, nframes: int) -> int:
    return VMD_BYTES_PER_ATOM_FRAME * natoms * nframes


def available_ram_bytes() -> int:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


def human(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024 or unit == "TiB":
            return "%.2f %s" % (n, unit) if unit != "B" else "%d B" % n
        n /= 1024.0
    return "%.2f TiB" % n


def suggest_stride(nframes: int, natoms: int, disk_limit: int,
                   ram_limit: int) -> int:
    """Smallest stride that brings both estimates under their limits."""
    need = 0.0
    if disk_limit > 0:
        need = max(need, estimate_disk_bytes(natoms, nframes) / disk_limit)
    if ram_limit > 0:
        need = max(need, estimate_vmd_bytes(natoms, nframes) / ram_limit)
    return max(1, int(math.ceil(need)))


def gate(nframes: int, natoms: int, warn_gb: float | None = None,
         ram_fraction: float = DEFAULT_RAM_FRACTION) -> dict:
    """Compute the size verdict. Pure: prompts and exits live in the caller."""
    if warn_gb is None:
        warn_gb = float(os.environ.get("PGD_LOAD_WARN_GB", DEFAULT_WARN_GB))
    disk_limit = int(warn_gb * 2 ** 30)
    avail = available_ram_bytes()
    ram_limit = int(ram_fraction * avail) if avail else 0

    est_disk = estimate_disk_bytes(natoms, nframes)
    est_ram = estimate_vmd_bytes(natoms, nframes)
    over = (est_disk > disk_limit) or (ram_limit and est_ram > ram_limit)

    return {
        "nframes": nframes,
        "natoms": natoms,
        "est_disk": est_disk,
        "est_ram": est_ram,
        "disk_limit": disk_limit,
        "ram_limit": ram_limit,
        "ram_available": avail,
        "ram_fraction": ram_fraction,
        "over": bool(over),
        "suggested_stride": suggest_stride(nframes, natoms, disk_limit, ram_limit),
    }


def render(g: dict) -> str:
    lines = [
        "Estimated trajectory: %s frames x %s atoms"
        % ("{:,}".format(g["nframes"]), "{:,}".format(g["natoms"])),
        "  on disk : %-10s (limit %s)" % (human(g["est_disk"]), human(g["disk_limit"])),
    ]
    if g["ram_limit"]:
        lines.append("  in VMD  : %-10s (limit %s = %d%% of %s MemAvailable)"
                     % (human(g["est_ram"]), human(g["ram_limit"]),
                        round(100 * g["ram_fraction"]), human(g["ram_available"])))
    else:
        lines.append("  in VMD  : %s (MemAvailable unknown)" % human(g["est_ram"]))
    return "\n".join(lines)
