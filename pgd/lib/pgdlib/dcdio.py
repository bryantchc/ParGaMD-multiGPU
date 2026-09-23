"""DCD header parsing, size arithmetic, and fixed-record byte splicing.

Why a byte splicer
------------------
After the strip cache normalizes every segment to the same atom count, all the
inputs to a stitch are DCDs with identical record geometry.  Concatenation is
then a pure byte operation: no float decode/encode, no precision loss, no
topology needed, constant memory, disk-speed.  It is also the only engine that
gets a GLOBAL stride right -- catdcd's -first/-last/-stride are global but
cannot be combined with per-file trimming, and cpptraj's `trajin` offset
restarts on every file.

Verified layout (little-endian, 32-bit record markers), e.g. a stripped
segment with 24,790 atoms:

    [84]['CORD'][20 int32 control words][84]      92 bytes
    [tb][ntitle][80*ntitle bytes][tb]             4+4+80*ntitle+4
    [4][natom][4]                                 12 bytes
    then per frame:
      [48][6 float64 unit cell][48]               56 bytes
      3 x [4*natom][natom float32][4*natom]       3*(8 + 4*natom)

  header 196 B + 100 * 297,560 B == 29,756,196 B   (matches every stripped seg)
  header 276 B + 100 * 2,814,668 B == 281,467,076 B (matches every full seg)

Timestep
--------
cpptraj rewrites ISTART/NSAVC/DELTA to its own defaults (1/1/dt=1.0 ps), so
every already-stripped segment on disk carries the wrong timing.  The OpenMM
originals carry the truth: ISTART=NSAVC=500, DELTA=0.0818193 AKMA = 0.004 ps.
We restore correct values in our own output.  We cannot and must not fix the
originals.
"""

from __future__ import annotations

import os
import struct
import time
from dataclasses import dataclass
from pathlib import Path

# AKMA time unit in picoseconds.  DELTA_akma = dt_ps / TIMESCALE
TIMESCALE = 0.04888821

CELL_BYTES = 56          # [48][6 x float64][48]


class DcdLayoutError(RuntimeError):
    """The file is not a DCD we can splice byte-wise."""


@dataclass(frozen=True)
class DcdProfile:
    path: Path
    header_len: int
    natoms: int
    nframes: int
    frame_bytes: int
    has_cell: bool
    ntitle: int
    istart: int
    nsavc: int
    delta_akma: float
    charmm_version: int

    @property
    def dt_ps(self) -> float:
        return self.delta_akma * TIMESCALE


def frame_bytes(natoms: int, has_cell: bool = True) -> int:
    return (CELL_BYTES if has_cell else 0) + 3 * (8 + 4 * natoms)


def header_bytes(ntitle: int = 2) -> int:
    return 92 + (4 + 4 + 80 * ntitle + 4) + 12


def dcd_bytes(natoms: int, nframes: int, ntitle: int = 2,
              has_cell: bool = True) -> int:
    """Exact on-disk size of a DCD. Verified against both real layouts."""
    return header_bytes(ntitle) + nframes * frame_bytes(natoms, has_cell)


def probe(path) -> DcdProfile:
    """Parse a DCD header and verify the size arithmetic closes exactly.

    The size check is the strong part: a partially written DCD (a segment the
    live simulation is still producing) cannot satisfy it.
    """
    p = Path(path)
    size = os.path.getsize(p)
    with open(p, "rb") as f:
        blk = f.read(4)
        if len(blk) != 4 or struct.unpack("<i", blk)[0] != 84:
            raise DcdLayoutError(
                "%s: not a little-endian 32-bit-record DCD (first marker %r)"
                % (p, blk))
        if f.read(4) != b"CORD":
            raise DcdLayoutError("%s: missing CORD magic" % p)
        ctrl = struct.unpack("<20i", f.read(80))
        if struct.unpack("<i", f.read(4))[0] != 84:
            raise DcdLayoutError("%s: control block not terminated" % p)

        tb = struct.unpack("<i", f.read(4))[0]
        ntitle = struct.unpack("<i", f.read(4))[0]
        f.read(80 * ntitle)
        if struct.unpack("<i", f.read(4))[0] != tb:
            raise DcdLayoutError("%s: title block not terminated" % p)

        if struct.unpack("<i", f.read(4))[0] != 4:
            raise DcdLayoutError("%s: natom block malformed" % p)
        natoms = struct.unpack("<i", f.read(4))[0]
        if struct.unpack("<i", f.read(4))[0] != 4:
            raise DcdLayoutError("%s: natom block not terminated" % p)
        hlen = f.tell()

    nset, istart, nsavc, nstep = ctrl[0], ctrl[1], ctrl[2], ctrl[3]
    delta = struct.unpack("<f", struct.pack("<i", ctrl[9]))[0]
    has_cell = bool(ctrl[10])
    fb = frame_bytes(natoms, has_cell)

    if fb <= 0 or (size - hlen) % fb != 0:
        raise DcdLayoutError(
            "%s: size %d with header %d is not a whole number of %d-byte "
            "frames (natoms=%d, cell=%s)" % (p, size, hlen, fb, natoms, has_cell))
    actual = (size - hlen) // fb
    if nset and actual != nset:
        raise DcdLayoutError(
            "%s: header claims %d frames but file holds %d -- truncated or "
            "still being written" % (p, nset, actual))

    return DcdProfile(
        path=p, header_len=hlen, natoms=natoms, nframes=actual,
        frame_bytes=fb, has_cell=has_cell, ntitle=ntitle,
        istart=istart, nsavc=nsavc, delta_akma=delta,
        charmm_version=ctrl[19],
    )


def dcd_natoms(path) -> int:
    """Atom count from the header alone. Cheap; used for topology dispatch."""
    return probe(path).natoms


def build_header(natoms: int, nframes: int, titles, dt_ps: float,
                 interval_steps: int, stride: int = 1,
                 has_cell: bool = True) -> bytes:
    """A DCD header with the timing metadata cpptraj throws away.

    Doesn't matter for VMD playback (VMD animates by frame index), but it does
    matter for cpptraj's time-dependent analyses, MDAnalysis ts.dt, and anyone
    who opens the file in six months. It costs 12 bytes.
    """
    step = interval_steps * stride
    ctrl = [0] * 20
    ctrl[0] = nframes                       # NSET
    ctrl[1] = step                          # ISTART
    ctrl[2] = step                          # NSAVC
    ctrl[3] = step * nframes                # NSTEP
    ctrl[9] = struct.unpack("<i", struct.pack("<f", dt_ps / TIMESCALE))[0]
    ctrl[10] = 1 if has_cell else 0         # CHARMM unit cell present
    ctrl[19] = 24                           # CHARMM version (match OpenMM)

    titles = list(titles)[:2] or ["pgd stitched trajectory"]
    tbuf = b"".join(t.encode()[:80].ljust(80, b"\0") for t in titles)
    tb = 4 + len(tbuf)

    out = struct.pack("<i", 84) + b"CORD" + struct.pack("<20i", *ctrl) \
        + struct.pack("<i", 84)
    out += struct.pack("<i", tb) + struct.pack("<i", len(titles)) + tbuf \
        + struct.pack("<i", tb)
    out += struct.pack("<i", 4) + struct.pack("<i", natoms) + struct.pack("<i", 4)
    if len(out) != header_bytes(len(titles)):
        raise AssertionError("header length %d != expected %d"
                             % (len(out), header_bytes(len(titles))))
    return out


def splice(sources, ranges, out_path, natoms_expect=None, titles=None,
           dt_ps=0.004, interval_steps=500, stride_for_header=1,
           progress=None) -> dict:
    """Concatenate frame ranges from several DCDs by raw byte copy.

    sources: [path]                       -- already solvent-normalized
    ranges:  [(first, stop_exclusive, step)] parallel to sources

    Every source must agree on record geometry; otherwise DcdLayoutError is
    raised and the caller falls back to the cpptraj engine.

    The header is written LAST, once the real frame count is known, so a crash
    leaves a file that fails probe() rather than a plausible-looking truncated
    one.  Output goes to <out>.partial and is os.replace()d on success.
    """
    profs = [probe(s) for s in sources]
    if not profs:
        raise ValueError("no sources")

    n0 = profs[0].natoms
    bad = [p for p in profs if p.natoms != n0]
    if bad:
        raise DcdLayoutError(
            "atom count mismatch: %d atoms in %s but %d in %s -- the strip "
            "cache should have normalized these"
            % (n0, profs[0].path, bad[0].natoms, bad[0].path))
    if natoms_expect is not None and n0 != natoms_expect:
        raise DcdLayoutError(
            "expected %d atoms (topology), sources have %d"
            % (natoms_expect, n0))
    if len({p.frame_bytes for p in profs}) != 1:
        raise DcdLayoutError("inconsistent frame record size across sources")
    if not all(p.has_cell for p in profs):
        raise DcdLayoutError("some sources carry no unit cell; use --engine cpptraj")

    fb = profs[0].frame_bytes
    out_path = Path(out_path)
    partial = out_path.with_suffix(out_path.suffix + ".partial")
    partial.parent.mkdir(parents=True, exist_ok=True)

    ntitle = 2
    written = 0
    t0 = time.time()
    with open(partial, "wb") as w:
        w.write(b"\0" * header_bytes(ntitle))       # placeholder, backfilled
        for i, (src, prof, (a, b, st)) in enumerate(zip(sources, profs, ranges)):
            if b > prof.nframes:
                raise DcdLayoutError(
                    "%s: asked for frames up to %d but file holds %d"
                    % (src, b - 1, prof.nframes))
            with open(src, "rb") as r:
                if st == 1:
                    # Contiguous run: one seek, one big read per file.
                    r.seek(prof.header_len + a * fb)
                    n = b - a
                    w.write(r.read(n * fb))
                    written += n
                else:
                    for fr in range(a, b, st):
                        r.seek(prof.header_len + fr * fb)
                        w.write(r.read(fb))
                        written += 1
            if progress:
                progress(i + 1, len(sources), written)

        w.seek(0)
        w.write(build_header(n0, written,
                             titles or ["pgd stitched trajectory"],
                             dt_ps, interval_steps, stride_for_header,
                             has_cell=True))

    os.replace(partial, out_path)

    check = probe(out_path)
    if check.nframes != written or check.natoms != n0:
        raise DcdLayoutError(
            "post-write verification failed: wrote %d frames / %d atoms, "
            "file reports %d / %d" % (written, n0, check.nframes, check.natoms))

    return {
        "path": str(out_path),
        "natoms": n0,
        "nframes": written,
        "bytes": os.path.getsize(out_path),
        "seconds": round(time.time() - t0, 2),
        "engine": "splice",
    }
