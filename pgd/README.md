# `pgd` — ParGaMD run analysis

One dispatcher, one file per capability. `./pgd <command>` from any run
directory; subcommands live in `cmds/` and share a library in `lib/pgdlib/`.

The motivating workflow: **from a populated point on the explored PES, open the
continuous trajectory back to the initial parent.**

```bash
./pgd doctor --scan-solvent                  # is this run ready, and what's in it
                                             # (also: which iterations are full-atom)
./pgd fes --pick                             # render the PMF, click a basin
./pgd find --cv 8.91,6.27 --tol 0.17         # which walkers visited it
./pgd strip-copy --for-lineage 60:220        # normalize full-atom ancestors (copies only)
./pgd load --tip 60:220 --colvars            # 60 segments -> vmd_2, with cv0/cv1 live
```

## Commands

| command | what it does |
|---|---|
| `doctor` | environment, tools, topologies, solvent inventory, bin-mapper churn, the write firewall |
| `find` | segments whose pcoord lands near a CV point, ranked by weight |
| `trace` | a walker's ancestry root→tip, with the exact frame slice each segment contributes |
| `info` | everything west.h5 and the disk know about an iteration or a segment |
| `strip-copy` | solvent-strip **copies** of full-atom segments into the cache |
| `load` | build a Tcl deck and open the lineage in `vmd_2` — writes no trajectory |
| `export` | materialize the lineage as one file (dcd/nc/pdb/xtc/trr/rst7/…) + provenance |
| `fes` | regenerate the PMF through the run's own `pyreweight.py`; `--pick` turns a click into a `find` |
| `colvars` | emit a VMD Colvars config computing this run's own progress coordinates |
| `cache` | `ls` / `du` / `verify --prune` / `clean` |

## Three things worth knowing

**There is no duplicated seam frame in this run.** The usual WESTPA convention
is that frame 0 of a child reproduces its parent's last frame, so a stitcher
drops it. Here the OpenMM DCDs carry `ISTART=NSAVC=500` — the first frame is
written 2 ps *into* the segment, so there is no t=0 frame to duplicate.
Measured over 3,788 non-root segments: zero exact pcoord matches at the seam.
`--seam auto` (the default) measures this rather than assuming it, and records
the verdict. Dropping frame 0 here would delete one real frame per segment and
open a 4 ps gap at every seam.

*(Note: `files_dist_cv/stitch_lineages.py` drops frame 0 by default, so the
existing `tica_chains/` were built that way.)*

**Original trajectory data is never modified.** Some iterations are still
full-atom (`post_iter.sh` strips with a 2-iteration lag, and the set moves as the
run advances — ask `pgd doctor --scan-solvent`, never assume), and the running
simulation depends on them. `pgd strip-copy` copies
each segment into a private work directory *before* cpptraj sees it, so no
`traj_segs` path appears in any deck; every write additionally goes through
`pgdlib.guard`, which realpath-resolves both sides and refuses anything under
`traj_segs` — defeating the run-dir symlink and `..` traversal alike. A
pre/post stat tripwire proves nothing changed. `pgd strip-copy --selftest`
asserts the firewall rejects a real `traj_segs` path and exits non-zero if it
does not.

`westpa_scripts/strip_iter.sh` is deliberately **not** invoked: it strips in
place (`mv -f` over the original), is iteration-scoped, and writes markers into
the live simulation tree. `pgd` reuses its cpptraj recipe, not its contract.

**Bin indices are not comparable across iterations.** The mapper changed 21
times mid-run (last at iteration 71), so `find` searches raw CV values. `info`
will print a bin index, always stamped with its iteration.

**pgd never holds the `west.h5` lock across a child process.** HDF5 does not set
`FD_CLOEXEC`, so a handle still open at `os.execvp` is inherited by VMD and keeps
the file locked for that session's whole lifetime — which makes `w_run` fail with
`BlockingIOError: [Errno 11] unable to lock file`. Every command calls
`cli.release(h5)` as soon as it is done reading and always before exec or fork;
`tests/test_h5_release.py` asserts that, and `pgd doctor` reports anyone else
holding the file. If WESTPA is running and you only want to read, `PGD_H5_NO_LOCK=1`
opens without taking the lock at all — opt-in, because an unlocked read of a file
being written can tear.

To check the end-to-end behaviour, substitute a probe for VMD:

```bash
cat > /tmp/fdprobe.sh <<'EOF'
#!/bin/bash
echo "PGD_FRAMES=0"; echo "PGD_ATOMS=0"
for fd in /proc/self/fd/*; do
    case "$(readlink "$fd" 2>/dev/null)" in *west.h5*) echo "LEAKED: $fd" ;; esac
done
echo "clean-or-leaked above"
EOF
chmod +x /tmp/fdprobe.sh
PGD_VMD=/tmp/fdprobe.sh ./pgd load --tip 113:46 --no-preflight
```

## Colvars: the run's own CVs, live in VMD

`pgd load --colvars` attaches the *same* progress coordinates the weighted
ensemble steered on, so you can watch cv0/cv1 as you scrub the trajectory:

```
frame 0      cv0=10.5050  cv1=19.9954
frame 5650   cv0=10.2096  cv1= 7.9267
```

Two helpers are installed in the VMD session: `cvs [frame]` prints the values at
a frame, `cvtrace <file> [stride]` dumps every frame to a file. A companion
`pcoord_recorded.dat` lands next to the deck with what `west.h5` recorded for the
same frames, so the live value is checkable against the pcoord of record.

It is a translation, not a reimplementation. `pgd colvars` imports the run's own
`cv_*.py` — the very modules `_pcoord_dispatch.py` feeds WESTPA — lets them
resolve their atom pairs against the topology, and reads the pairs back out. A
mean of N contact distances is a linear combination, which Colvars expresses
exactly as N `distance` components with `componentCoeff = 1/N`.

`pgd colvars --verify` closes the loop by loading a real segment in VMD and
comparing three numbers: what the CV module computes, what Colvars computes from
the generated config, and what `west.h5` recorded. Measured here: module vs
Colvars agree to **1.2e-06** (float32 coordinates), and both agree with `west.h5`
to **5e-05** — the floor set by `scaleoffset: 4` on the pcoord dataset in
`west.cfg`, which is lossy by design.

Limits worth knowing: `CV_MODE=softq` is **refused**, not approximated — it is a
logistic of each distance, and Colvars' linear component combination cannot
reproduce it (`coordNum` uses a different, rational switching function). Only CV
modules that expose their resolved pairs (the `cv_clamp` family) can be
translated; anything else gets a clear error naming the module, and you can pass
a hand-written config with `--colvars-config`. Attaching 121 distance components
makes Colvars print one setup line per atom group — noisy, harmless.

## Stride

`--stride` is applied on the *global* concatenated timeline, so the phase
carries across segment boundaries: `--stride 3` over 100-frame segments gives
true uniform 6 ps spacing even though 3 does not divide 100. Neither `catdcd`
(whose `-stride` is global but cannot be combined with per-file trimming) nor
`cpptraj` (whose `trajin` offset restarts per file) does this on its own.

## Size gate

Before loading, `pgd` estimates both the on-disk and the in-VMD footprint and
checks each against its own limit — a configurable threshold (`--warn-gb`,
`$PGD_LOAD_WARN_GB`, default 95 GiB) and a live fraction of `MemAvailable`. It
offers a stride that fits. With no tty it refuses rather than guessing, so a
batch script cannot OOM the machine. Exit 3 = size gate, 4 = declined,
5 = needs `strip-copy`.

## Export engines

- **splice** (default for DCD): raw byte concatenation. After the strip cache
  normalizes atom counts, every input shares a fixed record layout, so joining
  is a byte copy — bit-identical, constant memory, ~640 MB/s (3.13 GiB full
  lineage in ~5 s). It also restores the timestep cpptraj discards
  (`DELTA`→0.004 ps); every already-stripped segment on disk carries cpptraj's
  wrong 0.001 ps.
- **cpptraj**: selected automatically for any other format, `--mask`,
  `--align`, or `--autoimage`. A masked export also gets a matching `.parm7`,
  without which the trajectory is unusable later.

Every export writes `<out>.provenance.csv` mapping each output frame back to
`(iteration, segment, frame, weight, original time)`. `pgd load` bakes the same
map into the Tcl deck: type `prov` in VMD to identify the current frame.

## Layout

```
pgd/
  pgd              dispatcher (symlinked into each run dir as ./pgd)
  cmds/<name>.py   one file per command; line 2 is "# summary: ..." for --help
  lib/common.sh    root resolution, env activation, logging
  lib/pgdlib/
    paths.py       the ONLY two iteration formatters (h5 8-digit vs disk 6-digit)
    lineage.py     parent_id walk, completed-iteration cap, merge reporting
    frames.py      seam detection + global-stride slicing
    solvent.py     atom-count classification, strip-cache resolution
    dcdio.py       header parse/build, size arithmetic, byte splicer
    guard.py       the write firewall
    sizes.py       disk + RAM estimation and the gate
    pes.py         CV-point search
    vmdgen.py      Tcl generation
  tests/           standalone (no pytest in the pargamd env)
```

Run the tests with:

```bash
python tests/test_frames.py
python tests/test_guard.py
python tests/test_h5_release.py
./pgd strip-copy --selftest
```
