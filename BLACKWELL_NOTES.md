# Blackwell (consumer) multi-GPU notes

Hardware: TRX50 AERO D + RTX 5090 (sm_120, 32 GB) + RTX 5070 (sm_120, 12 GB).
Driver 580.95.05, CUDA 13.0, OpenMM 8.4 (CUDA platform), gamd-openmm.

This file documents three real bugs we hit getting ParGaMD onto multi-GPU
consumer Blackwell, the diagnoses, and the fixes. Everything below was
verified empirically against this branch (`multigpu`) on this workstation.

## 1. ReBAR must be enabled on both GPUs

**Symptom:** after a Gigabyte TRX50 BIOS update, `nvidia-smi` only showed
the 5090. Kernel log:

```
pci 0000:c1:00.0: BAR 1 [mem size 0x400000000 64bit pref]: can't assign; no space
NVRM: BAR1 is 0M @ 0x0 (PCI:0000:c1:00.0)
nvidia 0000:c1:00.0: probe with driver nvidia failed with error -1
```

The 5070's 16 GiB prefetchable BAR1 couldn't fit in the high-MMIO window
AGESA allocated by default. The 5090's 32 GiB window had already consumed
most of what was reserved.

**Fix:** in BIOS, set Advanced → PCI → SR-IOV → **Enabled**. This forces
AGESA to re-plan the high-MMIO window wide enough for both BAR1s. The
deeper AMD CBS → NBIO → MMIO High Size knob is the "right" answer but was
not exposed in this BIOS revision.

**Verify with** `lspci -vvv | grep -E "(RTX|BAR1)"` — both cards should
show a multi-GiB prefetchable BAR1, not a 256 MiB stub.

## 2. The `stepCount` readback after `loadCheckpoint` is garbage

**Symptom:** ~30–100% of segments (rate varies by config) fail with one of:

```
OverflowError: in method 'Context_setStepCount', argument 2 of type 'long long'
  ← from runners.py line `simulation.currentStep = old_step_count`
```

or

```
ValueError: RunningRates:  The save_rate and reporting_rate should
            evenly divide into the number of steps in the simulation.
```

**Diagnosis:** instrumented `runners.py` showed
`simulation.integrator.getGlobalVariableByName("stepCount")` returning
garbage floats like `1.36e+48`, `4.34e+102`, `-2.75e-174` immediately
after `loadCheckpoint()`. Meanwhile `state.getTime()` returned the
correct value (`99.999999...` ps for a 100 ps segment). So GPU position /
velocity / time readback is fine; only the integrator's scalar global
readback is corrupted.

Initially looked GPU-1-specific (the 5070 failed more often), but later
testing with the strengthened workaround showed the readback was wrong
on **every segment on both GPUs**. On non-Blackwell hardware the wrong
value happens to coincide with the correct one often enough that the
upstream code "works." Here it doesn't.

This is plausibly a latent design issue in gamd-openmm — the integrator's
`stepCount` global appears to track steps since the last context
creation, not absolute steps, and the upstream author left a commented
alternative on what is now line 415 of `runners.py` (`#current_step =
int(round(state_time / integrator_dt))`) suggesting some prior
uncertainty.

**Fix:** `common_files/gamd/runners.py` always trusts the
`state_time`-derived step count and logs `[WORKAROUND]` whenever it
disagrees with the integrator readback (i.e. always, on Blackwell):

```python
derived_step_count = int(round(state_time / integrator_dt))
if raw_as_int != derived_step_count:
    print("[WORKAROUND] ...", file=sys.stderr)
old_step_count = derived_step_count
```

The fix is in this branch only — upstream gamd-openmm is unchanged.

## 3. MPS routing: per-GPU daemons (`USE_MPS=1`) > shared sudo daemon (`USE_MPS=2`)

**Symptom:** with the shared sudo MPS daemon at `/tmp/nvidia-mps`
(`USE_MPS=2`), GPU 1 utilization in `nvidia-smi`/btop was near zero
during multi-GPU runs even though we were spawning 4 workers targeting
GPU 1 via `gamdRunner -d 1`. GPU 0 ran at 100% util sustained. Tools
showed 8 python procs on GPU 0 with ~524 MiB VRAM each, 4 procs on
GPU 1 with only ~184 MiB each (partial contexts that never advanced).

A direct A/B test (12 workers all on GPU 0 vs mixed 8+4 with MPS=2) put
the actual GPU 1 contribution at ~10% throughput. Whether the remaining
GPU 1 work was real or whether MPS was silently re-routing it to GPU 0
was never fully resolved — the data was consistent with either.

**Fix:** `USE_MPS=1` mode starts one user-mode MPS daemon per visible
GPU at `/tmp/nvidia-mps-$USER-$GPUID`. Each daemon is launched with
`CUDA_VISIBLE_DEVICES=$GPUID` so it serves exactly one physical GPU,
exposed as device index 0 in client space. In the worker spawn loop,
each worker is pinned to its GPU's daemon and uses `CVD=0`,
`gamdRunner -d 0`. No shared server, no cross-GPU routing ambiguity.

**Verified result** (5-iter chignolin test, 8+4 workers):

| config                   | GPU 0 mean util | GPU 1 mean util | success | iter 5 walkers/s |
|--------------------------|----------------:|----------------:|--------:|-----------------:|
| MPS=0 (no MPS)           | 80%             | 4%              | varies  | 0.046            |
| MPS=2 shared sudo daemon | 81%             | 3%              | 96%     | 0.284            |
| MPS=2 GPU 0 only         | 84%             | 0%              | 100%    | 0.257            |
| **MPS=1 per-GPU**        | **65%**         | **71%**         | **100%**| **0.286**        |

Aggregate throughput under MPS=1 with the strengthened workaround:
**~2470 ns/day** aggregate across 12 walker slots on chignolin
(100 ps WE segments). ~11% over the GPU-0-only baseline. Modest, but
real and verifiable.

## CUDA_VISIBLE_DEVICES + MPS semantics — easy to get wrong

The two-layer renumbering caused two false-start tests:

1. Daemon started with `CUDA_VISIBLE_DEVICES=1` (physical GPU 1)
   exposes its one GPU as device index **0** in the client namespace.
2. The client's `CUDA_VISIBLE_DEVICES` is interpreted against the
   daemon's namespace, not the host's.

So a worker connecting to GPU 1's daemon must set `CUDA_VISIBLE_DEVICES=0`
(matching the renumbered index) and `gamdRunner -d 0`. Setting
`CUDA_VISIBLE_DEVICES=1` gives `CUDA_ERROR_NO_DEVICE`; setting
`gamdRunner -d 1` gives `openmm.OpenMMException: Illegal value for
DeviceIndex: 1`. Both errors hit us before we sorted out the layering.

## Motherboard sensors on Gigabyte TRX50

The TRX50 AERO D ships with two ITE IT8695 Super I/O chips. Neither is
recognized by `sensors-detect 3.6.0` by ID, but the in-tree `it87`
kernel driver works with explicit `force_id`. To enable persistently:

```bash
# Module options at load time
sudo tee /etc/modprobe.d/it87.conf <<'EOF'
options it87 force_id=0x8695 ignore_resource_conflict=1
EOF

# Auto-load on every boot
echo "it87" | sudo tee /etc/modules-load.d/it87.conf

# Verify (no reboot needed)
sudo modprobe -r it87
sudo modprobe it87
sensors -A
```

You should see two `it87952-isa-0a40` and `it87952-isa-0a60` chip
groups with fan RPMs (`fan1..fan3`), motherboard temps (`temp1..temp3`),
and voltage channels (`in0..in6`, `3VSB`, `Vbat`).

**Reality check on what's reported:**
- **Fan RPMs and temps are accurate** — useful for catching case-fan
  failure or VRM heating early.
- **Voltage labels are NOT trustworthy in absolute terms**. Gigabyte
  uses board-level voltage dividers that the open driver doesn't know
  about, so a `+3.3V` channel may read e.g. 2.07 V — the chip ADC is
  reading correctly, the multiplier is wrong. There's no calibrated
  +12 V channel exposed at all.
- **Voltage trends still work**: a 10% drop in any in0..in6 channel
  under load is meaningful even if you don't know what rail it is.

`run_local.sh` writes timestamped `sensors -A` dumps to `board_health.log`
every 10 s alongside `gpu_util.log` whenever `sensors` is on PATH.
For full PSU-side monitoring you'd want a calibrated rail probe
(motherboard pinheader + ATX extension with sense lines) or a smart
plug at the wall — see the README "Health monitoring" section.

## Recommended production config

```bash
# Once per session: ReBAR confirmed enabled in BIOS, both GPUs visible.
nvidia-smi -L                       # should list both cards

# Stop any prior sudo MPS daemon (USE_MPS=1 starts its own, per-GPU).
echo quit | sudo nvidia-cuda-mps-control 2>/dev/null

# Launch
USE_MPS=1 CUDA_VISIBLE_DEVICES=0,1 WORKERS_GPU0=8 WORKERS_GPU1=4 ./run_local.sh
```

The cleanup trap shuts down the per-GPU user daemons on exit, so the
host is left in a clean state.
