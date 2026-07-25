# Contributing

This is a personal research renderer. Pull requests are welcome; expect strong
opinions and a request for a measurement.

## Install the pre-commit hook first

```
cp tools/hooks/pre-commit .git/hooks/pre-commit && chmod +x .git/hooks/pre-commit
```

This is not optional housekeeping. Every file in this tree is UTF-8 with no
BOM and the comments are dense with `—`, `×`, `→`, `≤`, `Δ`, `π`. Windows
PowerShell 5.1's default pair — `Get-Content` decoding a BOM-less file as
cp1252, `Out-File` writing UTF-8 *with* BOM — silently double-encodes every
non-ASCII character in a file it round-trips. It has already happened once
(`ca1f9b6` landed `src/main.rs` with 837 mangled sequences; `c39b6f3` repaired
it). Nothing catches it on its own: the damage is comments-only, so it compiles
and every gate passes.

So: **never round-trip a source file through a shell pipeline.** Pass
`-Encoding utf8` on both ends, or edit with a real editor.

The hook is deliberately installed per-clone rather than via `core.hooksPath`,
which would orphan git-lfs's four hooks — and a scene blob reaching plain git
is permanent bloat.

## Read the subsystem's notes before changing it

`CLAUDE.md` is the real design document (~400 KB, organised by subsystem). It
records why each decision was made, what was measured, and — often more useful
— what was already tried and thrown away. Start with `## Correctness invariants
(the bug class to guard)`, then the section for whatever you are touching. Each
section ends with the gates to run for changes in that area.

## The bar

```
cargo run --release -- --check          # required, always. No GPU, no DLLs.
cargo run --release -- --check-gpu      # if you touched GPU code
cargo run --release -- --check-dxr      # if you touched the DXR pipeline
```

`--check` renders a frame, re-traces every pixel with a `tmin = 0` reference
ray, and requires the false-sky and tmin-overshoot counters to be **exactly
zero**. If a change moves `check.png` at all, that is a finding to explain, not
a file to re-commit.

Whatever section of `CLAUDE.md` covers your change names the additional gates
it needs. Run those too.

## Two rules about numbers

1. **Never benchmark under `--profile quick`.** It exists for fast iteration
   and exercising the (perf-independent) correctness gates. Every performance
   number this project reports assumes `release`'s LTO settings.
2. **Interleave and take medians.** A 4090 spans 1.42–1.98 ms for one unchanged
   config; an Arc B70 repeats to ±0.002 ms but silently serves an *unoptimised*
   shader binary for the first several seconds of a fresh variant. A single
   sample is worthless on one and dangerous on the other. `CLAUDE.md`'s
   Profiling section has the details, including the traps that have already
   manufactured phantom results here.

## Style

Match the surrounding code. Comments here explain *why*, carry the measurement
that justified a constant, and name the failure mode a rule prevents — that is
the house style, and it is why the design document is as long as it is.
