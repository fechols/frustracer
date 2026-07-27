# Reference papers

Prior work the frustum tracer is measured against, kept here so the comparison
is reproducible without chasing a link that may rot. Each is a **third-party
publication redistributed under no explicit grant**; the copyright is the
authors' and their publishers', not this repository's.

These are committed to **plain git**, not LFS (see the `docs/papers/**` rule in
`.gitattributes`), so that the top-level README's links resolve for anyone
reading the repo on github.com — including from a fork or a
`GIT_LFS_SKIP_SMUDGE` clone. Keep the directory small and additive: a paper or
two that the README actually cites, never a library. Nothing here is required
to build or run the tracer.

| File | Paper |
|---|---|
| `mlrta105.pdf` | Reshetov, Soupikov & Hurley, *Multi-Level Ray Tracing Algorithm*, ACM SIGGRAPH 2005 — [public copy](https://www.eng.utah.edu/~cs6965/papers/p1176-reshetov.pdf) |
| `MIT-LCS-TR-740.pdf` | Teller & Alex, *Frustum Casting for Arbitrary Polyhedral Environments*, MIT LCS TR-740, 1998 |

Between them these are the antecedents, and they contribute different halves.
**TR-740 is the structural match**: its frustum descriptor — a shared point of
view, four extreme rays, four bounding planes — is the same object as
`TileFrustum`, subdivided by the same screen quadtree. **MLRTA contributes the
deep hierarchy entry point**, of which this renderer's node cut is a
strengthening (an antichain, not a single node). See "Relation to prior work"
in the top-level README for what frustracer adds, what it merely re-measures,
and which of the papers' remaining ideas were measured and found too small to
build.
