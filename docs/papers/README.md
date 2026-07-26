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

The closest antecedent to this renderer: image-space beams, adaptive tile
subdivision, and deep hierarchy entry points. See "Relation to prior work" in
the top-level README for what frustracer adds and what it merely re-measures.
