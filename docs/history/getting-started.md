# Getting started — build, THE WORLD, loading a model

The build commands, the flagless boot into THE WORLD (`src/world.rs`), loading an OBJ/glTF, and `--cam`.

Extracted verbatim from `CLAUDE_Historical.md`, which keeps a stub pointing here. Nothing in this file was rewritten.

```
cargo build --release
cargo run --release                   # interactive window (DLSS-RR on when supported) — flagless
                                      # boots THE WORLD (src/world.rs): the curated scenes under
                                      # scenes/ load through their normal loaders (warm .fcache
                                      # per part; a cold part STORES one so the next boot is warm;
                                      # glTF parts re-parse by design), get MERGED post-hoc into
                                      # one flat Scene — islands on a ring over one covering
                                      # ground quad, hour-ordered so a lap sweeps the day
                                      # (powerplant 06:30 -> sponza 08:30 -> rungholt 11:00 ->
                                      # helmet 13:00 -> san-miguel 15:30 -> bistro 17:30 ->
                                      # vokselia 22:00 — the last island is FULL night: the
                                      # antipodal moon is ~60 deg up as the one light, stars and
                                      # fireflies at full strength, so a lap sweeps sunrise ->
                                      # moonlit night; the old 18:45 civil-dusk cap left the moon
                                      # grazing the horizon at ~11 deg).
                                      # The per-island loads run CONCURRENTLY on the GLOBAL rayon
                                      # pool — one task per island, deliberately NEVER a pool of
                                      # their own, which every inner par_iter (texture decode, the
                                      # n2h solve, Bvh::build) would inherit through install,
                                      # narrowing the passes that do the real parallel work.
                                      # Results collect by INDEX off an INDEXED par_iter, so ring
                                      # order stays a function of the table and never of finish
                                      # order; world::self_test pins CURATED's hours STRICTLY
                                      # increasing (which makes the ring sort a TOTAL order, so a
                                      # stable sort can't fall through to load order) and its
                                      # paths UNALIASED (concurrent scene_cache::store calls are
                                      # safe per-FILE only — the tmp name is pid-suffixed, which
                                      # separates processes, not two entries resolving to one
                                      # file). That alias gate compares the RESOLVED form, which
                                      # is where the collision lands: resolve_scene_path maps p to
                                      # p.zst when p is absent AND NOTHING ELSE, so two entries
                                      # share a sidecar iff their paths are equal after stripping
                                      # one trailing .zst — an x.obj entry beside an x.obj.zst
                                      # entry are distinct strings naming one file, which a raw
                                      # string compare would miss. Stripping keeps the gate a pure
                                      # function of the table (calling the real resolver would
                                      # make it depend on which files are materialized).
                                      # The n2h solve's ~0.5 GB-per-4K-solve cap became ONE
                                      # process-wide pool for this (scene::n2h_pool); a per-call
                                      # pool multiplied the bound by the parts in flight.
                                      # progress::MultiGuard brackets the fan-out: its DROP is
                                      # what leaves multi mode, so a worker panic propagating out
                                      # of the collect can't leave that process-global armed for
                                      # whatever publishes next (set_multi is private — the guard
                                      # is the only way in).
                                      # Known-accept: per-part loud lines now INTERLEAVE on stderr
                                      # (nothing parses them, and --check* never loads the world);
                                      # the ring-ordered island summary still doesn't. MEASURED
                                      # (7950X3D, all 5 OBJ sidecars warm): load+merge 6.8 -> 4.6
                                      # s, whole cold-world boot 15.3 -> 13.5 s, and the
                                      # regenerated world.fcache BYTE-IDENTICAL — the determinism
                                      # gate (identical merge, id assignment, ring layout, world
                                      # BVH). The floor is the SLOWEST SINGLE ISLAND (bistro,
                                      # ~4.7 s of texture decode), so any further win lives INSIDE
                                      # a part, not in more fan-out.
                                      # MEMORY, measured the same config (`--cinematic
                                      # --cinematic-dry-run --cpu` returns AFTER the load and
                                      # BEFORE any render, and --cpu keeps the GPU BC7/BLAS spike
                                      # out of the sample; 100 ms working-set sampler): peak
                                      # 9.49 -> 10.13 GB, +6.7%. BOTH arms peak INSIDE THE LOAD
                                      # (t=3.9 s sequential, 3.0 s parallel of a ~5-6 s load), NOT
                                      # in the merge/world-BVH/sidecar phase after it, which tops
                                      # out ~9.1 GB either way. Read the cost against the SAME
                                      # RUN's post-fan-out resident floor — 7.86 GB, arm-
                                      # independent (it IS the merged parts): transient headroom
                                      # 1.63 -> 2.27 GB, +39%. So concurrent transients DO sum
                                      # where sequential ones only had to max, but the resident
                                      # floor dominates both, which is why the process ceiling
                                      # moves 6.7% and not a multiple — the pre-measurement guess
                                      # that this was a "sum not max" blow-up was wrong by an
                                      # order of magnitude, and the n2h pool being process-wide is
                                      # a large part of why. TRAP: a naive whole-process peak
                                      # sampler carries ~1 GB of run-to-run noise (seq 9.6/10.7,
                                      # par 10.3/9.9 — the arms OVERLAP and "measure" no effect at
                                      # all), which swamps the 0.64 GB signal; difference against
                                      # the same run's own baseline, never across runs
                                      # — and the flycam gains per-island TOD
                                      # ATTRACTORS: flying toward an island eases the GLOBAL tod
                                      # toward its theme hour at the manual-scrub rate (circular
                                      # weighted mean, so 22h+2h blends to midnight; converged =
                                      # writes stop, so still frames accumulate; the first manual
                                      # ,/. scrub takes the clock for the session, and an explicit
                                      # --tod disarms auto entirely). The merge is tile_scene's
                                      # idiom for HETEROGENEOUS parts: per-part vertex-index
                                      # rebase, tri_mat + texture-id offsets with the NO_TEX
                                      # sentinel preserved (each part's own ground quad is
                                      # stripped, its ground material left unreferenced), ONE
                                      # default_sun + ONE finalize_scalars (one sky by
                                      # construction; eps/ao_radius go world-diag ~69 — the
                                      # proven --tile regime). Curated skips: hairball (BVH-depth
                                      # pathology), intel-sponza (2.7 GB PNG wall), san-miguel
                                      # LOW-poly (the world takes full-res san-miguel 10M).
                                      # Missing files SKIP with one
                                      # loud line (fresh checkout degrades to fewer islands; zero
                                      # islands falls back to procedural_scene loudly). The world
                                      # has its OWN SIDECAR (scenes/world.fcache, gitignored by
                                      # the *.fcache patterns): the merged Scene + world BVH +
                                      # layout in one file, so a warm boot skips per-part loads,
                                      # the merge, AND the world BVH build. Multi-source key =
                                      # per-curated-entry identity + PRESENCE (an island
                                      # appearing on disk misses) + dependency stats (OBJ:
                                      # source+mtl; glTF: source + external buffers/images via
                                      # gltf_loader::dependency_files) + per-texture stats +
                                      # WORLD_VERSION/CACHE_VERSION/bvh::build_key/lever_word +
                                      # the ring constants — serialized by ONE writer and
                                      # blob-compared, so any drift is a miss by construction;
                                      # corruption is a silent miss (the per-scene budget/link
                                      # discipline). glTF textures ride it as INLINE
                                      # post-conversion texels (their synthetic "gltf:image:"
                                      # sources have no file to re-decode; flags verbatim, mips
                                      # rebuilt per the --no-mips lever via Texture::from_cached)
                                      # — the per-scene cache still deliberately skips glTF.
                                      # Lever flips thrash the one sidecar (per-scene class,
                                      # accepted); v1.1 fixed world cold boots writing per-scene
                                      # sidecars WITHOUT reclassify_spray (CACHE_VERSION 8→9
                                      # retired the wrong ones). Measured
                                      # (7950X3D + 4090): ~34.5M tris (30.1M in the low-poly-SM
                                      # era), cold-heavy first boot
                                      # ~35 s (26.5 load+merge + 8.8 BVH) + the sidecar store;
                                      # any_alpha/any_transmissive become UNIONS (San Miguel arms
                                      # cutout for the whole world — cost accept). The world is
                                      # NEVER the scene for --check*/--spin/positional runs
                                      # (flagless --check stays procedural — no structural gate
                                      # moves); an EXPLICIT --world combined with those exits 2
                                      # (the --fsr4 being-told shape). --no-world = today's
                                      # procedural flagless boot; --world = the default spelled
                                      # explicitly. world::self_test gates the merge math
                                      # (id offsets, NO_TEX preservation, ground-drop accounting,
                                      # ring determinism/disjointness, the circular attractor
                                      # mean) AND the sidecar (round-trip byte-fidelity incl.
                                      # inline/Path textures + rebuilt mips + BVH identity;
                                      # presence/hour/entry/layout-constant/dep-stat misses;
                                      # magic/version/zero-count/truncation corruption) in
                                      # --check, DLL-free. Follow-ons documented in the module:
                                      # per-island firefly scale, cross-part texture dedup,
                                      # zstd-compressing the sidecar
cargo run --release -- model.obj      # load an OBJ (auto-fitted onto ground plane; MTL map_Kd
                                      # textures + alpha-cutout foliage supported — see Real scenes)
cargo run --release -- model.obj --cam 4,2,4,0,1,0  # start camera: ex,ey,ez,tx,ty,tz (reproducible
                                                    # benchmark viewpoints; WASD flight still works)
```
