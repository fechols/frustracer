# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A "frustracer" — a frustum-tracer. The screen is a quadtree: each tile's frustum is traced against a BVH for a conservative nearest-possible-hit distance; on contact the tile splits into 4 children that inherit that distance as their ray `tmin` **and a node cut** (the parent's surviving BVH nodes, refined by `frustum::refine_cut`), bottoming out at 8×8 tiles of per-pixel rays seeded from the cut (`Bvh::intersect_multi`). Tiles whose frustum hits nothing are filled with sky (zero rays). While the camera moves, the same depth-first recursion runs to a **uniform depth cap** estimated each frame from the previous frame's time against a ~15 ms budget; tiles reaching the cap unresolved are flat-filled (dynamic resolution). The same divide-and-conquer also dispatches **bounce lighting** (`src/hemi.rs`, opt-in via `Quality::fb` / the H key): the hemisphere above a shading point is a spherical-triangle quadtree whose empty cells resolve analytically and whose leaf cells shoot cut-seeded stratified rays. Lighting is **one sky** (`src/sky.rs`): a Rayleigh+Mie scattering dome, carried as order-2 SH for its irradiance (`src/sh.rs`), plus a sun disc at infinity that is cone-sampled and shadow-rayed (at night the disc is the MOON, and the star field contributes its own smooth ambient floor — see "the star row") — see the "one sky" invariant below before touching any sky call site. See README.md for the algorithm write-up.

## Commands

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
cargo run --release -- --tod 17.5     # start time-of-day: float hours, wrapped into 0..24, along a
                                      # sun arc through the default sun's azimuth plane (06:00 east
                                      # horizon -> 12:00 zenith -> 18:00 west horizon). Flagless =
                                      # the default sun, BIT-IDENTICAL to the pre-TOD renderer
                                      # (scene::apply_tod is structurally unreachable then). Hold
                                      # `.` / `,` (or Xbox D-pad right/left) to scrub live at 1
                                      # game-hour per second (Ctrl/Shift/bumpers = finer, the
                                      # flight-speed divisors; the 500 Hz flycam thread integrates
                                      # it, the title shows HH:MM). The direct sun fades+reddens
                                      # through sunset by the dome's own transmittance
                                      # (sky::sun_fade); once it sets, the ONE light BECOMES the
                                      # antipodal full moon (sky::moon — same Sun struct, so
                                      # shadows/MIS/disc/dome-tint all just work), the dome falls
                                      # to a moonlit floor (Scene::sky_scale, sky::MOON_DOME_FRAC),
                                      # and a procedural twinkling star field fades in
                                      # (sky::stars — the POINTS are display-only; deterministic,
                                      # twinkle = f(frame index), zero rng). The field also LIGHTS
                                      # the scene: gather paths get its smooth analytic mean
                                      # (sky::star_glow, inside sky::gather), so night has an
                                      # ambient floor that does not depend on the moon's elevation
                                      # — see "the star row" in the one-sky invariant below for why
                                      # points-to-the-eye / mean-to-the-gathers is one delivery and
                                      # not a double count
                                      # A TOD delta = a SHADING change: plain-accumulation
                                      # reset (frame = 0) only — every upscaler/denoiser
                                      # history is KEPT (a held scrub fires per frame, and a
                                      # per-tick history reset starved RR into smearing the
                                      # night star field into cloud-shaped blotches; lighting
                                      # drift is the cloud-drift class the temporal
                                      # integrators absorb), and so are the temporal
                                      # cache/replay (geometry-only claims); GPU arms
                                      # re-derive via TraceGpu/DxrGpu::refresh_sky (cb_base
                                      # sun/sky rows).
                                      # Applied AFTER the .fcache load/store, so the cache always
                                      # holds the default day. --tod with --check* is
                                      # user's-own-risk (the --cam caveat class); scene::
                                      # tod_self_test gates the arc/fade/moon/star math in --check
cargo run --release -- --tod 22 --fireflies 24  # FIREFLIES (src/fireflies.rs, default ON — but they
                                      # only EXIST after dusk: brightness scales by Scene::night,
                                      # the stars' fade scalar, and a night==0 session snapshots
                                      # count=0, so every flagless day run is bit-identical
                                      # STRUCTURALLY, the apply_tod-unreachable precedent;
                                      # --no-fireflies is the kill lever, --fireflies N the count,
                                      # default 32, clamped loudly to MAX_FIREFLIES=64 = the CB row
                                      # cap). Placement spans the content box's WHOLE VOLUME (xz
                                      # footprint × full content height — floor-cleared/top-inset
                                      # by the exact displacement bounds, so poses stay inside the
                                      # box by construction; flat scenes keep at least the original
                                      # [FF_Y_MIN_K, FF_Y_MAX_K] floor band).
                                      # Every FF length is scale-relative to the CONTENT
                                      # diagonal (Scene::content_min/max — the geometry minus the
                                      # standard ground quad, derived in finalize_scalars; NOT
                                      # Scene::diag, which the procedural scenes' ±60 ground quad
                                      # inflates ~17× — placed off diag the swarm hovered over the
                                      # whole field instead of flitting among the models; the scale
                                      # rides the CB's ff_scale lane, ex-_pad0). Each firefly is a
                                      # REAL point light in shade()'s direct tier — windowed 1/d²
                                      # (exactly 0 at FF_RADIUS_K·scale,
                                      # near-field clamped under the f16 ceiling) + ONE HARD shadow
                                      # ray with FINITE tmax (dist − 2·eps; plain occluded — no cut
                                      # exists for that apex), diffuse into direct_d (albedo-free,
                                      # so FSR-RR's denoised dd/ds lobes carry it and the composite
                                      # identity closes untouched), GGX specular into direct_s at
                                      # w_l=1 (a point light has zero solid angle — the VNDF ray
                                      # can never deliver it, MIS does not apply) — plus a
                                      # depth-tested Gaussian GLOW splat on the display paths
                                      # (shade_traced hit/miss + fill_sky_rows + cs_sky/leaf/
                                      # reference/dxr-raygen; sized vs pixel_cone like sky::stars,
                                      # capped at FF_GLOW_L_MAX). Motion is CLOSED-FORM p_i(t):
                                      # sin-hash-free pcg placement + a time-shifted lookup into
                                      # the clouds' own static curl field (clouds::curl_offset
                                      # reused VERBATIM through fireflies::curl_dir — clouds.rs
                                      # unmodified) + hashed sine bob, bounded BY CONSTRUCTION
                                      # (the soft |v|<1 curl normalization is the exact bound; no
                                      # clamp), on the SHARED cloud_time clock (upscaler frames
                                      # advance, plain accumulation only at frame 0, --spin =
                                      # idx·CLOUD_SPIN_DT, every --check* pins CLOUD_CHECK_TIME).
                                      # ZERO rng draws anywhere — same-seed/replay/VisCtl-burn
                                      # contracts untouched (firefly rays bypass VisCtl and always
                                      # trace their own); poses bake once per frame on the CPU and
                                      # upload as f32 CB rows (FLAG_FIREFLIES=1024, ff_count rode
                                      # _pad3, ff[MAX_FIREFLIES] appended after sky_sh, CB_STRIDE
                                      # 1536→2048→2560 — the MAX_SPP-lockstep class), so CPU↔GPU
                                      # positions are bit-equal BY DATA. THE GATHER EXCLUSION (the
                                      # stars rule): fireflies never enter sky::dome/the SH
                                      # projection/hemi gathers/either GI reference — shade's
                                      # recursion, hemi.rs, and both estimators pass ff=None
                                      # (hemi_leaf.hlsl passes fireflies=false); like emissive,
                                      # they do not light bounce surfaces. Shading-only: no
                                      # visibility change, no MVs in the render G-buffers (drift =
                                      # shading change to the upscalers, the clouds accept — BUT
                                      # cloud drift is slow and firefly drift is fast+bright, so
                                      # raw-NGX FRAME GENERATION carries the one exception:
                                      # ngxfg_guides round 3 bakes closed-form firefly MVs into
                                      # the FG-ONLY MV plane at glow-dominated pixels — see the
                                      # --fg block; RR's plane and the ffx/XeSS-FG families still
                                      # see no firefly motion), temporal cache/replay KEPT.
                                      # Known-accepts: glow on the primary camera path only (none
                                      # in reflections/glass); no light through translucency; a
                                      # converging still freezes them mid-flight; slight RR ghost
                                      # risk on the glow (no emissive guide). fireflies::self_test
                                      # gates the off arms/determinism/bounds/falloff/glow energy
                                      # in --check; --check --tod 2 (+ --check-gpu/--check-dxr
                                      # --tod 2) run every gate with the swarm LIVE. MEASURED COST
                                      # (1080p --spin path --tod 2, release, 7950X3D): night
                                      # baseline 49.1 ms/frame, N=16 56.2 (+7.1), N=32 63.4
                                      # (+14.3) — ≈ +0.45 ms per firefly, dominated by the two
                                      # per-pixel linear scans (light rejection + glow), NOT rays
                                      # (sec rays +0.03/px, ray nodes +3% at N=16 — the windowed
                                      # radius works); the interactive budget controller absorbs
                                      # it as resolution, GPU sessions barely notice. One cost
                                      # lesson already measured — do not re-learn it: the glow's
                                      # Gaussian MUST reject on the exponent BEFORE calling exp
                                      # (a > 9.22 ⇔ g < 1e-4 — skipped pairs fail the post-test
                                      # anyway, survivors bit-identical); the unconditional exp
                                      # per (pixel, firefly) was +34 ms/frame at N=16, 4.8× the
                                      # whole feature's remaining cost. Past the 64 cap the
                                      # follow-ons are an SRV table + a per-leaf-tile cull (the
                                      # per-pixel scans are the linear-in-N cost — cull before
                                      # raising the cap again)
cargo run --release -- --emissive-lights  # ARM emissive surfaces lighting the scene — DEFAULT
                                      # OFF, the heightfield arming shape (src/emissive.rs,
                                      # 2026-08-01 — the fireflies template applied to
                                      # Ke/map_Ke/glTF-emissive geometry; bare flag = budget 32,
                                      # `--emissive-lights N` moves it, clamped loudly to
                                      # MAX_EMISSIVE_LIGHTS=64 = the CB row cap, and the token is
                                      # consumed only when all-digits — the --blas-split
                                      # optional-value idiom, so a following scene path is safe;
                                      # --no-emissive-lights spells the default. OFF by default
                                      # because the CPU shadow-ray cost is real (the MEASURED
                                      # block below: ~3.3 of the +5.5 ms is rays — irreducible,
                                      # they ARE the lights) while the PHYSICAL calibration's
                                      # pools are faint before true nightfall (the LOOK FINDING
                                      # below); if a feel-test lands an artistic boost, the
                                      # default is one constant to revisit.
                                      # At load, finalize_scalars enumerates emissive triangles
                                      # (per-tri power = area × Ke × a 4-tap map_Ke mean) and
                                      # clusters them — deterministic grid seed at EL_GRID_K ×
                                      # content diag, then min-power-into-nearest agglomerative
                                      # merge to the budget; SERIAL and index-ordered, so
                                      # byte-deterministic like the BVH build; derived-never-
                                      # serialized (the sky_sh precedent — warm loads re-derive,
                                      # no CACHE_VERSION move). Each cluster is a Lambertian DISC
                                      # light: irradiance C/π/(d²+rc²) — the +rc² denominator IS
                                      # the near-field softening (no hot spot beside a large
                                      # panel) — windowed by the fireflies' (1−d²/r²)² exact-zero
                                      # falloff at an influence radius derived from EL_MIN_E
                                      # (the ONE cost-vs-reach knob: the per-pixel scan pays a
                                      # shadow ray per in-range light), floored at 2·rc, capped
                                      # at EL_RMAX_K·diag_c, lum clamped under EL_E_MAX (f16
                                      # headroom, the sun-disc lesson). Sampled in shade()'s
                                      # direct tier as the third entry after sun + fireflies —
                                      # AFTER cloud sun-transmittance (a lamp under the slab is
                                      # local), BEFORE the prim export (the light rides FSR-RR's
                                      # denoised dd lobe; composite identity closes untouched);
                                      # ONE hard shadow ray per in-range light via transmittance
                                      # (tinted glass composes), stopping rc+2·eps SHORT of the
                                      # center so the emitter's own bulb geometry can't occlude
                                      # its own light (known-accept: nothing inside the cluster
                                      # sphere occludes it). ZERO rng draws anywhere — same-seed/
                                      # replay/VisCtl contracts untouched. DIFFUSE-ONLY by
                                      # design, not thrift: an emitter has real geometry, so its
                                      # specular image is already delivered by the traced VNDF
                                      # ray (the display color+=e at every depth); a firefly-
                                      # shaped w_l=1 specular term would double-count it (MIS'd
                                      # specular = the follow-on). THE GI RULE IS INVERTED vs the
                                      # sun disc: under fb.gi the hemi gather delivers emissive
                                      # transport EXACTLY (real soft shadows, textured emission),
                                      # so GI frames drop the cluster NEE instead of the gather —
                                      # CPU gates the one Some site on !q.fb.gi, GPU clears
                                      # FLAG_EMISSIVE=16384 at fb_mode==2; fb.ao keeps NEE (its
                                      # ambient is sky×AO). No tier double-counts, and hemi.rs /
                                      # hemi_leaf.hlsl / both GI references are UNTOUCHED.
                                      # Transport: CB rows (el_meta + el_a/el_b[64] appended
                                      # LAST after sway_mv_base; CB_STRIDE 2560→4608, the
                                      # MAX_SPP-lockstep class; MAX_EMISSIVE_LIGHTS injected by
                                      # spp_defs) — the root signature is 64/64 FULL, so an SRV
                                      # table is the documented follow-on before the cap ever
                                      # rises (the PER-LEAF-TILE CULL half shipped, next
                                      # paragraph). CPU↔GPU parity BY DATA (the
                                      # ff precedent); shade.hlsli's `fireflies` bool became
                                      # `cam_lights` (one camera-path-NEE gate for both light
                                      # families). An emissive-free scene (procedural, --stress,
                                      # powerplant) derives count=0 — and ANY unarmed session
                                      # runs the pre-feature kernels bit-identically (the flag/
                                      # None gates; check.png byte-verified).
                                      # Gates: emissive::self_test in --check (off arms,
                                      # determinism, power conservation through the merges,
                                      # budget cap, falloff zeros/monotone/clamp, the 4-tap map
                                      # identity — runs regardless of arming, pure math);
                                      # --check-gpu/--check-dxr must-fire emissive_rays > 0 on
                                      # emissive scenes WHEN ARMED (run them WITH
                                      # --emissive-lights on helmet/bistro — the --heightfield
                                      # checks-follow-the-session-flags pattern; CPU-side
                                      # counter, GPU liveness rides the radiance A/B) and
                                      # exactly 0 unarmed (the alpha-rej pattern, same --cam
                                      # caveat class — an armed pose outside every influence
                                      # radius trips it).
                                      # PER-LEAF-TILE CULL (2026-08-01, IN — the scan half of
                                      # the cost, recovered): armed HYBRID arms cull the light
                                      # set ONCE per leaf/capped tile instead of testing all N
                                      # per pixel — CPU emissive::cull_tile (a compacted copy
                                      # hoisted in shade_tile/sparse_fill beside ray_roots; the
                                      # frustum rebuilt from ctx.cam INSIDE the tile, which is
                                      # what keeps structure replay bit-identical for free),
                                      # GPU a group-uniform uint2 mask in leaf.hlsl (trace_
                                      # common::el_tile_culled; TF/tile_frustum HOISTED from
                                      # frustum.hlsli into trace_common so the zero-LDS leaf
                                      # kernel can build one — deliberately a SERIAL uniform
                                      # loop, NOT wave-cooperative: a lane-strided BitOr build
                                      # silently over-culls at FR_LGROUP below the wave width).
                                      # THREE tests, each independently conservative: the 4
                                      # side planes (TileFrustum::sphere_outside — zero
                                      # degenerate normals never cull), the inherited-claim
                                      # near ball (t >= t_start), and the camera FORWARD
                                      # half-space — the third exists because narrow side
                                      # planes provably cannot exclude the ANTIPODAL cone (dots
                                      # ≈ −dist·sin(half-angle)), so walked-past lamps never
                                      # culled without it. EXACT, not approximate: the windowed
                                      # falloff is exactly 0 at r_infl, so a culled light
                                      # contributed nothing (no rng, no counter) — CPU and GPU
                                      # cull INDEPENDENTLY (no bit-parity contract), the plain
                                      # reference / DXR / --defer-shade arms keep the full set
                                      # as unculled oracles, and the same-seed wavefront-vs-
                                      # reference A/B on armed helmet reads EXACT 0.00e0 with
                                      # the 3,006,754-ray anchor unchanged — the shipped proof.
                                      # FR_ABL=noelcull is the bit-identical cost probe (CPU
                                      # emissive::cull_abl + ABL_NO_EL_CULL in abl_defs — NOT
                                      # wavefront_ablation_defs, the probe-reach rule). Gates:
                                      # the self_test cull family (verdicts incl. the
                                      # antipodal-cone pin, conservativeness over a tile-ray
                                      # point grid, order preservation, degenerate-frustum
                                      # keep, near-ball disarm) + --check's `el-cull` must-fire
                                      # (armed: tiles > 0, snapshotted BEFORE the bench loop's
                                      # stats.clear; unarmed: exactly 0; stands down under the
                                      # lever). MEASURED (bistro armed N=32, --spin path 120f
                                      # prefix, interleaved min-of-3 with cooldowns): nocull
                                      # 54.71 -> cull 52.46 ms — the predicted ~2.2 ms scan
                                      # half; helmet check counters read ~30.5/32 culled per
                                      # tile. GPU wavefront: arms OVERLAP (medians 0.79 vs
                                      # 0.85 — noise band, per the wave-aggregation verdict on
                                      # micro-work; do not chase it).
                                      # Showcases: bistro dusk street lamps, DamagedHelmet's
                                      # visor. Calibration instrument the fireflies never had:
                                      # the same pose as a GI still frame IS ground truth for
                                      # the cluster tuning (EL_MIN_E/EL_GRID_K). MEASURED
                                      # (2026-08-01, 7950X3D + 4090, bistro Exterior --spin path
                                      # 120f min-of-2): CPU 49.6 -> 55.2 ms at N=32 (+5.5 ms ≈
                                      # +0.17 ms/light — HALF the firefly cost, as predicted:
                                      # one scan, no glow scan), N=16 54.0 (NOT linear in N —
                                      # merging halves the count but grows each cluster's
                                      # r_infl, so the IN-RANGE count barely moves); GPU
                                      # wavefront INSIDE the 4090's noise band (arms overlap,
                                      # the fireflies precedent). Suite evidence: helmet 32
                                      # clusters / 3.0M NEE rays / radiance A/B 0.043%; bistro
                                      # 32 / 1.0M / 0.003%; GPU-tier liveness proven by
                                      # FR_CHECK_AB_DUMP on/off image divergence. LOOK FINDING,
                                      # not yet resolved: the PHYSICAL calibration (C = area ×
                                      # map_Ke radiance, no boost) reads as glowing bulbs with
                                      # FAINT pools — bistro's lamp clusters total power 0.164,
                                      # so pools only beat the dome after true nightfall at
                                      # close range; an ARTISTIC per-cluster boost (the
                                      # MOON_E_OVER_PI/STAR_E precedent) is the likely retune
                                      # after the feel-test — one constant at the C_c fill in
                                      # emissive::derive_parts, self_test power-conservation
                                      # scales with it
cargo run --release -- --no-audio     # kill lever: no audio (it is ON by default in interactive
                                      # sessions — src/audio.rs: per-island CC0 ambience loops in
                                      # world mode, crossfaded by camera proximity with the TOD
                                      # attractors' inverse-square weight g = r²/(d²+r²) but over
                                      # FULL 3D distance (climbing away fades the island; the
                                      # clock ignores altitude on purpose); a directly-loaded
                                      # curated scene plays its own loop steadily
                                      # (audio::match_scene_path — san-miguel-low-poly,
                                      # intel-sponza, bistro Interior alias onto the curated
                                      # names); plus a PROCEDURAL wind swish everywhere — hash-
                                      # noise through a TWO-stage one-pole low-pass (−12 dB/oct;
                                      # one pole read as harsh static) whose gain (norm^1.5) AND
                                      # cutoff (150→1200 Hz log-lerp) track the REAL camera
                                      # speed, published by the flycam thread as a 500 Hz atomic
                                      # (flycam::speed_handle) so the wind stays responsive while
                                      # the main thread is blocked in a trace. One SDL3 callback
                                      # stream mixes everything (core sdl3 audio, NO mixer
                                      # feature; lewton pure-Rust vorbis decode at load: mono
                                      # downmix → wrap-aware linear resample to 48 kHz → RMS
                                      # normalize → 50 ms equal-power loop-seam bake). Assets:
                                      # scenes/audio/<island>.ogg via LFS (CC0, credited in
                                      # scenes/audio/credits_license.txt); a missing file is one
                                      # loud line + that island silent (the missing-island
                                      # degrade), no device = loud line + silent session.
                                      # DISPLAY-ONLY: no render-path, rng-stream, or gate-read
                                      # contact — every same-seed/replay contract structurally
                                      # untouched, --check* / --spin never construct the device
                                      # (AudioSys lives in run_window beside the FlyCam, so it
                                      # survives resize re-entries). audio::self_test gates the
                                      # pure math in --check (wrap-resampler, seam continuity,
                                      # proximity/wind anchors incl. exact-0.0 stillness, mixer
                                      # must-fires + all-off exact-silence — incl. the
                                      # post-activity GAIN_FLUSH re-latch, since an unflushed
                                      # smoother stalls at a denormal — + determinism, the
                                      # curated-name mapping incl. every CURATED island having a
                                      # loop). Follow-on documented in the module: TOD modulation
                                      # (rungholt birds → crickets after dusk, the fireflies
                                      # precedent)
cargo run --release -- --cloud-shadow 16  # slab-space cloud sun-transmittance cache, N cells per
                                      # cloud wavelength (DEFAULT 16; --no-cloud-shadow = off,
                                      # bit-identical). GPU wavefront AND DXR; the domain reduction
                                      # to F(M.x,M.z) is EXACT, only the bilinear fetch approximates
                                      # (measured worst 6.6% / mean 0.21% at a cloud edge — scale-
                                      # invariant cell/feature = 2/N). Snapshotted at TraceGpu/DxrGpu
                                      # construction; the grid is clouds::shadow_grid_row (one source
                                      # for both tracers + the gates). -21%/sample of the cloud bill.
                                      # Gated: clouds G13/G14 (pure Rust) + the --check-gpu/-dxr
                                      # fill-vs-oracle + on/off A/B
cargo run --release -- --sky-lod 4    # amortized cloud-march lattice, 1/K pixel pitch, power of
                                      # two (DEFAULT 4; --no-sky-lod = off, bit-identical). The
                                      # sharp half of the sky (sun limb, stars) stays per-pixel;
                                      # only the march runs at 1/K^2 and interpolates (0.14% whole-
                                      # image sky error, -9.8% frame at spp=16 / -1.0% at spp=1).
                                      # GPU wavefront AND DXR. The reference kernel reads the SAME
                                      # lattice (record_sky_lod fills it for record_reference), so
                                      # the exact-zero wavefront-vs-reference A/B stays bit-identical
cargo run --release -- --no-clouds    # A/B kill lever: no volumetric clouds (they are ON by
                                      # default — src/clouds.rs, a drifting slab of 2D coverage
                                      # carved by 3D erosion noise, marched TWO-PHASE: 6 coarse
                                      # occupancy probes of the cheap 2D cover (+ an analytic
                                      # interval-vs-column-top skip), then 3 fine sub-steps of
                                      # the eroded field per occupied interval, Beer-Lambert +
                                      # two-lobe HG + a multi-scatter floor, ONE sun probe and
                                      # lighting exp per occupied coarse step, an opaque-core
                                      # break. The march phase is DITHERED per (pixel, FRAME,
                                      # SAMPLE) (clouds::dither_jk — a pure integer hash +
                                      # k/spp stratification, u32-exact CPU↔GPU, consuming
                                      # nothing from any shading rng stream, so "ZERO rng
                                      # draws" still holds; never "clean it up" back to fixed
                                      # midpoints — fixed-phase sample altitudes are
                                      # ray-INDEPENDENT, and any smooth field marched on N
                                      # fixed planes renders as N nested contours, the
                                      # wedding-cake bug that shipped twice; and never back to
                                      # ONE shared j per pixel — N copies of one phase average
                                      # to themselves, which is why --spp used to leave the
                                      # dither untouched. Stratified per-sample phases + the
                                      # frame term make --spp, plain accumulation, and the
                                      # RR/XeSS temporal integrators all genuinely CONVERGE
                                      # the march grain; CLOUD_TEMPORAL_DITHER, mirrored
                                      # CPU/HLSL, is the frame-term kill const). Clouds are
                                      # MOVED by a static low-frequency 3D CURL-NOISE wind
                                      # field (clouds::curl_offset — v = ∇ψ1×∇ψ2 of two 3D
                                      # value-noise potentials, exactly div-free before its
                                      # soft |v|<1 normalization, sampled at raw world
                                      # coordinates so G6 never sees it): the whole field
                                      # (cover+prof+erosion, shadows included — G8 needs the
                                      # shared domain) is sampled through the displacement, so
                                      # clouds deform, wander off the straight wind line, and
                                      # billow vertically as the wind carries them through the
                                      # stationary field. The march folds ONE warp per RAY
                                      # into its origin (per-coarse-step warps measured
                                      # +21 ms/frame CPU and ~2× the wavefront's per-sample
                                      # cost; the fold also keeps the interval-skip EXACT in
                                      # field space — no conservative margin); the public
                                      # density_f/_lo_f keep the exact per-point warp for the
                                      # gates. Every length is scale-relative to Scene::diag.
                                      # Display paths see the backdrop (dome+disc+STARS) through
                                      # the layer's T along their own ray + its scatter; the direct
                                      # sun (dd AND ds — before the FSR signal capture) is scaled by
                                      # a 2-eval clouds::sun_transmittance, so the ground darkens as
                                      # a cloud crosses the sun and the visible disc dims in
                                      # lockstep (the shadow eases out below sun elevation
                                      # CLOUD_SUN_MIN_Y + CLOUD_FADE_BAND — a grazing sun casts no
                                      # cloud shadow, and a TOD scrub never pops one).
                                      # sky::dome() stays CLOUD-FREE (SH ambient, hemi
                                      # gathers, GI references — see the one-sky invariant). OFF is
                                      # bit-identical to the pre-cloud renderer (guarded branches;
                                      # clouds::self_test pins it), and so is every cloud-free RAY
                                      # even when on (the None arm / exact-1.0 transmittance).
                                      # main.rs owns the clock: upscaler/denoiser frames advance
                                      # every frame (clouds drift continuously in the default DXR
                                      # session), plain accumulation only at frame 0 (a converging
                                      # still frame integrates ONE sky), --spin = idx·CLOUD_SPIN_DT,
                                      # every --check* pins CLOUD_CHECK_TIME. Clouds have no MVs
                                      # (drift reads as shading change to the upscalers) and aren't
                                      # visible from above the slab — known accepts. At night the
                                      # sun IS the moon, so moonlit clouds come free; composes with
                                      # --tod. GPU twin: trace_common.hlsli's cloud block
                                      # (FLAG_CLOUDS; state rides cam_right.w/cam_up.w =
                                      # SCENE_DIAG/CLOUD_TIME), one port for wavefront + DXR + hemi.
                                      # MEASURED COST (1080p --spin path, 7950X3D): the CPU tracer
                                      # pays ~+24-28 ms/frame (default 50.7 vs 22.9, SM low-poly
                                      # 43.6 vs 19.3; the curl field + soft dither added ~+6-11
                                      # over the pre-curl 40.0/37.4, roughly half warp evals +
                                      # coverage shift, half LTO codegen drift that resisted
                                      # recovery) — the volumetric price; the interactive
                                      # budget controller absorbs it as resolution, GPU sessions
                                      # barely notice (wavefront per-sample 3.7 → 4.2 ms after
                                      # the per-ray warp fold; NEVER re-try per-coarse-step
                                      # warps). Cost knobs, cheapest first: CLOUD_CURL_AMP_K=0
                                      # (kills the warp evals — the dev A/B, not a flag),
                                      # CLOUD_THRESH (coverage ~ cost — clear
                                      # directions take the staged-cutoff fast path),
                                      # CLOUD_FINE (3 → 2 saves ~3 ms, coarser dither grain),
                                      # CLOUD_EROSION octave count, the 2D density_lo already
                                      # carries every shadow/lighting probe and the
                                      # reflection-miss march (along_rough: 2 steps, lo field —
                                      # a reflected sky is seen through the GGX lobe),
                                      # --no-clouds. Three cost lessons already measured — do
                                      # not re-learn them: fine samples REUSE the coarse cover
                                      # (re-evaluating the smooth 2D placement per fine sample
                                      # was ~4 ms for nothing), the sun/lighting exp pair is
                                      # per-COARSE-step (per-fine was a measured driver), and
                                      # the interval-window skip vs the column's own top is
                                      # mandatory (fine-marching the air above the tops was
                                      # +17 ms). WHERE THE LAYER'S TIME GOES (2026-07 ablation
                                      # A/B — one-line neutralizations, interleaved 600-frame
                                      # spins, drift floor ±0.3): the primary/sky-tile march
                                      # along_k ≈ 21.6 ms of the ~27 (≈ 80%), sun_transmittance
                                      # ≈ 2.6, along_rough ≈ 0 — its "largest single share" era
                                      # PREDATES the per-ray warp fold (header updated; don't
                                      # optimize it on the old comment) — with curl ≈ 6.5
                                      # cross-cutting inside those and CLOUD_FINE 3→2 ≈ 3.7.
                                      # ON THE GPU THAT DECOMPOSITION INVERTS — do not carry the
                                      # CPU shares across. Same ablation shape (B70, --spin path
                                      # 1080p, warm shader cache, per SAMPLE, clouds-off floor
                                      # 0.540): total cloud 0.301 ms, of which
                                      # **sun_transmittance 65%** (0.196), sky march along_k 26%,
                                      # along_rough 5% (0.015). The mechanism: sun_transmittance
                                      # runs on every LIT pixel — ~2/3 of the screen — with no
                                      # early-out and no divergence, while the sky march covers
                                      # ~1/4 and mostly takes its staged-cutoff fast path. A
                                      # scalar CPU is dominated by the branchy 24-step march; a
                                      # GPU by the uniform 2-eval cost on 3x as many pixels.
                                      # TWO SHIPPED CACHES ATTACK THOSE SHARES, both DEFAULT ON,
                                      # both GPU-wavefront AND DXR (promoted from the FR_SKY_LOD/
                                      # FR_CLOUD_SHADOW env levers — retired; the CLI flags are the
                                      # levers now), both bit-identical off. --sky-lod K (default 4)
                                      # amortizes the SKY march on a screen-space lattice at 1/K^2
                                      # (sharp half — sun limb, stars — stays per-pixel;
                                      # src/gpu/shaders/skylod.hlsli): -9.8% frame at spp=16, -1.0%
                                      # at spp=1, ~0.14% whole-image / ~0.85% sky-pixel mean rel
                                      # error, lattice pass 0.03 ms. It reaches only 31% of the
                                      # cloud bill because that is all the sky march is.
                                      # --cloud-shadow N (default 16) caches sun_transmittance in
                                      # SLAB space at N cells per cloud wavelength
                                      # (trace_common.hlsli — the reduction to F(M.x, M.z) is EXACT,
                                      # so there is no depth discontinuity to bilateral-filter):
                                      # **-21% per sample, -8.9% at spp=1, -19.6% at spp=16**, fill
                                      # pass 0.010 ms, 91% of what deleting the feature would buy.
                                      # Its cell is pinned to l0 = CLOUD_SCALE_K*diag (clouds::
                                      # shadow_grid_row — the ONE grid source, shared by both
                                      # tracers and the gates) and the grid SIDE derived per frame
                                      # from the footprint — fixing the side aliases a low sun's
                                      # projection, capping without growing the cell breaks coverage.
                                      # NOW GATED (the vacuity is resolved without a field-scale
                                      # hook): clouds::self_test G13 pins shadow_grid_row's geometry
                                      # (coverage, l0/N cell, the cap-grows-cell, frame-static snap)
                                      # and G14 is the executable large-cloudy-scene probe — bilinear
                                      # vs the exact sun_transmittance oracle over a MULTI-FEATURE
                                      # grid (anti-vacuity: the shadow must vary), MEASURED worst
                                      # 6.6% / mean 0.21% at N=16 (SCALE-INVARIANT: cell/feature =
                                      # 2/N, so a scene larger than one cloud feature sees ~6.6% in
                                      # the sun-transmittance term at a cloud edge — a soft penumbra
                                      # gradient, the accepted cache cost; raise N to shrink it).
                                      # --check-gpu/--check-dxr add the GPU fill-vs-oracle wiring
                                      # gate + an on-vs-off same-seed image A/B (sky-pixel + image
                                      # mean-rel bounds; EXACT off-vs-off when the session ran both
                                      # off). The reference kernel READS the lattice too (record_
                                      # sky_lod fills it for record_reference), which is what keeps
                                      # the exact-zero wavefront-vs-reference A/B bit-identical at
                                      # the default-ON K — keep reference.hlsl's sky_radiance_lod
                                      # arm textually identical to leaf.hlsl's.
                                      # The corner hashes are AVX2-BATCHED (clouds::hashx +
                                      # corner_hashes/corner_hashes3: one 8-lane pcg_mix per
                                      # vnoise3 cell, 4-lane per vnoise, u32-exact lane-wise ⇒
                                      # values BIT-IDENTICAL to the scalar arm, which stays as
                                      # the non-AVX2 fallback; curl's two octaves share one
                                      # floor/fade via vnoise3_grad2; self_test G12 pins the
                                      # lanes bitwise and the change moved check.png by ZERO
                                      # bytes): default spin 47.2 → 44.9 ms (layer 23.2 → 20.9,
                                      # ~−10%) — about the ceiling of VALUE-IDENTICAL work,
                                      # because the scalar hash chains were already ILP-friendly
                                      # and most cover evals take the staged cutoff (octave 0's
                                      # 4 hashes only). Anything bigger is look-affecting
                                      # (coarse/fine step counts, field cheapening) and needs
                                      # the HLSL mirror + the three-pose screenshot check
cargo run --release -- --stress 5000  # perf test: n-object procedural field (composes with --check*)
cargo run --release -- model.obj --tile 3       # replicate a loaded OBJ into a 3x3 grid (also NxM,
                                                # e.g. --tile 4x2): flattened copies, shared
                                                # materials/textures, stress-style camera/light
                                                # framing — the 100M-triangle path (see Big scenes;
                                                # composes with --check* as the loaded-scene gate class)
cargo run --release -- model.obj --no-bc7  # A/B lever: upload scene textures as raw RGBA8. BC7
                                        # block compression is ON BY DEFAULT (8 bpp vs 32 — GPU
                                        # upload only; measured live set incl. mips: Intel Sponza
                                        # 4608 -> 3072 MB, Bistro 2310 -> 1725; the CPU samplers
                                        # keep exact RGBA8 and alpha-masked cutout / height-
                                        # carrying textures NEVER compress — see Real scenes). The
                                        # DEFAULT arm is a GPU COMPUTE encoder (src/gpu/bc7gpu.rs
                                        # + shaders/bc7enc.hlsl: mode-6 PCA/LS fit + a 2-means-
                                        # ranked mode-1 two-subset arm; fxc cs_5_0, the bloom
                                        # no-DXC precedent) dispatched per band inside the scene
                                        # upload — what made default-on affordable: measured at
                                        # fast, SM-lp 117 ms / Bistro 229 ms (2.1 Gtexel/s) /
                                        # Intel Sponza 282 ms (3.8 Gtexel/s; rates count every
                                        # encoded level, mips incl.) vs the ispc CPU
                                        # arm's 0.8 / 9 / 20 s. --bc7-cpu keeps that ispc arm as
                                        # the A/B lever + independent cross-check (M11 worst on
                                        # SM-lp: cpu 33.0 dB vs gpu 32.0); --bc7-quality
                                        # ultrafast|fast|basic|slow = GPU effort tiers (0 = mode-6
                                        # no-refit, 26.3 dB; 1 = +2 LS rounds + CONDITIONAL
                                        # mode-1 top-4; 2/3 = mode-1 always, top-8/16). Encoder
                                        # construction failure = LOUD line + uncompressed RGBA8,
                                        # never an implicit CPU stall. Still no disk cache — the
                                        # per-load encode is now cheap by construction. Every
                                        # --check-gpu runs the bc7-gpu structural gate (flat
                                        # bit-exact + stride + ramp + two-cluster mode-1 proof,
                                        # synthetic — fires even on the untextured procedural
                                        # scene) and M11 runs whenever BC7 is armed on a
                                        # compressible-textured scene
cargo run --release -- --check        # headless: verify + benchmark + write check.png
cargo run --release -- --check-dlss   # headless: DLSS G-buffer MV/depth/matrix self-test (no GPU)
cargo run --release -- --dlss-dump    # --check-dlss + G-buffer PNG dumps
cargo run --release -- --no-dlss      # skip the DLSS-RR level of the ALWAYS-ON upscaler chain:
                                      # every session probes DLSS-RR -> FSR4-RR -> XeSS -> FSR3 in
                                      # that order and wires the FIRST supported level (--<x>
                                      # force-starts the chain at level x, --no-<x> skips a level;
                                      # upchain::self_test in --check gates the resolution algebra;
                                      # chain exhausted = one LOUD line + plain presentation)
cargo run --release -- --no-upscale   # plain presentation: no temporal upscaler at all — the
                                      # benchmark escape and the ONLY spelling of the old --no-dlss
                                      # plain path
cargo run --release -- --cpu          # the CPU frustum-tracer as the render mode — clears BOTH GPU
                                      # modes (--dxr is the DEFAULT session on NVIDIA/AMD: the eager
                                      # DispatchRays pipeline feeding the chain's wired upscaler;
                                      # --gpu is the wavefront tracer, and the FLAGLESS default on an
                                      # INTEL adapter — see the vendor-defaults paragraph in the DXR
                                      # section). Later flags win: `--cpu --gpu` = --gpu
cargo run --release -- --check-oidn   # headless: OIDN denoise self-test (needs the OIDN DLLs on disk)
cargo run --release -- --oidn-dump    # --check-oidn + before/after/G-buffer PNG dumps
cargo run --release -- --oidn         # start with OIDN denoising on (N toggles; DLSS off)
cargo run --release -- --oidn --oidn-no-temporal  # OIDN without the reprojection history (M toggles)
cargo run --release -- --oidn --oidn-quality high # OIDN RT-filter quality: fast|balanced|high (default balanced)
cargo run --release -- --oidn --oidn-no-clean-aux # don't declare the OIDN guides noise-free (A/B lever)
cargo run --release -- --check-xess   # headless: XeSS dynamic-res contract self-test (no GPU, no DLL)
cargo run --release -- --xess-dump    # --check-xess + G-buffer PNG dumps
cargo run --release -- --xess         # XeSS-SR dynamic super-resolution (X toggles; force-starts
                                      # the upscaler chain at XeSS — missing libxess.dll falls to FSR3)
cargo run --release -- --xess --oidn  # + OIDN pre-denoise at the dynamic render res (N cycles off/pre/post)
cargo run --release -- --xess --oidn-post  # + OIDN post-denoise on the upscaled window-res frame (A/B lever)
cargo run --release -- --xess --no-adaptive  # XeSS without the adaptive shading rate (uniform per-pixel shading)
cargo run --release -- --xess --xess-autoexposure  # XeSS computes exposure internally (A/B lever)
cargo run --release -- --check-fsr    # headless: FSR signal-split/encoding/MV/provider-pick contract self-test (no GPU, no DLL)
cargo run --release -- --fsr          # force-start the upscaler chain at FSR4 + Ray Regeneration
                                      # (K toggles; RDNA4 only — elsewhere the chain falls through
                                      # XeSS to FSR 3.1 upscale-only, cross-vendor; also flips the
                                      # default adapter preference to AMD)
cargo run --release -- --fsr4         # --fsr, but the level is REQUIRED, not merely force-started: a
                                      # chain fall-through (no RDNA4 / no Ray Regeneration provider /
                                      # wrong adapter) is a HARD ERROR — exit 2 with the probe's reason
                                      # and the flags worth trying (--fsr3, --prefer-amd, --fsr). The
                                      # one non-fallback in the codebase; being told IS the feature
cargo run --release -- --fsr3         # force-start the chain at the FSR 3.1 upscale-only level even
                                      # where FSR4+RR exists (A/B lever; no 3.1 provider = loud line
                                      # + plain, never a silent un-force)
cargo run --release -- --fsr3 --fg    # FRAME GENERATION, ffx family (W4 leg 1 of 3; legs 2-3 =
                                      # raw-NGX DLSS-G for DLSS sessions, XeSS-FG+XeLL for Intel
                                      # XeSS sessions — all three live). DEFAULT ON since
                                      # 2026-07-24 (--no-fg is the kill lever; --fg spells the
                                      # default; a defaulted fg under --quinlight disarms with a
                                      # loud line instead of the explicit pair's exit 2 — the
                                      # Opts::fg_explicit / mode_explicit pattern).
                                      # Exposed in the settings menu as the Upscaler page's
                                      # frame-generation row (restart-tier; the file drives the
                                      # DEFAULT arm only — never fg_explicit, so a menu click
                                      # can't make --quinlight fatal).
                                      # In an
                                      # FSR session (FSR4-RR or FSR3 wired) the swapchain is
                                      # WRAPPED by the FidelityFX frame-interpolation proxy at
                                      # creation (d3d12::SwapWrap — the wrap sits between
                                      # colour-space negotiation and RTV creation because
                                      # GetBuffer on the proxy returns the PROXY's backbuffers;
                                      # a PQ declare is re-asserted through the proxy, which ffx
                                      # supports — the FI swapchain reads the transfer function
                                      # off the chain's declared colour space, and the display
                                      # probe's real min/max nits ride the dispatch under
                                      # Hdr10 only) and ONE generated frame
                                      # is inserted per rendered frame (measured 4090, THE
                                      # WORLD, DXR->FSR3 vsync: 96 rendered -> ~195 presented
                                      # fps, the exact-halving pacing signature; same stack live
                                      # on the B70 and the AMD iGPU — AMD interpolation on Intel
                                      # silicon is the cross-vendor demo). The FG provider ships
                                      # in a DIFFERENT sample dir than the loader
                                      # (--fg-path/FRUSTRACER_FG_PATH; ffxshim_preload_dir skips
                                      # basenames already in the module list, so the primary
                                      # --ffx-path stays authoritative). fsr::pick_fg_version:
                                      # an FSR4 session prefers the 4.x ML frame generation,
                                      # everything else the 3.1 interpolation, other major =
                                      # fallback (the enumeration is device-filtered; the 4.0.1
                                      # provider DLL enumerates only "3.1.6" on non-RDNA4 —
                                      # RDNA4-gating observed live), never id 0 — gated in
                                      # --check-fsr. Per-frame contract: frame_id advances by
                                      # EXACTLY 1 (any other delta resets interpolation history
                                      # by ffx contract); the six FSR present arms record a
                                      # PrepareV2 dispatch (reversed-Z clip depth + the MV plane
                                      # with the SAME mv_scale their upscale dispatch uses — trio
                                      # = pixels (1,1), RR plane = UV-deltas (rw,rh); one MV
                                      # convention per session by construction) and configure the
                                      # FI swapchain live; fullscreen_to_backbuffer's HANDSHAKE
                                      # covers everything else — any frame presented WITHOUT a
                                      # prepare (plain arms, SPACE mode switches, the pause-menu
                                      # present_again hold) finds `prepared` unset and configures
                                      # the proxy DISABLED first (idempotent via `live`), so
                                      # pacing never runs against stale motion.
                                      # THE MODE-SWITCH STRADDLE (2026-07-31 — the AMD
                                      # mode-cycle-slowdown fix): carrying the prepare stream
                                      # SEAMLESSLY across a SPACE/F render-mode switch — a
                                      # reset=1 prepare + the depth/MV resource-set swap (each
                                      # arm feeds FG its OWN planes: CPU-upload vs
                                      # wavefront-pack vs DXR-pack) + a frame-time cadence jump
                                      # (66 -> ~5 ms in the trace that caught it), generation
                                      # enabled throughout — wedges the AMD provider's pacing
                                      # into a MASSIVE persistent slowdown after a few SPACE
                                      # laps (R9700, THE WORLD; NVIDIA/Intel never — they run
                                      # different FG families). NOT VRAM: measured 8.4/31.7 GB
                                      # while slow (the mode: vram line exists from this hunt).
                                      # Diagnosed by elimination, each arm user-measured:
                                      # --no-fg clean, F11 resize CURES (context rebuild), K
                                      # plain-toggle round trip CURES (disable/enable configures
                                      # with NO rebuild), FR_FG_CYCLE=recreate prevents. So
                                      # GpuContext::fg_mode_switch (fired from main.rs's
                                      # landed-switch hook only — a refused press straddles
                                      # nothing) DEFAULTS to the cheapest cure as prevention:
                                      # skip the next prepare, so the funnel hands the FI proxy
                                      # exactly ONE disabled passthrough present at the seam
                                      # (the K sequence compressed to a frame; frame_id
                                      # deliberately does not advance — the disable configure
                                      # reuses the last id, bit-identical to the K path).
                                      # FR_FG_CYCLE=off restores the carry-across repro arm,
                                      # =recreate is the heavy A/B (effect-context rebuild, the
                                      # resize straddle — also proven curative). Instruments
                                      # from the hunt, all shipped: the always-on
                                      # `fg: interpolation paused/resumed` transition lines
                                      # (pause counts + frame_id), FR_FG_TRACE=1 (reset
                                      # prepares + resource-set-swap lines), `mode: vram` per
                                      # switch, and per-tracer construction vram lines. NGX and
                                      # XeSS-FG deliberately keep their existing seam handling —
                                      # the wedge is ffx-family-only as measured. Resize: pending
                                      # paced presents retire, the display-size-bound FG effect
                                      # context rebuilds, the swapchain context survives
                                      # (ResizeBuffers forwards). Teardown: GpuContext::drop
                                      # waits presents with the queue live; `fg` is declared
                                      # AFTER `d3d` so the proxy refs release before the
                                      # swapchain context destroys the proxy. Known-accepts v1:
                                      # HUD is baked pre-present (interpolated with the frame —
                                      # static HUD ≈ invisible; the premul UI-resource
                                      # registration is plumbed in the shim, unwired); XeSS/plain
                                      # sessions wrap but present passthrough (their FG families
                                      # are legs 2-3); latency untouched (W5). FG COMPOSES
                                      # WITH --quinlight (2026-08-01): the family follows the
                                      # session, the quin present arms carry the per-frame
                                      # contract (gpu/mod.rs::quin_fg_tail — NGX interpolates
                                      # the FUSED image via ngxfg_target, ffx FI prepares from
                                      # the planes actually FED — the FSR4-RR pair when wired,
                                      # else the XeSS trio, NEVER a shared FSR3's own stale
                                      # planes — and XeSS-FG tags per present; VERIFIED live
                                      # 4090: NGX pair-present over [dlss-rr+fsr3+xess], ffx
                                      # 3.1.6 generating over [fsr3+xess] under --no-dlss).
                                      # Headless (--check*/--spin)
                                      # never consults it. Vendored: the v2.3.0 framegeneration
                                      # headers (FG kit 4.0.1 + FI swapchain 3.1.7) under
                                      # SDKs/fidelityfx-sdk/framegeneration/. Touch the shim FG
                                      # block/ffx.rs FG wrappers/fg_prepare/the wrap hook -> run
                                      # --check, --check-fsr, --check-gpu, then the interactive
                                      # smoke on 4090 + B70 (fg lines + the cadence-halving test)
cargo run --release -- --fg           # FRAME GENERATION, DLSS family (W4 leg 2): in a DLSS
                                      # session (the flagless NVIDIA default) fg — ON BY DEFAULT
                                      # — arms RAW-NGX DLSS-G, the ONE DLSS FG backend since the
                                      # Streamline retirement (the SL DLSS-G fallback — the
                                      # declines-to-insert open issue that also rejected scRGB —
                                      # is DELETED with the interposer; a build without the DLSS
                                      # SDK has no DLSS at all, RR included: one loud line, the
                                      # chain falls to FSR4/XeSS/FSR3):
                                      # RAW NGX (the DLSS SDK present at build —
                                      # FRUSTRACER_DLSS_SDK, default
                                      # ..\quinlight-player\SDKs\DLSS-SDK; never committed,
                                      # build.rs cfg(dlss_ngx) + stages nvngx_dlssg.dll AND
                                      # nvngx_dlssd.dll — one SDK, one gate, both features) —
                                      # VERIFIED GENERATING on the 4090: shim/dlssg_shim.cpp
                                      # (the quinlight-player blueprint, adapted with REAL
                                      # camera data) drives NVSDK_NGX_Feature_FrameGeneration
                                      # directly; the feature retains the previous rr.output
                                      # internally, one evaluate per frame writes the
                                      # in-between frame into fg_n.out, and ngxfg_tail
                                      # PAIR-PRESENTS: tonemap(interp) -> present_mid (Close+
                                      # Execute+Present+Reset on the same slot allocator, the
                                      # split_frame legality) -> tonemap(real) -> end_frame.
                                      # Under vsync the two presents land a vblank apart =
                                      # the pacing (measured: rendered 186 -> 93 fps while
                                      # presents hold ~174/s — the exact-halving signature).
                                      # NO handshake needed (nothing generates behind our
                                      # back), DLSS-RR runs in the SAME session (both raw NGX,
                                      # one refcounted init — shim/ngx_shared), and the
                                      # swapchain format is IRRELEVANT (no swapchain policing —
                                      # there is no swapchain hook; NGX sees only internal fp16
                                      # textures). THREE TRAPS, all
                                      # measured: (1) [SL-era, now moot — the init is
                                      # unconditionally refcounted through ngx_shared since
                                      # both consumers are ours] NGX could already be
                                      # initialized in-process by Streamline; two differently-
                                      # keyed inits on one device silently break each other;
                                      # (2) a null app-data
                                      # path fails init with 0xBAD0000F FAIL_UnableToWrite-
                                      # ToAppDataPath — pass %LOCALAPPDATA%\frustracer\ngx;
                                      # (3) motionVectorsInvalidValue must be FLT_MAX, not 0
                                      # (0 tags every static pixel invalid — the quinlight
                                      # lesson). THREE MORE, found chasing the DamagedHelmet
                                      # sky-reflection swim (generated frames only) — the
                                      # common root: quinlight's inputs were ZERO MVs, zero
                                      # jitter, and a synthetic [0,1] luma-depth, so NOTHING
                                      # motion-dependent in the blueprint was ever validated;
                                      # treat every "quinlight-settled" constant that way.
                                      # THAT ROOT BIT AGAIN on 2026-07-26 — TRAP 9, JITTER
                                      # SIGN: the evaluate was handed the NEGATED sample
                                      # offset, reasoned by analogy from Streamline's RR
                                      # (which does want it negated) on "same NGX family, one
                                      # sign". RAW NGX WANTS IT AS IS. quinlight's jitter was
                                      # (0,0), so every sign is identical there and the
                                      # blueprint could never disagree. A sign error misplaces
                                      # content by TWICE the jitter (~1 px) — invisible on
                                      # diffuse geometry, BLATANT on a small ultra-bright
                                      # specular highlight (the sun off DamagedHelmet's metal:
                                      # ~44,000 radiance against a ~1.0 scene turns 1 px into
                                      # a strobe). It predates the c66417d "swim FIXED"
                                      # commit — that binary reproduces it — so it was never a
                                      # regression, just never caught: the swim fix addressed
                                      # reflections DRAGGING, this is the highlight JUMPING.
                                      # Now `raw` by default; FR_NGXFG_JITTER=neg restores it.
                                      # HOW IT WAS FOUND, because the method transfers: every
                                      # environmental variable was eliminated by measurement
                                      # (resolution, frame rate, the resize path, the
                                      # virtual-image MVs, PAIR_BACKBUFFERS, the scene, and a
                                      # c66417d-era build), and then the SAME FRAME through
                                      # the ffx FI interpolator came back CLEAN — which
                                      # localized it to our NGX inputs and left the FR_NGXFG_*
                                      # levers to walk it down in two runs. A cross-vendor
                                      # A/B beats another mechanism hypothesis: three
                                      # plausible ones (DestroyParameters, the virtual-MV
                                      # blend, the sky-reflection distance) each explained the
                                      # symptom and each measured wrong.
                                      # (4) DEPTH: the snippet's Depth slot has DLSS-SR's
                                      # contract — a [0,1] buffer CONSISTENT WITH THE SUPPLIED
                                      # MATRICES — while RR's plane holds unbounded linear
                                      # view-Z (RR reads it via the LINEAR-depth tag, a
                                      # different contract). (5) MVEC SCALE: DLSSG.MvecScale
                                      # converts stored MVs to PIXELS — settled from
                                      # dlssg-to-fsr3, which hands it STRAIGHT to FSR3's
                                      # motionVectorScale across shipped SL titles; the SDK
                                      # header's "[-1,1]" comment is stale, and the
                                      # quinlight-era {1/rend} starved the snippet of geometry
                                      # motion ~2000× (why the depth fix alone changed nothing
                                      # visible). Our MV plane stores pixels ⇒ mv_scale {1,1}.
                                      # (6) REFLECTION MVs: surface MVs describe the SURFACE,
                                      # but a mirror pixel's CONTENT is the reflection — a
                                      # VIRTUAL IMAGE at path depth t_surf + t_refl (planar
                                      # unfold along the primary ray; a MISSED reflection is
                                      # the SKY, i.e. a virtual image at INFINITY with EXACTLY
                                      # zero translation parallax — the "reflection drifts
                                      # opposite the surface" strafe observation), so
                                      # warping with surface MVs drags the reflection with the
                                      # helmet on every generated frame. Both conversions run
                                      # in ONE fused pass, gpu/ngxfg_guides.rs (fxc cs_5_0,
                                      # the bloom no-DXC precedent — records inside
                                      # ngxfg_dispatch, the one site all three RR arms share):
                                      # clip depth d = A + B/z (the EXACT perspective_lh
                                      # z-mapping — deliberately NOT xess::view_z_to_clip_
                                      # depth, which is REVERSED-Z and inconsistent with the
                                      # matrices NGX gets) + an FG-ONLY MV plane
                                      # lerp(mv_surface, mv_virtual, w), w = lum(spec_alb)/
                                      # (lum(diff_alb)+lum(spec_alb)) damped over roughness
                                      # ROUGH_LO..ROUGH_HI (metal helmet ⇒ w≈1; RR's own MV
                                      # plane untouched — RR is trained for surface MVs + the
                                      # spec-hit guide; spec_hit_t is the reflection distance
                                      # source, 0 = no ray ⇒ passthrough). ONE LANE, TWO JOBS:
                                      # the pack clamps a MISSED reflection to CAM_FAR because
                                      # that lane's OTHER consumer is RR's depth delta, which
                                      # wants far — but "far" is a LIE as a reflection
                                      # distance (2*diag ≈ 138 world units is not infinity),
                                      # and feeding it the point form gave the sky real
                                      # parallax. The kernel now takes the analytic LIMIT for
                                      # t_r >= cam_far: as t_r → ∞ the virtual point becomes a
                                      # DIRECTION, so it projects with the translation column
                                      # dropped (w = 0) ⇒ rotation-only, exactly right.
                                      # True RADIANCE-
                                      # weighted w needs a dd/ds/ind_s-style capture in DLSS
                                      # sessions (the FLAG_FSR_SIG precedent) — the follow-on
                                      # if albedo-weighting leaves residue. ROUND 3 of the same
                                      # pass (the night-swarm strobe fix): fireflies move every
                                      # rendered frame with NO MVs anywhere (the glow is a
                                      # color-only add after the G-buffer capture), so FG warped
                                      # the bright blobs with the BACKGROUND's MV — and on
                                      # smooth/metal pixels the round-2 material-driven blend
                                      # confidently handed it the virtual-reflection MV at
                                      # exactly those pixels. Poses are closed-form, so the CPU
                                      # bakes per-firefly SCREEN-SPACE splat rows (ff_guide_rows:
                                      # cur px, prev px through the same world->prev-clip matrix,
                                      # view-Z, sigma-px, center lum — prev poses = the LAST
                                      # SUCCESSFULLY EVALUATED frame's swarm, retained beside
                                      # `primed`; a count mismatch reprojects the current pose,
                                      # camera-motion-only, never wrong-signed) and the kernel
                                      # lerps toward mv_i = prev_px - cur_px where glow luminance
                                      # dominates (w = S/(S+FF_MV_L_REF); analytic weight, never
                                      # an accum read — a 1-spp denominator would flicker the MV
                                      # plane; the exp-reject rides the fireflies +34 ms lesson
                                      # with a 1e-4 skirt so the weight is continuous at the
                                      # cut). MV constant across a splat (rigid translation —
                                      # per-pixel reprojection would contract the blob). The
                                      # table rides a root CBV (b1) on a FRAMES_IN_FLIGHT upload
                                      # ring; ffc=0 (day / --no-fireflies / lever-off) executes
                                      # the pre-round-3 kernel stream bit-identically.
                                      # FR_NGXFG_FFMV=off is the A/B (strobe returns on demand).
                                      # ROUND 4 of the same pass (the PARKED-CAMERA WATER
                                      # strobe) is round 3's shape again, not the swim's: a
                                      # MISSING MV, not a wrong one. Water's mirror normal
                                      # MOVES — ripple_normal tilts it every rendered frame
                                      # on the cloud clock (~14 deg of tilt ⇒ ~28 deg of
                                      # reflected swing, 1-3 deg per frame at 60 fps) — so the
                                      # reflected skyline slides across a surface whose
                                      # GEOMETRY is still, and every MV plane (camera motion
                                      # only) reports zero. Water was also the class most
                                      # exposed already: roughness 0.05 is below ROUGH_LO, so
                                      # it took 15-45% of the virtual-reflection MV with NO
                                      # roughness damping. The still-mirror unfold is
                                      # normal-FREE only because mirroring the reflected point
                                      # across the surface plane sends it back down the
                                      # primary ray — true while the normal holds still. The
                                      # field is closed-form, so the PREVIOUS normal is
                                      # computable, and reflecting the current content
                                      # direction off it collapses BOTH branches into one
                                      # expression: d = reflect(reflect(du, n_c), n_p).
                                      # reflect is an involution, so n_p == n_c gives d == du
                                      # EXACTLY — the finite arm reduces to org+du*(ray_t+t_r)
                                      # and the sky arm to du, i.e. the round-2/3 kernel
                                      # bit-for-bit. That identity IS the safety argument.
                                      # n_p is first order (the ripple SUBTRACTS the in-plane
                                      # gradient, so stepping the gradient back steps the
                                      # normal back); exact at dt=0, second-order otherwise
                                      # (<= ~0.8 deg), and a degenerate/horizon-crossing
                                      # reconstruction falls back to n_c = the exact
                                      # pre-round-4 unfold (coarser, never wrong-signed).
                                      # PLUS FRESNEL, scoped to ripple pixels so everything
                                      # else stays bit-identical: ls/(ld+ls) is an F0 proxy
                                      # that sits flat at ~0.15-0.45 on water while real
                                      # reflectance runs ~2% face-on to ~100% grazing, so
                                      # without it the fix applies a FRACTION of the correct
                                      # MV exactly where the sliding skyline is and HALVES the
                                      # strobe. Face-on stays low deliberately — the refracted
                                      # basin dominates there and has no MV at all, so the
                                      # near-zero surface MV is the honest answer.
                                      # PLUMBING: ripple_amp rides GBufExt.alb.w (the ONE
                                      # documented-unused lane ⇒ no stride change, and the ext
                                      # gates that skip lane 7 keep passing), reaching the
                                      # guide pass on an 8th RrResources plane delivered
                                      # through FEED_FSR_AO (u26 — same RWTexture2D<float>/
                                      # R16F, and an RR session never runs the FSR-RR kernel
                                      # that owns it; NOT an RR input, RR is never tagged with
                                      # it). The clock pairs with prev_ff beside `primed` (set
                                      # only on a SUCCESSFUL evaluate) and t_prev defaults to
                                      # t_cur, NEVER 0.0 — a session minutes into its clock
                                      # would otherwise inject a huge bogus delta on the first
                                      # armed frame, the confident-wrong-MV failure this whole
                                      # pass exists to avoid.
                                      # FR_NGXFG_RIPPLEMV=off is the A/B. Gate teeth: a
                                      # parked-camera probe scans dt until >= 5 px (loud if
                                      # never — anti-vacuity), the oracle is an INVERSE
                                      # ROUND-TRIP rather than a re-derivation, and the
                                      # pre-fix answer must FAIL the bound (measured 5.50 px
                                      # vs still-mirror 0.0000). The pack lane is gated in
                                      # --check-gpu/--check-dxr and PROVEN NON-VACUOUS on
                                      # rungholt (water px 1552) — the default and san-miguel
                                      # probe poses see NO water, so that gate alone would
                                      # have passed while proving nothing.
                                      # FG-ONLY: RR's MV plane, ffx FI and XeSS-FG unchanged
                                      # (their zero-MV glow drag is a documented accept); firefly
                                      # SPECULAR highlights still ride surface/virtual MVs
                                      # (half-vector geometry, out of scope). Gated in --check as
                                      # `ngxfg-guides` (clip-depth matrix-consistency sweep;
                                      # virtual-MV: static-camera zero, t_r=0 continuity vs
                                      # CamBasis::project itself, the strafe reflected-sky
                                      # collapse, weight anchors; round 3: off arms exact +
                                      # empty-table blend bit-identity, the moving-firefly gate
                                      # with anti-vacuity and TEETH pins — the pass-through
                                      # surface MV must FAIL the bound — occlusion/behind-camera
                                      # drops, the weight-continuity skirt pin, and the
                                      # projection-route/sigma-lum anchors vs CamBasis::project
                                      # and the shipped fireflies::glow). THE STRAFE GATE'S OWN
                                      # LESSON: it was RELATIVE (`mv_virt <= 0.05 * mv_surf`)
                                      # where the correct answer is EXACTLY ZERO, so any
                                      # percentage of a large surface MV passed; and it ran at
                                      # far = 5000 while the renderer ships far = 2*diag —
                                      # 36x more distant, making "reflection at far"
                                      # impersonate infinity far better in the gate than in
                                      # the product. It is now ABSOLUTE, sweeps
                                      # production-scale far values, and carries a TEETH pin
                                      # (the pre-fix point form must blow the bound). Note
                                      # this is the mirror of the --spp image A/B lesson,
                                      # where an ABSOLUTE limit was the wrong shape: pick the
                                      # form from what the true value is, not by habit. A SEVENTH trap, structural:
                                      # pair-present consumes TWO backbuffers per frame, so at
                                      # the shipped BACKBUFFERS=3 a buffer came back around
                                      # 1.5 frames later — under vsync with the DXGI present
                                      # queue full that re-renders into a buffer still queued
                                      # for scanout (stale-frame flicker; a timing race no
                                      # debug layer flags). Raw-NGX sessions now create the
                                      # swapchain at d3d12::PAIR_BACKBUFFERS=6, restoring the
                                      # exact 3-buffers-per-present ratio every other session
                                      # has (quinlight's pair-present had its own fence ring —
                                      # PAIR_PRESENT_FENCES — which the port had dropped).
                                      # THE INPUT CURVE (ngxfg_guides::TonePass, DEFAULT ON as
                                      # `reinhard`, 2026-07-31 — the sun-strobe fix): NGX's
                                      # flow estimator needs a DISPLAY-CURVE-shaped input, not
                                      # scene-referred linear radiance (sun disc ~810 vs scene
                                      # ~1) — diagnosed by elimination (camera/matrix/jitter/
                                      # depth/MV/HDR-declaration/magnitude/clouds/bloom/DRS all
                                      # measured out) with the ffx FI interpolator CLEAN on
                                      # identical content, then confirmed by the arms: `scale`
                                      # (ratios preserved) double-ghosts under rotation, `log`
                                      # (bounded but midtones crushed to 0.06) ghosts + bands,
                                      # `reinhard` (v/(1+v), a real tonemap operator putting
                                      # 1.0 at 0.5) correct parked AND rotating. So every --fg
                                      # session compresses rr.output into a scratch, hands THAT
                                      # to NGX, and expands the interpolated output in place —
                                      # presentation untouched, the REAL pair half bit-identical
                                      # in every mode. KNOWN-ACCEPT: the inverse is
                                      # ill-conditioned near the ceiling (one f16 quantum ≈ 490
                                      # radiance), which can band the outer bloom edge on
                                      # GENERATED frames only; the tracked follow-on (hand FG a
                                      # genuinely display-referred image + present through a
                                      # matching curve — no inverse pass) is blocked on
                                      # fullscreen_to_backbuffer render-target parameterization
                                      # and the bloom pyramid running on linear input. A second
                                      # SURVIVING artifact is NOT ours: straight piecewise-
                                      # linear banding in the smooth aureole under camera
                                      # motion, settling parked — NGX's own optical flow hitting
                                      # the aperture problem in a textureless gradient (per-
                                      # block flow drifts independently; would band along
                                      # curved iso-radiance contours if it were our round-trip
                                      # quantization, and it doesn't).
                                      # Empirical-settling + ELIMINATION env levers (the
                                      # FR_ABL read-only-probe idiom, loud on departure):
                                      # FR_NGXFG_TONEMAP=off|scale|reinhard|log (off = raw
                                      # linear to the evaluate — the sun strobe returns on
                                      # demand; scale/log = the diagnostic arms above),
                                      # FR_NGXFG_DEPTH=linear, FR_NGXFG_RMV=off (surface MVs —
                                      # brings the reflection swim back on demand),
                                      # FR_NGXFG_JITTER=0|raw, FR_NGXFG_MV=norm|neg|normneg
                                      # (scale/polarity walks), FR_NGXFG_FFMV=off (surface MVs
                                      # at firefly glow pixels — the night-swarm strobe A/B,
                                      # see round 3 above), FR_NGXFG_RIPPLEMV=off (a
                                      # STILL-mirror unfold on water — the parked-camera
                                      # water strobe A/B, see round 4 below),
                                      # FR_NGXFG_CAM=identity
                                      # (quinlight's proven identity-camera block — isolates
                                      # our matrix plumbing), FR_NGXFG_MAT=col (column-major
                                      # matrices — the majority was never validated: quinlight's
                                      # identities are transpose-invariant), FR_NGXFG_SHOW=
                                      # interp|real (present ONE side for both halves of the
                                      # pair: interp = inspect generated frames at full rate —
                                      # non-generating frames fall back to the real frame, so
                                      # a failed/skipped evaluate never re-presents a stale
                                      # out-texture; real = nothing NGX-made on screen, the
                                      # present-path null test — pacing identical in all
                                      # modes), FR_NGXFG_PACE=1 (per-frame pacing probe:
                                      # backbuffer indices of both pair halves + DXGI frame
                                      # statistics per rendered frame — diff pacing between
                                      # arms from a log; FOREGROUND window only, DWM retires
                                      # an occluded window's presents unthrottled). An
                                      # unrecognized lever value is LOUD and takes
                                      # the default (a silent no-op A/B walk is the failure
                                      # mode the levers exist to prevent).
                                      # Reset frames evaluate (to seed history) but
                                      # present real-only (`primed`); the feature is fixed-res
                                      # by creation (lazy-created at the frame's render res)
                                      # but FOLLOWS a render-res MOVE: a moved res that HOLDS
                                      # FG_RECREATE_STABLE=8 consecutive dispatches drains the
                                      # queue and FEATURE-SCOPE-recreates at it
                                      # (shim frdlssg_recreate: ReleaseFeature + CreateFeature
                                      # ONLY, then the guide planes re-ensure) — so FG survives
                                      # SPACE/F mode cycles (the CPU arm's quality-2/3 res vs
                                      # the GPU arms' native; the CPU renderer fills the same
                                      # MV/depth/guide planes, so CPU-rendered frames generate
                                      # too — the XeSS-FG composes-with---cpu precedent).
                                      # TRAP 8 [SL-era mechanism, discipline KEPT]: the
                                      # recreate must NEVER route through frdlssg_destroy
                                      # mid-session — destroy tore at the GetCapability-
                                      # Parameters map the in-process Streamline SHARED and
                                      # every subsequent RR evaluate failed 0xBAD00004
                                      # FeatureNotFound. The sharer today is our OWN DLSSD
                                      # session (same NGX-owned map), so the never-Destroy-
                                      # Parameters / feature-scoped-recreate rules still hold,
                                      # now by ownership discipline instead of SL archaeology.
                                      # A --lock-res dynamic RAMP changes res per
                                      # frame, never qualifies, and skips with a note (the
                                      # recreate-storm guard; a completed DRS step holds the
                                      # 90-frame dwell = one recreate per adoption).
                                      # RESIZE KEEPS THE FEATURE ALIVE and lets the
                                      # res-follow recreate adopt the new size — it does NOT
                                      # destroy (2026-07-26; FR_FG_RESIZE_DESTROY=1 restores
                                      # the old path for A/B). The SL-era crash that settled
                                      # this (destroy on an 8K resize -> the shared NGX state
                                      # torn under Streamline's live RR -> an AV inside _nvngx
                                      # surfacing as Present E_ABORT -> the session shed RR,
                                      # shed DXR, panicked at the plain present; NOT VRAM —
                                      # reproduced at 418 MB as readily as 5.8 GB) is
                                      # structurally impossible since the retirement, but the
                                      # cheap keep-alive shape stays — the DLSSD session
                                      # shares the same map. frdlssg_recreate therefore takes
                                      # DISPLAY dims as well as render dims — a window resize
                                      # moves both, and rend-only rebuilt the feature as old
                                      # display x new render, which NGX rejects at evaluate
                                      # with 0xBAD00005 FAIL_InvalidParameter.
                                      # Known-accepts: latency +~half frame
                                      # (the interpolation cost, W5 owns measurement). NOTE
                                      # the HUD is NOT among them on THIS path (it is on the
                                      # ffx one): NGX is handed `color: rr.output` — linear,
                                      # pre-tonemap, PRE-HUD — and both pair halves composite
                                      # the HUD themselves inside fullscreen_to_backbuffer, so
                                      # the UI is never interpolated and pHudless/pUI being
                                      # null costs nothing here.
                                      # RETIRED WITH STREAMLINE: the SL DLSS-G fallback
                                      # backend (contract-complete — Reflex/PCL markers, the
                                      # funnel mode-off handshake, the type-0 depth dual-tag —
                                      # yet SL's closed dlfg present layer DECLINED TO INSERT
                                      # on the dev box with every verifiable element green;
                                      # the elimination record, the FR_DLSSG_NO_RR isolate
                                      # lever, and the never-resolved open issue live in git
                                      # history at the SL-retirement commits). Its deletion is
                                      # half of why SL retired at all: the interposer's only
                                      # other job was evaluating RR, which the raw DLSSD shim
                                      # now does. Gates: --check, --check-dlss, --check-gpu
cargo run --release -- --prefer-intel # FRAME GENERATION, XeSS family (W4 leg 3 —
                                      # VERIFIED GENERATING on the B70; fg is ON BY DEFAULT, so
                                      # a flagless Arc session takes this leg — --no-fg opts
                                      # out): an Intel XeSS session
                                      # (the flagless Arc default) wraps its swapchain with the
                                      # XeSS-FG proxy (src/xess_fg.rs — libxess_fg.dll +
                                      # libxell.dll from the xess_path dir, the xess.rs
                                      # fn-table loader idiom; xefgSwapChainD3D12InitFromSwap-
                                      # Chain + GetSwapChainPtr at the same d3d12::SwapWrap
                                      # hook the ffx family uses). XeLL is created, sleep-moded
                                      # low-latency, and LINKED at wrap (a hard xefg
                                      # requirement); the three XeSS present arms tag depth +
                                      # MV (the XeSS trio planes, NPSR state) + row-major
                                      # frame constants per presentId (+1 per prepared frame),
                                      # fire all six XeLL markers (sleep + sim/renderSubmit at
                                      # prepare, present pair around Execute+Present), and the
                                      # funnel handshake disables generation on any unprepared
                                      # present (READ-not-consume, the DLSS-G shape). THE
                                      # OWNERSHIP TRAP (measured as a silent native crash):
                                      # unlike the ffx wrap, which CONSUMES the app swapchain,
                                      # the xefg proxy DELEGATES to it — the app-side ref must
                                      # stay alive until xefgSwapChainDestroy (XefgSwapchain
                                      # holds it; released LAST in Drop). XeSS-FG REJECTED the
                                      # old scRGB fp16 swapchain (InitFromSwapChain INVALID_
                                      # ARGUMENT — measured; no HDR flag exists in its API) but
                                      # ACCEPTS 10-bit (VERIFIED on the B70: R10G10B10A2 +
                                      # G2084 wraps and generates, gen result SUCCESS x2). Both
                                      # session defaults are now that same R10G10B10A2 format
                                      # (PQ on HDR-on, Sdr10 gamma on HDR-off — the Sdr10 wrap
                                      # is a byte-identical desc, not yet B70-smoke-verified),
                                      # so the old wrapper-forces-PQ-or-8-bit special case is
                                      # gone. If the wrap ever rejects the 10-bit chain,
                                      # D3d::with_queue rebuilds at 8-bit SDR and wraps AGAIN —
                                      # FG is why the session exists, so SDR with FG beats
                                      # 10-bit without it.
                                      # Verified on the B70 (SDK 1.2.2 + XeLL 1.2.1):
                                      # the library's own GetLastPresentStatus reports 2
                                      # frames presented per present / gen result SUCCESS, and
                                      # PresentMon shows ~174 presents/s over ~87 rendered.
                                      # Composes with --cpu (the CPU-fed XeSS arm carries the
                                      # prepare too — the biggest visual win, low source fps).
                                      # A status poll auto-disables on a negative gen result.
                                      # Gates: --check, --check-xess, --check-gpu
                                      # --prefer-intel. Touch xess_fg.rs / the xefg_* helpers /
                                      # the XeSS arms -> run those three + the interactive B70
                                      # smoke (fg lines + last-present status x2)
cargo run --release -- --fsr --fsr-max-radiance 10  # Ray Regeneration tuning (FfxApiConfigureDenoiserKey,
                                      # applied at denoiser creation): --fsr-max-radiance (the firefly
                                      # clamp — the highest-value knob for a 1-spp path tracer),
                                      # --fsr-stability-bias, --fsr-radiance-clip-k,
                                      # --fsr-disocclusion-threshold, --fsr-normal-strength,
                                      # --fsr-kernel-relaxation. Each unset = configure nothing = the
                                      # provider's own default, so a flagless session is unchanged
cargo run --release -- --check-nppd   # headless: NPPD neural-denoise self-test (needs onnxruntime.dll
                                      # + the exported model; the staging math is gated DLL-free by --check)
cargo run --release -- --nppd-dump    # --check-nppd + before/after PNG dumps
cargo run --release -- --nppd         # NPPD neural denoising (J toggles; mutually exclusive with G/N;
                                      # needs SDKs\onnxruntime\bin + SDKs\nppd\nppd_small.onnx — see
                                      # tools/nppd-export, export with --fp16; --nppd-device auto|cpu|dml[:n]).
                                      # IMPLIES --xess: trace at --lock-res (default native 100%), NPPD
                                      # pre-denoises at that render res, XeSS upscales; --no-xess keeps
                                      # the standalone window-res mode (also the automatic fallback when
                                      # libxess.dll is missing)
cargo run --release -- --xess --nppd  # same session spelled explicitly; J toggles the pre-upscale slot
                                      # (takes the slot OIDN's N-cycle pre placement uses)
cargo run --release -- --no-temporal  # A/B lever: disable ALL previous-frame quadtree reuse (no
                                      # temporal cache, no claim ring, no query skip, no structure
                                      # replay) — every frame proves its empty space from scratch
cargo run --release -- --no-replay    # A/B lever: temporal seeding stays, static-frame structure
                                      # replay (and its recording) off — on the CPU renderer AND
                                      # the GPU WAVEFRONT tracer. GPU replay (--gpu, still/
                                      # converging frames): when the CamBasis bit-equals the
                                      # previous producing frame's, record_frame re-dispatches the
                                      # persisted terminal queues (qleaf/qsky/cut_pool +
                                      # CTR_LEAF/CTR_SKY/CTR_CUT via cs_seed_replay) and skips
                                      # cs_seed + the whole level ladder — measured -43% GPU frame
                                      # span on a still 4090 spin (1.27 -> 0.72 ms; the ladder
                                      # vanishes, a `wavefront-replay` --gpu-timing region appears).
                                      # BIT-IDENTICAL to a fresh trace (the structure is a pure
                                      # function of scene/BVH/basis/rw,rh; spp/jitter/frame/clouds
                                      # ride the CB) — gated in --check-gpu (tbuf/info/accum diff 0,
                                      # ladder provably skipped, warm-frame + auto-predicate must-
                                      # fires). No DXR replay (that pipeline has no structure).
                                      # Invalidated on a hemi-probe seed (zeroes the terminal
                                      # counts) and any present error (a recorded-but-aborted
                                      # producing frame — gpu.invalidate_replay)
cargo run --release -- --no-adopt     # A/B lever: temporal seeding stays, query skip / cut
                                      # adoption (and CutStore production) off
cargo run --release -- --discard-seeds  # A/B/C lever: the whole temporal pipeline runs (lookups,
                                        # ring retries, cache + cut production) but nothing is
                                        # consumed — frames trace exactly like --no-temporal while
                                        # paying the machinery's cost. With --spin, wall-clock
                                        # differences isolate cost from benefit: (this −
                                        # --no-temporal) = pure cost, (default − this) = benefit
cargo run --release -- --no-hemi-share  # A/B lever: disable the shared hemisphere capture in fb (H)
                                        # frames — every shading point runs its own bounce tree
cargo run --release -- --no-bloom     # A/B lever: no glare. Bloom (`src/bloom.rs` + `gpu/bloom.rs`)
                                      # is a DISPLAY-stage pass on whatever the tonemap is about
                                      # to read — it never touches accum, the temporal cache, or
                                      # any upscaler guide, so every radiance gate is structurally
                                      # blind to it. It exists because the sun's limb is a HARD
                                      # ~650x step (physically correct) and the tonemap saturates
                                      # above radiance ~5, so the disc landed as a flat white
                                      # circle stamped on the aureole. Real suns look soft because
                                      # light scatters in the lens/eye, not because their edge is
                                      # soft — so the fix is the optics, not the sky. Mip pyramid,
                                      # 6 octaves, 3x3 tent upsample (a plain bilinear tap leaves
                                      # the box kernel's SQUARE footprint visible in the core), and
                                      # the composite is ENERGY-CONSERVING — `(1-s)·hdr + s·glare`,
                                      # so a uniform frame comes back unchanged and bloom can never
                                      # be tuned into an exposure change (`bloom::self_test` pins
                                      # exactly that, plus point-source energy and a monotone tail).
                                      # The GPU twin is gated too: --check-gpu's M13 runs the real
                                      # BloomGpu pyramid on a probe image and scores its HALO (the
                                      # pyramid's whole product) against `bloom::Bloom` — mean rel
                                      # <= 0.02 / worst <= 0.10, measured 0.0009/0.0024. It is a
                                      # WIRING gate (f16 + hardware bilinear will never match f32
                                      # exactly, but a bad weight/barrier/slot/pitch moves the halo
                                      # by tens of percent). Never widen those limits to pass a port
cargo run --release -- --gpu-debug    # D3D12 debug layer + GPU-BASED VALIDATION, draining to stderr
                                      # (`d3d12::drain_debug`, called from every present and every
                                      # headless submit). All three halves are load-bearing: the
                                      # layer writes to OutputDebugString, so without the drain it
                                      # armed validation and threw the findings away; and the BASIC
                                      # layer does not check the state a resource is IN when a shader
                                      # reads it through a descriptor table — that is GBV-only, and
                                      # it is exactly the class of bug that shipped here (a compute
                                      # dispatch sampling a texture left in PIXEL_SHADER_RESOURCE
                                      # instead of NON_PIXEL). GBV is slow by design; it is a
                                      # correctness flag, never a benchmark path. Applies to
                                      # --check-gpu / --check-dxr too
cargo run --release -- --no-mips      # A/B lever: no texture mip chains; every trilinear sample
                                      # degenerates to the pre-mip bilinear (see Mip-mapping below;
                                      # implies --no-aniso — mips are anisotropy's prerequisite)
cargo run --release -- --heightfield  # ARM relief rendering and start it ON wherever the scene
                                      # carries height data (see Heightfield relief below). The
                                      # DEFAULT session is UNARMED — structurally the pre-relief
                                      # renderer (no swept AABBs, no march), because the sweep's
                                      # all-axis edge pad wrecks BVH quality where EVERY triangle
                                      # carries height and tris are texel-scale (DamagedHelmet
                                      # close-up: 596 vs 146 ms/frame, 4×, WITH RELIEF OFF — the
                                      # armed-but-off tree paid the whole price, which is why
                                      # armed-by-default was reverted). In an armed session V
                                      # toggles relief ↔ plain normal-mapping live (a
                                      # shading+visibility change: frame + upscaler-history
                                      # reset, temporal cache/replay KEPT); unarmed sessions get
                                      # a V note instead. Armed state keys the .fcache
cargo run --release -- --no-heightfield  # the default, spelled explicitly (later flags win —
                                      # `--no-heightfield --heightfield` arms)
cargo run --release -- --no-h2n       # A/B lever: don't Sobel-convert grayscale map_Bump height
                                      # maps into normal maps at load (they are dropped — the
                                      # pre-conversion behavior; San Miguel carries exactly 1)
cargo run --release -- --no-n2h       # A/B lever: don't derive heightfields from normal maps at
                                      # load (the Frankot–Chellappa FFT inverse) — normal-map
                                      # alpha stays 255, height_amp stays 0, relief has no field
                                      # (all three levers key the .fcache — a warm load under a
                                      # different lever state is a cache miss, never a stale serve)
cargo run --release -- --no-tinted-shadows  # A/B lever: shadow/AO rays binary-block on transmissive
                                      # surfaces — the pre-feature renderer bit-identically. The
                                      # DEFAULT is TINTED SHADOWS: every LIGHT occlusion ray (sun
                                      # shadow, translucency back ray, firefly shadow, sampled AO,
                                      # hemi-AO leaf) accumulates transmission×albedo per
                                      # transmissive interface instead of blocking (see the
                                      # tinted-shadows paragraph in Real scenes — the fountain-
                                      # water-as-liquid-chrome fix)
cargo run --release -- --no-spray     # A/B lever: keep tiny transmissive islands (fountain
                                      # droplets) as clear glass — the pre-spray look, where a
                                      # clear millimeter droplet is invisible against a matched
                                      # background. Default ON: a load-time union-find (welded by
                                      # vertex POSITION bits, not index — per-block-unwelded
                                      # Minecraft water is one ocean, not 150k droplets) retags
                                      # transmissive components under SPRAY_MAX_K·diag (~4 cm on
                                      # San Miguel) as white-scatter spray (aerated water — the
                                      # reason games ship spray as white particles). Keys the
                                      # .fcache lever word (the h2n/n2h class)
cargo run --release -- --no-depth-tint  # A/B lever: no Beer–Lambert attenuation over the
                                      # transmission chain's interior segments. Default ON:
                                      # transmitted light attenuates by albedo^(d/(TRANS_DEPTH_K·
                                      # diag)) per interior segment — water is exactly
                                      # albedo-tinted at ~1 m of traversal, clearer above, darker
                                      # below (the depth term the per-interface tint can't carry;
                                      # shadow rays keep the per-interface tint — the clouds
                                      # two-transmittance bracket)
cargo run --release -- --no-coincident-cull  # A/B lever: keep transmissive faces exactly coincident
                                      # with an OPAQUE face (the pre-cull z-fight). Default ON:
                                      # scene::cull_coincident drops them at cold load (a face whose
                                      # 3 vertex positions bit-equal an opaque tri's, any winding —
                                      # the spray position-weld precedent; runs beside
                                      # reclassify_spray on direct loads AND per world island; keys
                                      # the cache lever word, bit 6). A transmissive face flush
                                      # against a solid transmits nothing physically, and keeping it
                                      # is worse than redundant: the two intersectors break an
                                      # exact-t tie DIFFERENTLY (CPU möller/BVH traversal order vs
                                      # hardware watertight order), and when the transmissive face
                                      # wins, the chain's eps-advanced continuation starts INSIDE
                                      # the solid and TUNNELS past it — with eps = 1e-4·diag and a
                                      # ground-quad-inflated diag, that advance is ~1.5 Minecraft
                                      # blocks. FOUND via "rungholt water is more transparent on
                                      # the CPU path": the loaders' ground quad ALSO sat exactly ON
                                      # the fit's rest plane (y = 0), where the model's whole base
                                      # layer z-fought it — the CPU resolved the ocean's y=0 tie to
                                      # the water-volume BOTTOM face and leaked the refracted chain
                                      # THROUGH the world's floor to sky (bright, stipple-latticed),
                                      # the GPU resolved it to the flat ground quad (dark,
                                      # featureless) — BOTH wrong, visibly different per render
                                      # mode. The quad now rests scene::GROUND_DROP = 1e-3 BELOW
                                      # the rest plane in the OBJ/glTF/--tile/world loaders
                                      # (procedural/stress deliberately keep y = 0: no transmissive
                                      # geometry, and their gate images are pinned byte-identical);
                                      # CACHE_VERSION 20→21. MEASURED at the rungholt water pose
                                      # (--cam 2.6,0.6,-1.5,3.4,0.05,-2.2 — ~45% water px, found by
                                      # scanning the OBJ for Stationary_Water bounds): the
                                      # --check-gpu 64-frame CPU-vs-GPU radiance A/B went 4.079%
                                      # FAIL → 0.028%, water-px mean color equal to 4 decimals.
                                      # NOTE rungholt's cull count is 0 — its open ocean has NO
                                      # modeled seabed, so the quad move alone closed the
                                      # divergence there (both arms now take the documented
                                      # interior-ray-to-sky leak: flat deep blue); the pass guards
                                      # the class wherever coincident pairs really exist.
                                      # scene::coincident_self_test gates drop/keep/winding/
                                      # lever-off in --check. Two diagnostics from the hunt stay:
                                      # FR_CHECK_AB_DUMP=1 makes the --check-gpu radiance A/B dump
                                      # check_ab_cpu/gpu/diff.png + a ripple-normal compare + a
                                      # per-term (dd/ds/ao/is/residual/color) water decomposition
                                      # against the sig-armed pack, and bvh::TRANS_PASS counts the
                                      # CPU's tinted-shadow crossings (CTR_TRANS_PASS's twin) for
                                      # crossing-count parity
cargo run --release -- --aniso 16     # max anisotropy, 1..=16 (DEFAULT 16; --no-aniso = --aniso 1).
                                      # The ray cone's elliptical footprint is resolved along its
                                      # major axis: CPU N-tap (texture.rs::sample_aniso), GPU
                                      # hardware SampleGrad + an ANISOTROPIC static sampler. 1 = off
                                      # is the isotropic ray-cone lod path VERBATIM, i.e.
                                      # bit-identical to the pre-aniso renderer (see Mip-mapping)
cargo run --release -- --defer-shade  # EXPERIMENT (off by default; measured no-win — see README's
                                      # "Deferred material-sorted shading"): plain-path leaf tiles
                                      # trace but defer shading; same-material runs merge up the
                                      # quadtree (≤ 64×64 px) and flush as material-sorted parallel
                                      # bursts. Bit-identical to fused shading (--check gates it on
                                      # any textured scene); untextured scenes structurally unchanged
cargo run --release -- --bvh-ctrav 3 --bvh-axes 3 --bvh-maxleaf 8  # BVH build knobs at their
                                        # defaults: SAH traversal/intersection cost ratio (the
                                        # MEMORY lever — halves the node array, speed-neutral),
                                        # axes searched by the binned SAH (the SPEED lever:
                                        # 3-axis is -33% ray nodes / -17% ms on San Miguel;
                                        # 1 = the historical widest-axis build), leaf-size cap.
                                        # Build params key the .fcache (bvh::build_key), so
                                        # sweeps never collide with a stale sidecar
cargo run --release -- --bvh-builder ploc  # ray-BVH builder bake-off: sah (default) | lbvh | ploc |
                                        # som — same Bvh type, all consumers/gates/.fcache work
                                        # unchanged (id rides bvh::build_key), all byte-deterministic.
                                        # Verdict (spin path, measured ray nodes — never SAH): sah
                                        # best-or-close everywhere and stays the default; ploc −34%
                                        # vs sah on San Miguel (dense clustering merit) but +121% on
                                        # --stress (sparse fields collapse; over-deep merge chains
                                        # get median-rebalanced at the TRAV_STACK point of no
                                        # return); lbvh the control, 2.7-4.4× worse; som — batch
                                        # 3D-lattice SOM as a LEARNED space-filling curve — is
                                        # WORSE than raw Morton on both scenes (BMU cell-boundary
                                        # jumps tear bit-prefix locality): the SOFM question,
                                        # settled with numbers. Caveat: lbvh trips the default
                                        # scene's hemi-share paired-GI limit (reclassification
                                        # fireflies on a coarser tree — topology-tuned gate; every
                                        # exact-zero soundness gate passes on all four builders)
cargo run --release -- --no-blas-split  # A/B lever (GPU only) BACK to ONE BLAS over scene.indices
                                        # in order + an identity instance. THE SPLIT IS THE DEFAULT
                                        # (65536 tris per BLAS, blas_split::DEFAULT_MAX_PRIMS;
                                        # --blas-split N overrides the cap): cut the ray BVH into
                                        # maximal subtrees of <= N tris and build ONE BLAS per
                                        # subtree, each instanced identity into the TLAS with
                                        # InstanceID = the chunk index — so the driver's structure
                                        # is ADDRESSABLE at BVH-node granularity (BlasPlan::
                                        # chunk_node is the instance <-> node map a cut-driven TLAS
                                        # rebuild would need). PrimitiveIndex() indexes a CHUNK,
                                        # not a triangle, so every intersector site goes through
                                        # trace_common.hlsli's tri_of(inst, prim) =
                                        # blas_tri[chunk_base[inst] + prim] — the chunk-major remap
                                        # (blas_tri/chunk_base ride t7/t8 space1, moving texs[] to t9;
                                        # TEX_TABLE_BUFS 7->9, lockstep with the HLSL). --no-blas-split
                                        # compiles tri_of as the IDENTITY (no BLAS_SPLIT define, the
                                        # ALPHA_CUTOUT precedent) and binds 4-byte dummies, which is
                                        # the pre-feature renderer bit-identically.
                                        # IT IS THE DEFAULT FOR ROBUSTNESS, NOT SPEED — and the
                                        # measurement that decided it is worth not re-deriving.
                                        # On NVIDIA it is NEUTRAL: 4090, THE WORLD, four static
                                        # poses, gpu-timing running means over thousands of frames,
                                        # tracer ms 1.692->1.698 (boot), 1.850->1.829 (island),
                                        # 1.894->1.888 (long view); --spin DXR -0.6% procedural /
                                        # -2.9% SM-lp, wavefront neutral. On INTEL IT IS THE
                                        # DIFFERENCE BETWEEN RUNNING AND NOT. BLAS scratch is sized
                                        # by the LARGEST SINGLE GEOMETRY, so THE WORLD's one
                                        # 34.4M-tri BLAS made the B70's driver ask 1891 MB of
                                        # scratch and REMOVE THE DEVICE mid-boot (0x887A0005 ->
                                        # "dxr: falling back to CPU tracing" -> XeSS disabled ->
                                        # panic at Present), where the same build asks NVIDIA for
                                        # 276 MB and survives. Split at 64k the scratch is a
                                        # function of one chunk — 3 MB — and the session runs
                                        # (dxr 7.27 ms, frame span 8.34). PROVEN to be the BLAS
                                        # size and nothing else by `--blas-split 40000000`: one
                                        # chunk through the ARMED path (no dummies anywhere)
                                        # reproduces the removal with the same 1891 MB. Intel's
                                        # compaction differs wildly too (4624->1576 MB vs NVIDIA's
                                        # 1844->668), so treat single-BLAS scratch as a vendor
                                        # cliff, not a constant.
                                        # THE RDNA4 INDEX-VALUE DEFECT (2026-08-01 — the
                                        # bistro-dusk shards): on the R9700 (driver
                                        # 32.0.31035.1003) a chunk BLAS whose index VALUES reach
                                        # past ~2^24 into the big shared vertex buffer builds
                                        # WRONG TRIANGLES — scattered sliver geometry,
                                        # deterministic per scene, BOTH GPU pipelines (they share
                                        # the one SceneGpu core), NVIDIA bit-clean on identical
                                        # inputs, the single-BLAS build (one huge geometry) never
                                        # trips it. Only scenes past ~16.7M VERTICES can reach it
                                        # (THE WORLD, big --tile runs), which is why every
                                        # committed-scene suite run missed it for a month. The
                                        # split therefore WINDOWS every chunk under
                                        # blas_split::SPLIT_INDEX_CEILING: REBASE to the chunk's
                                        # min id (free — nearly all chunks; the desc's
                                        # VertexBuffer.StartAddress slides to match) or GATHER
                                        # the <= 3*cap used vertices into a transient side buffer
                                        # (chunks whose id RANGE clears the ceiling — tile seams,
                                        # cross-island chunks; 9 chunks / 1.5 MB on tiled SM-lp,
                                        # 1 / 201 KB on the world). plan_windows is PURE and
                                        # pinned DLL-free by blas_split::self_test in --check
                                        # (rebase/gather dichotomy, bijective gather map, bit-
                                        # copied positions, every emitted value under the
                                        # ceiling, the disabled arm absolute). FR_SPLIT_NOREBASE=1
                                        # is the repro arm; FR_SPLIT_AUDIT=1 memcmps all three
                                        # streamed remap/index buffers against the CPU plan. The
                                        # hardware repro gate: `san-miguel-low-poly.obj --tile 3
                                        # --check-dxr --prefer-amd` read 287 divergent-t px
                                        # (max rel 1.04e-1) before, 0 (1.1e-5, NVIDIA-class)
                                        # after; T1's 0.01% threshold means a `--tile 2` dose
                                        # sits under the gate (16 px) — do not shrink the tile in
                                        # that repro. Eliminated on the way, each by measurement:
                                        # candidate loops (FR_ABL=noalpha,notrans still dirty),
                                        # foliage sway, remap-data corruption (audit bit-exact),
                                        # compaction, build serialization (per-build fences — the
                                        # shared-scratch UAV barrier is SOUND), arena overrun
                                        # (64 KB guard gaps). COSTS, paid on every GPU session:
                                        # a permanent 4 B/tri remap (+146 MB on the world), a
                                        # transient 12 B/tri reordered index stream during the
                                        # builds, and ~1 s of build time at 34.4M tris — against
                                        # which the scratch peak drops by 276 MB (NV) / 1888 MB
                                        # (Intel). --no-blas-split is the escape if a mega-scene
                                        # ever wants that 4 B/tri back.
                                        # THE CAP IS THE WHOLE DESIGN: 64k puts scenes in
                                        # the band drivers are tuned for (~1 chunk per ~40k tris —
                                        # MEASURED procedural 79.7k tris -> 2 chunks, San Miguel
                                        # low-poly 5.6M -> 152, --stress 5000 3.97M -> 157, THE
                                        # WORLD 34.4M -> 890, mean ~37k prims) and keeps compaction
                                        # affordable; a cap in the TENS gives ~25 single-use BLASes
                                        # per 1000 tris (~250k on San Miguel), two-three orders past
                                        # normal practice, each paying a header + an instance
                                        # transition — reachable as `--blas-split 64` precisely so it
                                        # can be measured, not argued. BLAS
                                        # 122 MB vs 124 MB single on SM-lp (same 300 MB pre-compaction),
                                        # build +0.4 s. Build shape mirrors the single-BLAS path
                                        # (worst-case arena + ALLOW_COMPACTION -> postbuild sizes ->
                                        # compact into an exact arena -> TLAS over the compacted VAs);
                                        # chunk BLASes SUB-ALLOCATE from one committed arena at
                                        # 256-B alignment (never one resource each) and build serially
                                        # through one max-sized scratch buffer — the UAV barrier
                                        # between builds IS that sharing's serialization, not a
                                        # removable pessimization. Vertex positions are SHARED (only
                                        # the index stream reorders, and it is dropped once the builds
                                        # run — a built AS is self-contained). A bare numeric that is
                                        # not a legal cap (0, past u32) exits 2 rather than arming at
                                        # the default and being read as an OBJ path; only a departure
                                        # from the default prints a lever line (the `gpu scene:` line
                                        # already reports the chunk count). A VRAM pre-flight vs
                                        # adapter::vram_info fails LOUDLY rather than letting WDDM
                                        # demote, and > 2^24 chunks is an error (the InstanceID
                                        # ceiling). NOTE what the VRAM failure costs: it CANNOT
                                        # degrade to the single-BLAS build — the lever is
                                        # session-global and both tracers bake blas_defs() into
                                        # their kernels/RTPSO, so a degraded ONE-BLAS core under the
                                        # SHARED Rc<SceneGpu> would have any armed shader (compiled
                                        # before or after the core) remapping every hit to garbage.
                                        # So the core upload fails and the session falls back to the
                                        # CPU renderer (an identity remap would make a real fallback
                                        # possible at 4 B/tri; deliberately not built — an untested
                                        # path reachable only by exhausting VRAM is how the
                                        # dummy-SRV device removal got in); the error text points at
                                        # --bc7 / --lock-res, and explicitly NOT at dropping the
                                        # split on Intel. blas_split::self_test gates the planner in --check
                                        # (cap, exact triangle partition, antichain-cut coverage,
                                        # determinism at the shipping cap, the single-chunk edge, and
                                        # a MUST-FIRE on the oversized-leaf split at cap
                                        # widest_leaf-1 — two chunks sharing a node id is the
                                        # observable proof it ran, and without it --bvh-maxleaf 1
                                        # would leave that branch dead while every other gate passed;
                                        # the sub-64 caps are skipped LOUDLY above 4M tris, where one
                                        # chunk per triangle would spike ~1 GB inside the gate);
                                        # the REMAP is proven by the existing suites, which now run
                                        # armed BY DEFAULT — --check-gpu/--check-dxr keep every
                                        # exact-zero counter at 0 with the same-seed image A/B
                                        # unchanged to the digit. Both suites FAIL on < 2 chunks
                                        # when the scene is OVER the cap (an over-cap run can't pass
                                        # vacuously) and print a NOTE when it is under (a small
                                        # scene is legitimately one chunk — the identity remap —
                                        # which is why the predicate is not simply chunks < 2).
                                        # Run --check-gpu/--check-dxr --no-blas-split to gate the
                                        # single-BLAS arm; --check* NEVER loads the world, so the
                                        # Intel removal above is reachable only interactively
cargo run --release -- --no-cut-rays    # A/B lever: cut-SEEDED rays (primary leaf-tile rays)
                                        # traverse from the BVH root instead; the inherited
                                        # t_start is a scalar and survives. Isolates what the
                                        # CUT itself is worth to the ray path (~10% procedural,
                                        # ~2.5% San Miguel after the root-order fix)
cargo run --release -- --cut-hemi       # re-enable hemi leaf rays seeding from their bounce cut
                                        # (the pre-M2 behavior): 64 scattered cut roots measured
                                        # 3-10% SLOWER than one coherent root descent on every
                                        # scene/tree tried, so root-first is the DEFAULT; the
                                        # bound queries still consume the cut either way, and
                                        # --check's hemi probe gates force seeding ON so the
                                        # cut-miss gate keeps exercising the cut machinery
cargo run --release -- --gpu --continuation-rays  # A/B measurement lever (default OFF;
                                        # --sw-rays is the technical alias, --no-sw-rays the
                                        # kill): the WAVEFRONT tracer's rays traverse the
                                        # SOFTWARE BVH — bvh.rs's loops ported to
                                        # rt_sw.hlsli, pasted IN PLACE of rt.hlsli's RayQuery
                                        # bodies (same three primitive signatures; off arm =
                                        # the exact pre-lever source lists) — so leaf
                                        # PRIMARIES seed traversal from the tile's node cut.
                                        # IT IS FRAMED AS A SEMANTIC PROTOTYPE OF A HARDWARE
                                        # SEAM THAT DOES NOT EXIST: the terminal beam
                                        # publishes ONE opaque TraversalFrontier
                                        # (shaders/continuation.hlsli — a cookie-tagged
                                        # uint2, v1 packing slot<<6 | len-1; concatenated
                                        # ahead of queues.hlsli at the ONE QUEUES_HLSLI site,
                                        # so no unit can compile a different producer/consumer
                                        # contract) and every ray AND spp sample in that leaf
                                        # record reuses it through trace_closest_frontier
                                        # (= intersect_multi's semantics, v1 pool-order roots
                                        # + running-tmax prune, behind the opaque seam). The
                                        # leaf shader CANNOT read a node id, pool slot, or
                                        # length — a native provider could swap the two words
                                        # for driver-owned traversal state without touching
                                        # LeafRec or the call site. An invalid cookie, an
                                        # out-of-domain token, an exhausted arena, and an
                                        # explicit root ALL degrade conservatively to root
                                        # traversal — never an out-of-bounds arena read,
                                        # never a dropped candidate. t_start is deliberately
                                        # NOT in the token: the empty-space proof stays valid
                                        # when a frontier coarsens to an ancestor or the
                                        # root. The
                                        # REFERENCE kernel swaps too (one intersector both
                                        # sides — and the wavefront-vs-reference same-seed A/B
                                        # then reads EXACT 0.00e0 / hot 0 on NVIDIA *and* AMD:
                                        # the TMin-re-origin ulp class disappears because no
                                        # re-origining exists). LeafRec grew 16→24 B (the
                                        # frontier's two words, written always, read only
                                        # armed; trace::LEAF_REC_BYTES ↔ queues.hlsli ↔
                                        # main.rs readback in lockstep, and --check-gpu audits
                                        # every record's cookie + token domain CPU-side before
                                        # trusting the consumer); under FTREE the slot-ref cut
                                        # is translated to binary node ids at leaf EMISSION
                                        # via the lever-only ft_bnode map (QFNode still drops
                                        # bnode for everyone else) into a second pool slot
                                        # (cap_cut ×2, overflow stays gated 0; exhaustion =
                                        # root fallback, counted). Stacks are per-lane SCRATCH
                                        # (96 = bvh::TRAV_STACK, injected; groupshared at
                                        # 32×96×8 B would LDS-cap a zero-LDS kernel — the
                                        # documented sweep). Composes: --no-cut-rays = software
                                        # from the root (SW_RAYS_LEAF compiles out, the CPU's
                                        # short-circuit); --no-ftree = binary cuts, no
                                        # translation. Secondaries/hemi rays go software from
                                        # the root (a primary cut is apex-specific — inheriting
                                        # it would light-leak; hemi cuts still drive bound
                                        # queries only). CTR_FRONTIER_HANDLES / _RAYS /
                                        # _ENTRIES (per leaf RECORD, never per ray — the
                                        # per-ray atomic would tax the very path the lever
                                        # exists to measure) are the --check-gpu must-fires:
                                        # non-root handles > 0, rays > handles (reuse IS the
                                        # claim), 1 <= entries/handle <= 64. They count
                                        # frontiers CONSUMED, which is why frontier_record_
                                        # reuse zeroes its flag on !SW_RAYS_LEAF while still
                                        # executing all three atomics: the root control pays
                                        # identical telemetry cost AND reports zero BY
                                        # CONSTRUCTION, which is what lets the off-lever gate
                                        # demand exact 0. Do not key that flag on the token
                                        # alone — a MIXED split (one child <= LEAF_TILE while
                                        # a sibling is not, reachable at a parent extent of
                                        # exactly 2*LEAF_TILE+1) mints a real frontier in
                                        # EVERY arm, so the gate would become a property of
                                        # its own resolution's split ladder. alpha/relief/
                                        # tint counters ride candidate_reject unchanged. THE
                                        # VERDICT, measured (--spin path 1080p wall, 600f, rep-2
                                        # warm per the Arc compile trap): hardware RayQuery WINS
                                        # EVERYWHERE — 4090 spp=1: hw 0.87 / sw 1.13; B70 spp=1:
                                        # hw 1.76 / sw 2.54 / sw--no-cut-rays 2.57 (the cut seed
                                        # recovers ~1% of a 44% gap); B70 spp=16: hw 13.54 / sw
                                        # 26.35 (~2× even amortized; sw marginal ≈ 3.3 vs hw
                                        # 1.0-1.6 ms/sample). So even on the vendor whose RT
                                        # cores are weakest, driver traversal beats this
                                        # software walk ~2×, and cut-seeding cannot close it —
                                        # the empty-space proof stays the quadtree's whole GPU
                                        # value (the leaf.hlsl t_start-ablation conclusion,
                                        # now proven from the other side). 2026-08-01 ABBA
                                        # re-run (identical protocol, B70, 1600+600, fresh
                                        # process per run, root/frontier/frontier/root): root
                                        # vs frontier now IDENTICAL — ±0.004 ms per 120-frame
                                        # window across the whole lap — while both arms run
                                        # 7-11% faster than the 07-26 recording (wave-atomics
                                        # + leaf-kernel restructurings landed in between), so
                                        # the README's 6.5%-leaf/3.2%-frame frontier margin is
                                        # RETIRED; the frontier is proven LIVE the same day
                                        # (check-gpu: 768/768 non-root handles, 468.8
                                        # rays/handle, 0 root fallbacks) — the machinery works
                                        # and buys ~0 time on this workload. Known-accepts v1:
                                        # BLAS/TLAS still build (SPACE→DXR works; AS-skip +
                                        # dropping the RT-1.1 requirement — running on non-RT
                                        # GPUs — is the documented follow-on), require_caps
                                        # unchanged, no scene-cache key (GPU-only, the
                                        # blas-split class). THE CONTROL ARM is
                                        # --continuation-rays --no-cut-rays: same intersector,
                                        # shading, and inherited t_start, rays from the root.
                                        # It is NOT the same quadtree — SW_RAYS_LEAF also
                                        # gates the terminal-cut skip (see the wavefront queue
                                        # treatments below), so the control refines strictly
                                        # FEWER cuts (800x600 gate frame: 65 vs 449) and any
                                        # measured continuation win is a CONSERVATIVE bound.
                                        # Follow-ons: FR_SW_SORT
                                        # group-cooperative front-to-back root order,
                                        # groupshared-stack sweep, --cut-hemi re-measure on
                                        # GPU (HemiCellRec already carries cut_slot/cut_len),
                                        # hybrid sw-primary/hw-secondary. Gates: --check-gpu
                                        # [--sw-rays [--no-ftree|--no-cut-rays|--stress|
                                        # san-miguel-lp|--heightfield]] all PASS (exact-zero +
                                        # bit-identical A/B), --check-dxr untouched (dxr.rs
                                        # pastes neither queues nor frustum). Pre-existing,
                                        # NOT this lever: the AMD iGPU fails check-gpu's spp
                                        # readback with and without --sw-rays (environment)
cargo run --release -- --no-ftree       # A/B lever: hemi bound queries back on the binary BVH.
                                        # Default is the 8-wide frustum tree (src/ftree.rs) —
                                        # lazily collapsed from the ray BVH on the first hemi
                                        # query (only fb sessions pay its build/memory), returns
                                        # BIT-IDENTICAL bounds (self-test-pinned), measured
                                        # -15/-17% hemi-ao and -4/-8% hemi-gi ms/frame; cuts are
                                        # slot-refs, translated by Accel::ray_roots iff a ray
                                        # seeds from them (--cut-hemi)
cargo run --release -- --ftree-tiles    # A/B lever: the CPU tile recursion on the wide tree too
                                        # (tile_step/adopt_step; leaf tiles translate their cut
                                        # to binary ray roots once). Default OFF — unlike the GPU
                                        # tile kernels (-23%), CPU tiles measured wall-NEUTRAL on
                                        # San Miguel and ~10% slower on --stress no-temporal
                                        # (fat singleton-entry cuts, short descents — the
                                        # short-query regime again) despite -21..45% counted
                                        # frustum nodes; --check's `wide-tiles` gate verifies the
                                        # wired path every run so the lever can't rot
cargo run --release -- --no-wide-levels # A/B lever (GPU): every quadtree level runs one THREAD per
                                        # tile (the pre-cooperative ladder). Default ON = the shallow
                                        # levels (d < trace::WIDE_LEVELS) give one TILE a whole 32-lane
                                        # group sharing a breadth-first frontier (wavefront.hlsl::
                                        # bound_query_wave / cs_level_wide) — the ladder was under-
                                        # occupied (level 0 is one lane descending the whole BVH). A
                                        # BFS, so node counts differ, but `best` is an order-independent
                                        # min, so the same-seed image A/B comes back to the digit (a
                                        # pure perf A/B). Works on both frustum structures (binary +
                                        # ftree), unlike the old FTREE-only draft. See the Profiling
                                        # section for the measured -7..30% and the WIDE_LEVELS crossover
cargo run --release -- --spp 4         # multi-sampling: N primary samples per pixel per frame (1..128,
                                       # default 1; U doubles live), averaged into ONE splat
                                       # before the frame reaches the upscaler/denoiser. All three
                                       # render modes. Sample 0 is the frame's REPORTED sample (same
                                       # position rule, same rng seed, the only one that writes
                                       # tbuf/info/G-buffers/MVs — so spp=1 is bit-identical to a
                                       # single-sample frame and the upscaler's jitter contract stays
                                       # literally true); samples 1.. take dlss::jitter_for_sample
                                       # (the same Halton sequence at a phase-coprime stride, so the
                                       # reported 72-phase coverage is untouched) and contribute
                                       # color only. SOUND because every sample lands inside the same
                                       # pixel, hence inside the tile frustum: it consumes the SAME
                                       # inherited t_start/cut as sample 0 (the leaf-tile argument —
                                       # gated per sample, see --check). Pinned to 1 on fb (H) frames.
                                       # --defer-shade defers to the fused path at spp > 1 (a deferred
                                       # leaf stages ONE Traced per pixel — deferring a multi-sampled
                                       # tile would drop every sample but the first; coarser, never
                                       # wrong, and the two levers compose).
                                       # UNDER FSR RAY REGENERATION the presented color is
                                       # RECONSTRUCTED from the signal planes (dd⊗kd + ds⊗f0 +
                                       # residual), never from accum — so the residual must be the
                                       # exact remainder against the AVERAGE, not sample 0's color, or
                                       # --spp would be a costly no-op there. The GPU feed kernel gets
                                       # this for free (it subtracts from averaged accum); the CPU path
                                       # rewrites the sig after the average (render.rs::write_fsr — the
                                       # ONE fsr_buf write site, called again by shade_pixel at
                                       # spp > 1). Known accept, both feeds: the DENOISED lobes
                                       # (dd/ds) are the probe sample's, so the other N−1 samples'
                                       # direct light rides the un-denoised residual — --spp buys RR
                                       # less than it buys RR-less upscalers. Averaging dd/ds would
                                       # need the DXR pack write hoisted out of chs_shade (the
                                       # PrimSurf would have to ride the payload).
                                       # The 128 cap is NOT a math limit: it is the size of the
                                       # jitter table in FrameCb (MAX_SPP × 8 B, must fit CB_STRIDE) —
                                       # raise those two in lockstep (the HLSL cbuffer's row count is
                                       # INJECTED from MAX_SPP by trace::spp_defs, so it follows).
                                       # The extra samples' Halton index
                                       # runs FREE (not mod JITTER_PHASE, which bounds only the
                                       # sequence the UPSCALER sees): a wrap would alias sample 72
                                       # onto sample 0, so --check gates 128/128 distinct positions
                                       # and re-verifies the LAST sample at spp=128 (CPU and GPU).
                                       #
                                       # WHERE THE RETURNS STOP (measured; both benches print the fit).
                                       # Frame time is affine in the sample count: ms(n) = F + m·n,
                                       # F = the once-per-frame quadtree, m = one sample's rays+shading.
                                       # So amortization(n) = ms(n)/(n·ms(1)) = m/(F+m) + F/((F+m)·n):
                                       # an asymptote plus a 1/n term — HALF the fixed cost is diluted
                                       # away by spp 2, 90% by spp 10, 99% by spp 100. The amortization
                                       # is therefore spent by ~8-16 spp; past that every sample pays
                                       # the full marginal price m, while QUALITY improves only as
                                       # 1/√n. spp 128 is honest supersampling, not a free lunch.
                                       #   HISTORICAL GPU wavefront table (1080p, interleaved medians).
                                       #     It predates the shipping (LEAF_TILE, LEAF_GROUP)=(32,256)
                                       #     frontier and the gputime async-compile-bias fix. Retained as
                                       #     experiment provenance only; do NOT cite its ratios/crossovers
                                       #     as current. The t_start ablation below remains the durable result.
                                       #     These rows were post the earlier wave64 leaf-lane repair:
                                       #     4070 Ti: hybrid = 1.32 ms fixed + 0.464 ms/sample (floor 0.26×)
                                       #              plain  = 0.15 ms fixed + 0.420 ms/sample (floor 0.74×)
                                       #     R9700:   hybrid = 1.50 ms fixed + 0.690 ms/sample (floor 0.32×)
                                       #              plain  = 0.15 ms fixed + 0.544 ms/sample (floor 0.78×)
                                       #     hybrid/plain floor 1.11× (NV) / 1.27× (AMD) — on those two
                                       #     vendors the quadtree does not win primary visibility (its
                                       #     marginal sample stays dearer than an RT-core root traversal,
                                       #     and only the marginal cost survives at high spp) — but the
                                       #     margin is small, where it used to read 1.33× on a 4090 and
                                       #     2.56× on AMD. Most of that gap was the wave64 lane waste,
                                       #     not the algorithm.
                                       #   ON INTEL IT WINS, AND THAT IS THE FIRST GPU WHERE IT DOES.
                                       #     Arc Pro B70, post cs_sky + cs_level_wide:
                                       #       default: hybrid 1.28 fixed + 1.046 ms/sample
                                       #                plain  0.18 fixed + 1.134 ms/sample -> 0.92×
                                       #       stress : hybrid 1.51 fixed + 0.732 ms/sample
                                       #                plain  0.21 fixed + 0.842 ms/sample -> 0.87×
                                       #       powerplant 12.8M tris:
                                       #                hybrid 1.33 fixed + 0.869 ms/sample
                                       #                plain  0.19 fixed + 0.934 ms/sample -> 0.93×
                                       #     i.e. the quadtree makes each SAMPLE 7-13% cheaper than an
                                       #     RT-core root traversal there, and at spp=16 the hybrid beats
                                       #     the control outright (18.01 vs 18.32 default, 13.22 vs 13.69
                                       #     stress). Same asymptote is 1.37×/1.31×/1.36× on a 4090 — so
                                       #     this is a property of the HARDWARE BALANCE, not of the scene:
                                       #     the quadtree trades RT-core work for shader-core work, and
                                       #     Intel's RT is weak relative to its shader cores (plain
                                       #     reference 1.31 ms vs the 4090's 0.36 — 3.6×, far wider than
                                       #     the gap in shading-bound work), so there is more traversal to
                                       #     save and it is worth more.
                                       #     THREE THINGS THAT ARE **NOT** TRUE, EACH MEASURED:
                                       #     (1) It does not scale with scene complexity. The ratio is
                                       #         FLAT (0.87-0.93 Intel, 1.31-1.37 NV) across 80k -> 12.8M
                                       #         tris, a 160× range. What little variation there is tracks
                                       #         SPARSITY, not size — --stress 5000's 5000 separate objects
                                       #         (0.87) beat one dense powerplant mesh (0.93), which is
                                       #         what you would expect of a structure whose product is
                                       #         proving space empty.
                                       #     (2) It is not the INHERITED DISTANCE BOUND doing the work.
                                       #         Ablating t_start to 0 while keeping the quadtree costs
                                       #         only +1.7/+1.1/+1.7% on the B70 (and straddles zero on a
                                       #         4090: -3.5/-7.1/+5.2%, i.e. free and worth nothing, the
                                       #         verdict AMD already had). That is 7-21% of the advantage;
                                       #         the other ~80-93% is TILES PROVEN EMPTY TRACING NO RAYS —
                                       #         a conservative screen-space occupancy mask would buy most
                                       #         of it, with no quadtree at all. See leaf.hlsl.
                                       #     (3) It does not help the config that ships. At spp=1, the
                                       #         interactive default, the quadtree still LOSES on Intel
                                       #         (1.77× default, 2.12× stress, 1.96× powerplant); the win
                                       #         needs the once-per-frame quadtree cost amortized and
                                       #         crosses over at ~spp 16.
                                       #     Both halves moved when cs_sky was fixed: the B70 asymptote was
                                       #     1.29× before it, because the sky fill's per-sample cloud
                                       #     marching (--spp averages sample positions in cs_sky) was
                                       #     inflating the MARGINAL cost, not just the fixed one.
                                       #   CPU (default scene): hybrid = 0.9 ms fixed + 9.6 ms/sample,
                                       #     plain = ~1 ms + 10.6 — the mirror image: almost no fixed
                                       #     cost to amortize (floor 0.91×), but the quadtree makes
                                       #     each SAMPLE ~10% cheaper, and that discount does NOT decay
                                       #     with spp. (--spin path SM low-poly: 17.1 → 61.5 ms at 4 spp.)
                                       # Noise: 1.9-2.0× quieter at 4 spp (--check's stability gate).
                                       # --dxr traces from the TLAS root: no claim to inherit, so
                                       # there it is plain supersampling (quality only)
cargo run --release -- --lock-res dynamic  # step-wise dynamic render resolution (DLSS-RR and XeSS)
cargo run --release -- --lock-res 0.75     # lock the render res to a fixed scale of the window; the
                                           # default is `native` (100%, xess::DEFAULT_LOCK_SCALE —
                                           # DLAA-shaped: the wired upscaler still denoises/
                                           # antialiases, it just doesn't upscale) for
                                           # EVERY render mode — CPU, --gpu and --dxr alike, so F/SPACE
                                           # cycling arms never moves the render res. HISTORY: until
                                           # 2026-07-26 the GPU arms defaulted to native through a
                                           # second `Opts::gpu_lock_scale` (that field and the split
                                           # are gone); from then until 2026-07-31 the ONE default was
                                           # `quality` (2/3), so numbers recorded in that window are at
                                           # 0.444x the PIXELS a flagless session traces — 2/3 is a
                                           # LINEAR scale, 1920x1080 -> 1280x720, so pixel-proportional
                                           # costs scale by 0.444 and per-tile ones (the level ladder)
                                           # barely move at all. `--lock-res quality` still spells the
                                           # 2/3 arm (the preset vocabulary is decoupled from the
                                           # default constant — see xess::lock_scale). Headless
                                           # `--spin` stays at native unless --lock-res
                                           # is passed — benchmarks must not have defaults move under
                                           # them, the vendor-default rule (native happens to coincide
                                           # with today's default, but the rule is independent of it).
                                           # The G/X/K accept from the 2/3 era is moot at native:
                                           # toggling the upscaler OFF mid-session presents the LOCKED
                                           # res, which is now the window res anyway; and
                                           # vendor_defaults' Intel entry is measured at the res the
                                           # flagless session traces again (its old OPEN DEBT is
                                           # resolved). Also takes quality (2/3)
                                           # |balanced|performance|ultra-performance or a ratio in (0, 1]
cargo run --release -- --gpu          # GPU-resident tracing: the whole quadtree + shading in D3D12
                                      # compute with DXR RayQuery rays (needs the DXC DLLs + RT tier 1.1;
                                      # falls back to the CPU renderer with the reason on stderr).
                                      # Composes with the chain's wired level (GPU-born G-buffers,
                                      # zero CPU readback — RR/XeSS/FSR4-RR/FSR3 all GPU-fed);
                                      # --no-upscale = plain. Wins over the --dxr default
cargo run --release -- --gpu --xess   # GPU tracer -> XeSS-SR composition (implies --no-dlss); the
                                      # render res is LOCKED per session (--lock-res, default native
                                      # 100% — `--lock-res dynamic` is not honorable under --gpu, it
                                      # locks at that same default with a loud line)
cargo run --release -- --gpu --nppd   # GPU tracer -> GPU-RESIDENT NPPD -> XeSS: ONNX Runtime executes
                                      # on the tracer's own queue (DML1) with the staging buffers bound
                                      # as tensors — zero per-frame CPU traffic (pack/warp/crop are
                                      # nppd.hlsl kernels). J toggles; XeSS-only (--no-xess forces it off)
cargo run --release -- --check-gpu    # GPU tracer gate suite + bench (needs a real GPU + the DXC DLLs;
                                      # composes with --stress; exit 2 = environment, 1 = a gate failed)
cargo run --release -- --dxr          # the by-the-book DXR pipeline (RTPSO + SBT + DispatchRays with
                                      # raygen/closest-hit/miss shaders) — the DEFAULT render mode
                                      # on NVIDIA/AMD (--cpu opts out). ON AN INTEL ADAPTER the
                                      # flagless default is the WAVEFRONT tracer instead
                                      # (main::vendor_defaults — measured 2.6-5.1x, see the vendor-
                                      # aware defaults paragraph; DXR stays armed as the automatic
                                      # fallback if the wavefront init fails), so --dxr is no
                                      # longer a pure no-op: it PINS the DXR start against that
                                      # policy (mode_explicit). F toggles CPU <->
                                      # DXR live. COMPOSES with the chain's wired upscaler: DXR-fed
                                      # DLSS-RR / FSR4-RR / XeSS / FSR3 — tracing at the LOCKED
                                      # --lock-res scale (default native 100% — the ONE
                                      # session default, every mode; window-res when plain
                                      # either way). Needs the
                                      # DXC DLLs + RT tier 1.0; falls back to the
                                      # CPU renderer with a loud line (the chain's upscaler stays
                                      # wired). SPACE cycles all three render modes live (see the
                                      # interactive-keys paragraph); the CLI flags pick the mode a
                                      # session STARTS in. This pipeline's rays ride --dxr-inline
                                      # (below): DEFAULT 1 = inline RayQuery secondaries — 2 on an
                                      # INTEL adapter (main::vendor_defaults, 2026-08-01)
cargo run --release -- --dxr-inline 0 # A/B lever: the DXR pipeline back on ALL-TraceRay dispatch —
                                      # the pre-W2 by-the-book build, bit-identical library. The
                                      # CROSS-VENDOR DEFAULT is 1: primary TraceRay -> chs_shade,
                                      # every secondary an inline RayQuery inside the hit shader
                                      # (MaxTraceRecursionDepth 1) — promoted because it strictly
                                      # DOMINATES 0 at every measured point on both vendors
                                      # (spp=1 tracer: B70 9.05 -> 2.35 ms, 4090 1.34 -> 0.26;
                                      # never slower at any spp). 2 = everything inline in raygen
                                      # (DispatchRays as a bare launch grid) — the measurement arm
                                      # that proved launch overhead ~ 0, and THE INTEL DEFAULT
                                      # since 2026-08-01 (main::vendor_defaults: mode 2 beats 1 on
                                      # the B70 at every measured point — 1.41/1.22/1.29 vs
                                      # 2.35/1.64/1.94 spp=1, world span 4.77 vs 5.36 — while the
                                      # 4090 prefers 1; and mode 1's fat hit shader pays occupancy
                                      # per sample, B70 marginal 2.2 ms/sample vs mode 2's 1.11,
                                      # so high spp widens it). ANY explicit --dxr-inline N — 1
                                      # included — sets dxr_inline_explicit, the policy's veto
                                      # (presence-not-value, the --spin-frames doctrine), and a
                                      # settings-file value vetoes too (the renderer.mode
                                      # precedent: the menu writes it). Armed modes need tier
                                      # 1.1/SM 6.5 (lib_6_5); lesser hardware degrades to 0 with
                                      # one loud line. The cross-vendor default stays quiet; 0/2
                                      # print (on Intel the mode-2 line names the vendor route +
                                      # the opt-out); an illegal value exits 2 (CLI) / warns
                                      # (settings file). Headless (--check*/--spin) never runs
                                      # the vendor policy — gates stay a pure function of the
                                      # command line. See the DXR section's ablation table.
                                      # 3 = THIN CHS + DEFERRED COMPUTE SHADE (2026-08-03, built on
                                      # the mechanism campaign's finding that Arc executes a fat
                                      # shader hosted in a raygen/CHS stage at 3-4.5x its compute
                                      # cost — FR_ABL=nosec collapsed mode-1 DXR 2.395 -> 0.478 ms,
                                      # BELOW the compute reference's 0.604, with the component
                                      # ablations sub-additive and `noglass` "saving" 0.28 on a
                                      # glassless scene: an occupancy/spill tax, not ray work; the
                                      # inherited t_start measured EXACTLY 0.000 via the new
                                      # FR_ABL=tzero lever). Raygen fires ONLY the bare-hit primary
                                      # (HgHit — cutout any-hit + relief re-march inherited) and
                                      # writes a 20 B record at u7 (the wavefront's dead qleaf
                                      # register, the cloud-cache u5/u6 precedent); dxr_shade.hlsl
                                      # (cs_6_5) shades from the record with rt.hlsli's inline
                                      # secondaries; one sample per pass pair (index in the b1 push
                                      # constants), cross-pass sum at u8, one store-or-add splat on
                                      # the last pass. MEASURED (spin path 1080p spp=1, dxr core,
                                      # default/stress/SM-lp): B70 1.39/1.56/1.56 vs mode 1's
                                      # 2.51/1.67/2.20 — THE BEST DXR ARM ON ARC, and the thin
                                      # dispatch is finally cheap (dxr-rays 0.23-0.35; THE WORLD
                                      # 0.54 vs mode 1's 2.87). NOT promoted, two measured reasons:
                                      # 4090 mode 1 still edges it (0.224 vs 0.243; spp=16 mode 2
                                      # 2.31 vs 3.53 — 2N RTPSO rebinds), and on Arc the DEFERRED
                                      # KERNEL now pays the codegen tax the CHS used to (dxr-shade
                                      # 1.124 vs the reference kernel's 0.603 for strictly MORE
                                      # work there — Arc compute codegen is knife-edged; the D2
                                      # lottery read 1.41/1.77/2.46 across identical builds), so
                                      # the wavefront still wins every Arc point (0.745 spin /
                                      # 3.25 world vs D3 1.39/4.73) and the Intel vendor default
                                      # STANDS (see vendor_defaults' 2026-08-03 paragraph).
                                      # Follow-on that would change that: split the deferred
                                      # kernel (hit/sky — the wavefront's own leaf+sky lesson) or
                                      # find its register cliff; dxr-shade < reference is the bar.
                                      # KNOWN REFUSAL: mode 3 + --heightfield on Intel driver
                                      # 32.0.101.8805 hangs the device (DEVICE_HUNG, GBV silent,
                                      # 4090 passes the identical suite) — DxrGpu::new degrades
                                      # the combo to mode 1 with a loud line; re-test on newer
                                      # drivers. COMPARISON-TARGET NOTE: with mode 2 now the Intel
                                      # DXR default, mode 2 is D3's Arc bar — JUDGED PER BINARY
                                      # 2026-08-04 (merged tree, B70, same binary, ABBA ±0.01):
                                      # D3 wins ONLY the default scene (span 1.40 vs 1.80, −22%);
                                      # D2 wins stress (1.40 vs 1.44), SM-lp (1.94 vs 2.20), THE
                                      # WORLD parked (4.93 vs 5.08), and every spp=16 point by
                                      # 26-85%. The thin half works everywhere (dxr-rays
                                      # 0.25-0.52, world 2.70 -> 0.52); the deferred kernel is
                                      # the whole loss (dxr-shade 1.12/1.10/1.81/2.35 vs the
                                      # ~0.60 reference class) — NO promotion, mode 2 keeps the
                                      # Intel default, dxr-shade < reference stays the bar; both
                                      # arms are lottery-prone, re-judge per binary.
                                      # Gates: --check-dxr --dxr-inline 3 green on
                                      # default/smlp/stress (B70) and smlp+relief (4090); 4 new
                                      # cargo-test source pins (miss-sentinel-before-consumers,
                                      # no-TraceRay-in-the-cs-unit, rt_dxr guards intact + inst
                                      # guarded, thin-raygen-writes-only-the-record)
cargo run --release -- --dxr-sbt 1    # EXPERIMENT lever (default 0 = off): the many-record,
                                      # MATERIAL-SORTED SBT ladder — the Intel-brief Q4
                                      # counterfactual (the TSU sorts by shader RECORD; our SBT
                                      # had effectively one). 8 field-derived shading classes
                                      # (src/shadeclass.rs — the STRIPS table IS the soundness
                                      # argument: a class may strip a shade arm only when its
                                      # membership predicate forces that arm's guard data-false,
                                      # self-tested in --check + re-verified on the LIVE scene
                                      # at upload; anything not provably strippable lands in
                                      # uber) partition every blas-split chunk into per-class
                                      # SUB-CHUNK instances (blas_split::refine_by_class —
                                      # INSTANCE-keyed, never multi-geometry: PrimitiveIndex()
                                      # restarts per GEOMETRY, which would break tri_of on BOTH
                                      # pipelines and drag GeometryIndex()'s SM 6.5 floor into
                                      # the lib_6_3 mode-0 path; instance-keying keeps the remap
                                      # contract with ZERO shader edits, and the wavefront
                                      # ignores hit groups entirely so the grown TLAS is
                                      # transparent to it). Each instance carries
                                      # InstanceContributionToHitGroupIndex = class*3 into a
                                      # class-major [HgShade_ck, HgHit, HgOcclude]x8 SBT — every
                                      # TraceRay call site untouched (multipliers stay literal
                                      # 0). The sway tail is RELABELED, never split (the
                                      # cells-parallel contract); sub-chunks stay under the cap
                                      # by construction; windows/stream/FR_SPLIT_AUDIT derive
                                      # from the mutated plan and need no changes. MODE 1 =
                                      # ALIAS records: 8 ExportToRename aliases of the ONE
                                      # chs_shade — identical code, distinct sort keys, zero new
                                      # compiles — isolating the PURE record-sort/repack effect
                                      # (plus the sibling sub-chunk AABB overlap cost, the
                                      # structural price of instance-keying). MODE 2 =
                                      # SPECIALIZED records: one extra DXIL library per class
                                      # PRESENT in the scene (k != uber), compiled with
                                      # shadeclass::strip_defines(k) prepended — shade.hlsli
                                      # gained SHADE_MAT_* macro seams over every material-
                                      # feature guard whose #ifndef defaults ARE the verbatim
                                      # expressions (all five pasting units stay semantically
                                      # identical unarmed — the same-seed wavefront-vs-reference
                                      # bit A/B is the drift tooth; REFL's seam carries the MIS
                                      # coupling: refl_ray feeds the VNDF block AND the w_l
                                      # reweight, so a strip keeps w_l=1 and light sampling
                                      # delivers the whole sun specular, rng pair inside the
                                      # gate so streams never need a burn). Each specialized
                                      # library exports exactly {chs_shade_ck <- chs_shade};
                                      # lib 0 aliases only uber + ABSENT classes (exported
                                      # names are state-object-unique; ah_*/misses resolve
                                      # cross-library). The identifier audit's REQUIRED set
                                      # narrows to specialized ∪ uber, and a dedupe there fails
                                      # HARD on every vendor (different libraries folding is a
                                      # defect, not a quirk) — MEASURED 2026-08-04: NVIDIA
                                      # mints DISTINCT identifiers for specialized libraries
                                      # (3/3 default scene), so it genuinely joins the ladder
                                      # at this rung; mode-2-vs-mode-1 accum drift is the
                                      # predicted DXC-rescheduling class (default scene: ~1% of
                                      # channels at max |d| 7.6e-6 — noise-scale, which is why
                                      # that compare is REPORT-ONLY and the statistical suite
                                      # is specialization's gate: a mis-routed class strips a
                                      # LIVE arm and blows T2's 2% loudly). MODE 3 = RECURSIVE
                                      # class dispatch: rung 2's records dispatched the way
                                      # production titles feed the TSU — every reflection/
                                      # glass continuation is a REAL TraceRay at
                                      # RayContribution 0, so the hit instance's class*3
                                      # contribution lands it in the hit surface's OWN
                                      # specialized closest-hit (routing = SBT arithmetic,
                                      # zero shader-side dispatch; rt_dxr.hlsli::trace_shade).
                                      # shade_split's DXR_SBT_RECURSE arm collapses the lap
                                      # loop to one iteration — the hardware ray stack
                                      # replaces the stash; Beer–Lambert multiplies the
                                      # RETURNED radiance (the CPU's own association); ind_s
                                      # becomes the literal rtput*child_color; rng round-trips
                                      # through the payload so the stream keeps the CPU DFS
                                      # draw order; depth+cone ride the repurposed sp lanes
                                      # (no payload growth past the 32 B config). HYBRID:
                                      # shadow/AO occlusion stays inline RayQuery (rt.hlsli
                                      # rides along; lib_6_5 + tier 1.1 or degrade to 2),
                                      # which is what caps MaxTraceRecursionDepth at 5
                                      # (primary 1 + refl 1 + the depth<TRANS_MAX_DEPTH=4
                                      # chain's 3 — the pipe_cfg derivation; exceeding a
                                      # declared depth is device removal, so the bound is
                                      # soundness, not tuning). Continuation misses take the
                                      # miss_rec SENTINEL (miss index 3 — the 4th record
                                      # fills the SBT's [64,192) miss gap to the byte): t=INF
                                      # and NO sky, because a reflection miss needs the
                                      # PARENT lobe's MIS weight — the parent keeps its own
                                      # miss arms. Arms only at --dxr-inline 0 (inline modes
                                      # have no TraceRay continuations to redirect; asked-for
                                      # anyway degrades to 2 loudly). Unbuilt/unarmed rungs
                                      # degrade loudly at DxrGpu construction. A dev MEASUREMENT
                                      # lever, the --sw-rays class: no vendor policy, no
                                      # settings row, loud on every armed mode, off-state
                                      # byte-identical (source AND instance descs). Must be set
                                      # at parse — the SceneGpu core bakes contributions at
                                      # UPLOAD (a partition-free core degrades the pipeline to
                                      # the one-record SBT with one loud line); --dxr-inline 2
                                      # composition is VACUOUS (zero TraceRay dispatches no
                                      # record) and --dxr-inline 3 nearly so (only the thin
                                      # bare-hit record dispatches — the sorted SHADING records
                                      # never run), both said loudly. Gates: shadeclass::self_test (the
                                      # strip-soundness must-fire + all-8 anti-vacuity) and
                                      # blas_split's refine spec-replay + grow must-fire in
                                      # --check; `--check-dxr --dxr-sbt 1` adds T1d — the
                                      # construction audit (>=2 live classes; PAIRWISE-DISTINCT
                                      # alias identifiers — MEASURED 2026-08-04: NVIDIA DEDUPES
                                      # the 8 aliases to ONE identifier on every scene while
                                      # the Intel B70 mints all 8 distinct, so rung 1 is an
                                      # INTEL-ONLY instrument (the vendor the TSU experiment
                                      # exists for) and NVIDIA joins at rung 2 where genuinely
                                      # different libraries cannot dedupe; the gate is HARD on
                                      # Intel, a recorded loud note elsewhere) and the
                                      # alias-vs-off same-seed A/B through a
                                      # SECOND SceneGpu core (the partition changes the plan, so
                                      # the T1c one-core flip is insufficient): BIT-identical
                                      # accum/tbuf/info on tint-free scenes — aliases run
                                      # identical code — with transmissive scenes printed
                                      # ungated (any-hit tint order is hardware-arbitrary by
                                      # contract, and the partition moves exact-t ties). NOTE
                                      # the routing wiring's real teeth arrive with rung 2:
                                      # under aliasing a mis-routed class is image-neutral by
                                      # construction; specialized records make it fail T2.
                                      # `--check-dxr --dxr-sbt 2` swaps T1d's image arms (the
                                      # off-core bit A/B cannot hold under rescheduling): (a)
                                      # rebuild DETERMINISM, bit-exact and HARD — a second
                                      # armed pipeline on the SAME core (partition identical
                                      # across armed modes; only RTPSO/SBT differ) must
                                      # reproduce accum/tbuf/info to the byte — and (b) the
                                      # mode-1 comparison, report-only (above). Both suites
                                      # pass armed on default/stress/SM-lp, NVIDIA + B70.
                                      # `--check-dxr --dxr-inline 0 --dxr-sbt 3` gates mode 3
                                      # the same way (armed rows REQUIRE the explicit
                                      # --dxr-inline 0 — the parse default is 1 and headless
                                      # never runs the vendor policy, so without it the row
                                      # silently gates rung 2) — green on default/stress/
                                      # SM-lp × NVIDIA + B70 + the GBV run. DEEP-CHAIN
                                      # LIVENESS is pose-bound (the committed poses recurse
                                      # only to depth 2 — the SM-lp default frame's drift
                                      # report is bit-equal mode 2's, the tell): the glassware
                                      # close-up (--cam 0.71,1.55,0.45,0.71,1.25,-0.35,
                                      # SM-lp) is the depth-proof pose — radiance A/B 0.031%
                                      # NV / 0.054% B70 with 444800 hit px of glass chains
                                      # live and no depth violation; its exit 1 is ENTIRELY
                                      # the documented mv_selftest close-up caveat (median
                                      # 3.156 vs 0.17 limit, vendor-independent, pre-existing
                                      # — read the log, not the exit code, at that pose).
                                      # MODE 2'S KNOWN APPROXIMATION, measured at that same
                                      # pose: a specialized record's lap loop also shades its
                                      # CONTINUATION surfaces, which can be a different class
                                      # (a tex-opaque parent's strips drop a glass child's
                                      # transmission) — mode-2-vs-mode-1 drift max |d| 9.61e-3
                                      # (~3% of channels) vs the 5.96e-8 fp floor; real,
                                      # bounded under T2, the documented price of an occupancy
                                      # instrument. Mode 3 closes it BY CONSTRUCTION (every
                                      # surface shades in its own class record) — the second
                                      # reason that rung exists.
                                      # THE LADDER, MEASURED (2026-08-04, --spin path 1080p,
                                      # min of 2 reps forward/reversed, spans in ms at
                                      # default/stress/SM-lp; CSVs + protocol in the session
                                      # scratchpad's matrix1): at --dxr-inline 0 — the
                                      # by-the-book all-TraceRay pipeline, the TSU's regime —
                                      # B70 spp=1 reads sbt0 8.02/5.31/6.79 → sbt1
                                      # 8.60/5.39/6.67 (FLAT with 8 genuinely distinct sort
                                      # keys) → sbt2 2.51/2.38/2.17 (−55..−69%) → sbt3
                                      # 1.49/1.78/1.29 (−66..−81%); 4090 1.17/0.83/1.15 →
                                      # flat → 0.25/0.34/0.24 → 0.19/0.27/0.20. Per-sample
                                      # marginals (spp16−spp1)/15: B70 7.02/4.54/5.92 → sbt3
                                      # 0.93/1.03/0.81 (5-7x); 4090 1.26/1.48/1.00 → sbt3
                                      # 0.15/0.29/0.14. At --dxr-inline 1, sbt2 is −20..−30%
                                      # on the 4090 (0.186/0.228/0.202) and −50..−60% on the
                                      # B70 (1.05/0.87/0.97) — which BEATS the same-day
                                      # inline-2 (1.80/1.40/1.94) and inline-3 (1.40/1.44/
                                      # 2.20) bars: `--dxr-inline 1 --dxr-sbt 2` is the
                                      # fastest DXR configuration measured on Arc (the
                                      # wavefront still wins outright — 0.64/0.78 recorded).
                                      # FOUR READINGS: (1) sort keys ALONE buy ~0 on both
                                      # vendors — mode 1 is flat even where the TSU has 8
                                      # distinct keys, because sorting identical fat shaders
                                      # has nothing to gain; (2) SPECIALIZATION is the prize —
                                      # thin per-class hit shaders recover 55-80% of the
                                      # by-the-book pipeline's cost, refining the launch-tax
                                      # story: most of the tax was the FAT UBER SHADER hosted
                                      # in RT pipeline stages, not TraceRay itself; (3) the
                                      # recursion rung lands the textbook pipeline at parity
                                      # with the inline hybrids (B70 sbt3 1.49 vs the same-day
                                      # inline-3 1.40; 4090 0.19 vs inline-1-sbt-2's 0.186) —
                                      # noting sbt3-vs-sbt2-at-inline-0 confounds recursion
                                      # with inline occlusion; (4) specialization also
                                      # STABILIZES Arc codegen — the >15%-spread rows in the
                                      # rep-trust check are all fat-shader configs (mode 0/1
                                      # SM-lp), the specialized rows repeat tight.
                                      # ENVIRONMENT (2026-08-04): the AMD iGPU ("Radeon(TM)
                                      # Graphics", driver 32.0.21018.14) AVs 0xC0000005 inside
                                      # CreateStateObject/identifier query on ANY armed mode —
                                      # mode 1 (Commit A code, single library) crashes
                                      # identically, mode 0 passes, the SAME run passes under
                                      # --gpu-debug (the debug layer masks it), and NVIDIA's
                                      # debug layer validates the identical descs clean — so
                                      # the driver chokes on ExportToRename itself, the
                                      # pre-existing-iGPU-environment class (the spp-readback
                                      # precedent). Deterministic (2/2 reps). The vendor
                                      # rename triptych: NVIDIA dedupes, Intel mints distinct,
                                      # this AMD iGPU crashes. Not coded around — the ladder
                                      # is a dev lever and the iGPU is not a measurement
                                      # target; re-probe when an RDNA4 discrete card returns.
cargo run --release -- --check-dxr    # DXR pipeline gate suite (needs a real RT GPU + the DXC DLLs;
                                      # composes with --stress; exit 2 = environment, 1 = a gate failed)
cargo run --release -- --dxc-path <d> # DXC DLL directory (default SDKs\dxc\bin\x64; or FRUSTRACER_DXC_PATH)
cargo run --release -- --prefer-intel # pick that vendor's adapter for the D3D12 device (also
                                      # --prefer-nvidia / --prefer-amd; default NVIDIA, or AMD under
                                      # --fsr). A preference, not a requirement: features the picked
                                      # GPU can't support (DLSS/FSR/RT tiers) fall back with a log
                                      # line, per the existing probes. Applies to --check-gpu /
                                      # --check-dxr too
cargo run --release --features tracy  # Tracy CPU profiling (see the Profiling section; off = zero-cost)
cargo run --release -- --quinlight    # REGISTERED CONSENSUS (a port of quinlight-player's
                                      # consensus_registered.comp): suspend the chain's
                                      # first-hit-wins rule, wire EVERY supported level at once
                                      # (DLSS-RR + FSR4-RR + XeSS + FSR 3.1), run them all over the
                                      # SAME traced frame, and present the LK-registered winsorized
                                      # consensus of their outputs. GPU-fed only (--dxr/--gpu).
                                      # --quin-anchor N picks the engine that defines the spatial
                                      # frame (default 0 = the highest wired level). The chain flags
                                      # still compose: --quinlight --no-dlss fuses XeSS + FSR3
cargo run --release -- --spin path    # headless deterministic benchmark/profiling workload: the
                                      # interactive frame contract on a closed-loop Catmull-Rom
                                      # camera (still | path; --spin-frames n, default 2000 — a
                                      # DEFAULTED count is extended so the timed span covers a
                                      # whole SPIN_LAP=600-frame lap past the warm-up, an
                                      # EXPLICIT one is obeyed verbatim; --spin-warmup n
                                      # excludes leading frames, default 20 but 1600 on an
                                      # INTEL adapter, where the driver's async shader
                                      # recompile lands ~600-1500 frames in — see the Arc
                                      # measurement trap in Profiling, and note that every
                                      # Intel --spin number recorded in this file predates the
                                      # warm-up and was taken at WARMUP=20; --spin-hybrid /
                                      # --spin-plain pick the quadtree or the per-pixel
                                      # root-traversal reference arm for the CPU and --gpu
                                      # runners, and print a note under --dxr, which has only
                                      # its one DispatchRays arm; pose is
                                      # a pure function of the frame index — bit-repeatable A/Bs;
                                      # composes with --no-temporal / --no-replay / --no-adopt).
                                      # Drives the GPU arms too: `--gpu` (wavefront) or an EXPLICIT
                                      # `--dxr` runs the same pose loop through HeadlessGpu
                                      # (record -> execute -> block, no swapchain), at the
                                      # `--lock-res` scale — NATIVE unless --lock-res is passed,
                                      # deliberately independent of the interactive default (a
                                      # benchmark's res must not move under recorded numbers; the
                                      # default happens to be native again since 2026-07-31, but the
                                      # rule predates and outlives that) — with the SAME
                                      # per-frame contract as
                                      # the CPU arm (1-spp upscaler quality, accumulate off,
                                      # frame-uniform Halton) so --cpu/--gpu/--dxr rows compare
                                      # directly. `opts.dxr` defaults ON, so it takes an explicit
                                      # --dxr — a bare --spin still drives the CPU renderer. This is
                                      # the deterministic GPU benchmark the tree lacked: the
                                      # `gpu hybrid` bench row is warm-clock noisy (its own spp
                                      # sweep interleaves and takes medians for that reason) and an
                                      # interactive --gpu-timing table depends on where the camera
                                      # happened to be. It measures the TRACER (no G-buffer pack,
                                      # no feed/upscale — those need a swapchain and are constant
                                      # across tracer changes); pair with --gpu-timing, whose
                                      # per-pass table prints every 120 frames and at exit
cargo run --release -- --cinematic tour --cinematic-res 3840x2160 --cinematic-fps 60 \
                        --cinematic-frames 1200 --cinematic-hdr
                                      # MEDIA MODE (see the Cinematic capture section): headless,
                                      # deterministic stills and camera-spline sequences for the
                                      # README/release. Presets hero|islands|tour|orbit|foliage|hud|list,
                                      # or a JSON shot-list path; bare --cinematic = hero + the
                                      # catalogue. Writes a numbered PNG sequence + manifest and
                                      # PRINTS the exact ffmpeg commands (--cinematic-encode runs
                                      # them). The GPU arms capture through the upscaler chain BY
                                      # DEFAULT (DLSS-RR -> FSR4-RR -> XeSS -> FSR3 at 100%
                                      # render scale — DLAA-grade; the frame written is the
                                      # model's RECONSTRUCTED output; chain flags steer it,
                                      # --no-upscale / GI shots / --cpu / chain exhausted fall
                                      # back to accumulation loudly — see the cinematic section).
                                      # Sub-flags: -res WxH (odd dims round down — yuv420p),
                                      # -samples N (sub-frames per OUTPUT frame: reconstruction
                                      # warm/converge passes, or plain accumulation on the
                                      # fallback arms — composes with --spp, a different axis
                                      # that amortizes the
                                      # quadtree), -frames, -fps (drives the cloud clock AND the
                                      # encode), -island, -gi (forces the wavefront: DXR has no
                                      # hemi stage), -overlay, -hud off|hud|menu|settings:<Group>,
                                      # -hdr (16-bit PQ frames + HDR10 HEVC; stills also get a
                                      # linear EXR master + a PQ-tagged AVIF), -out, -encode,
                                      # -exposure EV (STOPS, -8..=8, applied to linear radiance at
                                      # the ONE write site so the SDR PNG / PQ frames / EXR master
                                      # are one exposure by construction; 0.0 returns EXACTLY 1.0
                                      # and the call site branches around the copy, so every
                                      # pre-exposure capture is bit-identical. It exists because the
                                      # tonemap is anchored at a fixed paper white while the
                                      # interesting parts of these scenes are ENCLOSURES whose sun
                                      # is occluded by construction: a physically correct San Miguel
                                      # patio at 15:30 is 2-3 stops under a lit exterior — correct,
                                      # and unpublishable. Brightening the sky or the curve would be
                                      # a lie about the lighting; opening the aperture is what a
                                      # photographer does),
                                      # -dry-run. LOADS THE WORLD by default (it is a media mode,
                                      # not a gate — and it is exclusive with --spin/--check*,
                                      # which keep their own scenes so no must-fire gate moves).
                                      # The only path that can render a moving camera WITH hemi GI,
                                      # because every output frame is a static accumulating pose
cargo run --release -- --no-settings  # ignore frustracer-settings.json for this run (the pause
                                      # menu's saved settings — loaded as DEFAULTS the CLI flags
                                      # override; auto-saved on every menu edit; headless
                                      # --check*/--spin runs always ignore it. ESC opens the menu,
                                      # F1 toggles the HUD — see the HUD/pause-menu section)
cargo run --release -- --no-vsync     # uncapped presentation (Present sync interval 0 on a tearing
                                      # swapchain when DXGI supports it) so interactive frame times
                                      # measure the renderer, not the monitor refresh; composes with
                                      # every mode/upscaler
                                      # (the 10-bit swapchain is ON BY DEFAULT — see the HDR section)
cargo run --release -- --no-hdr       # A/B lever: force the legacy 8-bit B8G8R8A8 swapchain (also
                                      # the FG wrap-failure fallback; the swapchain is otherwise
                                      # ALWAYS 10-bit R10G10B10A2 — the scRGB f16 chain is deleted)
cargo run --release -- --hdr10        # force the PQ declaration (R10G10B10A2 +
                                      # G2084, tone::ToneMode::Pq — 709->2020 matrix + ST 2084 at
                                      # the end of the one curve) in ANY session — which only ADDS
                                      # the HDR-off case, because PQ is the DEFAULT on an HDR-ON
                                      # display; on an HDR-OFF display the same buffer defaults to
                                      # its gamma-2.2 reading instead (Sdr10 — DXGI's default
                                      # interpretation of an undeclared UNORM chain; see the HDR
                                      # section for the bytes-per-present measurement).
                                      # Override-wins like --hdr-peak, including over an "HDR off"
                                      # probe verdict; a REFUSED G2084 declare relabels the session
                                      # Sdr10 on the same swapchain (no rebuild — the buffer's
                                      # default reading IS Sdr10). The swapchain flags are a
                                      # THREE-way (8-bit SDR | Sdr10 | HDR10 — one 10-bit format,
                                      # two curves) spelled as toggles, later flags win across the
                                      # pairs: `--no-hdr --hdr10` = PQ, `--hdr10 --no-hdr` = 8-bit,
                                      # and `--no-hdr10` = Sdr10 — "10-bit but NOT PQ", needed
                                      # because PQ is the HDR-display default so the gamma arm
                                      # isn't reachable as "neither flag" there (cli::self_test
                                      # pins the three-way and the fact that each arm still wins
                                      # from any predecessor). `--hdr` returns to the
                                      # display-probed default. Exposed in the settings menu as the
                                      # Display page's hdr10 row (restart-tier; its OFF state
                                      # means Sdr10, mirroring --no-hdr10; files written before the
                                      # scRGB retirement stored false meaning scRGB — it now reads
                                      # as Sdr10, deliberately unmigrated).
                                      # The wrapper-FG families need no format special-case any
                                      # more (XeSS-FG rejected scRGB fp16 but takes 10-bit —
                                      # VERIFIED GENERATING at HDR10 on the B70; the Sdr10 wrap is
                                      # the same desc and falls to 8-bit + rewrap if refused).
                                      # See the HDR section
cargo run --release -- --hdr-paper-white 120  # where linear 1.0 lands, in nits (default 200);
                                              # LOWER = more highlight headroom above white
cargo run --release -- --hdr-peak 1000        # override the display's reported peak (A/B lever).
                                              # WINS over the probe, including over an "HDR off"
                                              # verdict — the probe can be wrong, and an override
                                              # that no-op'd in exactly that case would be no
                                              # escape hatch at all
cargo run --release -- --check-gpu --gpu-timing  # the same timestamps over the DETERMINISTIC workload: a
                                            # per-pass table under every bench row (drained per row, so the
                                            # reference kernel's frames can't dilute the wavefront's mean),
                                            # plus a spp=1-vs-spp=16 pair — which is what separates a pass's
                                            # FIXED cost from its per-sample MARGINAL cost. The per-pass
                                            # AMD-vs-NVIDIA diff this makes possible is what found the
                                            # LEAF_GROUP wave64 bug
cargo run --release -- --gpu --pix-markers  # PIX events on the D3D12 lists (needs WinPixEventRuntime.dll,
                                            # --pix-path / FRUSTRACER_PIX_PATH, default SDKs\pix\bin\x64)
cargo run --release -- --gpu --gpu-timing   # D3D12 timestamp queries around the SAME marker brackets,
                                            # printed as a per-region GPU-ms table every 120 frames. No
                                            # DLL, every vendor — and the ONLY per-pass GPU numbers
                                            # available on Intel, whose captures PIX cannot analyze at
                                            # all (see Profiling)
```

There are essentially no unit tests — the exception is four `#[cfg(test)]` SHADER-SOURCE gates at the end of `gpu/trace.rs` (run by `cargo test`, which CI runs alongside the executable suites), which assert ordering/monotonicity statements inside the HLSL that are load-bearing for soundness but that no CPU-only gate can reach (`--check-gpu`/`--check-dxr` need a real adapter): the ftree overflow fallback rechecking stale distances before it lowers `best`, the empty-scene guards preceding the first `bvh_nodes[]` read, relief's full hardware interval + payload-carried logical bounds, and the continuation seam (LeafRec carries ONLY the opaque token, the leaf passes it through untouched, and the software provider validates it BEFORE its first `cut_pool[]` read — an ordering no CPU gate can see). They are deliberately narrow — never assert formatting — and the live HLSL remains the executable specification. **Otherwise `--check` is the test suite**: it renders a hybrid frame at **full depth and again through the depth-capped driver**, re-traces every pixel with a tmin=0 reference ray, and exits nonzero unless the `false-sky` and `tmin-overshoot` counters are exactly 0. In the capped pass, coarse (cell-flooded) pixels are excluded from every counter but must be > 0, and each capped tile's per-cell point samples (`KIND_LEAF`, see `sparse_fill`) ARE inside the gates like any leaf pixel and must also be > 0 whenever coarse pixels exist — proof the capped path ran (deterministic; no wall clock involved). It then warms a temporal cache with one frame and verifies five consumer passes the same way (static replay, forward dolly, dolly capped, dolly+yaw, pure yaw), asserting the temporal path demonstrably fired where that is structural (static: seeds and sky-tiles > 0; dolly and dolly-capped: seeds > 0; pure yaw: sky-tiles > 0). **`bvh::empty_self_test`** (the `empty-bvh` gate) covers the degenerate scene: a zero-triangle build must be an empty depth-0 hierarchy, and every scalar and cut-seeded traversal entry point must take its CLEAR-SPACE identity — `intersect`/`intersect_multi` `None`, `occluded`/`occluded_multi` `false`, `transmittance`/`transmittance_multi` exactly `Vec3A::ONE`, `nearest_geometry_distance_within` `None`, `refine_cut` empty — **without visiting a node**, which is what the gate actually asserts (`visits == 0`) and what a deliberately invalid `[u32::MAX]` root proves it does before dereferencing a caller-supplied id. That matters because the empty build is a COUNT-ZERO SENTINEL root, which is indistinguishable from an internal node to any consumer that reads it: on the GPU the same job is done at compile time by `trace::empty_defs`' `#define SCENE_EMPTY` (frustum.hlsli's `bound_query`/`refine_cut` and rt_sw.hlsli's three primitives return their identities ahead of the first `bvh_nodes[]` read; the wide tree instead ships one physically-present zero-occupancy root, see the two-tree bullet). The bounce integrators get the same treatment on a deterministic probe set: `sphcell::self_test` (closed-form Ω/PSA identities, exact partition, in-cell sampling), hemisphere AO and GI gates (`psa-viol`, `false-empty`, `tmin-overshoot`, `cut-miss` — all exactly 0) plus A/Bs against high-sample cosine references (AO: mean |Δ| < 0.02 and signed mean < 0.005 — the estimator is unbiased; GI: mean rel < 5% vs a reference running the same depth-1 policy `hemi::BOUNCE_Q` — that reference must integrate `sky::dome`, in lockstep with `hemi.rs`'s leaf miss, or it is scoring two different functions). The one-sky model has two closed-form gates: **`sh::self_test`** (basis orthonormality — projecting each band must return a unit coefficient, which pins every constant; the uniform-sky convention pin, radiance L in ⇒ exactly L out, which is what makes SH irradiance a drop-in for the old `AMBIENT · ao` and comparable to `hemi::gi`; accuracy vs a brute-force cosine-weighted reference, measured 0.035% mean / 0.188% worst; and projection determinism, since the byte-identical-build contract depends on it) and **`sky::self_test`** (the disc's radiance↔irradiance round-trip — the classic place to be off by 4π; cone sampling staying inside AND covering the disc; the disc test agreeing with the cone the sampler draws from; **the dome carrying no disc**, tested RELATIVE to the disc's own radiance since the Mie aureole legitimately peaks well above the dome's average; and the resulting ambient landing in a physically sane, blue-dominant band — the gate that pins `DOME_SCALE`). **Hemi sharing** gets its own family on 2×2 probe groups built under the renderer's exact predicate: the four zero-counters re-run PER MEMBER (each member re-validates the rep's folded empty claims, every leaf-ray tmin, and every recorded-cut traversal from ITS OWN apex), a paired same-seed shared-vs-unshared A/B (identical rng streams make common rays — fireflies included — cancel exactly; AO mean Δ < 0.005, GI rel Δ < 0.01; an unpaired construction was tried and measured the baseline estimator's firefly skew, not the sharing), same-seed share-on frame determinism and an fb-ao replay-vs-trace bit-identity pass (both with groups > 0 anti-vacuity), and `hemi-ao/gi (share off/on)` bench rows whose must-fires (groups > 0, fallbacks > 0, strictly fewer hemi queries) double as the KILL CRITERION guard: if share-on is not measurably faster on both the default scene and `--stress 5000`, the feature does not merge. Run it after any change to `frustum.rs`, `bvh.rs`, `render.rs`, `camera.rs`, `temporal.rs`, `hemi.rs`, `sky.rs`, `sh.rs`, `sphcell.rs`, or `shade.rs`. It also A/B benchmarks hybrid vs hemi-ao vs hemi-gi vs plain with node/ray counters and smoke-tests depth-capped dynamic frames at several caps. Three temporal-reuse gate families follow: **structure replay** (exact terminal pixel accounting; replay-vs-trace bit-identity of tbuf/info/accum at frame 0 AND at a warm jittered frame 1; a post-replay dolly verify on the frozen producer cache), the **claim ring** (exact pan-back must-fire ring hits — off the newest screen, answered verbatim by an older entry — plus a near-pose correctness pass), and the **query skip** (T1/T2 adoption must-fires; a 4-step dolly chain where the age cap must force requeries by step 4, every step reference-verified; and `dolly warm (adopt off/on)` A/B rows at preset q and 1-spp — the kill-criterion regression guard). **Multi-sampling** (`--spp`) gets its own `spp` gate family, at a FIXED spp=4 so a plain `--check` can never stop exercising it: the frame is rendered once per sample with `primary_sample = k` (the sample whose t lands in tbuf) and `verify_sampled` re-traces THAT sample's ray from tmin=0 — `false-sky`/`tmin-overshoot`/`hybrid-extra` exactly 0 for every k, which is the proof that an extra sample may ride the inherited t_start/cut (it is the same bug class, so it gets the same reference-ray proof); an accounting must-fire (primary rays exactly ×spp while frustum queries/nodes/tiles stay BIT-IDENTICAL — that inequality IS the amortization claim, and a regression there means multi-sampling started re-tracing the quadtree); a stability must-fire (mean inter-frame |Δ| strictly lower at spp=4 — if it isn't measurably quieter it isn't doing its job; structural, default scene); a pure-math distinctness gate (all MAX_SPP sub-pixel positions pairwise distinct — the Halton index must not wrap) plus a verify pass of the LAST sample at spp = MAX_SPP (the edge of the GPU's CB jitter table, invisible at spp=4); and an `spp=1|2|4|8|16` sweep whose printed cost-model fit (`ms(n) = F + m·n`) is where the "when do the returns stop" answer comes from. `--check-gpu` and `--check-dxr` carry the GPU halves (the same probe sweep — on the wavefront with the exact-zero gates AND the same-seed wavefront-vs-reference image A/B re-run at spp=4; on DXR as a per-sample CPU-vs-GPU t compare, which is what pins the CB's Halton jitter table to `dlss::jitter_for_sample`), plus interleaved-median spp bench rows (that row is warm-clock noisy — a cold first row can "measure" a physically impossible speedup). **The spp image A/B is gated RELATIVE to the image's own magnitude (mean |Δ| / mean |ref| < 1e-4), never absolutely** — and this is a correctness property of the gate, not a convenience. At spp=1 the two kernels are BIT-IDENTICAL; the divergence at spp>1 is per-sample fp rounding between two compile units' summations, which averages DOWN ~1/√N (the signature of independent rounding noise, not a bias) and scales with scene RADIANCE. So an absolute limit is a different limit on every scene: the original 1e-5 passed the default scene by 15% and failed `--stress 5000` outright (mean 1.46e-5) for no reason but that the stress field is brighter. The relative error is flat across scenes and vendors (default 1.95e-5, San Miguel 1.93e-5, stress 2.34e-5 NVIDIA / 2.69e-5 AMD — all at spp=4, the worst spp), so 1e-4 sits ~3.7× above the worst fp noise and ~100× below the ~1e-2 a real shading divergence produces. Gate-teeth are pinned: injecting a 0.1% error into the reference kernel's shade fails it at rel 1.02e-3 while the hot-channel count stays 0 — a systematic bias is invisible to the max/hot half and caught only by the relative mean, which is why BOTH halves exist. `--check --stress n` runs the same suite on the n-object stress field (deterministic sin-hash placement, `scene::stress_scene`) with the zero-counter gates intact but the "must fire" structural assertions skipped — those are tuned to the default scene's topology. Loaded OBJ scenes (`model.obj --check`) get the same structural skip: a real scene can lack the required features outright (a skyless view can't fire sky-tiles; a dense view legitimately overflows the replay recording arena), while the zero-counter gates run everywhere.

**Build config.** `.cargo/config.toml` links with `rust-lld` (bundled with the rustup toolchain — nothing to install; mold is ELF-only and cannot link a PE binary, so it is not an option here). It must keep emitting a PDB: `debug = "line-tables-only"` exists so Tracy's sampler and PIX can symbolize release frames, so any link-arg that drops debug info is off the table. `release` is `lto = "thin"` + `codegen-units = 16` (was 1 until 2026-07-23 — cgu=1 serialized ThinLTO into a fat-LTO-shaped single pass; 16 measured a one-line-touch rebuild 198 → 45 s at a priced +1-2% CPU-tracer ms/frame, interleaved `--spin path` medians in the Cargo.toml comment — CPU numbers recorded before that date carry the offset). `[profile.quick]` (`lto = false`, `codegen-units = 16`) is an ITERATION-ONLY profile — a one-line touch rebuilds in ~25 s vs ~45 s under `release`, because the ThinLTO whole-program pass, not the linker, is the cost. **Never take a benchmark number from `quick`**: every measurement this project reports (the `--check` A/B bench rows, the hemi-share kill criterion, the adopt on/off regression guards) is only meaningful under `release`'s `lto`/`codegen-units`. Correctness gates are perf-independent and run fine there.

**Source encoding.** Every file in this tree is **UTF-8 with no BOM**, and the comments are dense with `—`, `×`, `→`, `≤`, `Δ`, `π`. That makes the tree unusually sensitive to one Windows failure mode, which has already cost a repair: `ca1f9b6` landed `src/main.rs` double-encoded (837 occurrences across 29 sequences — an em-dash `e2 80 94` had become `c3 a2 e2 82 ac e2 80 9d`, "a-euro-rightquote") plus a leading BOM; `c39b6f3` repaired it. **The cause is Windows PowerShell 5.1's default pair, and it reproduces byte-for-byte**: `Get-Content` decodes a BOM-less file with the system ANSI codepage (cp1252), while `Out-File`/`>` writes UTF-8 **with** BOM — so `Get-Content f | ... | Out-File f` double-encodes every non-ASCII character in the file and prepends `ef bb bf` (it also converts LF→CRLF, which git normalizes away, so that half leaves no trace in history). The existing advice to pass `-Encoding utf8` when WRITING covers only half of it; the read side is the silent one. **So never round-trip a source file through a shell pipeline** — pass `-Encoding utf8` on BOTH ends, or do bulk edits with `python3` or the editor. It is the LARGEST file that is at risk, because that is the one somebody scripts a bulk edit on rather than editing normally: main.rs was the only casualty of a 16-file commit whose other files (CLAUDE.md included, 382 em-dash lines) came through clean. Nothing catches it on its own — the damage is comments-only, so it compiles and every gate passes; it surfaces as line noise in the source, in the RUNTIME's own stderr, and as a conflict on every em-dash line (rebasing a 21-line change across it cost one). `tools/hooks/pre-commit` is the guard — install per clone with `cp tools/hooks/pre-commit .git/hooks/pre-commit && chmod +x .git/hooks/pre-commit`, deliberately NOT via `core.hooksPath`, which would orphan git-lfs's four hooks and a scene blob reaching plain git is permanent bloat. Its header carries the repair recipe: the cp1252-encode/UTF-8-decode inverse applied per non-ASCII run, leaving runs that FAIL that round-trip untouched, which is what protects text that is already correct (hand-editing the characters only fixes the lines you happen to notice). Two traps if you ever touch the detector: `git grep -P` matches CHARACTERS, not bytes — a byte-class pattern silently matches nothing, and the hook's first draft "passed" against the known-bad blob for exactly that reason — and GNU `grep -P` is unusable in this environment at all ("supports only unibyte and UTF-8 locales", even under `LC_ALL=C`), so git's own matcher is the one to use.

Always benchmark in `--release`; debug builds are ~10× too slow to judge anything (but one debug `--check` run arms the `debug_assert`s in the cut logic). In the interactive window, **SPACE** cycles the render mode live — CPU frustum tracer → GPU wavefront (`--gpu`) → DXR (`--dxr`) → CPU — in EVERY session (the CLI flags only pick the starting mode): each GPU tracer is lazily built on first entry (DXC load + kernel/RTPSO compile), but the SCENE half (streams + BLAS/TLAS + textures) is ONE shared `Rc<SceneGpu>` cached in GpuContext — uploaded by whichever tracer comes first (the single `gpu scene:`/vram line per session), so the second tracer pays only its kernels + window-sized planes, plus (wavefront only) its own software trees (`SwTreesGpu`, the `gpu sw-trees:` line — sw BVH + ftree stay per-TraceGpu because DXR never binds them). This replaced the old both-tracers-own-a-copy design after 8K fullscreen on the B70 measured ~24 GB committed and the second tracer's BLAS pre-flight failing (GPU mode memoized off for the session); a failed init is still memoized (`trace_failed`/`dxr_failed`) and the cycle SKIPS that mode with a loud line (RT-tier-1.0-only hardware cycles CPU ↔ DXR; no DXC = CPU only), with the cached core EVICTED when no tracer is live so a CPU-fallback session doesn't strand gigabytes. Every entry restores that arm's session-default sub-mode (the F contract), resets `frame` + the target arm's histories, and entering CPU declares the discontinuity to the CPU upscaler histories; the temporal ring is dropped every GPU/DXR frame so a return to CPU never resumes against stale claims. **R** toggles hybrid vs the brute-force reference, **T** toggles dynamic resolution vs fixed half-res while moving, **O** shows the quadtree overlay, **C** runs verification on the current view, **G** toggles DLSS Ray Reconstruction, **N** toggles OIDN denoising (mutually exclusive with G), **J** toggles NPPD neural denoising (mutually exclusive with G/N/X; in a XeSS session it toggles the pre-upscale placement instead), **F** toggles the DXR DispatchRays pipeline (the DEFAULT mode on NVIDIA/AMD; Intel defaults to the wavefront — see the DXR section's vendor-defaults paragraph) against the CPU tracer — its historical CPU/GPU → DXR, DXR → CPU semantics, now valid from every mode; it composes with the chain's wired upscaler, and inside DXR mode G/X/K toggle that upscaler vs plain DXR (N/J stay CPU-side), **K** toggles FSR (whichever flavor the chain wired — FSR4+RR on RDNA4, FSR 3.1 upscale-only elsewhere), **M** toggles the temporal reprojection history in OIDN mode, **H** cycles hemisphere frustum bounces (off → AO → GI; still frames only), **U** doubles samples per pixel (1 → 2 → … → 128 → 1 — `--spp` live, in every render mode; a sample-count change resets accumulation and every temporal history, like a quality preset, and never on camera motion), **V** toggles heightfield relief vs plain normal-mapping in `--heightfield`-armed sessions (every render mode; the quality-preset reset set — frame + upscaler histories, never the temporal cache/replay; unarmed sessions — the default — print a note; see the Heightfield relief section), **B** toggles the GPU tonemap A/B (non-DLSS/OIDN mode), **ESC** opens the Slint PAUSE MENU (Resume / Settings / Exit — see the HUD section below; ESC no longer quits directly — the window X and the menu's Exit do, and a session whose HUD failed to init keeps the historical ESC-quits), **F1** toggles the HUD (compass + clock + the motion-gated keymap panel; display-stage only, NO reset of any kind), **F11** toggles borderless desktop fullscreen, **`.` / `,`** (held) fast-forward / reverse time-of-day at 1 game-hour per second (Xbox D-pad right/left does the same; Ctrl/Shift/bumpers scrub finer — see the `--tod` entry above; a TOD delta resets plain accumulation only — upscaler/denoiser histories and the temporal cache/replay are KEPT, since a held scrub fires per frame and lighting drift is a shading change the temporal integrators absorb). The window is resizable (maximize button works): any client-size change, once settled for 250 ms, is a TRUE NATIVE resize — the session exits and re-enters `session()` (main.rs) at the new size, rebuilding every window-sized buffer/controller/history, while `GpuContext::resize_output` keeps the device/queue alive and rebuilds the swapchain (ResizeBuffers — forwarded through any FG-family proxy) + upscaler contexts at the new output, re-querying their render-res ranges so the `--lock-res` ratio re-applies (the ratio is of the NEW window — the native default just means it IS the new window). Camera pose and all mode toggles survive via `Persist`; accumulation/temporal/histories intentionally restart. During a drag the old-size swapchain presents DWM-stretched (soft, never broken); a minimized window (0-dim) never commits a resize. Under `--gpu`/`--dxr` a resize re-pays kernel compile + the window-sized planes, but NOT the scene upload/BLAS build — the shared `Rc<SceneGpu>` core deliberately survives `resize_output` (device+queue live across it; only `drop_scene_tracers` — the live-scene-edit path — clears it).

## HUD, pause menu, and persisted settings (Slint software renderer)

The first on-screen UI: a **HUD** — compass + clock + a **render-mode pill** (CPU | GPU | DXR, derived at the call site from the `gpu_trace`/`dxr_on` pair — no other mode→string mapping exists; a SPACE/F mode change also WAKES the HUD like camera motion, else the switch would repaint invisibly) + a **sci-fi FPS graph** (40 bars × 125 ms buckets = a 5 s window of bucket-average FPS on a FIXED 0..120 scale, so the 60-fps reference line is a static element and a spike clamps instead of rescaling history; each bar is a STACKED PAIR `ui::FpsBar { base, fg }` — base = rendered FPS color-banded cyan ≥ 60 / amber ≥ 30 / red below (banded by RENDERED fps: FG doesn't cheapen tracing), fg = the frame-generation surplus in violet, fed per frame from `GpuContext::fg_display_mult()` (the title bar's own presented-per-rendered source) so `fg = 1000·(Σmult−n)/Σms` averages a mid-bucket FG flip honestly; the readouts deliberately don't invert — fps-now = presented (base+fg), ms-now = RENDER frame time; at > 120 rendered fps the stack clamps and the violet is invisible by design; fed `last_ms` — the PREVIOUS frame's render time — at the one `Hud::frame` call site; the Rust-side ring samples ALWAYS so a woken graph is current, but the Slint writes go through ONE persistent `VecModel` via per-row `set_row_data` — never a fresh ModelRc, which would rebuild all 40 for-items — and gate on hud-live + visible, so a FADED graph freezes and the idle-clean contract below survives a per-frame data source; sampling pauses under the menu hold, whose ~140 Hz `present_again` re-entry carries a STALE `last_ms`; all chrome is static, and the glow is a radial-gradient rect because the software renderer SILENTLY IGNORES drop-shadow-* and no-ops per-element rotation — hence L-bracket corner accents) + a keymap/controller panel, ALL activity-gated: the keymap fades IN while the camera moves and out ~2.5 s after it stops (help appears exactly when the pilot wants it), and the compass+clock+mode+graph wake on camera OR time-of-day activity (a scrub, the world attractors, the menu's TOD row) with their own ~4 s linger — longer, because heading/hour are what you glance at just after stopping — plus once at boot so the HUD announces itself; an idle screen is clean, and a faded HUD costs zero repaints and zero uploads (nothing updates its properties while both signals are idle; measured post-graph: live = one ~35-60 KB graph-region upload per 125 ms tick, then the 400 ms fade's full-element frames, then TOTAL silence) — and an **ESC pause menu** (Resume / Settings / Exit) whose Settings pages expose every CLI-exposable option. Both are rendered by **Slint's SOFTWARE renderer** (`src/hud/` — a custom `slint::platform::Platform` + `MinimalSoftwareWindow`; no winit/skia backends compile, the slint dep is `default-features = false` + `renderer-software` + `software-renderer-systemfonts`, UI authored in the `slint!` macro so build.rs is untouched; Royalty-Free license) into a persistent premultiplied-RGBA8 CPU buffer, and composited by **one alpha-blended fullscreen draw inside `fullscreen_to_backbuffer`** (`src/gpu/hud.rs`, SRV slot 9, `hud.hlsl`) — the single funnel every present arm passes through, so the HUD works in EVERY mode (CPU/GPU/DXR × every upscaler), the first overlay that does (the O quadtree overlay stays CPU-resolve-only).

**TRUE dirty rectangles, three layers deep** — the design constraint, verified by `FRUSTRACER_HUD_STATS=1` (one line + a `hud-buffer-dump.png` ground-truth per non-empty upload): Slint's `RepaintBufferType::ReusedBuffer` re-rasterizes ONLY the dirty region and `draw_if_needed` returns false on a clean frame (zero raster); `hud::Hud::frame` packs only the reported rects' bytes (inputs quantized — whole compass degrees, whole clock minutes, the SPACE/F-only mode string, 125 ms FPS buckets whose Slint writes additionally gate on hud-live — so a still frame dirties nothing); `HudGpu::record_upload` memcpys only those bytes into the upload ring and records ONE `CopyTextureRegion` per rect (non-zero DstX/DstY + source `D3D12_BOX`; every rect shares the frame's full-window placed footprint at a 512-aligned slice offset, so per-rect copies need no offset alignment). Measured: first frame 8.3 MB (full window, forced — the texture starts undefined; same on resize), a keymap fade = its own 660×76 rect (200 KB) per animation frame, a settled HUD = **0 rects, 0 bytes** (125 dirty frames out of ~3500 in a flight test). Blend space: SDR and Sdr10 backbuffers = display space, texel passes through; HDR10 = the PS un-premultiplies, decodes 2.2, PQ-encodes, re-premultiplies — the display-space-blend compromise at PQ's own encoding.

**ESC pause menu.** ESC opens (ESC in settings backs out; on the main page it closes); while open: the **flycam thread pauses** (it reads raw OS key state, so typing in a text field would otherwise fly the camera — the session-rebuild pause gate reused; paused ticks keep the dt clock so closing never teleports), SDL events route to Slint instead of the toggle handler (`input::Input::poll(menu)` — `hud/events.rs` translates pointer/wheel/TextInput/nav keys; toggle keys structurally can't fire; quit/resize/display/F11 keep their edges), SDL text input runs for the menu's text fields, and the frame loop **skips tracing entirely**: `GpuContext::present_again()` re-presents the last frame + overlay (the `present_hold` contract generalized — every tonemap source rests in PIXEL_SHADER_RESOURCE; `last_present` records what to re-run) at ~140 Hz, no history advances, no accumulation, so closing the menu needs ZERO resets. A live menu edit falls through to ONE normal frame so its key handler runs and the user sees the change behind the menu; a TOD set likewise (`FlyCam::set_tod` write-through → the existing `sun_moved` path; it also takes the clock from the world-mode attractors, like a manual scrub).

**Settings rows are a declarative table** (`settings::menu_items()` — id/label/group/tier/control + get/set accessors over the JSON schema; `GROUPS` = Display/Renderer/Upscaler/Effects/Scene/Advanced). **Live-tier rows apply through SYNTHESIZED `Edges` fields** — the exact key-handler code paths, so reset semantics cannot drift from the keys (mode=SPACE, spp=U, bounce=H, quality=1-3, G/X/K/N/M/J/R/T/O/B/V all ride their edges; `settings::MenuFx` is the mapping) — plus direct atomics for the keyless levers: bloom (display-stage, NO reset), clouds/fireflies/firefly-count (`frame = 0`, histories kept — the TOD-scrub precedent), TOD, HUD visibility. **Restart-tier rows** (chain wiring, lock-res, HDR/vsync, adapter, scene/world, BVH knobs, load-time levers, aniso, A/B levers, SDK paths — everything decided at GpuContext::new/scene load) edit the file only and badge "restart". **Every menu edit auto-saves; keyboard toggles deliberately never persist** (a key press is experimentation, a menu click is a preference).

**The settings file** (`frustracer-settings.json` next to the exe, `src/settings.rs`): all-`Option` sparse serde schema — `None` = never set, only deliberate choices serialize; enum-ish fields carry the CLI's own strings. Loads at the top of `main()` and applies BEFORE the arg parse, so precedence reads **compiled defaults < settings file < CLI flags** — and since the CLI moved to `src/cli.rs` that is a DATA FLOW rather than the ordering accident it used to be: `apply_to_opts` and the parse loop both write `Opts` FIELDS, `main`'s lever block is the single place any of them reaches a process global (see the CLI bullet in Architecture notes). Headless runs (`--check*`, `--*-dump`, `--spin*`) and `--no-settings` ignore the file with one loud line — the gates stay a pure function of the command line. Corrupt/BOM-prefixed files are a loud line + defaults, never a panic; invalid field values warn and fall back per-field; the file's scene choice applies only when the CLI named no scene source (never an exclusivity error against a flag). `settings::self_test` (in `--check`) pins the serde round-trip/sparseness/forward-compat, every enum vocabulary against its real consumer (`xess::lock_scale`, `bc7::Quality::parse`, the `parse_*` mirrors of the CLI arms), the menu descriptor's invariants (unique ids, known groups, cycle options its consumers accept, restart-Toggle round-trip), and the headless predicate.

Known-accepts: P screenshots and `--check` PNGs contain NO HUD (they read pre-composite sources — deliberate); the HUD has no MVs (upscalers see it post-upscale, so nothing to absorb); a menu-open hold re-presents one frame (converging stills freeze, like the existing holds); glow-through: `PrintWindow` captures of the flip-model swapchain can show a stale pre-menu buffer (capture quirk, not a compositing bug — the CPU buffer dump is ground truth). Structurally `--check`-safe: everything hangs off `GpuContext`/`Hud`, which only exist in the interactive window; headless paths never construct either. Touch `src/hud/`, `src/gpu/hud.rs`, `hud.hlsl`, `src/settings.rs`, `src/cli.rs`, `input.rs`, `flycam.rs`'s set_tod/manual_tod, or the menu block in `session()` → run `--check` (settings + cli self-tests + the untouched-gates proof), then the interactive smoke: HUD in every arm, ESC menu open/close with a held W (camera must stay frozen), a live row click (applies + `frustracer-settings.json` updates), a restart row (badge + file), relaunch picks the file up, `--spp 2` on the CLI overrides a file `spp`.

## HDR display output (ONE 10-bit swapchain — PQ or gamma by the display probe)

The renderer was always HDR internally and only SDR at the very last step: every upscaler output (DLSS-RR, both FSR flavors, XeSS), the GPU tracer's resolve target, the DXR output, and the CPU `HdrUpload` are linear `R16G16B16A16_FLOAT`, and all **16 present arms funnel through one function** (`GpuContext::fullscreen_to_backbuffer`) and **one shader** (`tonemap.hlsl`) — which then crushed that signal into an 8-bit `B8G8R8A8_UNORM` backbuffer. The 10-bit swapchain replaces that last step; nothing upstream changes.

**ONE format, `R10G10B10A2_UNORM` (4 B/px), two transfer curves** (2026-08-01 — the scRGB `R16G16B16A16_FLOAT` swapchain is DELETED; this used to read "PQ on HDR-on, scRGB f16 everywhere else"): declared `G2084_NONE_P2020` it is **HDR10/PQ** (the HDR-on-display default); left UNDECLARED it reads as gamma-2.2/`G22_NONE_P709` — DXGI's default interpretation of a UNORM chain — which is the **Sdr10** deep-colour arm, the HDR-off-display default. The reason is BYTES PER PRESENT: fp16 was 8 B/px, and the present is the entire frame budget whenever the display hangs off a **different GPU than the renderer** — DWM must then COPY every frame across, which at 7680×3969 is 244 MB at fp16 vs 122 MB at 10-bit. Measured on this box (world, 8K, display driven by an Intel B70 while a 4090 renders): **6.1 → 10.0 rendered fps, ~80 → ~51 ms per present** just from the format halving, while the GPU itself was doing only 14.7 ms of work per frame — the wire, not the renderer, was the ceiling. Caveat: the win is ~nothing when the display hangs off the RENDER GPU (that present is a flip, no copy). Quality is not traded away: 10-bit + gamma 2.2 is the classic deep-colour SDR pipeline — strictly finer steps than the 8-bit path everywhere — so it keeps the no-banding property scRGB used to buy on SDR panels, and PQ is what HDR titles ship. (PQ on an HDR-OFF display was rejected as the one-arm answer: DWM *can* composite PQ down but applies its own assumed-nit tone mapping — washed out and unpredictable — hence the gamma arm.) **`--no-hdr10` is Sdr10's explicit spelling** — on an HDR-on display PQ is the default, so the gamma arm needs its own flag or the arm the gates rest on would be unreachable there (`cli::self_test` pins exactly that, plus that each of the three still wins from any predecessor). `--no-hdr` forces the legacy 8-bit path (the A/B lever, and the FG wrap-failure fallback). A refused G2084 declare is a RELABEL, not a rebuild: the 10-bit buffer already exists and Sdr10 is its default reading, so the session just becomes Sdr10 with one loud line — 8-bit is reachable only via `--no-hdr` and a failed FG wrap. `--check`'s PNGs are unaffected — still written through `ToneParams::SDR`, and `check.png`/`check_gi.png` came through the scRGB deletion byte-identical.

**The curve** (`src/tone.rs` — the single source of truth, ported term-for-term into `tonemap.hlsl`). Everything is in units of **paper white** (`--hdr-paper-white`, default 200 nits — the scene is authored so linear 1.0 ≈ diffuse white; the exposure anchor is now the sun's own irradiance plus `sky::DOME_SCALE`, which `sky::self_test` pins into a physically sane band), with a knee `k` and headroom `w = peak_nits / paper_white`:

```
f(x) = x                                            for x <= k
f(x) = k + (w - k) * (1 - exp(-(x - k) / (w - k)))  for x >  k
```

Monotone, C¹ at the knee (slope exactly 1 on both sides — a break here rings around every highlight), and asymptotic to `w`, so **the curve is bounded by construction and can never exceed the display's peak**: a physical sun disc (radiance ~44,000) pins at the asymptote instead of blowing up. The load-bearing property is that at `k = 0, w = 1` it collapses to `1 - exp(-x)`, **the curve the renderer has always shipped** — SDR is not a separate path, it is the degenerate case. `tone::self_test` (in `--check`) gates that degeneracy **bit-for-bit**; it is the regression guard that `--hdr` did not move the default.

- **SDR** = `k=0, w=1`, then `^(1/2.2)`, 8-bit pack. Byte-identical to the pre-HDR build.
- **Sdr10** (`tone::ToneMode::Gamma22`, same as SDR) = the SAME curve and gamma at a 10-bit pack — `ToneParams::SDR` verbatim; only the wire width differs. The image matches the SDR one by construction (both packs quantize the same `shape()` output; `tone::self_test` pins that output into [0,1] — the packs' clamp-free precondition — and M12 gates the real 10-bit wire).
- **HDR10** (`tone::ToneMode::Pq`) = the `k=1` rolloff (everything up to paper white reproduced *exactly*, only highlights compressed), then `PQ(M_709→2020 · f(x) · paper_white/10000)` — the BT.2087 gamut matrix and the SMPTE ST 2084 inverse EOTF, both literal-mirrored in `tonemap.hlsl`/`hud.hlsl` (the clouds-wind idiom; `pq_encode(1.0) == 1.0` exactly because `c1+c2 == 1+c3`, a self_test anchor). `w` from the display probe.

**The three-way** (`d3d12::PresentSpace` — Sdr | Sdr10 | Hdr10, the one fact every encode site reads) also simplified the wrapper-FG story: XeSS-FG rejected scRGB fp16 outright (InitFromSwapChain INVALID_ARGUMENT, measured) but takes 10-bit — **verified generating at HDR10 on the B70** (gen result SUCCESS ×2 through the R10G10B10A2/G2084 proxy) — and since BOTH defaults are now R10G10B10A2 the old wrapper-forces-PQ-or-8-bit special case is GONE (`want == without_fg` always; the Sdr10 wrap on the B70 is the one arm not yet smoke-verified). `--hdr10` forces PQ in any session (override-wins like `--hdr-peak`, including over an "HDR off" probe verdict); on an HDR-off display `tone_pq` degenerates to the SDR rolloff PQ-encoded at paper white (`display::DisplayHdr::tone_pq`). The failure ladder lives in `D3d::with_queue` and never presents mis-declared: an FG wrap that rejects the 10-bit chain (Sdr10 and Hdr10 share the format, so there is no intermediate rung) → loud rebuild at 8-bit + wrap AGAIN (FG is why the session exists); a wrap that succeeds but whose G2084 re-declare through the proxy FAILS (only reachable at Hdr10 — Sdr10 declares nothing) → relabel the session Sdr10 and KEEP the proxy: the proxy's fresh chain sits at its default gamma reading, so no rebuild, no unwind (the old `FgHook::unwind` teardown path is deleted with scRGB). Raw-NGX DLSS-G polices nothing (no swapchain hook — its fp16 is internal NGX textures).

**The display is a fact, not a guess** (`src/gpu/display.rs`): `MonitorFromWindow` on the HWND we already own → matched against `IDXGIOutput6::GetDesc1()`'s `Monitor` → `ColorSpace` (G2084 ⇒ Windows HDR is on for that output), `MaxLuminance`, `MaxFullFrameLuminance`. Deliberately **not** `GetContainingOutput`: an FG-family session's swapchain is a wrapper proxy (ffx FI / XeSS-FG), and whether a proxy forwards that call is not worth depending on. The rolloff aims at `MaxLuminance` (a small-window highlight peak — exactly the regime the values up there live in); aiming at `MaxFullFrameLuminance` would throw away most of the panel's headroom for precisely this content. Measured here: 427 nits peak / 254 full-frame ⇒ 2.1× headroom at 200-nit paper white (**lower** `--hdr-paper-white` to buy more).

**A display change is a retune, not a rebuild — and only the PQ arm has anything to retune.** The gamma arms' curve is the static `ToneParams::SDR` (no peak, no paper-white), so `GpuContext::refresh_display` early-returns unless the session is Hdr10; there, a monitor move is a **field write of `self.tone`** (read by `fullscreen_to_backbuffer`, which is why none of the 16 arms have a tone parameter): no `ResizeBuffers`, no PSO rebuild, no resource realloc, and — deliberately — **no upscaler-history reset** (a change of output device is not a change of scene, the same reason camera motion never resets it). Re-probed on SDL's `DisplayChanged`, on `Moved` (a window can straddle two monitors and change owner without `DisplayChanged` firing), on the resize/F11 seam, and on a **1 Hz poll** — the poll is the *only* thing that catches the user toggling Windows HDR on the monitor the window is already sitting on, because no window event fires for that at all. The session's SPACE stays what boot negotiated (a PQ session dragged to an SDR monitor keeps presenting PQ through DWM's conversion); a live colorspace re-declare on display moves is the noted follow-on — cheap now that both 10-bit spaces share one format, so it would be `SetColorSpace1` + a `ToneMode` flip, no swapchain work.

**The CPU-presented arms** (OIDN, NPPD, XeSS-post, plain resolve) tonemap on the CPU into `CpuPresent` (main.rs), which holds two wires and picks one **from the swapchain, not the CLI flag** (`GpuContext::encoding()` — the G2084 declare can be refused, in which case the session relabels Sdr10). SDR fills `u32 0x00RRGGBB`; the 10-bit sessions fill `wire10` — packed 10-bit u32 (`r | g<<10 | b<<20 | 3<<30` — **R in the LOW bits**, R10G10B10A2's lane order, opposite BGRA8's) via `render::present_px_sdr10` (gamma) or `present_px_pq` (matrix + ST 2084). The blit PS stays a **passthrough** in every case — the CPU owns the encode on this path. `BlitUpload` therefore has its own `BLIT_FORMAT`/`BLIT_FORMAT_10BIT` consts and must **never** follow `SWAPCHAIN_FORMAT`: each pack's byte order is only valid for its own format, and a shared const would silently reinterpret those bytes.

The **O overlay's tints are display-space [0,1] colours**, authored to sit on a gamma-encoded image, so `render::overlay_px` is composited in that space by every path and nowhere else. The SDR AND Sdr10 paths get that for free (`shape` under Gamma22 already applied the gamma — `present_px_sdr10` composites DIRECTLY; copying the PQ round-trip there would double-encode); the PQ path is still *light* after `shape` (its encode lives in `tone::encode`), so `present_px_pq` pays an explicit `pow(1/2.2)` → composite → `pow(2.2)` round-trip, **only when the overlay is on**, so the normal path is never perturbed by a pow/pow⁻¹ pair. Compositing in linear instead would tint highlights in proportion to their magnitude rather than uniformly, and the overlay would not match the SDR build. The HUD composite makes the same split in `hud.hlsl` (mode 1 = display-space passthrough — SDR and Sdr10 alike, 2 = decode → PQ-encode → blend in PQ space; mode 0 was the scRGB linear blend, retired).

Screenshots (P) and `--check` PNGs stay **SDR 8-bit** regardless of the session — a PNG has nowhere to put a nit. `read_hdr_output` and the screenshot arm both call `tone::map(ToneParams::SDR)`, which killed the third open-coded copy of the curve. Under the 10-bit wires the CPU arms hold a pack the PNG can't carry, so P **re-resolves each arm's own linear source** through the SDR curve — and must re-resolve *the source that was actually presented*: XeSS + OIDN **post** presents the DENOISED window-res image, not the raw `xess_hdr` upscale (only the post placement reaches that arm at all — the other placements take the GPU-readback path), and the overlay rides along exactly as it does on screen, because an SDR session saves its present buffer verbatim with the overlay in it and the two sessions must agree about what P captures.

**Gates.** `--check` runs `tone::self_test` (the SDR degeneracy bit-for-bit — the `--no-hdr` path and every PNG still reproduce the pre-HDR curve exactly; the Sdr10 range pin — `shape(_, SDR)` lands in [0,1], the clamp-free precondition of both integer packs, which is the testable half of the deep-colour claim (the refines-never-diverges half is structural: one `shape()` output, two step widths); plus below-knee exactness, monotonicity, the headroom asymptote out to radiance 1e6, and C¹ at the knee, all on the `hdr10` parameterisation now that it is the one wide constructor). `--check-gpu` **M12** is the HLSL twin gate — the REAL pixel shader through the REAL PSO and SRV slot over a synthetic geometric ramp **pinned to end at 6e4** (a physical sun disc is ~44,000, so the tail is the point; an anti-vacuity assert fails the gate if the ramp stops short — an earlier growth-factor formulation silently topped out at ~215, covering none of the regime the gate exists for, and 6e4 is the largest round number under `f16::MAX` = 65504, which the RGBA16F source texture must hold), compared against `tone::map` on **all three wires** (SDR ≤ 1 UNORM LSB, measured 2.2e-3 of its 3.9e-3 budget; Sdr10 ≤ 1 ten-bit LSB + 1e-4, measured 5.4e-4 — the Gamma22 curve through a real 10-bit RTV, the genuinely new wire; HDR10 ≤ 2.5e-3, measured 5.5e-4 — never widen past 5e-3 without investigating). `tone::self_test` also carries the PQ block (signal-1.0 bitwise, the 100-nit ≈ 0.508 anchor, monotone + numeric EOTF round-trip, matrix white, the hdr10 paper-white anchor and no-headroom degeneracy). The oracle is fed the **f16-rounded** ramp, not the exact f32: the shader reads a real RGBA16F texture, and charging the port for the wire's rounding (a step of 32 radiance at the top) would gate the upload, not the curve. Run `--check` + `--check-gpu` after touching `tone.rs`, `tonemap.hlsl`, `gpu/display.rs`, the swapchain format in `d3d12.rs`, or the resolve/present chain.

## The upscaler chain (always-on temporal upscaling)

**Temporal upscaling is always on.** Every session probes the chain **DLSS-RR → FSR4-RR → XeSS → FSR3** in that fixed order and wires the FIRST level whose support probe passes; exactly one upscaler is wired per session (`GpuContext::wired()` derives it from the live state, so it can never disagree with the contexts actually held). Reaching the end of the chain is a LOUD line + plain presentation, the same shape as any other unsupported-feature fallback — the only quiet plain path is the explicit `--no-upscale`.

**The rung BELOW the whole ladder is the present itself, and it degrades too** (`present_or_shed!` in `session()`). Every level above sheds loudly and keeps rendering — RR/FSR/XeSS fall to plain, DXR falls to the CPU tracer, NPPD/OIDN switch themselves off — and each of those landings ends at one of five present sites that used to be `.expect("GPU present failed")`, i.e. a panic at the one rung with nothing beneath it. That is not hypothetical: an 8K resize wedged Present at `E_ABORT`, the session correctly shed RR and then DXR, arrived at the plain CPU present, and killed the process with every other fallback already spent — and the panic then unwound through Streamline's teardown and took a SECOND access violation on the way out, so the crash the user saw had nothing to do with the cause. A failed present now costs THAT FRAME only (nothing is advanced by presenting); `PRESENT_FAIL_LIMIT = 120` consecutive failures end the session as a clean `SessionEnd::Quit`, never a panic, and the loud line names which arm failed (`xess`/`oidn`/`nppd`/`gpu-tonemap`/`plain`).

The intent half is pure data (`src/upchain.rs::UpChain`, gated DLL-free by `upchain::self_test` in `--check` — availability is *injected*, never probed there). Flag algebra: `--<x>` **forces** the chain to start at level x (every level above is disabled; the levels below stay as fall-through), `--no-<x>` **skips** that one level (`--no-fsr` skips both FSR levels), `--no-upscale` is the empty chain. Later flags win, matching left-to-right parse order — so `--xess --fsr` resolves FSR4 (with XeSS still below it), and `--fsr --no-fsr` resolves XeSS. The one flag outside this algebra is **`--fsr4`**: it forces level 2 like `--fsr`, but also *requires* it — a fall-through exits 2 instead of resolving the next level (see the FSR section).

DLSS is the top of the chain by POLICY only (the Streamline retirement killed the old STRUCTURAL reason — slInit-before-any-DXGI-factory is gone; the raw-NGX probe runs on the ordinary device like every native level's, and every session is all-native: one device, one queue, no proxies). The native levels (FSR4-RR, XeSS, FSR3) are then first-hit-wins on the real device. The probe lives in `GpuContext::new`; `resize_output` rebuilds **what was wired** and never re-probes, so a resize can't switch upscalers mid-session.

The one-upscaler rule is a POLICY, not a hardware constraint, and `--quinlight` is where it is deliberately suspended (see the registered-consensus section). It was long asserted here that a live DLSS level means no native upscaler can coexist; that is FALSE, and since the SL retirement trivially so — every context (the raw-NGX RR evaluate, XeSS, both ffx flavors) lives on the one native device and records into the one native command list. A `--quinlight` session on the dev 4090 wires DLSS-RR + XeSS + FSR 3.1 simultaneously.

Every render mode consumes the same wiring: the `--dxr` default, the CPU-renderer fallback, and `--gpu` all feed whichever level came up (the GPU tracers via `FeedKind`; the CPU renderer via the `record_upload` paths). In-session, G/X/K toggle the wired upscaler against plain — you cannot toggle *into* an upscaler the session didn't wire, and the keys say so.

## Real scenes (textures + alpha cutout + classified PBR materials)

Game-like benchmark scenes come from the McGuire Computer Graphics Archive (https://casual-effects.com/data/ — OBJ+MTL+PNG, the format `tobj` parses) into `scenes/`, committed via **git LFS** (run `git lfs install` once per clone or checkouts materialize pointer files) with the meshes stored as zstd-compressed text (`.obj.zst`, ~6× — see .gitattributes; only the original download archives stay untracked). The loader decodes `.obj.zst` transparently and a bare `model.obj` argument falls back to its `.zst` sibling, so the commands above work verbatim on a fresh checkout; the standard scene here is **San Miguel** (`scenes\san-miguel\san-miguel.obj`, ~10M tris, heavy alpha-masked foliage; a `-low-poly` variant loads much faster for iteration). The OBJ loader keeps per-vertex UVs (`Scene::texcoords`, parallel to `positions`, zeros where absent — sound because `single_index` gives one unified index stream) and decodes each `map_Kd` once (deduped, rayon-parallel with **largest-first LPT scheduling** — WebP decodes slower per file than PNG, so the big-file tail would otherwise dominate load time; all three decode sites sort by file size, the two order-sensitive ones through an index permutation so texture ids never shift — `src/texture.rs`: RGBA8 storage, bilinear repeat-wrap sampling through an sRGB→linear LUT, V flipped once at load). **Committed textures are LOSSLESS WebP** (`scene::resolve_texture_path`: a manifest-referenced `foo.png` that is absent on disk resolves to its `foo.webp` sibling — the texture flavor of the `.obj.zst` convention, ~30% under PNG, decoded RGBA bit-identical; convert with Pillow `lossless=True, exact=True` — **`exact` is mandatory**: without it libwebp zeroes RGB under A==0 texels and `sample_bilinear` blends those at cutout edges; the fallback is pinned by a `gltf_loader::self_test` case and JPGs stay JPG — lossless re-encode of JPEG grows). A texture **replaces** Kd (exporters set Kd = 1 alongside map_Kd; multiplying would double-darken); Kd stays as the flat untextured fallback. `MatKind::Textured` is dispatched at the same shade.rs albedo match as Marble, so `PrimarySurface`/all upscaler guides pick textured albedo up automatically. **The GPU paths sample textures too** (`--gpu` AND `--dxr`): `SceneGpu` uploads texcoords + one `R8G8B8A8_UNORM_SRGB` Texture2D per scene texture (the FULL CPU-generated mip chain — CPU-trilinear parity; see the Mip-mapping section) into an SRV table in register **space1** (`RP_SCENE_TEX`, appended after the NPPD params, 62/64 root-signature DWORDs, two static wrap samplers — trilinear `samp_lin` + anisotropic `samp_aniso`, see Mip-mapping; static samplers cost 0 DWORDs; the descriptors extend the tracer's own shader-visible heap at `TEX_HEAP_BASE` — only one CBV_SRV_UAV heap is bindable), and `shade.hlsli`'s albedo match gains the `MAT_TEXTURED` arm, so the G-buffer pack / every upscaler guide / NPPD's albedo plane get textured albedo automatically (gated: the `albedo A/B` rows in `--check-gpu`/`--check-dxr` compare the pack's albedo plane against a CPU `GBufs` render — mean |Δ| ≤ 0.02/channel + a > 64-distinct-values must-fire on textured scenes). **Alpha cutout** lives in `bvh.rs::moller_trumbore` — the single choke point every CPU ray funnels through (hybrid, verify reference, shadow, AO, hemi), so the exact-zero gates stay like-for-like — gated on `Scene::any_alpha` (false on procedural/stress scenes: the hot loop is untouched there and `--check` output stays bit-identical). On the GPU the same test (`trace_common.hlsli::alpha_cutout`, an ALU mirror of `alpha_nearest` over `.Load` — deliberately not a sampler, so both GPU intersectors agree bit-for-bit) compiles in per scene via a `#define ALPHA_CUTOUT` prepended to the kernel concats when `any_alpha`, with the BLAS geometry flag switching OPAQUE → NONE: rt.hlsli's RayQuery wrappers grow candidate loops and the DXR pipeline gains three any-hit shaders plus a real any-hit-only `HgOcclude` hit group replacing the null occlusion record — opaque scenes compile the FORCE_OPAQUE originals verbatim and keep the null record, so procedural/stress sessions stay structurally untouched. Cutout agreement is near-exact (measured San Miguel: T1 class-mismatch 0 vs the CPU on both pipelines); `--check-gpu` prints `alpha-cutout rejections` and must-fires it on alpha-masked scenes (caveat: a `--cam` pose with no masked geometry in view trips it — the canopy-caveat class). Soundness: rejection only *removes* hits, so the true nearest hit only moves farther — frustum bounds built from solid AABBs stay conservative lower bounds, inherited tmin never overshoots, and hemi cells become at most less provably-empty. MTL `d`/`Tr`/`map_d` are ignored (San Miguel's masks are in the map_Kd alpha).

**Classified PBR materials** (`src/matclass.rs`, load-time only): OBJ materials get per-class roughness/metallic + the lobe fields below instead of the old flat 0.8/0.0. San Miguel's `newmtl` names are anonymous and its Ks/Ns exporter-flattened, so `classify` keys on the **map_Kd filename stem** (whole-token matching over an ordered first-match table — rust > metal > clay > stone, foliage last; Spanish vocabulary + the `lef/pet/stm/bark` plant-library tokens), then the material name (catches the untextured `CafeChair_Metal`), then an Ns/illum tier (illum 4, Ns ≥ 500, or a whole-token `glass`/`pane`/`portal` NAME = glassware → `transmission` + an albedo lift toward white — transmitted light tints by albedo and the MTL's dark glass Kd would render black; the name admission exists because Minecraft/bistro exporters write windows as illum 2 / low-Ns, which no Ns/illum signal can ever admit — rungholt's `Glass` and bistro's `MASTER_Glass_*` rendered opaque matte; 100 ≤ Ns < 500 = opaque glossy), then the old default. The loader prints one `obj materials:` count line per class — the tuning signal; `matclass::self_test` (in `--check`) pins the precedence/token cases. Three **new BRDF lobes** ride `Material` fields that default to 0.0 (the structural guarantee procedural/stress scenes shade bit-identically): `sheen` (Charlie NDF + Ashikhmin visibility in the direct loop, kd scaled by 1−0.157·sheen, zero rng draws), `translucency` (thin-surface diffuse transmission in the `ndl <= 0` arm — the back ray starts at `hit − n·eps`, the mirror of the front convention, via plain `occluded`: no cut exists for that apex; under VisCtl::Apply the rep's bit is reused — segment occlusion is normal-independent within 2·eps), and `transmission` (a Snell-refracted continuation chain at `GLASS_IOR` 1.5, ≤ `TRANS_MAX_DEPTH` interfaces, exact dielectric Fresnel — Schlick pops at the TIR handoff — TIR continues as an internal mirror bounce; **shading-only**: glass still HITS, so frustum bounds/inherited tmin/temporal claims are untouched — a visibility-level design would false-sky structurally; shadow/AO rays pass through glass with a tint — see the tinted-shadows paragraph below). All three draw **zero rng** (the su/sv draws already precede the ndl test), which is what keeps replay/same-seed bit-identity and the VisCtl burn accounting intact. ALL THREE lobes are ported to `shade_split` (wavefront, GPU upscaler chains, AND the DXR pipeline — same pasted source). Transmission on the GPU is a **flattened DFS**, not recursion: the CPU tree has reflection at depth 0 only and transmission at depth < 4, so only the root has two children — one stash slot and a ≤ 9-lap loop reproduce the CPU's exact DFS order (reflection subtree, then the root's transmission chain), and on DXR every continuation routes to HgHit (bare hit record) under `--dxr-inline 0`, so **`MaxTraceRecursionDepth` never exceeds 2** (the old "couldn't host the chain" claim assumed payload recursion; at the `--dxr-inline 1` default the continuations are inline RayQueries and the declared depth is 1). With transmission = 0 everywhere the loop degenerates to the old two laps and `kt = 1 - transmission` multiplies as exact 1.0 — the same-seed wavefront-vs-reference bit gate held through the restructure unchanged. Measured on a San Miguel glassware close-up (`--cam 0.71,1.55,0.45,0.71,1.25,-0.35`, low-poly, 5199 glass px): GPU-vs-CPU radiance mean rel 170% before the port, 0.042% after. Touch matclass.rs/the lobes → run `--check`, `--check-xess` (the adaptive same-seed gates are the rng-alignment proof), `san-miguel.obj --check`, and `--check-gpu` + `--check-dxr` for the HLSL side, plus `san-miguel[-low-poly].obj --check-gpu`/`--check-dxr` (the textured/cutout/transmission proof — T1 class-mismatch 0, T2 ≤ 2%, the albedo A/B, the alpha-reject must-fire). Known gate caveat (pre-existing, measured identical before the GPU material work): `mv_selftest` inside `--check-gpu`/`--check-dxr` is pose-sensitive — close-up interior `--cam` poses put geometry ~10× nearer than the default pose while the MV-check dolly stays 0.02·diag, and the fixed pixel-error limits trip (default OBJ poses pass); same class as the canopy caveat below. Known gate caveat: the **unpaired** hemi-GI A/B is pose-marginal under dense canopy, INDEPENDENT of the lobes — a classic interior view of San Miguel's ficus (`--cam 0.5,0.4,0.1,-1.5,0.35,0.1`, low-poly) measures mean rel 0.0715 PRE-feature vs 0.0731 post (limit 0.05, signed −0.003 fine both ways), so custom canopy `--cam ... --check` runs could already fail before the material work. The signed half is additionally firefly-skewed (relative error bounded −1 below, unbounded above): a synthetic straight-down canopy pose tripped it on a single +14.8× probe post-feature (+0.0129 vs +0.0011 pre) while `hemi.rs:770` provably shades both arms through the same `shade`/`BOUNCE_Q`. ALL exact-zero gates stay 0 at every measured pose — they are the authoritative soundness signal; the mandated suite (default pose, --stress, OBJ default poses) passes throughout. The firefly-robust fix is now IN: the unpaired GI A/B trims 2% from each tail before taking its means (the limits are unchanged — never widen them; `worst raw` still prints). Real emissive content is what forced it: DamagedHelmet's emissive visor put a single +8× relative outlier in the probe set (signed +0.027 raw vs the ±0.01 limit) while every exact-zero gate stayed 0 — the same class as the synthetic canopy probe, now reproducible with a stock Khronos asset.

**Tinted shadows** (default ON, `--no-tinted-shadows` kills — the fountain-water fix): San Miguel's water is correctly classified glassware (illum 4 → transmission 0.9), but under the old "glass is opaque to shadow/AO rays" rule the pool hard-shadowed its own basin into blackness, and a black base under a roughness-0.05 GGX lobe reads as liquid chrome. Now every LIGHT occlusion ray — sun shadow, translucency back ray, firefly shadow, sampled AO, hemi-AO leaf — runs **`Bvh::transmittance[_multi]`** instead of `occluded`: RGB throughput ONE when clear, ZERO at any opaque hit, `×= Material::shadow_tint()` (= `transmission × albedo` — the ONE tint source; the load-time albedo lift composes) per transmissive interface crossed, floored at `SHADOW_TP_MIN = 1e-3`. **The two-query split is the design spine**: `occluded` KEEPS binary any-geometry semantics and still serves every GEOMETRIC oracle (hemi `check_empty`/false-empty references, relief self-tests, the GPU `check_empty_cell`) — a glass-containing cell must never verify "empty". Visibility is untouched (glass still HITS — primary/reflection/glass-chain rays unchanged, all frustum/tmin/temporal soundness carries over), and **zero rng draws** anywhere, so every same-seed/replay/VisCtl-burn contract holds. `VisRecord` stores per-sample **throughputs, not bits** (`vis: [Vec3A]`; uniform = all bit-equal — a consistently tinted cell stays COARSE, mixed values still declassify; below-horizon stores ZERO + the poison); AO folds RGB to gray by mean-of-components with a TRUE divide (`3.0/3.0 == 1.0` exactly — opaque scenes keep the old integer counts bit-identically, and `x·1.0` keeps the direct loops bitwise). Structural off-state: `Scene::any_transmissive` (derived in `finalize_scalars`, lever folded in) gates everything — procedural/stress scenes and lever-off sessions run the `occluded` arm verbatim, no CACHE_VERSION bump (all derived at load). GPU twin: `transmit_q` in rt.hlsli (candidate loop; transmissive candidates multiply and are NOT committed — ACCEPT_FIRST_HIT only ends on opaque commits) and rt_dxr.hlsli (`ShadowPayload` grew `float3 tint`; `ah_shadow` multiplies + `IgnoreHit()` — payload writes persist, the standard pattern; the `SHADOW_RF` macro drops FORCE_OPAQUE on occlusion rays only, closest rays keep `OPAQUE_RF`), fed by the per-material `mat_shadow` buffer (`float4(tint, transmission)`, t6 space1 — `TEX_TABLE_BUFS` 6→7 moved `texs[]` to t7), compiled per scene by `#define TRANS_SHADOW` (`trace::trans_defs`, the alpha_defs pattern). **The BLAS gains `NO_DUPLICATE_ANYHIT_INVOCATION` when armed** — D3D12 may legally surface one triangle twice to any-hit/candidate code, and a tint MULTIPLY (unlike the idempotent cutout reject) must not run twice; do not remove the flag as a "perf cleanup". Gates: `bvh::tinted_shadow_self_test` in `--check` (single/double-interface tint bitwise, opaque termination, the floor, the primary-visibility pin, `occluded`'s binary contract, the cut-seeded twin, the lever-off block); `--check-gpu` must-fires `CTR_TRANS_PASS` on transmissive scenes and gates it 0/compiled-out elsewhere (the alpha-rej pattern, same `--cam` caveat class); the hemi-AO A/B references and the estimator switched to `transmittance` in LOCKSTEP (the cut-miss oracle compares tinted throughputs with a 1-ulp-scale slack — ≥3-interface products may associate differently between cut and root traversals; binary stays exact). Known-accepts: shadow rays travel straight (no refraction bending, no caustics — a water shadow is a uniform tint); textured transmissive materials tint by their FLAT albedo (no UV fetch in the occlusion loop); hemi-GI bounce rays still hit and shade glass (GI sees the surface, not through it); glassware casts ~45%-deep tinted shadows instead of hard ones. Touch `transmittance`/`shadow_tint`/`transmit_q`/`ah_shadow`/`mat_shadow` → run `--check`, `--check --stress`, `--check-xess`, `san-miguel[-low-poly].obj --check`, `--check-gpu` + `--check-dxr` (+ the san-miguel flavors). **Part 2 — the standard game-water model, completed on real rays** (Fresnel split and true refraction already existed; these add the depth fog and the aerated spray): (1) **spray reclassification** (`scene::reclassify_spray`, default ON, `--no-spray`): tinted shadows made the airborne fountain droplets correctly clear — and therefore invisible (their old look was "dark bead against lit surround": the chain used to end in the self-shadowed black basin). Real spray reads white because it is AERATED, so a load-time union-find over shared vertex POSITIONS (exact f32 bits — welding by INDEX shipped first and retagged rungholt's entire per-block-unwelded ocean as ~150k one-block "droplets"; grid exporters repeat the same coordinate per shared corner, so bit-keying welds the sea back into one over-limit component, and a missed weld only degrades toward the index status quo; transmissive tris only, the `derive_heights` cold-load slot — warm loads inherit the retag from the .fcache, `CACHE_VERSION` 8/16, the lever keys the cache lever word) retags components under `SPRAY_MAX_K·diag` (6e-4 ≈ 4 cm on SM; droplets ~3e-4·diag, smallest glassware ~1.5e-3·diag — the load-time histogram line is the tuning signal) to a per-source-material deduped SPRAY clone: albedo lerped 0.6 toward white, transmission 0, translucency 0.35, roughness 0.4. Pools/streams/glassware stay glass; scenes with no transmissive geometry are structurally untouched; `scene::spray_self_test` gates islands/dedup/overrides/lever-off in `--check`. (2) **Beer–Lambert depth tint** (`shade::depth_attenuation` + the shade.hlsli twin, default ON, `--no-depth-tint` = `FLAG_DEPTH_TINT` 2048 on the GPU): the chain's per-interface tint carries no path length (1 mm of droplet attenuated like 2 m of pool), so each INTERIOR segment (entering or TIR — a clean exit travels outside) now also attenuates by `albedo^(d/(TRANS_DEPTH_K·diag))`, `TRANS_DEPTH_K = 0.015` ≈ 1 m: exactly albedo-tinted at the reference depth, clearer above, darker below. The CPU multiplies the child's returned radiance, the GPU folds it into the child's throughput (the child hit is traced in the same DFS lap — same product, different fp association, absorbed by the statistical CPU↔GPU gates); the per-interface tint STAYS (thin glassware keeps its look — dimming, never gaining); the interior-ray-to-sky miss stays unattenuated (leaked geometry, status-quo shape); zero rng draws. `shade::depth_tint_self_test` pins the closed-form anchors (seg 0 ⇒ exactly ONE, seg D_ref ⇒ exactly albedo, monotone, white passthrough). Touch `reclassify_spray`/`depth_attenuation`/the chain's interior logic → the same run list as above, plus a COLD San Miguel load (the CACHE_VERSION bump). **Part 3 — the WATER CLASS + ripples, the actual de-chroming** (parts 1-2 gave water depth-fog and spray but the SURFACE still read as liquid chrome: mirror-flat, and — because the loader lifts every transmissive Kd toward neutral white for glassware — colorless, so a smooth plane mirroring the sky IS polished metal). The fix is material parameters, not the BRDF (which is correct — exact Fresnel, F0 0.04). A **water refinement of the glass tier** (`matclass::WATER`, `Pbr::water`): only a material that already classifies glassware (`illum 4 || Ns ≥ 500 ||` a `glass`-token name) can become water, keyed by an OBJ `water`/`agua` object/group name (`o Water` → `materialo` in San Miguel; NOT `o Fountain`, the basin) OR a chromatic MTL `Tf` (`tf_chromatic`, `max−min > 0.05` — `materialo`'s `Tf 0.5 0.4 0.2` fires, neutral glassware stays glass; the Tf color itself is untrusted, only its chromaticity — which is why a whole-token `glass`/`pane`/`portal` NAME vetoes the Tf cue: vokselia's `Glass`/`Glass_Pane`/`Portal` are illum 4 with chromatic Tf and must be glass, not rippling water; the name-based water signals stay unconditional). `classify` gained `water_hint`/`tf` args (the loader scans tobj `models` names + parses `Tf` from `unknown_param`, the `norm`/`Pr` precedent) AND a `water_on` arg — `--no-water` gates ALL FOUR signals through it (this text used to claim `false, None` at the call site disarmed "the whole path"; that gated only hint+Tf, and name/stem-classified Minecraft water — rungholt's `Stationary_Water`, vokselia's `Water` — ignored the lever entirely until 2026-07-31; `matclass::self_test` now pins the lever-off arm on exactly that shape) and keys `scene_cache::lever_word` bit 4. Three sentinel-gated `Material` fields (the `NO_TEX`/`height_amp` idiom — non-water shades bit-identically): **`trans_tint`** (`splat(-1.0)` sentinel = "use albedo"; `Material::trans_tint_or(albedo)` returns albedo VERBATIM for every existing material — the ONE tint source for the per-interface glass tint, Beer–Lambert, and `shadow_tint`, all switched in lockstep CPU+HLSL; water = `WATER_TINT (0.75, 0.92, 0.96)` light blue-green so the Beer–Lambert exponent does the depth work, red extinguishing fastest), **`ior`** (default 1.5 == the old fixed `GLASS_IOR`, `1.0/1.5f32` bit-identical; water `WATER_IOR = 1.33`; the const `GLASS_IOR` is retired for `mat.ior` in both `shade.rs` and `shade.hlsli`), **`ripple_amp`** (0.0 = off, water `WATER_RIPPLE_AMP = 0.25`). Water is EXEMPT from the dark-glass albedo lift (`scene.rs`: `transmission > 0 && !pbr.water`) so its raw dark Kd stays, killing the `kd·(1−T)` neutral wash. **Ripples** (`shade::ripple_height`/`ripple_grad` + `ripple_normal`; the GPU twin is its OWN file, `shaders/ripple.hlsli`, pasted ahead of shade.hlsli at all four concat sites AND into the fxc FG guide kernel — three consumers, one copy): ONE domain-warped directional swell + 3 octaves of scrolling gradient-of-value-noise chop (`clouds::vnoise_vg`, fresh octave ids 16-19, so it inherits the AVX2 corner path and stays u32-exact). It replaced 3 fixed sinusoids, which beat against each other on a fixed lattice — the interference REPEATED and a large expanse read as TILED. The field must stay the analytic GRADIENT OF A SCALAR HEIGHT (⇒ consistent virtual heightfield, no impossible-normal shimmer): **never reach for curl noise to de-tile it** — divergence-free is the exact opposite property. Four layers scroll at their own velocities (one velocity reads as a rigid sliding pattern); a sum of scalars is still a scalar, so integrability holds at every instant and the field stays closed-form in t, which is what FG round 4 needs to evaluate it at two times. Constants LITERALS so the HLSL twin is identical by construction (the clouds-wind precedent), lengths `Scene::diag`-relative, animated on the SHARED cloud clock (`cl.time`/`CLOUD_TIME`; `--check*` pins `CLOUD_CHECK_TIME`, `--spin` uses `idx·CLOUD_SPIN_DT`), ZERO rng. It perturbs the SHADING normal n_s only (composes ON the normal map: full-res water has none, low-poly's `water_bump.png` perturbs first and the ripple tilts on top — the n_g/n_s split holds, geometric n keeps eps offsets/hemi/back ray). The refraction chain ALSO perturbs its Snell axis (`n_snell = ripple_normal(n, n, …)` — bends the basin image, the dominant water cue) GUARDED: a refraction must cross the geometric surface (`tdir·n < 0`), a TIR mirror stay on the near side (`tdir·n > 0`); a ripple that flips the side recomputes on the geometric n (which provably passes both — coarser, never wrong), and the eps offsets stay on geometric n. `GpuMat`/HLSL `Mat` grew to **100 B** (append `trans_tint`/`ior`/`ripple_amp`; stride comment lockstep both files), `DiskMat` gained the three fields (`CACHE_VERSION` → 10). The GPU `mat_shadow`/rt.hlsli/rt_dxr.hlsli/dxr.hlsl need ZERO changes (`shadow_tint` is data-fed). Known-accepts: no caustics; ripple motion has no MVs in the RENDER G-buffers (shading change the upscalers absorb — the clouds precedent; slight RR ghost risk on glints) — **retired for raw-NGX FRAME GENERATION, which strobed on it: `ngxfg_guides` round 4 reconstructs the previous frame's mirror normal from the closed-form field and writes real ripple MVs into the FG-ONLY plane (see the `--fg` block); RR's own plane, ffx FI and XeSS-FG still see no ripple motion**; a converging still freezes the ripple phase; the refracted basin wobbles with zero geometry motion (and has no MV at all — face-on water deliberately keeps the near-zero surface MV). Gates: `matclass::self_test` (the water signals, the `material_79`/`materialn` neutral-glass pins, refinement-only, token safety), `shade::ripple_self_test` (off-state bit-identity, horizon guard, **INTEGRABILITY — central-difference `ripple_height` vs the analytic `ripple_grad`, the mechanized proof the sinusoids never had and the one property no image gate can see** — slope bound, determinism, and an APERIODICITY probe at 32 whole swell wavelengths, where the old plane-wave field repeated exactly), `bvh::tinted_shadow_self_test` (the trans_tint override), `scene::spray_self_test` (clone sentinel fields). Touch `matclass`/the water class/`trans_tint_or`/`ripple_normal`/the Snell IOR → run `--check`, `--check --stress`, `--check-xess`, `san-miguel[-low-poly].obj --check` COLD (the CACHE_VERSION bump) + WARM, `--check-gpu` + `--check-dxr` (+ the san-miguel flavors, the HLSL twin), `--check-fsr` (the shading rides the existing signal planes), and the three-pose screenshot check (looks are gate-blind — a fountain close-up `--cam` over the pool, default pose, low-poly, day + `--tod 17.5`, one `--spin`/interactive pass to see ripples animate).

**PBR material maps** (normal / roughness / metallic / emissive, MTL-fed): `Material` gained map-index fields defaulting to `NO_TEX` plus `emissive`/`normal_scale` — the structural guarantee that unmapped materials shade bit-identically (all map logic branches on the sentinel and draws ZERO rng; the same-seed gates rely on it). The loader reads `map_Bump/bump` (+`-bm s`) and `norm` as normal maps, `map_Pr`/`map_Pm` (+ the `Pr`/`Pm` scalars) and `Ke`/`map_Ke` from tobj's first-class fields / `unknown_param`; textures now carry a `srgb` role (color maps sRGB, normal/rough-metal LINEAR — a linear map must never go through the sRGB LUT, and `alpha_masked` is Kd-role-gated so a stray alpha channel can't arm the cutout pipeline; dedup key is `(path, srgb)`). Grayscale `map_Bump` files are HEIGHT maps (San Miguel carries exactly 1) — detected post-decode (`Texture::is_grayscale`) and **Sobel-CONVERTED into normal maps at load** (`Texture::height_to_normal`: 3×3 Sobel with WRAP addressing, `n = normalize(−K·∂h, 1)` at `K = HEIGHT_NORMAL_STRENGTH` = 2 texel-widths, green stored PRE-NEGATED so `NORMAL_MAP_Y_SIGN`'s decode lands the true normal — pinned end-to-end by `tangent_self_test`'s ramp cases — and the exact source height kept in ALPHA; `height-maps converted` in the `obj materials:` line, which also counts the map families; `--no-h2n` restores the old drop). The inverse also runs: **every real normal map gains a derived heightfield at load** (`Texture::apply_n2h` — the discrete Frankot–Chellappa least-squares integration by FFT over the wrap-repeat domain, where periodic BCs are exact; rustfft, pure Rust, `--check` stays DLL-free; the curl part of hand-authored maps is discarded by construction, `N2H_HIGHPASS` damps the ill-conditioned lowest bins, and the [0,1]-normalized height packs into the map's own alpha channel with the PEAK at 1.0 = the surface plane — the inward-only convention; amp in texel-widths → `Material::height_amp` × the `-bm`/glTF scale; `scene::derive_heights` is the shared OBJ+glTF post-load pass, `--no-n2h` kills it). Both conversions persist through the scene cache by per-texture flag bytes (`h2n`/`n2h` — the warm-load re-decode re-applies them; CACHE_VERSION 7, and the three levers ride a cache-key lever word so an A/B never serves a stale sidecar). **Height-carrying textures never BC7-compress** (`bc7::should_compress` — the opaque presets would flatten the alpha field; the predicate must stay equal to the `mat_height` fill in trace.rs, the `mat_cutout` agreement argument). Semantics: **factor × sample** (glTF's), roughness = `.g` / metallic = `.b` (glTF channels — grayscale MTL maps satisfy both via to_rgba8 replication); with a map present the flat factor is the MTL's own scalar (default 1.0), bypassing the matclass constant, which stays the no-map fallback. **The n_g/n_s split is load-bearing**: `shade()` computes the shading normal n_s (`perturb_normal` — tangents derived ON THE FLY from the triangle's positions+UVs, Gram-Schmidt vs n, handedness from the UV winding, zero storage; per-triangle tangents facet at UV seams, accepted) and feeds it to the BRDF frame / N·L / VNDF / `PrimarySurface.n`, while the GEOMETRIC n keeps every visibility-adjacent use — eps offsets, the translucency back ray, the ENTIRE hemi tier (a perturbed apex normal can put the own triangle inside the "open" hemisphere ⇒ false-empty), and the glass chain (`cos_i = v·n`). Reflection acceptance is `rdir·n_s > 0 && rdir·n > 0` (a perturbed lobe must not fire a ray that re-enters the surface), and **the reflection-lobe GATE keeps reading the FLAT roughness/metallic** — a texture-driven gate would make the two conditional VNDF draws depend on two different bilinear implementations (CPU LUT vs hardware) and skew the statistical A/Bs. Hemi sharing still fires on normal-mapped 2×2 groups (its predicate compares n_g, and the shared hemisphere is over n_g — valid). **Emissive** adds AFTER the kd·(1−transmission) factor at every depth (emitters appear in reflections/through glass), guarded (no unconditional `+0.0` — bit-identity); the add itself rides in color, so FSR's exact-remainder residual absorbs it with zero G-buffer change (known accept: possible slight RR ghosting on bright emitters — no emissive guide). **Emitters CAN light other surfaces since 2026-08-01, opt-in** (src/emissive.rs — see the `--emissive-lights` command-block entry; DEFAULT OFF): the display add here is the `sky::radiance` half, and when armed the direct tier's clustered-light NEE is the gather half (under fb.gi the hemi gather takes over instead — it picks this very add up off bounce hits). The green channel is NEGATED (`NORMAL_MAP_Y_SIGN` in shade.rs — the loader V-flips, so OpenGL-convention +Y maps point against our +v rows), pinned by `shade::tangent_self_test` (run by `--check`: analytic tangent directions, sign pin, mirrored-UV handedness, degenerate skip). GPU: `GpuMat` is **100 B** (stride comment lockstep with shade.hlsli's `Mat`; grew from 80 B for the water class's trans_tint/ior/ripple_amp), scene textures pick `_SRGB` vs `_UNORM` per `Texture::srgb` (resource + SRV via `GetDesc().Format`), and shade.hlsli mirrors the whole block (perturb_normal, effective rough/metal, per-lap emissive) — pasted into the wavefront AND DXR pipelines. Touch any of this → `--check` (tangent self-test), `--check --stress` (bit-identity), `--check-xess` (rng alignment), `san-miguel[-low-poly].obj --check`/`--check-gpu`/`--check-dxr` (T1 class-mismatch 0, T2 ≤ 2%, albedo A/B). Bump `scene_cache::CACHE_VERSION` whenever the Material/Texture on-disk repr moves.

**glTF 2.0 scenes** (`src/gltf_loader.rs`; `.gltf`/`.glb` on the positional arg — same CLI slot as OBJ, so exclusivity/default camera/structural-skip/`--tile`/the scene cache all compose): materials carry REAL PBR data, so `matclass` is bypassed — baseColor/metallic/roughness factors + normal/metallicRoughness (roughness=G, metallic=B, one shared map)/emissive textures map straight onto the extended `Material`; KHR_materials_transmission → `transmission` (NO albedo lift — glTF baseColor is authored for tinting, unlike San Miguel's dark exporter Kd), KHR_materials_emissive_strength multiplies, sheen reads the raw KHR_materials_sheen JSON (no crate feature exists in gltf 1.4), KHR_materials_ior is logged (GLASS_IOR fixed), unlit → matte, Blend → Mask + note. The three glTF-vs-OBJ traps, all deliberate: **NO V flip** (glTF UVs are top-left origin — the convention our texels land in after the OBJ path's load-time flip), **no glass albedo lift**, and **OPAQUE materials must not cutout** (spec ignores their baseColor alpha; `alpha_masked` is cleared unless a MASK/BLEND material references the texture — junk JPG alpha would punch holes in walls). Node graph is FLATTENED (transforms baked; normals by inverse-transpose; negative-determinant nodes flip winding so geometric face normals keep agreeing); authored TANGENTs are ignored (one tangent source: shade.rs's on-the-fly derivation); COLOR_0/TEXCOORD_1/occlusionTexture/KHR_lights_punctual ignored (the engine's one sky — scattering dome + sun disc — is the lighting model); images decode through the texture.rs rayon pipeline (only slot-referenced images; GLB buffer views, external files, and base64 data URIs); external BUFFERS resolve through `gltf_loader::resolve_buffers` with the OBJ path's `.zst` sibling fallback — committed scenes carry `.bin.zst` (raw vertex/index buffers zstd ~2-3×, measured Intel Sponza 133.5 → 60.7 MB; textures are already-deflated PNG/JPG and stay as-is), a plain `.bin` still loads verbatim, and the sibling fallback is pinned by a `self_test` case. `gltf_loader::self_test` (run by `--check`, download-free — a GLB assembled in code) pins node flattening, the mirrored-winding flip, u16 index widening, the welded-normal fallback, and the factor mapping; real-scene gates are download-optional like san-miguel (Khronos glTF-Sample-Assets DamagedHelmet/Sponza, Intel Sponza, Bistro — .gitattributes already LFS-tracks `scenes/**/*.glb|bin|gltf|jpg|jpeg`). One `gltf:` summary line prints prim/tri/material/texture counts + everything skipped.

**BC7 scene textures** (ON BY DEFAULT; `--no-bc7` kills, `--bc7-cpu` = the ispc A/B arm; `src/bc7.rs` owns the mode/predicate + the CPU arm over the `intel_tex_2` crate — prebuilt ISPC binaries, an ordinary static dep like `image`, so every `--check*` stays DLL-free): block-compresses the OPAQUE scene textures on upload, **GPU upload only** — the CPU samplers keep the exact RGBA8 texels, so the feature moves ONLY the statistical GPU-vs-CPU gates (measured San Miguel/Sponza/Bistro: albedo A/B 0.0001–0.0004 vs the 0.02 limit, radiance ≤ 0.007%) while class-mismatch/`t_viol`/alpha-rejections stay **bit-identical** — the carve-out proof. **The DEFAULT arm is a GPU compute encoder** (`Bc7Mode::Gpu(Fast)`; `src/gpu/bc7gpu.rs` + `shaders/bc7enc.hlsl`, fxc cs_5_0 — the bloom no-DXC precedent, so it exists before any tracer kernel and works in both Submit harnesses), dispatched per band inside `SceneGpu::new_uploaded`: the ring stages the RGBA8 source rows, the kernel (one thread per 4×4 block) writes `block_pitch`-strided blocks into a reused UAV buffer, `CopyTextureRegion` lands them in the committed BC7 mip — blocks never touch the CPU. Encoder shape: mode-6 (single-subset 7.7.7.7+P) PCA fit — covariance power iteration seeded from the largest-diagonal COLUMN, never a fixed vector ((1,1,1) is exactly perpendicular to an anti-correlated axis and collapsed red↔green blocks to near-solid, the measured max-198-LSB class) — plus 2 least-squares refinement rounds, plus a **mode-1 two-subset arm** (64 spec partitions ranked by 2-means SSE — an UPPER-bound predictor: rank with it, never prune — top-N full-fitted, lower-SSE mode wins): mode-6-only measured 26.4 dB worst on San Miguel's patterned plate vs ispc's 33.0 — the gap was the second subset, not the fit. `--bc7-quality` maps to effort tiers: ultrafast = mode-6 no-refit (26.3 dB), fast = + conditional mode-1 (top-4, only when mode-6 SSE > ~2.5 LSB RMS — 32.0 dB), basic/slow = mode-1 always at top-8/16 (32.5/32.8). MEASURED at `fast`: SM-lp 117 ms, Bistro 229 ms (2.1 Gtexel/s), Intel Sponza **282 ms at 3.8 Gtexel/s vs the ispc arm's ~20 s** (rates count every encoded level, mips included) — the 70× that made default-on affordable; there is still deliberately **no BC7 disk cache**. Determinism: the GPU arm claims per-(device, driver) only (per-block-independent kernel, no atomics) and NOTHING depends on more — no disk cache, and M11 runs the session's own encoder in-session; the CPU arm keeps its full determinism pin. Fallback ladder: `Bc7Enc::new` failure = LOUD line + uncompressed RGBA8 for the upload (never an implicit CPU-encode stall); in `--check-gpu` the same failure is a suite FAIL. `bc7::should_compress = !alpha_masked && !h2n && !n2h && 4-aligned dims`, all arms load-bearing: **alpha-masked cutout textures must stay RGBA8** — `alpha_nearest < 128`/`alpha_cutout` is a hard binary threshold on one texel, BC7 alpha lines reach only 4/8/16 levels per block (an authored 128 CANNOT be encoded and snaps across — a *visibility* divergence, and San Miguel's masks are antialiased), and the predicate mirrors `mat_cutout`'s (`trace.rs`) so the id set `.Load` can reach is exactly the RGBA8 set — their agreement IS the soundness argument (the GPU kernel excludes alpha by construction too: mode 6 pins alpha endpoints, mode 1 carries none — the ispc `opaque_*` twin); the 4-align arm is the D3D spec ("a block-compressed texture must be created as a multiple of size 4 in all dimensions" — odd dims measured working on the dev NVIDIA driver, but that is tolerance, not contract; free where it matters: Bistro/Sponza are 100% aligned, San Miguel keeps its odd-dim 63% of opaque MB as RGBA8). Mixed BC7+RGBA8 in the one `texs[]` table needs zero shader changes (SRV format reads back off the resource; the `_SRGB`/`_UNORM` role split carries over; the GPU arm's copy-out and the CPU arm's block-row staging both speak `d3d12::footprint_block`). Gating: `bc7::self_test` (in `--check`) pins the predicate/block math/CPU-arm determinism/`Bc7Mode` flag algebra; **every `--check-gpu` runs the `bc7-gpu` structural gate** (synthetic, fires even on the untextured procedural scene: an all-even flat color must round-trip the hardware decoder BIT-EXACT — representable exactly via e0 == e1, so any loss is wiring; every flat block byte-identical to block 0, the stride catch; a gradient ramp ≥ 30 dB at every effort tier; and a two-CLUSTER block whose small max error proves the mode-1 arm fired with partition/anchor tables + packing the hardware decoder agrees with); **M11** (armed + compressible textures) encodes with the session's arm, uploads as `BC7_UNORM`, `.Load`-decodes back (BC7 *decode* is spec-bit-exact) and per-texel diffs vs the CPU texels, worst-texture RGB PSNR ≥ 25 dB (a WIRING gate: pitch/footprint/format errors land ~10–20 dB; worst honest measured at `fast`: gpu 32.0 / cpu 33.0 on SM-lp, Bistro 41.9, Intel Sponza 37.4; `FR_BC7_DUMP=1` prints per-texture PSNRs). Touch bc7.rs, bc7gpu.rs, bc7enc.hlsl, the trace.rs texture upload loop, or `footprint_block` → run `--check` plus `san-miguel-low-poly.obj --check-gpu` and `--check-dxr` (defaults armed), `--check-gpu --bc7-cpu` (the ispc arm), and `--check-gpu --no-bc7` (the RGBA8 baseline).

## Mip-mapping + trilinear + 16× anisotropic (all three renderers)

Scene textures carry a **CPU-generated mip chain** (`texture.rs::build_mips`: floor-halving 2×2 box filter to 1×1, `Texture::mips`) and every sampler is trilinear with an explicit **ray-cone LOD** (Möller 2019, curvature-free). One chain, one formula, three consumers — the CPU renderer (`sample_trilinear`/`sample_trilinear_linear`), the `--gpu` wavefront, and `--dxr` (both through the shared `shade.hlsli`, whose `tex_lod_base`/`tex_lod` are term-for-term mirrors of `shade.rs::tri_lod_base`/`Texture::lod_dims` — change both together). The GPU uploads the chain verbatim (per-mip subresources through the same staging ring; SRV `MipLevels` = the chain; the static sampler is `MIN_MAG_MIP_LINEAR`), so **the parity axiom is upgraded, not broken**: identical texels at identical lods, and the `--check-gpu`/`--check-dxr` albedo A/B (mean |Δ| ≤ 0.02/ch) now compares trilinear against trilinear — measured 0.0000/ch on San Miguel, unchanged tolerances.

The filter runs in **LINEAR space** (sRGB texels decode through `SRGB_LUT`, average, re-encode via `encode_srgb`; linear maps average raw u8) — a gamma-space box filter darkens mid-tones, and `texture::self_test`'s 2×2-checker gate rejects it (mip must be `encode(0.5)` ≈ 188, not ~128). LOD:
`lod = 0.5·log2(uv_area/world_area) + 0.5·log2(w·h) + log2(cone_w) − log2(max(|n·d|, 0.05))`, with `cone_w = w0 + t·spread` and the triangle term computed **on the fly** from the hit's vertices (the `perturb_normal` precedent — a cached per-tri array would cost ~400 MB at 100M-tri tiling scale). Zero rng draws, a pure function of the hit — which is what keeps every same-seed/replay/VisCtl-burn contract intact. Primary spread = `CamBasis::pixel_cone()` (the GPU gets the same f32 through `FrameCb::pixel_cone`, single source); reflection/glass continuations start their cone at the parent hit's width (the HLSL flattened-DFS keeps its own `cone_o`, and the stashed transmission child carries `st_cone` so reflection laps can't advance it); hemi-GI bounce hits use the fixed broad `HEMI_CONE_SPREAD` (octant-scale footprint — over-blurred bounce albedo is variance reduction, coarser never wrong), mirrored in `hemi_leaf.hlsl`.

**`lod ≤ 0` is bit-identical to the pre-mip bilinear renderer** (both samplers early-out to `sample_bilinear`; on the GPU, lod 0 on a MIP_LINEAR filter reads level 0 only) — that is the compatibility contract, gated in `texture::self_test`, and it is why magnified/close-up views and every existing tolerance gate are unmoved. **`--no-mips`** is the A/B lever (`texture::set_mips`, consumed before scene load like the `--bvh-*` knobs): no chains are built, every trilinear sample degenerates to the old bilinear.

**Alpha cutout never sees mips**: `bvh.rs::alpha_nearest` and `trace_common.hlsli::alpha_cutout` both stay nearest-texel at level 0 — visibility parity across CPU/RayQuery/DXR is a correctness contract, and BC7's cutout carve-out depends on it. **BC7 (on by default)** compresses every level — the GPU arm edge-replicate-clamps sub-4/odd tail mips inside the kernel, the `--bc7-cpu` arm pads via `bc7::encode_level`; `should_compress` still gates on base dims only — a BC7 resource cannot mix formats per mip.

Measured (1080p `--spin path`, 7950X3D): mips are **faster** — Intel Sponza 34.69 → 32.56 ms, San Miguel interior 7.72 → 7.49 (fewer DRAM-miss taps under minification pays for the extra tap), memory +33%, load +the chain build. Shimmer drops as intended: `FRUSTRACER_STAB=1` on a still XeSS Sponza view reads 0.65 → 0.57 /255. Run `--check` (the `texture` gate), `--check --stress`, `san-miguel-low-poly.obj --check`, `--check-xess`, `--check-dlss`, and `--check-gpu`/`--check-dxr` (± `--bc7`) after touching `texture.rs`, the LOD math in `shade.rs`/`shade.hlsli`, the texture upload loop, or the sampler.

**Anisotropy (`--aniso N`, default 16)** is the refinement of that lod, not a rival filter. The cone is a CIRCLE; projected along `d` onto the surface it is an ELLIPSE — `cone_w` across the direction of travel, `cone_w / |n·d|` along it — and the scalar lod above can only describe a circle, so its `− log2(max(|n·d|, 0.05))` term takes the MAJOR axis and blurs the minor one with it. `shade::tri_grads` (mirrored by `shade.hlsli::tri_grads`) keeps both axes and returns them as **normalized-UV gradients**, so ONE footprint serves all five maps on a material regardless of their dims — the gradient form of the `tri_lod_base`/`lod_dims` split, and `SampleGrad`'s own contract. The world→UV inversion is Cramer's rule against `tri_uv_basis` (∂P/∂u, ∂P/∂v) — the on-the-fly basis `perturb_normal` already derived, now factored out and shared (one derivation, two consumers; still zero storage). Pure hit geometry, **zero rng draws** — the same-seed/replay/VisCtl-burn contracts are untouched. Pinned by `tangent_self_test`'s **reduction gate**: on a conformal UV map the major axis in texels IS `tri_lod_base + lod_dims` to 1e-4. If that drifts, the two paths have diverged and `--no-aniso` has stopped being a clean A/B.

Consumers: the CPU averages `ceil(ratio)` trilinear taps stepped along the major axis at the MINOR axis's lod (`texture::sample_aniso`, capped at `MAX_TAPS = 16`); the GPU hands the same gradients to `SampleGrad` on a **second static sampler** (`samp_aniso`, `s1 space1`, `D3D12_FILTER_ANISOTROPIC`, `MaxAnisotropy` = the CLI value — `trace.rs::create_root_signature`, shared with DXR). **`SampleLevel` cannot be anisotropic** — it hands the TMU one scalar lod and no gradients, so flipping the old sampler's filter alone would have been a silent no-op; the gradients ARE the feature. `FLAG_ANISO` (a session constant from `texture::max_aniso()`) gates it, but WHICH rays use it is a call-site decision: primary + reflection/glass continuations yes, hemi-GI bounce laps no (`shade_split`'s `aniso` arg / `Cone::aniso = 1.0` — the bounce cone is octant-coarse by design). `--no-mips` forces `--no-aniso` (no chain ⇒ nothing to prefilter the minor axis with). **`aniso = 1` runs the isotropic lod path VERBATIM** — that is why `--no-aniso` is bit-identical to the pre-aniso renderer by construction rather than by gate, and the whole suite re-run with it is the regression proof.

Measured: the GPU/CPU albedo A/B (hardware aniso vs the CPU's N-tap — different tap distributions, so exact agreement was never expected) lands at **0.0001/ch against the unchanged 0.02 limit**, on both pipelines and under `--bc7`; the software-N-tap fallback that was held in reserve was not needed. CPU cost is real and pose-dependent (16 taps where the footprint is 16:1): Intel Sponza `--spin path` 24.14 → 26.25 ms (+8.7%), San Miguel interior 15.95 → 16.13 (+1.1%); `--aniso 4|8` is the lever if a CPU session wants the time back. GPU cost is **below the `gpu hybrid` bench row's noise floor** (that row spans 1.46–3.47 ms for one unchanged config — never trust a single sample of it).

## Heightfield relief rendering (V — inward-only displacement at the intersectors)

Surfaces whose normal map carries an alpha-channel heightfield (`Material::height_amp` > 0 — every normal-mapped material after the load-time conversions above) can render as REAL displaced geometry: a tangent-space ray march at the intersector choke point of **all three renderers**, silhouette-correct because a ray that exits the prism untouched REJECTS the hit and traversal continues (the alpha-cutout monotonicity precedent, and the same single-choke-point discipline — every ray type marches: primary, shadow, AO, hemi, verify reference, so the exact-zero gates stay like-for-like). **Inward-only is the soundness spine**: h ∈ [0,1] with 1 = the flat triangle plane, displacement only recedes, so front-side hits only move FARTHER and every frustum/tmin/temporal claim stays a conservative lower bound. Two repairs make that literally true: (1) height-carrying triangles' AABBs are **swept inward at BVH build** (`bvh::grow_height_sweep` — a recessed pit wall pokes below the flat tri's box, and a wall hit at `t' < t_plane` from a recessed apex would otherwise fire the exact-zero gates; every claim consumer inherits soundness from this one site, and the lever keys `.fcache`); (2) while relief is live, both GPU intersectors enumerate the **full positive base-triangle ray** — the hardware culls at the PLANE t, but the marched delta is `depth / |dot(ray_dir, geometric_normal)|`, so the old `FrameCb::height_max` ±world-depth widening was not conservative at grazing. The inline RayQuery loops then re-apply the ORIGINAL logical `(tmin, tmax)` to the marched candidate t, and DispatchRays carries both bounds in `HitPayload` / `ShadowPayload` for the same any-hit re-check. This covers inherited tile/hemi nonzero TMin as well as finite AO/shadow TMax; primary TraceRay was already `(0, FLT_MAX)`. The march (`bvh::height_march`, bit-mirrored in `trace_common.hlsli`): ĥ and the barycentrics are exactly AFFINE in t (Cramer rates — no UV inversion anywhere), `HEIGHT_COARSE` = 16 linear steps + `HEIGHT_REFINE` = 5 bisections + secant, field sampled by NESTED-LERP ALU bilinear on level-0 alpha (`Texture::height_bilinear` — exact on constant plateaus, which is what makes a 255-alpha region an exact-1.0 field and the flat-field identity BITWISE; hardware lerp is not bit-reproducible, hence no sampler). Entry rules unify every orientation: plane entry hits at the first g ≤ 0 (255-plateau ⇒ the ORIGINAL hit verbatim); below entry hits the underside at the first g ≥ 0 (floors opaque from below); interior entry (the recessed-apex secondary-ray case) takes the two-phase POM shadow rule — skip solid, then hit — which kills eps-offset acne and lets pits see their own sky. GPU: RayQuery candidate loops reject-or-commit-at-plane-t then re-march the winner; DXR any-hits reject (IgnoreHit — silhouettes real there too) and chs_* re-march. **Interior-edge cracks are FIXED by the march's edge extension** (`HEIGHT_EDGE_EXTEND` = 4×amp texels of uv travel past the footprint exit, wrap-sampling the shared chart — which IS the neighbor's field on a continuous atlas; hits report edge-clamped bary; the swept AABBs pad every axis by `EXTEND·depth` and the march CLAMPS its per-ray budget to that same world size — the budget's texel rate is DIRECTIONAL while the pad's depth is geometric-mean, so without the clamp an anisotropic chart's sparse UV axis could out-travel the pad — which together is what keeps extended hits inside claimed-occupied boxes): without it, a ray drifting across a shared edge mid-march fell through — the coplanar neighbor NEVER surfaces as a candidate (one plane crossing, inside this triangle only, on the CPU AND the RT hardware), and on real meshes (triangle fans) every edge leaked a dark band one relief-depth wide. Known-accepts: GPU plane-t commit ordering can mis-sort candidates within one relief prism's displaced ray interval (the world-space prism stays bounded, but its ray-t width grows at grazing); chart SEAMS get a ≤-budget wrong-field fringe instead of a crack; a texel-scale extension fringe at true silhouettes; grazing rays undersampling texel-thin spires; marched parallax has no MVs (the clouds precedent). **The DEFAULT session is UNARMED** — structurally the bit-exact pre-relief renderer (no swept AABBs, no march; V prints a note). Armed-by-default was tried and REVERTED on measurement: the sweep's all-axis `HEIGHT_EDGE_EXTEND`·depth pad wrecks BVH quality wherever every triangle carries height and triangles are only a few texels wide (DamagedHelmet close-up `--spin path`: 596 armed vs 146 unarmed ms/frame — 4×, with relief OFF; the pad alone is ~95% of it, the inward sweep ~15%; armed build SAH@1 10.0 / mean leaf 5.48 vs 5.9 / 4.25 — SAH stops splitting into the overlap). `--heightfield` ARMS the session and starts relief ON — swept AABBs built, march compiled in — and **V** then toggles relief ↔ plain normal-mapping live (shading+visibility change: frame + upscaler-history reset; temporal ring/replay KEPT — claims live on the swept boxes in both modes and replay re-marches through `shade_tile`; armed stays true across the toggle, so no rebuild). `--no-heightfield` spells the default explicitly (later flags win). Shrinking the pad (lateral-only, or topology-on-tight-boxes) is the open follow-on that could make armed cheap enough to re-default. Zero rng draws anywhere — every same-seed/replay/VisCtl contract holds. Gated by `bvh::height_self_test` in `--check` (flat-field bitwise identity, closed-form marched hits incl. bary rates, side-exit/silhouette reject must-fires, interior escape + pit-wall occlusion with `t' < t_plane`, underside crossing, grazing finite-interval endpoints, the build-vs-march depth pin) plus the whole existing verify/GPU suites running WITH relief on for any height-carrying scene when `--heightfield` is passed (the checks follow the session flags, so a bare `--check` gates the unarmed pre-relief path and `--check --heightfield` the relief path (`height_self_test` self-arms either way); measured: San Miguel `--check-gpu`/`--check-dxr` class-mismatch 0, claim-violation 0). Touch `height_march`/`tri_height_depth`/`grow_height_sweep`/`height_bilinear` or their HLSL twins → run `--check`, `san-miguel-low-poly.obj --check --heightfield` + `--check-gpu --heightfield` + `--check-dxr --heightfield --dxr-inline 0`.

## Big scenes (tiling, scene cache, 100M+ triangles)

**`--tile NxM`** (`scene::tile_scene`) reaches 100M+ triangles by **flattened replication** — duplicated transformed geometry in the one flat Scene/BVH, deliberately NOT instancing (a two-level BVH would need instance-aware tokens in the frustum node cuts, world-space temporal claims, hemi cuts, and both GPU pipelines' positional stream binding — a deferred epic; Moana-class scenes sit behind it). Tiling runs AFTER the diag-10 fit and re-derives `diag`/`eps`/`ao_radius` over the tiled extent via `scene::finalize_scalars` (tiling before the fit would shrink eps below float precision); the ground quad is rewritten to cover the grid, materials/textures are shared untouched, and the light/camera scale stress-style. Measured: San Miguel low-poly ×4×4 = **89.9M tris, 39M-node BVH (2.1 GB) built in ~23 s at the M2 defaults** (depth 53 of the 96-entry stack; the pre-M2 single-axis/c_trav=0 build was 109M nodes / 5.2 GB in ~14 s — 3-axis binning costs ~3× the binning work per launch until the tiled BVH is cached), traces headlessly; `--check` passes at ×2 (22.5M) as the loaded-scene gate class. GPU sessions print a `gpu scene:` line with stream/BLAS/VRAM accounting (`adapter::vram_info` — WDDM demotes over-budget commits silently, so check it when a big scene runs slow).

The supporting machinery, all gated: (1) the **BVH build is a deterministic two-phase parallel build** (`bvh.rs`: sequential top splits to `par_threshold(n)` — a pure function of n, never thread count — then rayon per-subtree local arenas over disjoint `tri_idx` slices, stitched with a uniform link rebase; byte-identical across runs AND thread counts, which `--check` gates by building twice — `Bvh::identical`); the traversal stacks are `TRAV_STACK = 96` entries with a hard `max_depth() <= TRAV_STACK` assert at build (the hot loops stay branch-free; never "fix" an overflow by dropping the far child — that breaks the lower-bound soundness contracts). (2) A **binary scene+BVH sidecar cache** (`src/scene_cache.rs`, `<resolved-source>.fcache`, gitignored): manual POD format keyed on source+MTL size/mtime, EVERY texture file's size/mtime (`alpha_masked` and the height-map skip are texture-content decisions — an edited texture misses the whole cache rather than resurfacing stale flags), + `CACHE_VERSION` (bump on ANY Scene/Material/Texture layout, BVH build, loader, or matclass change); a corrupt/truncated sidecar is a silent miss, never a panic (counts capped against the file size, cross-array links validated before use); texels are NOT cached — texture paths + `alpha_masked` re-decode in id order (rayon), a failure substitutes 1×1 white so material tex ids never shift. Measured San Miguel low-poly: cold 5.4 s → warm 0.58 s; under `--check` the build-twice gate doubles as cache-integrity (loaded BVH vs fresh build). The cache stores the UNTILED 1× scene; tiled runs re-tile + rebuild. (3) **GPU scene upload streams through one reusable 256 MB staging ring** (`SceneGpu::new_uploaded`, blocking chunks via the `d3d12::Submit` trait — HeadlessGpu and D3d both implement it): no full-scene repack Vecs, no full-scene upload-heap commit; the **BLAS builds with ALLOW_COMPACTION** and is replaced by its compact copy before init returns (~40-80% smaller; the TLAS builds against the compacted BLAS), and **DXR-only sessions skip the software BVH upload** entirely (`sw_bvh: None` — dxr.rs never binds t0/t1). Practical GPU reach on 24 GB: DXR ~100M+, wavefront less (it carries the software BVH for frustum queries); the CPU path handles ×20 (~200M) in ~25 GB RAM.

Scene acquisition workflow (per new scene): download from the McGuire archive into `scenes/<name>/`, `zstd --rm -19 <name>.obj` (MTLs stay plain text; the loader resolves them as plain siblings), convert PNG textures to lossless WebP (Pillow `save(..., "WEBP", lossless=True, exact=True, quality=100, method=6)` + a round-trip pixel compare before deleting each PNG — the scratchpad script pattern; manifests keep their `.png` names, the loader falls back to the `.webp` sibling), **verify LFS before committing** — `git check-attr filter scenes/<name>/<name>.obj.zst` must say `lfs` and `git lfs status` must list the files (a scene file reaching plain git history is permanent bloat) — then smoke `cargo run --release -- scenes/<name>/<name>.obj --check` (the bare-.obj form works via the .zst sibling fallback). Alternate source formats some drops bundle (FBX/MAX/USD — Intel Sponza ships all three) are gitignored under scenes/; env maps and straggler texture formats (hdr/exr/tga/dds) have LFS guard patterns so nothing binary can reach plain git.

**Committed scenes** (all smoke `--check` PASSED at their default poses): `scenes/san-miguel/san-miguel[-low-poly].obj` (10M/5.6M, the standard textured+cutout scene), `scenes/powerplant/powerplant.obj` (12.8M untextured; **×8 ≈ 102M — the 100M smoke scene**), `scenes/rungholt/rungholt.obj` (6.7M textured Minecraft city; + `house.obj`), `scenes/bistro/Exterior/exterior.obj` (2.8M; **38 normal + 16 emissive maps live, matclass-classified** — the PBR-map showcase; + `Interior/interior.obj` 1.0M; MTLs reference the sibling `*Textures/` dirs — the G3D multi-zip layout; upstream ships 4 dangling map refs, warned + flat-Kd fallback), `scenes/vokselia-spawn/vokselia_spawn.obj` (1.9M), `scenes/hairball/hairball.obj` (2.9M, pathological BVH depth — the stack-margin test), `scenes/damaged-helmet/DamagedHelmet.glb` (15k, all 4 glTF map types), `scenes/sponza-khronos/Sponza.gltf` (262k, 69 textures).

**NOT committed — fetch it yourself:** `scenes/intel-sponza/main_sponza/NewSponza_Main_glTF_003.gltf` (3.7M tris, 2.7 GB of 4K PNGs — the texture-memory stress; COLOR_0/authored tangents ignored by design). Dropped from the repo 2026-07-25 for two independent reasons, either sufficient: its `credits_license.txt` grants "personal use and educational use" and does NOT grant redistribution, and at 2.05 GB across 147 LFS files it was **56% of the repo's entire LFS payload**. It is also the one ex-committed scene the default world never loads (the `CURATED` skip — a texture-memory wall, not a place), so its absence costs the default experience nothing. Everything about LOADING it is unchanged: download from Intel's graphics-research samples page, extract to `scenes/intel-sponza/`, and the positional-argument path above works exactly as before (`.gitignore` keeps it untracked). Measurements taken on it — the `--defer-shade` negative result, the mip and anisotropy costs, the BC7 encode rates — remain valid and are still quoted throughout; they are just no longer reproducible from a bare clone.

## DLSS Ray Reconstruction (raw NGX)

Presentation is SDL3 + D3D12 (`src/gpu/`, `src/input.rs`; minifb is gone; the `sdl3` crate is pinned at 0.18 — it self-describes as API-unstable, so version bumps may need call-site fixes). **Streamline is RETIRED** (2026-08-01): DLSS-RR is driven DIRECTLY through NGX — `shim/dlssd_shim.cpp` (a sibling of the FG shim, both over `shim/ngx_shared`'s one refcounted `NVSDK_NGX_D3D12_Init_with_ProjectID` per device) wraps `NGX_D3D12_CREATE_DLSSD_EXT` / `NGX_D3D12_EVALUATE_DLSSD_EXT` / `NGX_DLSSD_GET_OPTIMAL_SETTINGS`, and `src/gpu/ngxrr.rs` is the Rust session/feature wrapper (`NgxRr::open` = the chain's level-1 probe via `SuperSamplingDenoising.Available`; `RrFeature` created eagerly at the DRS range max, recreated across resizes with the queue drained). BUILD-gated like FG: `FRUSTRACER_DLSS_SDK` + `cfg(dlss_ngx)` — one SDK, one gate, both features; build.rs stages `nvngx_dlssd.dll` + `nvngx_dlssg.dll`; a non-SDK build has NO DLSS at all (one loud line, the chain falls to FSR4/XeSS/FSR3, and the old SL-zip runtime dependency is gone with the intent that nothing fetches SL anymore). What died with the interposer: the proxy device/queue/swapchain split and its startup assertion, the slInit-before-any-DXGI-factory ordering constraint (DLSS is chain-top by POLICY only now), `HeadlessGpu::new_sl` (the plain harness serves the cinematic capture — raw NGX needs no queue hook), the whole TRAP-8 shared-in-process-NGX-state class (one NGX client family, refcounted — the never-DestroyParameters discipline survives as ownership hygiene, see dlssg_shim.cpp), the SL DLSS-G declines-to-insert fallback, and Reflex/PCL. Every gate stays DLL-free headless (`--check`/`--check-dlss` never touch NGX).

DLSS mode rules (`main.rs`): every frame is a fresh **1-spp hybrid frame at a render resolution inside the range** the DLSSD optimal-settings query reports at startup (optimal/min/max; RR upscales + denoises to the window size — a failed or degenerate query collapses to opt == min == max, which main.rs reads as "DRS off, fixed res" and falls back to DLAA at native — the raw create then says `dlaa` honestly instead of via a degenerate range). DLSS-RR is **level 1 of the always-on upscaler chain** (see the command block) by policy. By default the resolution is **locked** to the `--lock-res` scale (`xess::lock_scale`; default `native` = 100% — DLAA — `quantize_res`-clamped into the range) — one fixed res for the session, controller and StepLimiter never consulted; `--lock-res dynamic` opts into step-wise DRS instead (both upscaler paths share the flag). **Step-wise DRS**: the resolution comes from the SAME `xess::ScaleCtl` + `quantize_res` controller as XeSS (shared pure math), rate-limited by `xess::StepLimiter` (dwell `STEP_DWELL = 90` frames between APPLIED steps; expiry adopts the current target in ONE multi-quantum decision; an emergency shed on a badly blown frame bypasses, growth never does — without this, the slow climb after a shed crossed each quantum as its own step). An adoption is a decision, not a jump: the limiter **ramps** the output resolution from the previous endpoint to the adopted one over `RAMP_FRAMES = 24` frames (height lerped linearly, width derived per frame from the window aspect via `xess::width_for_height` — the same single width source `quantize_res` uses, so the ramp's final frame reproduces the endpoint bit-exactly; `RAMP_FRAMES < STEP_DWELL` is const-asserted so a ramp always completes before the next adoption). An emergency shed still snaps — its gate compares the target against the ramp's CURRENT output, not the endpoint (else a blown frame mid up-ramp could GROW resolution), and it also fires when the target already equals a down-ramp's endpoint, fast-forwarding the ramp instead of descending through more blown frames. Ramp intermediates change res per frame, so the temporal cache drops per distinct-res frame (rounded heights repeat: ≤ |Δh| drops per ramp) and `GBufs::set_res`/`DlssPrev` rebuild run per distinct-res frame — all cheap reinterprets/CPU math by design; `StepLimiter::new(0)` is the per-path kill switch back to snapping. **A step is a scale change, not a scene change — do NOT reset**: no `dlss_reset`, no prev drop; `gbufs.set_res` reinterprets, and `DlssPrev` (which carries the previous `Camera`) rebuilds its basis/matrices at the NEW resolution so MVs land in current-res pixels; XeSS mode likewise derives the MV basis from the prev Camera at each frame's own res. History survives the step via the extents. Step-resets were a shipped bug: every step wiped the upscaler history and the image re-converged patchily — stochastic shadows visibly "danced" (the SL note that RR re-initializes on input-res change at `ProgrammingGuideDLSS_RR.md:410` is RR's own internal affair, not a reason to also wipe). `FRUSTRACER_STAB=1` prints a mean inter-frame |Δ| of the upscaled output every 15 frames — the numeric dancing detector; hold the camera still and healthy output trends toward ~0. The temporal cache still drops on steps via `tprev_res` — that one is a correctness contract, not a quality choice. `FrameCtx::accumulate = false` with `frame` still advancing (pinning `frame` would freeze the RNG noise pattern and RR would treat it as signal); frame-uniform Halton(2,3) jitter via `FrameCtx::frame_jitter` (offset ∈ [-0.5,0.5) render-res pixels keeps samples inside pixel footprints, so all quadtree/temporal invariants hold; `dlss::JITTER_PHASE = 72` covers the 3× ratio at both DRS paths' range floors, 8·ratio²); half-res and the depth-cap budget path are disabled; the temporal cache participates at whatever res the frame traces (each is a full-depth hybrid frame at one fixed res; `tprev_res` drops the prev cache on any res step, including the G toggle); the adaptive shading rate stays **XeSS-only** (`adaptive: mode == Xess`) — on RR the 2×2-correlated shadow noise and per-frame cell reclassification presented as patchy dancing (RR's network preserves block-correlated noise as structure instead of integrating it); revisit with an RR-friendly classifier before re-enabling. Everything the evaluate sees — camera matrices, jitter, MVs, mv_scale — is in **render-res pixel space**; only the RR output color is window-sized. DRS is expressed **only** through the per-evaluate `InRenderSubrectDimensions` (`rr_ngx_sequence` passes `{fc.rw, fc.rh}`; the `MVLowRes` create flag carries what SL used to infer from extents): `gpu/rr.rs` allocates the input planes at the range **max** and `record_upload` converts/copies just the frame's `rw×rh` sub-rect. G-buffers (`src/dlss.rs::GBufs`, capacity = range max, reinterpreted per step via `set_res`; every plane stores **f16 bit patterns** (`dlss::ld16`/`st16`, the only conversion sites) EXCEPT `depth`, which stays f32 — its wire format is R32_FLOAT on both upscalers, reprojection and the XeSS sky-exact-0.0 encode consume it, and the RR feed gate is bit-equal; the f16 planes upload as raw bit copies, albedo still converts f16→unorm8; `spec_alb` is full-RGB F0 = lerp(0.04, albedo, metallic), 3/px — the RR guide wants linear 3-channel specular albedo, an achromatic mean loses metals' specular tint) are written only at the primary-hit fill sites in `render.rs` behind `ctx.gbuf: Option<_>` — `shade()`'s `PrimarySurface` out-param is `None` for the recursive bounce, which structurally keeps secondary rays out of the capture. Motion vectors reproject the exact hit point through `prev_cam` (`CamBasis::project`), current→previous in render-res pixels; depth is linear view-Z (`t·dot(dir, forward)`), sky = `far` from `dlss::near_far` (single source). `dlss_prev` is deliberately separate from `tprev_basis` (different contracts). The glam→NGX row-major transpose happens ONLY in `gpu/mod.rs::row_major`; jitter/MV conventions live on the `ngxrr::NgxRr` session (the `FR_NGXRR_JITTER`/`FR_NGXRR_MV`/`FR_NGXRR_DEPTH`/`FR_NGXRR_EXPO` levers, loud-on-departure), nowhere else — the jitter handed to the evaluate is the **NEGATED** sample offset (settled empirically 2026-08-01, like SL and UNLIKE raw-NGX FG — trap 9's each-feature-keys-its-own-polarity rule cut both ways: negated reads STAB 0.11-0.16/255 static, raw 0.26-0.50; the renderer's own jitter is never changed). RR uses preset E (latest transformer model, stamped on every per-mode preset key at create). Run `--check` AND `--check-dlss` after touching `render.rs`, `camera.rs`, or `dlss.rs`; touch `dlssd_shim`/`ngxrr.rs`/`rr_ngx_sequence` → those plus `--check-gpu`/`--check-dxr` (the RR feed gates) and the interactive STAB smoke.

## OIDN (Intel Open Image Denoise)

The secondary, vendor-independent denoiser (N toggles; mutually exclusive with DLSS). The OIDN 2.5 SDK lives under `SDKs/oidn.x64.windows` (Apache-2.0 but gitignored with the rest of `SDKs/*`); `src/oidn.rs` keeps the runtime-DLL footprint policy in pure Rust — nothing links the SDK, `OpenImageDenoise.dll` is `LoadLibraryExW`'d at runtime (the tbb/core/device DLLs are preloaded by absolute path so OIDN's lazy device loading resolves from the module list), every entry point goes through a fn-pointer table, and headless builds/`--check` stay DLL-free. There is deliberately no C++ shim: OIDN is a plain, unversioned C API.

OIDN mode rules (`main.rs`): the normal render loop keeps running — temporal cache, depth-cap budget frames, hemi bounces — at forced full-res (the half-res moving prefix would misalign the full-res-stride G-buffers), with `ctx.gbuf` set to a **separate window-res `GBufs`** (`oidn_gbufs`, lazily allocated ~62 MB — f16 planes + f32 depth, 30 B/px; the DLSS `gbufs` is render-res and must not be shared). Two sub-modes, **M** toggles (`frame = 0` in BOTH directions — accum semantics flip; `--oidn-no-temporal` starts off):

- **Temporal (default)**: every frame is a fresh 1-spp frame (`accumulate = false`, free-running `oidn_seq` as the RNG frame index, `jitter` always on — the DLSS pattern; pinning `frame` would freeze the noise) folded into a reprojected per-pixel EMA history (`src/reproject.rs::History`, ~77 MB, lazily allocated) that is the **sole accumulator and the denoiser color input**. Reprojection is recomputed **from depth** in the history pass (`p = origin + ray_dir(center)·t`, validated against the previous frame's view-Z within 5% relative), deliberately NOT from `GBufs::mvec` — a coarse quad's MV projects the quad's shared center-ray hit point (an MV fetch would collapse the quad's history to one texel) and behind-plane is stored as mv (0,0); `prev_cam` stays `None` and render.rs is untouched. A static camera takes the bit-equal-basis identity path (`CamBasis::PartialEq`, the temporal.rs precedent) with the length cap lifted from `L_MAX = 32` to MAX_SAMPLES, so still frames converge statistically like the plain accumulation — no moving↔static seam (the price: while-moving-quality samples wash out of the EMA at ~3%/frame after stopping, a brief crossfade instead of a pop). Budget frames still run and sprinkle real point samples (one per 16×16 cell, `render::sparse_fill`) that blend into the history at their exact pixels; a `KIND_COARSE` (cell-flooded) pixel with valid history keeps the history at blend weight 0 (reprojection visually hides the coarse cells), without history it takes its cell sample's color at L = 1. The history invalidates on every setting change that resets `frame` (R/T/H/1-3/N/G/M) EXCEPT camera motion and the budget↔normal transition. `History::prev_basis` is its own state — not `dlss_prev`, not `tprev_basis`. The history is **presentation-side only**: it never feeds `accum`, `tbuf`, the temporal cache, or any tracing decision; in this mode `accum` holds the last 1-spp frame (the DLSS-mode carve-out to "accum is a pure sum").
- **Plain (M off)**: the accumulation average goes to the filter directly; shimmers while moving (the RT filter is not temporally stable), converges sharp when still.

Either way the color + first-hit albedo (`clamp(diff_alb + spec_alb, 0, 1)` per channel — spec_alb is RGB F0, per OIDN's albedo guidance; sky is already 1) + world normal go through the RT filter (`hdr=true`; `cleanAux=true` by default — the guides are deterministic primary-hit values, `--oidn-no-clean-aux` is the A/B escape; quality balanced, `--oidn-quality fast|balanced|high` overrides) via OIDN buffers — shared host pointers are not guaranteed device-accessible on GPU backends — with all four images bound `OIDN_FORMAT_HALF3` (f16 staging: color narrows clamped to f16::MAX so HDR never becomes +Inf, normal is a bit copy of the f16 plane, the denoised half output widens into `out_f32` so `resolve_hdr` keeps its `&[f32]` contract) — and the denoised HDR is CPU-tonemapped by `render::resolve_hdr` (same curve as `resolve`, single-sourced in `present_px`) into the present buffer. The denoised output is **never written back into `accum` or the history**; images are bound and the filter committed once at startup, per-frame is write→execute→read only. `--oidn-device default` lets OIDN rank devices and may not pick the fastest on a multi-GPU box (it chose SYCL over the faster CUDA here) — `--oidn-device cuda|cpu|...` overrides. `--check-oidn` (the only OIDN check that needs the DLLs) renders accumulated frames through the exact interactive contract and gates the filter (output finite, ≠ input, mean |Laplacian| strictly drops, mean preserved within 2×, second denoise proves filter reuse), then runs the temporal gates: `reproject::self_test` (closed-form: projection roundtrip, static replay = exact running mean, analytic strafe, behind-plane/depth rejection, coarse keep/reset, L-accounting — also run by `--check`), static replay (identity path fires, history == sum/N within 1e-4, zero rejections), forward dolly (0 < rejected < 10%, world-point reprojection agreement vs the previous frame's depth, history error < 0.7× the fresh-1-spp error against a 16-frame converged reference), pure yaw (sky reprojects; the exposed edge band rejects), a capped d=4 budget frame (coarse-kept fires; sparse samples blend; the moving L cap holds), and L-accounting after every update (`#(L==1)` ∈ [rejected+coarse_reset, +coarse_kept]). `--check-oidn --stress n` skips the must-fire structural halves, mirroring `--check`. Run `--check-oidn` after touching `oidn.rs`, `reproject.rs`, or the resolve/present chain in `main.rs` (and `--check` for the self-test hook).

## XeSS-SR (Intel XeSS Super Resolution)

The third upscaler path (`--xess`, X toggles; mutually exclusive with DLSS — the context lives on the **native** D3D12 device, so `--xess` implies `--no-dlss`). The SDK sits under `SDKs/XeSS-SDK` (Intel Simplified license, gitignored with the rest of `SDKs/*`); `src/xess.rs` mirrors the OIDN footprint policy — nothing links the SDK, `libxess.dll` is `LoadLibraryExW`'d at runtime into a fn-pointer table, headless builds/`--check-xess` stay DLL-free, and there is no C++ shim (plain C API; the `#[repr(C)]` structs are hand-transcribed from `xess.h`/`xess_d3d12.h` — pack(8) == natural x64 layout).

XeSS mode is the all-in "**a pixel is a sample**" mode: **block filling is gone completely** — `render_frame_capped`/`sparse_fill`, the half-res moving prefix, and CPU accumulation never run. Every frame is a fresh jittered 1-spp **full-depth** hybrid trace at the session's render resolution, and XeSS's temporal accumulation of the jittered sample stream (frame-uniform Halton via `FrameCtx::frame_jitter`, free-running `xess_idx` — pinning would freeze the noise) is the only spatial reconstruction. By default the resolution is **locked** to the `--lock-res` scale (`xess_lock` in main.rs, `quantize_res`-clamped into the SDK range; default `native` = 100%) — the controller/StepLimiter are never built. With `--lock-res dynamic` the resolution is **dynamic**: while still, the scale creeps to the range max and the jitter accumulates into genuine super-resolution, so "converged" never falls back to a CPU path. The dynamic resolution comes from `xess::ScaleCtl` (the `depth_est` controller transplanted into log2-scale units: cost ~ area, slow-up/fast-down, deadband above 60% of the ~15 ms budget, no wall clock read in the renderer) quantized by `xess::quantize_res` (height in `RES_STEP = 36` px quanta — divides 1080 and is a multiple of 9, so 16:9 widths are exact — width derived from the window aspect, both clamped **hard** into the `xessGetOptimalInputResolution` min/max range; the quantization plus `xess::StepLimiter` (shared with the DLSS path) IS the hysteresis: a res step drops the temporal cache via `tprev_res` and reinterprets `xess_gbufs` in place via `GBufs::set_res` — capacity is the range max, no realloc. Adopted steps are **ramped** over `xess::RAMP_FRAMES = 24` frames instead of snapped (see the DLSS section — same limiter, same rules; intermediates are exact-aspect via `xess::width_for_height` and need no RES_STEP quantum). A step is a scale change, not a scene change: no `reset_history`, no prev drop — `xess_prev` stores the previous CAMERA and the MV basis is derived at each frame's own res, staying correct across steps). Dynamic resolution is first-class in the SDK: input planes (`gpu/xr.rs::XessResources` — color RGBA16F, mvec RG16F, depth R32F) are allocated once at the range **max**, every execute names its own `input_width/height` sub-rect.

Contracts: everything XeSS sees is in **input-res pixel space**; only the output (window-res RGBA16F UAV → `SRV_SLOT_XESS` tonemap) is window-sized. MVs come from the same `GBufs::mvec` fill sites as DLSS (pixels, y-down, current→previous, `prev_cam = xess_prev` — its own contract, not `dlss_prev`, not `tprev_basis`); depth is converted at upload from linear view-Z to [0,1] reversed-Z clip depth by `xess::view_z_to_clip_depth` (single source; `INIT_FLAGS` carries `XESS_INIT_FLAG_INVERTED_DEPTH`; sky's `view_z = far` lands exactly on 0.0; `--xess-autoexposure` ORs in `ENABLE_AUTOEXPOSURE` at init — A/B lever, default off). The undocumented sign/flag polarities live ONLY in the constants at the top of `xess.rs` (`JITTER_SIGN` — start unnegated, unlike SL's negated jitter; `VELOCITY_SCALE`; `INIT_FLAGS`) — settle empirically like the SL jitter sign was (wrong jitter sign = 2× wobble on a static view; wrong MV polarity = directional smear under motion). `reset_history` fires on every predicate that resets `frame` EXCEPT camera motion (surviving motion is the upscaler's job). The temporal cache participates every frame (each is a full-depth hybrid frame at one fixed res, the producer/consumer contract).

**N composes** (unlike DLSS) and is a **3-state cycle in XeSS mode: off → pre → post** (`XessOidn` in main.rs, independent of the plain-mode `oidn_on`; `--oidn` starts pre, `--oidn-post` starts post). XeSS-SR is a TAA-upscaler, not a denoiser, so raw 1-spp shading noise shimmers. **Pre** (the recommended default) inserts the OIDN pipeline at the dynamic render res, before upscaling — guides match, subpixel jitter detail preserved, cheaper. **Post** is the A/B experiment: raw 1-spp → XeSS upscale → the window-res output is pulled back to the CPU (persistent `XessResources::readback` + `D3d::submit_and_wait`, a frame submission WITHOUT a Present) → OIDN at window res with **nearest-upscaled guides** (`GBufs::upscale_guides_from` copies exactly the planes `oidn::run_filter` reads) → `resolve_hdr` → `present_cpu` as the frame's single Present. Post costs a synchronous ~16 MB readback plus a window-res denoise per frame and has no pre-EMA history (M prints a note; XeSS is the temporal integrator in that ordering). In XeSS sessions OIDN's device auto-pick skips SYCL (CUDA then CPU): the SYCL runtime and libxess.dll drag conflicting Intel compute stacks into one process and abort() natively at first use (observed on OIDN 2.5 + XeSS 2.0.2); an explicit `--oidn-device` is honored as given, including `sycl` at your own risk. OIDN is resolution-agile for this: buffers/staging are allocated at the construction res and `OidnContext::set_res` rebinds the filter images + recommits on a res step (weights stay loaded — cheap); the reprojection `History` is capacity-allocated at window res and `History::set_res` reinterprets in place + self-invalidates on any res change (a fresh history is an invalidated one; XeSS's accumulation hides the blip) — no realloc, which matters now that a DRS ramp changes res per frame. The denoised/upscaled output is never written back into `accum` or the history; in XeSS mode `accum` holds the last 1-spp frame (the DLSS-mode carve-out).

**Adaptive shading rate** (`FrameCtx::adaptive`, XeSS frames only; `--no-adaptive` forces uniform per-pixel shading — visibility is unchanged either way, the flag only disables the 2×2-cell shadow/AO sharing and HOT top-ups): the VRS transplant done the way temporal upscalers can consume — **visibility stays per-pixel always** (every pixel of every leaf tile traces its own primary ray with the inherited cut/t_start; tbuf and all G-buffer guides are bit-identical to a non-adaptive frame — gated), only shading effort adapts, per 2×2 cell (`render::shade_cell`): **COARSE** cells (geometrically coherent vs a per-frame-rotating representative: same `tri_mat`, close t, agreeing normals) reuse the rep's shadow/AO rays via `shade::VisCtl::Capture/Apply` — the record shares the *rays* (same light points, occlusion bits, AO scalar) while N·L/albedo/GGX/reflection stay per-pixel, and **self-declassifies in penumbras and at the light terminator** (fractional captured visibility ⇒ per-pixel fallback, and a below-horizon capture sample poisons `uniform` outright — its "occluded" was never traced, so replaying it onto a neighbor whose own N·L is positive would darken lit pixels; the fractional half is only meaningful at ≥ 2 shadow samples — the interactive 1-spp preset is trivially uniform and relies on temporal laundering instead). Apply burns the rng draws it skips so the stream stays aligned — that is what keeps `spec_hit_t` (the one rng-dependent guide plane) bit-identical, and all six planes are gated bit-exactly by `--check-xess`; **HOT** cells (high in-cell luminance spread) take a second full sample per pixel at its own in-pixel position (`TOPUP_SALT`), averaged locally and **splatted once** (double-splatting would corrupt both accum semantics), with meta/G-buffer writes suppressed so guides stay tied to the reported frame jitter. Never share view-dependent terms. No budget ledger: coarse savings drop frame time and `ScaleCtl` re-spends them as resolution. Counters (`adapt_*` in stats.rs) print as the `adapt:` segment.

`--check-xess` (DLL- and GPU-free, unlike `--check-oidn`) gates the dynamic-res contract: depth-encoding monotone-decreasing roundtrip with exact endpoints (reversed-Z: near → 1.0, sky = far → 0.0), `quantize_res` range-clamp/quantum/aspect, `ScaleCtl` shed/creep/clamp/deadband on a scripted frame-time sequence, the guide nearest-upscale (destination texels bit-equal their nearest source texels at identity/integer/non-integer ratios), the adaptive-rate gates (BASE vs ADAPTIVE same-seed frames: **tbuf bit-identity**, mean relative luminance diff < 2%, cell/primary-ray accounting exact, and — default scene only — coarse/hot/penumbra/rays-saved must-fires at a 2/2-sample quality), and the `--check-dlss` MV/depth/matrix self-test (`mv_check_at`) swept across quantized and odd-dimension render resolutions. `--check-xess --stress n` skips the must-fire halves. Run it (plus `--check` and `--check-dlss`) after touching `xess.rs`, `gpu/xr.rs`, `shade.rs`'s VisCtl paths, `render.rs`'s cell loop, the resolution/present arms in `main.rs`, or the G-buffer fill sites.

## NPPD (Neural Partitioning Pyramids)

The neural denoiser path (`--nppd`, J toggles). **`--nppd` implies `--xess`**: the default NPPD experience is the composition — trace at the `--lock-res` scale (default native 100%), NPPD pre-denoises at that render res, XeSS upscales to the window (J toggles the pre-upscale slot, displacing the OIDN N-cycle's pre placement and vice versa). The standalone window-res mode (mutually exclusive with G/N/X) survives as the automatic fallback when `libxess.dll` is missing, or explicitly via `--nppd --no-xess`. NPPD (Bálint et al., SIGGRAPH 2023, Apache-2.0 code) is a recurrent spatiotemporal network; the deployable graph is produced offline by `tools/nppd-export/export.py` from the pretrained checkpoint (`small_2_spp` by default; the WEIGHTS carry no explicit license — neither the checkpoint nor the exported `.onnx` may be committed; both live in gitignored `SDKs/nppd/`). **Export with `--fp16`** (the deployed default here; fp32 kept as `nppd_small_fp32.onnx` for A/B and the CPU EP): I/O stays fp32 via keep_io_types so nothing outside the graph changes — but know the measured shape of this model: it is **launch-bound on ~1600 tiny shape/slice ops**, not conv FLOPs, so fp16 alone is only ~10% — the real levers are the frozen session dims and (on `--gpu`) device-resident I/O (94 → 26 ms at 1280×736, see tools/nppd-export/README.md). Inference runs through **ONNX Runtime + the DirectML EP** (vendor-neutral D3D12; CPU EP automatic fallback, `--nppd-device` forces): `src/nppd.rs` mirrors the OIDN footprint policy — nothing links the SDK, `onnxruntime.dll` (Microsoft.ML.OnnxRuntime.DirectML NuGet ≥ 1.22 — the exported graph is IR v10) + `DirectML.dll` (Microsoft.AI.DirectML ≥ 1.15 — an old DirectML under a new ORT fails the U-Net's Resize node at run time; verified pairing 1.24.4 + 1.15.4), both dropped in gitignored `SDKs/onnxruntime/bin`, are `LoadLibraryExW`'d at runtime, and ONE export (`OrtGetApiBase`) bootstraps the whole versioned `OrtApi` fn-pointer struct. The `OrtApi` layout was extracted from onnxruntime v1.17.3's `onnxruntime_c_api.h` **by script, in header order** (append-only versioned struct — ordering IS the ABI; `GetApi(17)` returning null is a loud version error, never a mis-laid call; regenerate from the pinned header, never hand-edit). The DML EP contract (mem-pattern off + sequential execution) is baked into session creation, and every session **freezes the padded dims** via `AddFreeDimensionOverrideByName` — that is what lets ORT constant-fold the launch-bound shape chains (~25%); the price is that `NppdContext::set_res` to a different PADDED res rebuilds the session (hundreds of ms of DML re-specialization — rare outside `--lock-res dynamic` ramps, which already print a note).

NPPD mode rules (`main.rs`): the `neural` predicate extends the upscaler frame contract to a CPU-presented denoiser — every frame is a fresh jittered 1-spp **full-depth window-res** hybrid frame (`accumulate = false`, free-running `nppd_seq`, per-pixel jitter, fixed cheap 1-spp preset, fb pinned OFF, no budget frames, never idles), with `ctx.gbuf` a separate window-res `GBufs` and — the first CPU-denoiser mode to do this — `prev_cam` set (`nppd_prev`, its own contract like `dlss_prev`/`xess_prev`): the recurrent state is **backward-warped in Rust** (`nppd::warp_temporal`, bilinear + zeros padding — exactly upstream's `F.grid_sample`; the warp is excised from the graph so DirectML never sees a GridSample op) by `GBufs::mvec`, whose "pixels, current → previous" convention IS noisebase's `motion = prev − current`, i.e. the fetch offset itself (`MV_SIGN`, gated in `self_test`, not empirical). The packer (`nppd::pack_inputs`) reproduces the two Noisebase dataloader transforms the checkpoints were trained on (settled from `noisebase/projective.py`, not empirical): depth = `ln(1 + 1/d)` of the EUCLIDEAN camera distance (view-Z undone through the pixel-center ray; sky = exact 0) and CAMERA-space normals `(n·forward, n·right, n·up)`; diffuse albedo and HDR color pass raw (`normalize_radiance`/`clip_logp1` live inside the graph). Frames are padded to /32 (`PAD_MULT`, the K=5 pyramid needs /16) by **edge replication** (a zero border is a synthetic edge the U-Net would denoise against); the graph writes `output` and `temporal_out` straight into pre-created output tensors (no post-run copy of the ~38-plane state; ~800 MB total staging at 1080p — the recurrent state dominates). The recurrent state resets on every predicate that resets `frame`/the OIDN history (the `hist_stale` list + J itself) and NEVER on camera motion — surviving motion is what the warped state is for; a step in the XeSS composition's dynamic render res also invalidates it (`set_res`), which is why `--lock-res dynamic` prints a note there (each step additionally re-specializes the DML graph). The denoised HDR goes through `render::resolve_hdr` → `present_cpu` and is never written back into `accum`.

**GPU-resident NPPD** (`--gpu --nppd`; XeSS composition only — RR stays excluded, it is itself a denoiser): the whole stage moves onto the tracer's device. `nppd::NppdGpu` opens the SAME frozen-dims session but with `SessionOptionsAppendExecutionProvider_DML1` — ORT executes on the tracer's own `ID3D12CommandQueue` (an `IDMLDevice` is created over the same parent device via a runtime-resolved `DMLCreateDevice`; the `IID_IDMLDevice` GUID is transcribed from DirectML.h) — and the four I/O tensors are bound ONCE over `trace::NppdRes`'s buffers (`CreateGPUAllocationFromD3DResource` + IoBinding; default-heap raw buffers, ALLOW_UNORDERED_ACCESS, resting in UNORDERED_ACCESS, ~340 MB at 1080p/quality). **The `gpu_lock_scale` retirement (2026-07-26) closed a res mismatch here, and the native default (2026-07-31) RE-OPENS it**: this section, the ~340 MB figure, and the export README's own 94 → 26 ms measurement all describe the QUALITY lock — 1280×736 being exactly 2/3 of 1080p padded to /32 — which was the flagless res only during the 2026-07-26..31 quality-default window. A flagless `--gpu --nppd` session now traces at native (1920×1088 padded at 1080p), a larger frozen-dims graph with proportionally larger staging and per-frame cost than the recorded numbers; `--lock-res quality` reproduces the measured configuration. Bindings are fixed by design: the warp kernel reads `state` → writes `warped`, the graph reads `warped` → writes `state` back in place — no ping-pong, no per-frame rebinding, zero per-frame CPU traffic. The staging kernels (`src/gpu/shaders/nppd.hlsl`, root UAVs u23..u26 appended after RP_GBUF) are term-for-term ports of the Rust pure math: `cs_nppd_pack` = `pack_inputs` (edge-replicate padding, Euclidean log-depth, camera-space normals), `cs_nppd_warp` = `warp_temporal` (bilinear, zeros outside, mirrored fp order), `cs_nppd_zero` zeroes the WARPED buffer on reset (the graph fully rewrites `state`, so only what it reads needs zeroing), `cs_nppd_out` crops the planar output into the XeSS color plane while `cs_feed_xess_dm` (feed.hlsl) writes only the guide planes. The frame **splits** around the inference (`D3d::split_frame`: Close + Execute + list Reset on the same allocator — legal, only *allocator* reset needs the fence): list A = trace + pack + warp submitted without a Present, ORT's DML work lands on the same queue behind it, list B = feed + crop + XeSS + tonemap + the one Present — **single-queue FIFO order is the only synchronization** (cross-ExecuteCommandLists visibility is implicit). State resets ride `gpu_reset` (quality/R/X/C, J itself — never motion) via `nppd_state_valid` in GpuContext. A stage failure sheds NPPD (loud line, plain GPU-XeSS), never the session.

**`--check-nppd`** is the DLL-half gate suite (needs `onnxruntime.dll` + the exported model — the only NPPD check with external deps; a `--random-init` plumbing export passes the session/run wiring but fails the quality gates by design). It runs `nppd::self_test` (also run DLL-free by `--check`: pad table, NCHW pack/crop bit-gates, the depth/normal transform anchors — center-pixel `ln 2`, +Y-normal→camera-up — warp identity/integer-shift/bilinear-midpoint/zeros-outside, MV-sign convention), then gates one recurrent step at a time through the exact interactive contract: frame 0 from a reset state (finite, ≠ input, mean |Laplacian| strictly drops, mean ratio ∈ [0.5, 2], state flips valid and is not all-zero), frame 1 static (state advances; **structural, default scene only**: the identity-warped recurrence must not roughen the output), frame 2 under the 0.02·diag dolly with real MVs (gates hold), then `reset_temporal` + re-denoise (the reset path). `--check-nppd --stress n` skips the structural halves, mirroring `--check`. The GPU composition is gated by `--check-gpu`'s M10 section: the pack/warp kernels vs the CPU oracles on the SAME readback inputs (depth ≤ 1e-4 abs with sky bit-zero + must-fire, normals ≤ 1e-5, copy channels bit-equal, warp interior ≤ 1e-6 with the padded border bit-zero — DLL-free, plain compute), then the end-to-end DML1 interop vs the CPU-staged `NppdContext` on the same model (mean rel ≤ 1e-2, measured 5e-4; state-advanced; skipped with one loud line when the DLLs/model are absent). Run `--check-nppd` (plus `--check` for the self_test hook) after touching `nppd.rs`, the NPPD arms in `main.rs`, or the G-buffer fill sites — and `--check-gpu` when the change touches `nppd.hlsl`, `NppdGpu`, `NppdRes`, or the present_trace_xess split; the `OrtApi` struct may only be regenerated from a pinned header, never hand-edited.

## GPU-resident tracing (--gpu)

The full algorithm — quadtree frustum tracing with inherited tmin + node cuts, sky proving, per-pixel leaf rays, full shading, and the hemisphere AO/GI integrator — re-hosted in D3D12 compute (`src/gpu/trace.rs`, `src/gpu/shaders/*.hlsl(i)`), presented through the existing tonemap PS (`SRV_SLOT_GPU`). The CPU renderer is untouched and remains the reference. Toolchain: kernels are `cs_6_5` compiled at startup by a runtime-loaded DXC (`src/gpu/dxc.rs` — `LoadLibraryExW` of `dxcompiler.dll`+`dxil.dll` from the gitignored `SDKs/dxc/`, instance via `GetProcAddress` only, the OIDN/XeSS footprint policy; the windows-rs `Dxc` feature supplies interface types, never the linked import). There is no `#include` — trace.rs concatenates the `.hlsli` prelude into each kernel's source, and each compile unit re-declares the (sometimes differently-typed) buffers its registers mean in that phase. Hard gates at init: RT tier ≥ 1.1, SM ≥ 6.5, `ID3D12Device5` (`trace::require_caps`); anything missing = loud line + CPU fallback, no degraded half-mode.

**Work generation is wavefront**: the CPU records one fixed command list per frame — seed → `depth_full` × (prep-args → ExecuteIndirect level kernel) → leaf + sky fills → (hemi batches) → compose → resolve — and every scheduling decision after the root is a GPU-written counter feeding a dispatch-only command signature (D3D12's DispatchIndirect). Zero readbacks, zero CPU decisions; empty levels/batches dispatch zero groups. Queues are sized to the structural worst case (rects at depth d ≤ 4^d; terminals ≤ 4^depth_full; one cut slot per split) so the primary queues **cannot overflow** — the overflow counter is still gated == 0. Cut-pool exhaustion reuses the parent's slot (an ancestor cut is valid for any descendant frustum — coarse, never wrong; counted, not gated). Records are fixed-size self-contained POD, deliberately work-graph-shaped for a later SM 6.8 backend.

**A leaf tile is ~32 px, not 64 — the leaf kernel is ONE WAVE wide (`trace::LEAF_GROUP_DEF` = 32), and this is a correctness-of-performance invariant, not a tuning knob.** `depth_full` is driven by the WIDER screen axis, so at 1920×1080 a leaf rect is 1920/2⁸ = 7.5 by 1080/2⁸ = 4.2 — about 32 pixels. The kernel originally dispatched **64** lanes per tile and let the surplus half `return` immediately. On a wave32 GPU that is nearly free (the all-idle second wave retires at once); on **wave64 it is catastrophic** — the idle lanes sit inside the SAME wave and waste half its RT throughput. That single mismatch was most of the AMD-vs-NVIDIA gap: per extra sample the leaf kernel cost **2.27× its own reference kernel on RDNA but only 1.24× on Ada**, for identical work (same rays, same `shade_full`). leaf.hlsl now grid-strides over the tile's pixels, so the group width is free. Measured (`--gpu-timing`, leaf+sky, 1080p, 64 → 32): spp=1 **AMD 1.63 → 1.01 ms (−38%)**, NVIDIA 2.24 → 1.38 (−38%); spp=16 **AMD 19.7 → 11.4 (−42%)**, NVIDIA 10.2 → 7.6 (−25%) — a win on BOTH vendors (a 64-thread group reserves registers for 64 threads on Ada too, so halving it doubles the blocks in flight). 32 is a FLOOR: RDNA's wave is 32 lanes minimum, so a 16-wide group is a half-empty wave again (measured worse), and a tile bigger than 32 px simply takes a second full lap — the same lane utilization a 64-wide group would have had. Bit-neutral: NVIDIA's same-seed wavefront-vs-reference image A/B stays exactly 0.00e0 / hot ch 0 across the change. **Two things this ruled OUT, both worth not re-litigating**: the inherited `t_start` is *exactly free* on AMD as a RayQuery TMin (20.719 vs 20.714 ms with TMin forced to 0 — AMD's documented re-origining costs nothing, and equally buys the ray path nothing); and while the fb arm's register pressure was real (`leaf.hlsl`'s `LEAF_NO_FB`: `fb_mode` is a cbuffer value, so a runtime branch inlines shade_split at both call sites and the kernel's VGPR allocation is the max of the two — worth −11% AMD / −16% NVIDIA, and kept), it was a *shared* cost, not the AMD-specific one.

**The intersector split**: frustum machinery (`bound_query`/`refine_cut` in `frustum.hlsli` — term-for-term ports with the same slacks, `precise` on the accumulating math, per-lane groupshared stacks) runs on our own uploaded BVH; **every actual ray is DXR inline RayQuery** against a driver BLAS/TLAS (geometry OPAQUE, no cull — möller-trumbore is two-sided). **`PrimitiveIndex() == tri` is NO LONGER the shipping configuration**: the scene is one BLAS per maximal BVH subtree of ≤ 64k triangles by default (see `--no-blas-split` in the command block — it defaulted on because a single 34.4M-tri BLAS removes the device on Intel), so the primitive index indexes a CHUNK and every site reads `tri_of(InstanceID(), PrimitiveIndex())` — the chunk remap by default, the identity under `--no-blas-split`. Both GPU pipelines and all of `rt.hlsli`/`dxr.hlsl` route through that one function; if you are adding an intersector site, it is the only correct way to get a triangle id. Leaf primary rays consume the inherited claim as `TMin = t_start` (sound: hit acceptance is strictly beyond TMin and tc is shaved 1e-4 at the advance); cut-*seeded* rays (`intersect_multi`) don't exist on the GPU in the DEFAULT configuration — RT-core root traversal replaces them, and the cut is consumed only by the bound queries. The ONE exception is `--sw-rays` (see the command block): rt_sw.hlsli replaces the RayQuery bodies with bvh.rs's software loops and leaf primaries DO seed from the tile cut — built as the measurement lever that settled the question, and the answer is that hardware root traversal wins ~2× everywhere (even Intel), so the default stands. There is deliberately NO cut-vs-full-tree bound-compare gate: the cut legitimately encodes ancestor culling (e.g. the hemi root's tangent half-space) that a descendant's own frustum can't express, so a conservative full-tree query can report a *lower* bound than the tighter, correct cut. Cut fidelity is gated where claims are consumed: `false-empty` and `tmin-overshoot`, both with real RayQuery rays.

**Hemi on GPU** (H cycles off → AO → GI; still frames, wavefront path only): its own wavefront over `HemiPointRec`s appended by the leaf pass (fb mode splits shade: `partial` = ambient-free color, `ambw` = kd; `compose` is the single accum splat site), processed in `HEMI_BATCH` slices whose per-batch counter/pool reset bounds transient memory (queues sized to batch × 4^(depth−1) — bounded, cannot overflow). Results land in a fixed-point (2^18) atomic accumulator — order-independent adds ⇒ run-to-run determinism; leaf-ray RNG is keyed by (pixel, hemi path, frame, salt), never the atomic slot index. All hemi.rs soundness contracts carry over verbatim (apex t_start = 0 not eps, root cut `[0]`, blocked-must-subdivide, AO's None-means-open-within-radius); `sky_cell` is ported iteratively (no HLSL recursion).

**`--check-gpu` is the GPU test suite** (needs real hardware, unlike the other checks): M1 dispatch plumbing; the vanilla reference tracer vs the CPU plain reference (visibility classification + 64-frame radiance A/B — *statistical*, hardware watertight intersection ≠ möller-trumbore at edges and the RNG streams differ by design); the resolve pass vs accum; then the transplanted **exact-zero gates**: **claim-violation** (THE soundness contract, asserted DIRECTLY: the tile's inherited `t_start` must not exceed the true nearest hit, ground-truthed as the EARLIEST t either intersector reports — the most pessimistic bar available, since a hit either one finds is a real triangle inside the tile frustum), false-sky, tmin-overshoot, hybrid-extra, exactly-once pixel coverage (an info sentinel), and queue accounting (leaf+sky rects partition the screen exactly, tile queues drained, overflow 0). **The wavefront-vs-reference comparison is NOT exact-zero, and must not be made so** — see the AMD note below; it is `claim-violation` that guards the contract, and the old `wave_t > ref_t => overshoot` inference was a *consequence* of an overshoot, not the invariant (a consequence has other causes). The same-seed image A/B is gated in three parts that together are strictly stronger than the old mean/max pair: mean < 1e-5 (a systematic shading bug moves it), a HOT COUNT of channels past 1e-2 bounded by the same 0.05% two-intersector allowance (a localized bug lights up far more), and all-finite (a catastrophic single pixel the counts would miss). The wavefront runs the same quadtree as the CPU (its tile counts matched `--check`'s when hand-compared; no gate asserts that).

**AMD RT hardware re-origins the ray at TMin** (measured on a Radeon AI PRO R9700; NVIDIA does not). Two consequences, both benign, both permanent — do NOT "fix" either by tightening the gates back:
  1. *A grazing shared edge can flip accept/reject with TMin.* The reference kernel (TMin = 0) and a leaf ray (TMin = t_start) are the SAME intersector but not the same ray, so at a pixel where a ray grazes a triangle edge they can hit different geometry. Measured: default scene px (381,258), cpu t 28.546 / gpu-ref t 21.897 / wavefront t 28.546 — i.e. the wavefront agrees with the CPU and the *reference* takes the edge, while `t_start` (6.499) sits 15 units short of either. Such pixels are ALREADY known-disagreeing from the reference-vs-CPU gate, so `--check-gpu` masks them out of the wavefront comparison by that mask and bounds their count at the same 0.05%.
  2. *The reported t differs by 1-2 ulp for one ray at two TMins.* That shifts the hit point by ulps, which shifts the shadow/AO ray origin, which at a grazing angle flips a BINARY occlusion bit — 2 ulp of geometry becomes ~0.02 of color. Measured 18 hot channels of 1.44M on the default scene, 3 on San Miguel. No amount of correct code prevents this: a discrete decision on a continuous input is discontinuous by construction. It is why the image max is a bounded COUNT and not an absolute limit.
Both `--check-gpu` and `--check-dxr` pass on NVIDIA and AMD (`--prefer-amd`), on the default scene, `--stress`, and San Miguel; on NVIDIA every counter above is still exactly 0 / 0.00e0, so nothing was traded away where the hardware permits bit-identity. Hemi gates run on a CPU-generated probe set with `FLAG_VERIFY` (psa-viol / false-empty / tmin-overshoot == 0, PSA accounts to π in H.w) plus multi-seed A/Bs at the CPU suite's tolerances (AO mean |Δ| < 0.02, signed < 0.005; GI mean rel < 5% vs BOUNCE_Q references), a frame-level accounting gate (hemi points == hit pixels), and a bench (1080p, RTX 4090: GPU hybrid ~2.0 ms, GPU plain reference ~0.3 ms, +hemi-GI ~113 ms, CPU hybrid ~36 ms — the plain-reference number is the honest baseline: RT-core root traversal is cheap enough that software frustum queries cost more than they save for primary visibility; the wavefront's shallow levels underutilize the GPU, the noted optimization being a wave-cooperative level kernel). `--check-gpu --stress n` skips the structural must-fires, mirroring `--check`. In the interactive window `--gpu` keeps R (wavefront vs reference — both write the pack, so the A/B works under RR/XeSS too), C (on-GPU verify of the current view at the SESSION render res — clobbers and resets the accumulation), H (plain sub-mode only; upscaler frames pin fb OFF), P (screenshots the upscaler's window-res output; plain saves the render-res hdr at its own dims), G/X (toggle the session's WIRED upscaler vs plain — wiring is init-time, you can't toggle into an upscaler the session didn't start with), and 1–3 (plain sub-mode; upscaler frames pin the 1-spp quality); T/O/N/M print notes; SPACE/F leave the arm live (the mode cycle — --gpu is no longer a locked session). `FRUSTRACER_STAB=1` works in the GPU arm (healthy statics: RR ≈ 0.14/255, XeSS ≈ 1.0/255 — both match their CPU-fed baselines). Run `--check-gpu` after touching anything under `src/gpu/shaders/`, `src/gpu/trace.rs`, or `src/gpu/dxc.rs`; run `--check` too when the change mirrors CPU semantics (the kernels are ports — the CPU files are their source of truth).

**GPU-born G-buffers → the chain's wired upscaler.** With `--gpu` the upscaler composes ON the tracer, whichever level the chain wired: DLSS-RR, FSR4-RR, XeSS, or FSR3 (`--no-upscale` = plain); OIDN stays CPU-only. The render res is **LOCKED per session** (`--lock-res`, `quantize_res`-clamped into the upscaler's queried range; `TraceGpu` is built once at that res — no DRS, no per-frame rw/rh). The default is the session's one — **native (100%)**, `xess::DEFAULT_LOCK_SCALE`, the SAME scale the CPU renderer takes (the ONE-scale-for-every-arm rule dates to the 2026-07-26 `Opts::gpu_lock_scale` retirement: an upscaler is the reconstruction stage in every arm, so which tracer happens to be live is not a reason to change how many pixels get traced, and F/SPACE cycling arms never moves the render res. The unified default was quality 2/3 from then until 2026-07-31, when it moved to native — a flagless GPU session is DLAA-shaped again, the upscaler denoising/antialiasing without upscaling). `--lock-res quality` spells the 2/3 arm, and any explicit `--lock-res` means the same thing in whichever arm renders (F toggles between them — the res discontinuity is already a history-reset latch). **Two notes from the quality-default window, both now mostly moot at native but load-bearing for reading numbers recorded 2026-07-26..31.** (1) **"Plain" meant two resolutions there**: toggling the wired upscaler OFF mid-session (G/X/K) leaves the tracer at its LOCKED res, so under a sub-native lock the plain image is a stretched blit through `fullscreen_to_backbuffer`, `P` saves the locked dims, and `C` verifies at them (the title's `grw×grh` field is the ground truth). At the native default the locked res IS the window res, so flagless G/X/K plain is full-res again; the stretched-blit behavior remains reachable via any explicit sub-native `--lock-res`. (2) **`main::vendor_defaults`' Intel entry is measured at the res the flagless session traces again** — during the 2/3 window it wasn't, and the arms don't scale together (`depth_full` is 6 at 1920 AND 1280, so the level ladder is res-independent while `leaf`/`sky` and all of `dxr-rays` are per-pixel); the RESOLVED DEBT paragraph in `vendor_defaults` records that history. The leaf/sky/reference kernels write a **SPLIT G-buffer pack**: `GBufCore` at u15 (`trace::GBUF_STRIDE` = **16 B/px** — mv.xy | view_z | prev_z), stored by every upscaler session, plus `GBufExt` at u32/`RP_GBUF_EXT` (`GBUF_EXT_STRIDE` = **72 B/px** — normal+rough | diff_alb | spec_alb+spec_hit_t | sig | sig2), stored only under **FLAG_GBUF_EXT** (a wired RR/FSR-RR feed, or NPPD; derived per frame like `fsr_sig`, since the tracer is built BEFORE `wire_feed` runs and `--quinlight` wires several kinds at once). It was one 88 B/px record until 2026-07-24; splitting it is worth **−33% of the B70's world frame** because an XeSS or FSR 3.1 session reads 12 of those 88 bytes and now stores 16 instead of 88 — see the pack-split paragraph in Profiling for the numbers, the write-allocate mechanism, and the two probe traps it exposed. Both halves go through `gbuf_write_hit/_sky` in trace_common.hlsli — term-for-term ports of render.rs's `write_gbuf_*`, MVs from `project_prev` (the HLSL `CamBasis::project` against a prev-camera CB block; FrameCb is 1488 B / CB_STRIDE 1536 — the base 304 B, minus the 32 B of rect-light rows the sun's three replace, plus the spp block and its MAX_SPP jitter table, plus 9 float4 rows of order-2 SH sky; near/far ride the prev rows' w slots, and `scene.eps`/`ao_radius` were rehomed onto `sun_e.w`/`sun_l.w` when the rect light's rows died). `shade_split` exposes the capture as a `PrimSurf` out-param filled at lap 0 only — a pure copy of already-computed values, **zero added rng draws**, so the same-seed wavefront-vs-reference bit-identity gates hold. Every write is gated by `FLAG_GBUF`: the plain-session pack is a GBUF_STRIDE-byte dummy and root-descriptor UAVs have no bounds check — that branch is memory safety, not an optimization. **The last three pack fields are FSR-RR-only** (`FLAG_FSR_SIG`, armed solely by a `FeedKind::FsrRr` wiring): `mv.z` = the prev-camera linear view-Z of the same hit point (sky stores CAM_FAR, so the denoiser's depth delta is exactly 0), `sig` = the f16x2-packed **demodulated** `direct_d`/`direct_s`, and `sig2` = the same packing of `ao` and the demodulated `ind_s` (`shade.hlsli`'s lap-0 export — assignment-only, zero rng draws; the demodulation divides by the un-floored WIRE F0 via `sqrt_wire`, `fsr::split_signals`' twin with the pack's f32 storage in place of the CPU's f16 hop). `ind_s` is the one capture the flattened GPU DFS cannot read off a single lap: a reflected ray that hits glass continues its own transmission chain, so `shade_split` carries an `in_refl` tag through its lap/stash state and accumulates EVERY contribution descending from the root's reflection child (the CPU gets it for free — its `tput * rcol` is one recursive call). Every other wiring writes zeros there and its pack bytes are unchanged — gated by a same-seed **FLAG_FSR_SIG on/off accum bit-identity** pass. Per frame one **feed kernel** (`feed.hlsl`; texture UAVs u16..u27 on a second range of the u14 descriptor table — the range grew for `cs_feed_fsr_rr`'s five extra planes, shifting NPPD's root UAVs to u28..u31, whose nppd.hlsl register literals must stay in lockstep with `NPPD_REG_BASE`; wired per session by `trace::wire_feed` with a typed-UAV-store format gate) fans pack+accum into the live upscaler's input planes **in place** — rr.rs/xr.rs/ffx_rr.rs/ffx_up.rs planes gained ALLOW_UNORDERED_ACCESS, and `record_feed`'s NPSR→UAV→NPSR transitions double as the write→read sync and keep the NGX/XeSS state-at-use contracts truthful. The CPU convert-and-upload paths (`rr/xr/ffx_rr/ffx_up::record_upload`) still serve the CPU renderer; the GPU chains (`present_trace_rr`/`_xess`/`_fsr_rr`/`_fsr3`) have **zero per-frame CPU→GPU traffic**. Upscaler frames follow the CPU contracts exactly: fresh 1-spp (accumulate off, free-running index, frame-uniform Halton via `frame_jitter`), q pinned (1 shadow / 1 AO / reflections / fb OFF), reset on quality/R/G/X/C but never on motion, and ONE camera pose feeds both the shader MVs and `frame_constants`' prev matrices. In RR sessions the whole list — ExecuteIndirect, DXR RayQuery, AS builds, feed, the raw-NGX evaluate — executes on the one native queue (the SL proxy died with the retirement). XeSS depth is a `precise` HLSL port of `view_z_to_clip_depth` (the quotient must be `precise` too or DXC lowers it to rcp+mul); RR depth is raw view-Z passthrough. `--check-gpu` gates (all DLL-free): the pack read back into CPU `GBufs` through the **exact existing `dlss::mv_selftest`** at odd dims (533×400) plus coverage (every px view_z > 0, sky == far bit-equal); XeSS feed depth vs the CPU encode ≤ 4 f32 ulp (D3D's divide is ~2.5 ulp even `precise`; observed max 2) with **sky bit-equal 0.0** and mvec/color ≤ 1 f16 ulp; RR feed depth **bit-equal** and EVERY other plane gated too (color/normal-rough/spec-hit ≤ 1 f16 ulp, albedo/spec-albedo ≤ 1 UNORM LSB) so a wiring-order swap cannot pass the suite; the **FSR3 feed** through the same `gate_xess_feed` over `FeedKind::Fsr3`-wired planes; the **FSR4-RR feed** over all eleven planes (see the `--check-dxr` list below — same `gate_fsr_rr_feed` helper). One tolerance there is deliberately NOT an ulp count: the **residual** is a near-cancellation (`color − dd⊗kd − ds⊗f0 − ao·amb(n)⊗kd − is⊗f0`), so its f32 slop is bounded by the CANCELLED terms (every subtracted product enters the bound), not by the tiny result — gating it at 1 f16 ulp of the remainder would shrink the limit toward zero exactly where the error doesn't, and lit pixels whose residual lands near 0 fail on ordinary rounding (measured: 19/640k channels). The bound is absolute — a few f32 ulps of the largest cancelled term plus the residual's own f16 step — and still catches any wiring/formula error, which moves the residual by the size of a whole term. The sky gates carry `must_fire` anti-vacuity (sky px > 0 on the default scene). The rr/xr upload rings (and the XeSS post-denoise readback) allocate lazily on first CPU use — GPU-fed sessions never commit that memory. Run `--check-gpu` + `--check` + `--check-dlss` + `--check-xess` + `--check-fsr` after touching feed.hlsl, nppd.hlsl, the pack helpers in trace_common.hlsli, shade.hlsli's PrimSurf, rr.rs/xr.rs/ffx_rr.rs/ffx_up.rs plane creation, or the present_trace_* chains. `--gpu --nppd` composes the GPU-resident NPPD stage into the XeSS chain (see the NPPD section — the frame splits around the ORT run on the shared queue; gated by M10).

**Two wavefront queue treatments in `level_finish`, both revertible via `FR_ABL`** (`wavefront_ablation_defs` pastes them into the TILE-RECURSION unit ONLY, so toggling one cannot perturb the reference/leaf shader cache — the inverse of the `abl_defs` probe-reach trap, and legitimate here precisely because these two defines change nothing outside `wavefront.hlsl`). **(1) The terminal split refines no cut** (`FR_ABL=oldcut` restores it): a split whose every child is a leaf has no `TileRec` to inherit a cut, and hardware `RayQuery` accepts only `TMin`, so the refined cut had no consumer — it was allocated, written, and dropped. The predicate is `ceil(parent_extent / 2) <= LEAF_TILE` on BOTH axes, which is the largest child extent and therefore conservative on odd and one-pixel dimensions; a MIXED split (one child under the cap, a sibling over it — `2*LEAF_TILE+1`) falls through to the normal path. Terminal leaves get `frontier_root()`. **`SW_RAYS_LEAF` disables it**, because that arm's rays DO consume the cut — which is also why the continuation control arm is not quadtree-identical (see the lever entry). Measured on the 800x600 gate frame: materialized cuts **257 -> 65** at 257 splits. **(2) Homogeneous child batching** (`FR_ABL=nobatch` restores the loop): a regular level emits four internal children or four leaves, so one `InterlockedAdd` reserves the contiguous queue range instead of four contended ones; the counter total and per-child overflow reporting are preserved exactly, and mixed splits keep the original per-child loop. Both are PIXEL-IDENTICAL by construction and verified so — `--check-gpu` reads claim-violation / false-sky / tmin-overshoot 0 and same-seed image `0.00e0` in both the shipping and the `FR_ABL=oldcut,nobatch` arm.

**GPU structure replay** (`--no-replay` kills, the CPU replay's twin). A still/converging frame whose `CamBasis` bit-equals the previous producing frame's re-dispatches the persisted terminal queues instead of re-running seed + the level ladder — the ladder is the wavefront's ~0.5–1.5 ms fixed cost, and replay deletes it (measured **−43%** GPU frame span on a still 4090 spin, 1.27 → 0.72 ms; the level regions read ~0 occupancy and a `wavefront-replay` `--gpu-timing` region appears). Soundness is the CPU replay's exactly: the terminal structure (`qleaf`/`qsky`/`cut_pool` + `CTR_LEAF`/`CTR_SKY`/`CTR_CUT`) is byte-intact between producing frames (only `cs_seed` + the ladder write it; leaf/sky/hemi passes only read it, hemi rebinds transients at u7/u9, and the reference/feed/nppd units declare no counters or queues — so the R toggle round-trips replay for free), and it is a pure function of (scene, BVH, basis, rw, rh) — spp/jitter/frame/fb/quality/clouds all ride the CB, so a replay frame's shading is fresh and the result is BIT-IDENTICAL to a fresh trace. Implementation: `FrameParams.replay` (per-call, never a global — headless gates set false so nothing silently switches paths under a measurement); `TraceGpu::last_struct` (set by every `record_wavefront`); `cs_seed_replay` (zeroes every counter EXCEPT the terminal three); `record_terminal_fills` factored out and shared by `record_wavefront`/`record_wavefront_replay`; `record_frame` dispatches replay when `p.replay && last_struct == Some(p.cam)`. **Two invalidation hazards, both handled**: `run_hemi_probes`' `cs_seed_probes` zeroes `CTR_LEAF`/`CTR_SKY` (calls `invalidate_replay`), and a producing frame recorded-then-aborted (present error) would leave the key claiming a structure the GPU never built (main.rs calls `gpu.invalidate_replay()` on every GPU present-error arm). No DXR replay (that pipeline has no quadtree structure). Gated in `--check-gpu`: frame-0 tbuf/info/accum bit-identity + exactly-once coverage + terminal-count preservation + the ladder provably not running (`CTR_SPLIT`/`CTR_TILE_A`/`CTR_TILE_B` == 0), a warm jittered frame-1 bit-identity via the re-produce sequence, and the auto-predicate must-fire through `record_frame` (same-basis replays, a 0.02·diag dolly re-produces). Touch `cs_seed_replay`/`record_terminal_fills`/`record_wavefront_replay`/the record_frame predicate → run `--check-gpu` (+ `--sw-rays`, the cut_pool-consuming replay flavor).

## DXR pipeline (--dxr / F)

The by-the-book DXR flavor next to the wavefront tracer — and **the DEFAULT render mode on NVIDIA/AMD adapters**: an RTPSO state object, a shader binding table, and `DispatchRays` with raygen / closest-hit / miss shaders (`src/gpu/dxr.rs`, `src/gpu/shaders/rt_dxr.hlsli` + `dxr.hlsl`). **On an Intel adapter the flagless default is the wavefront tracer instead** (`main::vendor_defaults` — the vendor-keyed defaults live there, TWO since 2026-08-01: the Intel mode preference below, and Intel `--dxr-inline` = 2; its doc comment carries the bar for adding more): the mode preference measurably INVERTS by vendor — `--spin path` 1080p GPU frame span, warm-cache interleaved medians, DXR/wavefront = **5.14×/2.61× on the B70** (default/stress — Arc's RT is weak relative to its shader cores, so proving space empty with shader-core work is worth more than RT-core root traversal) vs **0.54×/0.72× on the 4090** — and a crossing ratio, not a mere difference in margin, is the bar for a vendor default. (W2 caveat: that table is the `--dxr-inline 0` all-TraceRay pipeline; under the mode-1 default the Intel ratio straddles 1.0 by scene at spp=1 — 1.34×/0.81×/0.94× default/stress/SM-lp — most of the old gap having been secondary TraceRay dispatch. THE WORLD re-measure (2026-07-22) reads wavefront 4.15 ms tracer / 5.21 span vs mode-1 DXR 3.80/4.88 — DXR 0.92× on the flagless scene itself, so the policy now rests on the ≥spp-3 regime + the wavefront's H/R/C/O + an imperceptible ~8% margin, never on spp=1 perf — see the `--dxr-inline` paragraph and `main::vendor_defaults` for the full table. THAT PARITY IS RETIRED (2026-08-01): the leaf frontier + pack split + compose skip were wavefront-side, and the world re-measure now reads wavefront 3.30 span parked / 3.4-3.5 moving vs mode-1 DXR 5.36 either state — the hybrid 1.5-1.6× moving or parked, so the perf grounds are back; vendor_defaults carries the dated table.) The policy fires only when the DXR default is actually in force (never against `--cpu`/`--gpu`/`--dxr` — `Opts::mode_explicit` — nor against the OIDN/NPPD parse-time opt-outs), keys off the PICKED adapter (`GpuContext::adapter_vendor`, a fact) rather than `--prefer-*` (a request that can fall back), and leaves `opts.dxr` armed so a wavefront init failure still falls to DXR before the CPU tracer (session() reads the pair as a preference order — the wavefront carries the software BVH for its frustum queries, so a mega-scene can exceed VRAM there while DXR still fits). Headless paths (`--check*`, `--spin`) never consult it — benchmarks must not have defaults move under them. On its home vendors the default session builds DXR EAGERLY at startup (`--cpu` opts back into the CPU frustum-tracer — it clears both GPU modes, later flags winning; explicit `--gpu` also wins); **F** still toggles it live against the CPU tracer (the F path lazily builds when the eager init was skipped; a failed init is memoized in `dxr_failed` and the session continues as the CPU renderer with the chain's upscaler wired — the loud-fallback shape). **It composes with the chain's WIRED upscaler**: DXR-fed DLSS-RR / FSR4-RR / XeSS / FSR3, whichever level the session probe wired — DXR availability follows the wiring because every upscaler context (the raw-NGX RR session included) lives on the one native device/queue, so there is nothing to re-create. In an upscaler sub-mode the pipeline traces at the **session-locked render res** (the same `--lock-res`/`locked_render_res`/`lock_scale` math as `--gpu`; default native = 100%; `--lock-res dynamic` prints a note and locks at that default — `DxrGpu`'s buffers are sized once, no DRS) and each frame follows the `--gpu` upscaler contract verbatim: fresh 1-spp (`accumulate` off, free-running `dxr_up_idx`, frame-uniform Halton via `frame_jitter`), q pinned to `upscaler_1spp`, reset on quality/F/G/X/K but never motion, ONE camera pose feeding both the shader MVs (`dxr_prev_cam` — its own contract, like `gpu_prev_cam`) and the upscaler's constants; DispatchRays + feed + evaluate/denoise/upscale + tonemap ride one command list (`present_dxr_rr`/`present_dxr_xess`/`present_dxr_fsr_rr`/`present_dxr_fsr3` in gpu/mod.rs, clones of the `present_trace_*` chains — the DXR feed reuses the same `feed.hlsl` kernels compiled at this pipeline's cs_6_3 floor, and `trace::wire_feed_targets`/`record_feed_dispatch` are the shared owner-independent halves both tracers wrap; the FSR chains share `record_fsr_rr_sequence`/`record_fsr3_upscale` with the CPU-fed presenters). **Inside DXR mode G/X/K toggle wired-upscaler ↔ plain** (the `--gpu` semantics; each F enable restores the session default), the CPU arms' DLSS/XeSS/FSR state stays untouched (F-off resumes it, with reset latches for the res discontinuity), and N/J stay mutually exclusive (CPU-side denoisers). Plain sub-mode frame semantics mirror the `--gpu` plain sub-mode: `while_moving()` quality with `frame` pinned at 0 while moving, accumulate + converge when still, re-present the resolved image at MAX_SAMPLES. Every DXR frame drops the temporal ring (`tr.end(false, ..)`) so F-off never resumes CPU tracing against stale claims. O/H/C are CPU/wavefront features and no-op with a note; P saves the upscaler's window-res output in upscaler sub-modes, the render-res hdr in plain (the `--gpu` P behavior). The G-buffer capture is `chs_shade`/raygen calling the same `gbuf_write_hit/_sky` pack helpers the wavefront uses — pure copies after `shade_full` returns, zero rng draws, FLAG_GBUF-gated so plain sessions (GBUF_STRIDE-byte dummy pack) stay bit-identical.

**Reuse is the design**: shading parity is inherited, not re-ported — the DXR library pastes the SAME `trace_common.hlsli` + `shade.hlsli` the compute tracer runs, and the trace primitives (`trace_closest`/`occluded_q`/`transmit_q`) are the only swap point: under the `--dxr-inline 1` DEFAULT the secondaries are `rt.hlsli`'s inline RayQuery bodies pasted ahead of `rt_dxr.hlsli` (whose TraceRay flavors compile out), while `--dxr-inline 0` compiles `rt_dxr.hlsli`'s `TraceRay` flavors so every ray rides the hardware pipeline — either way the closest-hit shader (`chs_shade`) runs `shade_full` unmodified. Never fork the shading: shade.rs is the source of truth, shade.hlsli its one GPU port. The compute root signature doubles as the DXR **global** root signature (same registers; identifier-only SBT records, no local root signatures — nothing is per-hit-group with one mesh), `SceneGpu` provides the buffers + BLAS/TLAS, `FrameCb::base/with_frame` the constants, and the resolve kernel + tonemap present ride `SRV_SLOT_DXR`. SBT layout (`dxr.rs` byte offsets == `rt_dxr.hlsli` TraceRay indices — keep in lockstep): hit groups [HgShade, HgHit (bare hit info for the reflection/transmission continuations — routing them to chs_shade would double-shade), null (occlusion, SKIP_CLOSEST_HIT_SHADER + miss_shadow) — on alpha-masked scenes the three groups gain the `ah_*` cutout any-hit shaders and the null record becomes the any-hit-only `HgOcclude` (see Real scenes)], miss [radiance, shadow, hit_info]. `MaxTraceRecursionDepth` = 1 at the `--dxr-inline 1` default (raygen's primary is the only TraceRay — chs_shade's secondaries are inline) and 2 under `--dxr-inline 0` (raygen level 1, chs_shade's shadow/AO/reflection/transmission rays level 2); either way the CPU's recursion (one reflection bounce + the transmission chain) is shade_split's flattened DFS loop inside chs_shade, not payload recursion. The pixel's RNG stream rides the payload (raygen draws the jitter, chs_shade continues the stream — the reference kernel's one-stream contract). Caps: tier 1.0 + SM 6.3 (`dxr::require_caps`; the wavefront needs 1.1/6.5 — this pipeline runs on strictly more hardware).

**The cloud shading caches ride the reuse too** (`--cloud-shadow` / `--sky-lod`, default ON — see the command block): the DXR library injects `CLOUD_SHADOW_N` + `SKY_LOD`/`SKY_LOD_LOG` into its assembly and pastes `SKYLOD_HLSLI` after `trace_common.hlsli`, which arms trace_common's cached `cloud_sun_transmittance` (u6) + `sky_radiance_lod` (u5) for every shade path — parity inherited, not re-ported (u5/u6 are the wavefront's tile queues, unbound in the DXR root signature, so no root-signature change). `DxrGpu` snapshots the two levers at construction (like `TraceGpu` — a mid-process A/B can't desync a kernel from its fill), compiles `cs_sky_lod`/`cs_cloud_shadow` from the shared `trace::sky_unit_src`, fills `cloud_grid` in `write_cb` via `clouds::shadow_grid_row`, and dispatches both fills in `record_frame` (the ONE DispatchRays site — session chains AND `--check-dxr` route through it, so the fill-must-precede-the-read contract holds structurally) before `DispatchRays`, binding `cloud_shadow`@u6 / `cloud_lod`@u5 which persist for the ray dispatch. The two sky consumer sites (`miss_radiance` + the inline-mode-2 raygen miss) take `sky_radiance_lod(dir, id.x, id.y)` under `SKY_LOD > 1`. Gated by `--check-dxr`'s on-vs-off same-seed A/B (sky/image mean-rel bounds; EXACT off-vs-off) — measured identical to the wavefront's numbers (sky 8.5e-3, image 1.4e-3), the radiance A/B unmoved at 0.046%.

**`--check-dxr`** (needs a real RT GPU + the DXC DLLs, like `--check-gpu`): DXC/caps/RTPSO/SBT/AS init, then one unjittered DispatchRays frame vs the CPU plain reference (hit/sky class-mismatch ≤ 0.05%, rel-t > 1e-3 ≤ 0.01% — statistical: watertight hardware intersection ≠ möller-trumbore at edges; must-fire hit > 0 and sky > 0), a 64-frame converged radiance A/B (per-channel mean rel ≤ 2%, HDR finite/non-negative), and the resolve link (texel == accum/samples within f16). Then the composition gates (all DLL-free, the transplanted `--check-gpu` M7–M9 shapes on a `gbuf_full` pipeline at odd dims 533×400 under the upscaler contract): pack coverage (every px view-z > 0, sky depth far bit-equal, sky must-fire) + the exact existing `dlss::mv_selftest` on a frame-A/dolly-frame-B pair, the XeSS feed (depth ≤ 4 f32 ulp with sky bit-equal 0.0, mvec/color ≤ 1 f16 ulp), the FSR3 feed (the same `gate_xess_feed` over `FeedKind::Fsr3`-wired `Fsr3Resources` planes — pins the kind mapping and wiring order), the **FSR4-RR feed** (frame B re-traced with `FeedKind::FsrRr` wired, which arms FLAG_FSR_SIG: accum must come back BIT-IDENTICAL — the sig capture is assignment-only — then all eleven planes vs pack-readback oracles: dd/ds/ao/indirect-spec bit-equal the pack's sig+sig2 f16 halves, the indirect-specular plane's A channel within 1 f16 ulp of the pack's spec_hit_t lane (a typed UAV store to a FLOAT16 format has rounding latitude the CPU's `from_f32` does not — the same tolerance `gate_rr_feed`'s spec-hit plane takes; measured max 1), linear depth bit-equal, clip depth ≤ 4 ulp + sky 0.0, mvec ≤ 1 f16 ulp with sky depth-delta exactly 0, oct-normals/albedos ≤ 1 LSB, residual ≤ 1 f16 ulp with the wire factors taken from the PLANE BYTES the GPU stored — GPU sqrt has 1-ulp latitude, a recompute can land across a quantization boundary — plus sky-sig-zero and must-fires on sig, an occluded-AO pixel and a reflection pixel), and the RR feed (depth bit-equal, every other plane at its storage tolerance — the shared `gate_xess_feed`/`gate_rr_feed`/`gate_fsr_rr_feed` helpers both suites call). `--check-dxr --stress n` skips the must-fires, per convention. Run it after touching `dxr.rs`, `rt_dxr.hlsli`, `dxr.hlsl`, the shared sources they paste (`trace_common.hlsli`, `shade.hlsli`, `resolve.hlsl` — those also need `--check-gpu`), `feed.hlsl`, rr.rs/xr.rs/ffx_rr.rs/ffx_up.rs plane creation, or the `present_dxr_*` chains.

**`--dxr-inline 0|1|2` (DEFAULT 1 — the W2 promotion): which of this pipeline's rays ride recursive TraceRay vs inline RayQuery, and the measurement that located WHERE the DXR overhead lives.** Mode 1 — THE DEFAULT — keeps the primary `TraceRay` → `chs_shade` but compiles `rt.hlsli`'s RayQuery primitives in place of `rt_dxr.hlsli`'s TraceRay flavors (both implement the same `trace_closest`/`occluded_q`/`transmit_q`, which is what makes the swap surgical), so every secondary `shade.hlsli` fires — shadow, AO, reflection, the transmission chain, the translucency back ray — runs INLINE inside the closest-hit shader and `MaxTraceRecursionDepth` drops to 1. Mode 2 additionally takes the `DXR_INLINE_SEC == 2` arm in `dxr.hlsl`'s raygen: no TraceRay anywhere, DispatchRays as a bare launch grid over the reference loop (raygen + chs_shade + miss_radiance fused; the relief re-march rides inside the inline `trace_closest` exactly as in reference.hlsl). Mode 0 is the pre-W2 all-TraceRay pipeline, kept as the A/B escape — it assembles the exact pre-lever source list, so the legacy arm is byte-identical by construction. Armed modes compile at lib_6_5 and need RT tier 1.1 (the wavefront's own floor; lesser hardware degrades to 0 with one loud line — a preference, never the `--fsr4` shape); the RTPSO/SBT layout is unchanged in every mode (unreached hit groups stay exported — identifier-only records, no cost); only a departure from the default prints a lever line (the blas-split precedent), and an illegal value exits 2. The knob rides `dxr::set_inline_mode` (the `texture::set_aniso` knob idiom), so every path that builds a `DxrGpu` — session, F/SPACE, `--spin --dxr`, `--check-dxr` — reads one source of truth, and the suites now gate MODE 1 BY DEFAULT (run `--check-dxr --dxr-inline 0` for the TraceRay arm, the `--no-blas-split` pattern). MEASURED (`--spin path` 1080p spp=1, GPU frame span ms, warm-cache interleaved medians; B70 repeats ±0.002):

| | mode 0 (ship) | mode 1 | mode 2 | wavefront |
|---|---|---|---|---|
| B70 default | 9.051 | 2.353 (−74%) | 1.411 (−84%) | 1.760 |
| B70 stress | 5.297 | 1.635 (−69%) | 1.215 (−77%) | 2.021 |
| B70 san-miguel-lp | 6.748 | 1.938 (−71%) | 1.292 (−81%) | 2.054 |
| 4090 default | 1.335 | 0.261 (−80%) | 0.289 | 2.088 |
| 4090 stress | 0.789 | 0.253 (−68%) | 0.267 | 1.084 |
| 4090 san-miguel-lp | 1.175 | 0.338 (−71%) | 0.343 | 1.000 |

The distilled claim, numbers attached: **Arc executes DispatchRays and inline RayQuery just fine; what it hates is re-entering the scheduler from a hit shader.** Recursive TraceRay secondaries multiply the tracer 4.4–6.4× on the B70 vs 3.0–4.6× on the 4090 (scene-matched ~1.4–1.5× NVIDIA's relative penalty), and that factor COMPOUNDS with Arc's weaker RT-core baseline into the 5.14× DXR-vs-wavefront gap that motivated the vendor-mode default. Three findings behind it. (1) **Secondary TraceRay dispatch is 68-84% of the whole pipeline's cost on BOTH vendors** (B70 default: 6.70 of the 7.64 ms mode-0-vs-mode-2 gap). The "DXR is fine on NVIDIA" reading of the vendor table was only ever true relative to mode 0 — the 4090 pays the same proportion at its own scale. (2) **DispatchRays launch itself is ≈ free**: mode 2 lands at the compute plain reference's own cost on both vendors — the overhead is per-TraceRay, not per-DispatchRays. SCENE-DEPENDENT CAVEAT (2026-08-01, THE WORLD on the B70, parked boot pose, same-day sweep): with the world's fat cutout-armed shaders the DXR execution model is NOT free — the identical full-screen traversal reads 1.28 ms as the compute reference kernel, 2.59 as mode-2 raygen, 3.13 as mode-1 dxr-rays (a 2-2.4× DispatchRays-vs-compute tax that the small procedural scene never showed; shader source byte-identical, same TLAS — MECHANISM CLOSED 2026-08-04: live state × the RT launch regime's spill pricing, see THE DISPATCHRAYS-REGIME CAMPAIGN in the Intel section; AUDITED + RE-MEASURED 2026-08-05 like-for-like — the old `reference` bracket was ~3% fat vs `dxr-rays`, the corrected world read is 1.178 vs 2.808 = 2.38× at --tod 11 with the ratio banding 2.07-3.30× across scenes/builds, see THE FINDING-1 AUDIT paragraph there). Mode 2 also now BEATS mode 1 on the world at spp=1 (span 4.77 vs 5.36), unlike the 07-22 tie (3.83/3.80) — mode 2 is the Intel pick for world-class scenes at any spp (the vendor default since 2026-08-01). This tax, not the quadtree (hybrid-vs-reference is a ±10% wash on the world too: 1.15-1.4 vs 1.28 trace ms), is ~90% of the README's ~65% hybrid-vs-DXR world margin. (3) The primary is the one place TraceRay earns anything: on the 4090 mode 1 beats mode 2 (keeping the coherent primary on the hardware pipeline is worth a few %), while the B70 prefers zero TraceRay (mode 2 < mode 1 everywhere). Gates: the full `--check-dxr` suite passes at every mode on both vendors with statistics identical to baseline (class-mismatch 0, radiance A/B 0.045% — same hardware traversal, same shading code), including `san-miguel-low-poly` armed, where the ALPHA_CUTOUT/TRANS_SHADOW candidate loops run inline inside `chs_shade`. **Why mode 1 is the default and mode 2 is not** (the promotion decision, 2026-07-22): mode 1 strictly dominates mode 0 — never slower at any measured (vendor, scene, spp) point — while keeping the pipeline architecturally real (payload/closest-hit/SBT/any-hit all still do their jobs for the primary; mode 2 is the reference loop wearing a raygen hat, kept as the measurement arm and the manual high-spp-Intel pick). The spp crossing is MEASURED (default scene, spp 1/4/16, same harness): on the B70 mode 2 beats the wavefront only below spp≈3 (1.41 vs 1.76 at spp=1, 4.71 vs 4.48 at spp=4, 18.10 vs 14.72 at spp=16 — the quadtree's marginal sample is 0.86 ms vs the reference-shaped 1.11, so amortization takes over almost immediately), while on the 4090 inline-DXR dominates EVERY spp measured (mode 1 0.26/0.83/3.53 vs wavefront 2.11/2.96/6.36 — the quadtree never catches up there, consistent with its 1.36× marginal-ratio asymptote). A register-pressure lesson INSIDE the curve: mode 1's marginal sample on the B70 is 2.2 ms — DOUBLE mode 2's — because the candidate-loop-fattened chs_shade pays occupancy per sample where mode 0 paid per dispatch; a fat closest-hit shader is fine at spp=1 (the upscaler contract every composed DXR frame pins) and ruinous at spp=16, which is why mode 2 was the right manual pick for a high-spp Intel DXR session — AUTOMATED 2026-08-01: `main::vendor_defaults` now defaults Intel sessions to mode 2 outright (mode 2 also wins Intel at spp=1 on every measured scene, procedural included), with any explicit `--dxr-inline` — 1 included — or a settings-file value as the veto (`dxr_inline_explicit`). **W1 INTERACTION**: the vendor-mode default's justifying table (`main::vendor_defaults` — B70 DXR/wavefront 5.14×/2.61×) was measured against MODE 0; under the mode-1 default the Intel ratio straddles 1.0 by scene at spp=1 (default 1.34×, stress 0.81×, san-miguel-lp 0.94×). The Intel→wavefront default is KEPT (it still wins the default scene at spp=1, every scene from ~spp 3 up, and owns H/R/C/O), but the clean crossing is gone — settled by THE WORLD RE-MEASURE (2026-07-22; the actual flagless scene, which `--spin` never loads, so it took interactive boot-pose sessions: B70 native 1080p spp=1, `--gpu-timing` running means over 6–30k frames, 2 interleaved reps, spread 1–5%): wavefront 4.15 ms tracer / 5.21 span vs mode-1 DXR **3.80/4.88** (mode 2 3.83/4.92 — cutout-fattened inline loses its edge over the coherent primary; mode 0 7.28/8.37, matching the BLAS-era 7.27/8.34 record). At spp=1 the wavefront now loses on every scene measured except the sky-heavy procedural default (world 0.92×, stress 0.81×, SM-lp 0.94×), so the Intel default stands on the feature/multi-sample grounds above, never on spp=1 perf — a conclusion RETIRED 2026-08-01: the frontier/pack-split/compose wins that followed were wavefront-side (the ladder now ~0.22 ms), and the world re-measure reads wavefront 3.30-3.49 vs DXR 5.36 span, hybrid 1.5-1.6× moving or parked, so spp=1 perf argues FOR the Intel wavefront default again; `main::vendor_defaults` carries the table and the one-line flip instruction if that trade ever reverses. Beware the Select-Object `-First` pipe-kill when checking armed exit codes (it terminates the process mid-run and manufactures phantom crashes — this session lost 20 minutes to a "teardown crash" that was the pipe closing).

## Profiling

Three instrumentation layers, all zero-cost and dependency-free by default (the same footprint discipline as the SDKs):

- **Tracy (CPU)**: `cargo run --release --features tracy` — without the feature the `tracy-client` crate is not even compiled and every `crate::zone!` / `crate::plot!` / `prof::*` hook is an inert no-op. The crate version in Cargo.toml pairs 1:1 with a Tracy server protocol (tracy-client 0.17.x ↔ Tracy **v0.11.1** GUI) — bump both sides together or the GUI refuses the connection. Zones are **frame-phase granularity only** (`trace-full`/`trace-capped`/`replay`, `temporal-admin`, `resolve`, `present-*`, `oidn-filter`, `oidn-history`, `rr-upload`/`rr-eval`, `xess-upload`/`xess-eval`, `gpu-wait`) — NEVER per-tile or per-pixel (tens of thousands of zones/frame is itself ms-class overhead). The breakdown *inside* a trace zone (bound query vs refine vs replay-recording vs shading) comes from Tracy's statistical sampling, which is why `[profile.release] debug = "line-tables-only"` is always on (run the Tracy capture elevated for kernel sampling). Per-frame plots: frame ms, fr-queries, adopts, replay leaves, render height.
- **PIX (GPU)**: `--pix-markers` (default off — unprofiled sessions stay byte-identical) puts named Begin/End events on the D3D12 command lists: the wavefront (`wavefront` → `level {d}` → `leaf+sky` with nested `leaf`/`sky` sub-brackets around the two EIs — no barrier separates them, so `sky` under-reports whatever overlapped `leaf`, while `leaf` is honest bottom-of-pipe at its EI's drain — → `hemi` → `compose`), `wavefront-replay`, `feed`, `resolve`, `reference`, `dxr` → `dxr-rays` (the `DispatchRays` alone, so the parent's residual isolates `bind_common` + `SetPipelineState1`) / `dxr-cloud-shadow` / `dxr-sky-lod`, `rr-upload`/`rr-eval`, `xess-upload`/`xess-eval`. `WinPixEventRuntime.dll` is `LoadLibraryExW`'d from `--pix-path` / `FRUSTRACER_PIX_PATH` (default `SDKs\pix\bin\x64`, gitignored — drop it there from a PIX install or the WinPixEventRuntime NuGet); a missing DLL prints one loud line and leaves markers off, never an error. Nothing links it, so every `--check*` stays DLL-free.
- **GPU timestamps (`--gpu-timing`, `src/gpu/gputime.rs`)**: D3D12 timestamp queries printed as a per-region GPU-ms table every 120 frames (plus a `= gpu frame span` row). **The table is WINDOWED as well as cumulative** — `win ms`/`win max` cover the last 120 frames, `mean ms`/`max ms` cover the whole run. Read the window column; the cumulative one divides by every frame since the first, so on Intel it permanently carries the async-compile fallback (measured: a fresh variant's first window reads 2.003 ms with a 10.9 ms max, and the cumulative mean is still 3% high 480 frames later while the windows sat flat at ~1.3-1.5). That bias is UNEVEN per region, so it distorts the shape and not just the total, and it is baked into every pre-2026-07-24 number recorded here. The window also makes the fallback self-diagnosing (a step down to a flat floor, not a slow drift) and makes each table an independent sample, so a 60 s run yields ~90 of them and a real IQR — measured ±0.07 ms / ~1.6% on an interactive world session, far tighter than the "1-5%" that was previously assumed. `take_regions` (the `--check-gpu` per-row drain) is unaffected. **The PIX marker brackets ARE the timing brackets** — `pix::scope` opens a `gputime` region of the same name, so the two instruments cannot drift apart and a new marker is timed automatically. This exists because **PIX cannot analyze a capture on an Intel adapter at all**: its replay engine fails `D3D12EnableExperimentalFeatures` (with Developer Mode on, and with `--disable-gpu-plugins`), so on Arc there is no way to get numbers out of a `.wpix`. Timestamps are vendor-neutral, so this is the only per-pass GPU profiler that works there — and that neutrality is load-bearing in the other direction too: the same numbers come back from an NVIDIA and an AMD adapter, so `--prefer-amd` vs `--prefer-nvidia` gives a **per-pass** diff. That diff is how the LEAF_GROUP wave64 bug was found — "AMD is 2.2× slower per sample" is not actionable; "AMD is 2.2× slower *in the leaf kernel only, while its reference kernel is FASTER than NVIDIA's*" points straight at the kernel. Readback is free of stalls by construction: `D3d::begin_frame` has already waited on the slot's fence, so frame N's timings are collected at the top of frame N+`FRAMES_IN_FLIGHT`; `HeadlessGpu::run` gets the same wait-then-collect on one slot, which is what arms the flag inside `--check-gpu`/`--check-dxr` (there `gputime::take_regions` drains the table per bench row — the running mean `report` prints divides by the FRAME count, so folding the reference kernel's frames in with the wavefront's would dilute every region by the frames that never ran it). The wall-clock bench rows carry the headless loop's per-frame submit+fence overhead, so GPU time can be well under the row — compare GPU time to GPU time. Cross-validated against the suite's own bench (`--gpu` wavefront 3.38 ms vs `--check-gpu`'s 3.49 ms row, same scene/res). Opt-in and inert when off — no query heap, no name allocation, no `EndQuery` — so default sessions and every `--check*` stay byte-identical.

**The Intel Arc Pro B70 numbers, and the bug the vendor diff found.** Re-measured on today's tree with `--spin path` (the deterministic GPU workload — `--spin` drives `--gpu` and an explicit `--dxr` too, so `--cpu`/`--gpu`/`--dxr` rows are directly comparable; see `run_spin_gpu`), 1080p, GPU frame span over 360 frames:

| | B70 wavefront | B70 DXR | 4090 wavefront |
|---|---|---|---|
| default scene | **1.56** | 7.85 | **1.25** |
| `--stress 5000` | **2.81** | 5.02 | **1.96** |

(The DXR column is the all-TraceRay pipeline of its era — today's `--dxr-inline 1` default reads 2.35/1.64 on the B70; see the DXR section's ablation.)

Two claims that used to live here are now WRONG and are recorded as such so nobody re-derives them: *"DispatchRays is pathologically slow on Arc"* (it is not — those numbers predate the current driver, and DXR is a perfectly respectable arm), and the reading of the level ladder as the wavefront's problem. What was actually happening is the third bullet below, and it was worth more than everything else on this page combined.

**THE WORLD's frame, decomposed (B70, native 1080p, spp=1, 2026-07-24).** `--spin` never loads the world, so this needs interactive sessions: `--world --prefer-intel --gpu --xess --no-fg --no-vsync --lock-res native --no-settings --gpu-timing --cam <boot pose>`. Every one of those flags is load-bearing — `--prefer-intel` because the adapter preference defaults to NVIDIA, `--no-settings` because the pause-menu file applies BEFORE the arg parse, `--no-fg` because XeSS-FG contends on the same queue AND its vsync cadence-halving idles the GPU into higher boost clocks (i.e. it makes every region read optimistically fast), `--no-vsync` for steady clocks and 3× the sample rate. The boot pose now prints itself as a paste-ready `--cam` line at load (it is a function of `world.field_half`, so it moves silently when the curated set changes). Medians over ~100 windows, attribution closing to within 0.002 ms:

| region | at rest (replay) | moving (producing) | DXR |
|---|---|---|---|
| `= gpu frame span` | 4.158 | 5.644 | 5.430 |
| tracer total | 3.072 | 4.554 | 4.304 |
| `leaf` | 1.939 (47%) | 1.894 | — |
| `sky` | 0.902 (22%) | 0.896 | — |
| `level 0..7` | **0.000** | 1.415 (25%) | — |
| `compose` | 0.190 | 0.188 | — |
| `dxr-rays` | — | — | 4.271 |
| `feed` | 0.546 (13%) | 0.546 | 0.585 |
| `xess-eval` | 0.541 (13%) | 0.546 | 0.542 |

Three findings worth not re-deriving. (1) **Structure replay INVERTS the wavefront-vs-DXR comparison at rest, and nobody had re-measured since it landed.** On producing frames DXR still edges the wavefront (4.304 vs 4.554 = 0.95×, consistent with the recorded 0.92-0.94×), but DXR has no replay, so on the resting frame the wavefront runs 3.072 vs DXR's 4.304 — **29% faster on the frame a parked user actually gets**, with the ladder and all 8 levels reading exactly 0.000/`calls 0.0`. `main::vendor_defaults`' Intel entry is therefore better justified than its own comment claims ("features + multi-sample, never spp=1 perf"). It also means **any ladder optimization is worth exactly ZERO at rest** — measure `--no-replay` for the producing frame before valuing one. (2) `dxr-rays` (the new region around `DispatchRays`) shows DXR's 4.304 is 4.271 of *rays*: setup, `bind_common` and `SetPipelineState1` together are 0.033 ms, so there is nothing to win there. (3) **`feed` moves far more than it uses**: 88 B/px read + ~16 B/px written, and `cs_feed_xess` consumes 12 of those 88 bytes. That became the pack split below.

**THE G-BUFFER PACK SPLIT (2026-07-24) — and two probe traps worth more than the optimization.**
The pack was one 88 B/px `GBufPx` record written by every upscaler session; `cs_feed_xess` consumes
12 of those bytes. It is now `GBufCore` (16 B/px, always) + `GBufExt` (72 B/px, only under
FLAG_GBUF_EXT). Measured on THE WORLD, medians over ~150 windows, at the (32,256) frontier:

| | B70 / XeSS | 4090 / DLSS-RR |
|---|---|---|
| `= gpu frame span` | 3.536 → **2.359 (−33.3%)** | 3.883 → 3.669 (−5.5%) |
| `leaf` | 1.486 → 1.213 | 0.957 → 0.745 (−22%) |
| `sky` | 0.736 → **0.099 (−87%)** | 0.092 → 0.091 |
| `feed` | 0.544 → **0.231 (−57%)** | 0.385 → 0.385 |

`check.png` is BIT-IDENTICAL (the pack is display-invisible), `--spin` is unmoved (it runs
`gbuf_full = false`, so no pack at all — it can prove no-regression but can never show this win),
and the RR arm IMPROVES even though it writes both halves: two contiguous stores beat one 88 B
record.

**Why a split and not a mask.** Storing a single 16-byte member of the old record measured
**1.791 ms in `leaf` vs 1.486 for storing all 88** — writing 16 B at offset 48 of an 88 B stride is
an unaligned scatter, and a partial-line write costs a write-allocate (fetch the line, merge, write
back). The arithmetic fits: `1.075 + 2×0.411 = 1.90` predicted vs 1.791 measured. **Contiguity, not
size, is what the memory system rewards** — 32 lanes × 16 B = 512 contiguous bytes = 8 whole lines.
Do not "simplify" the two buffers back into one struct with skipped members.

**THE TRAP, and it is the transferable lesson: an ablation that cannot reach its target answers
CONFIDENTLY.** `FR_ABL` defines only reach compile units whose source list includes `defs`.
`leaf_of` did; `sky_unit_src` and `feed_src` did **not**. So an `FR_ABL=nogbuf` probe reported `sky`
unchanged and an `FR_ABL=nopack` probe reported `feed` unchanged — both were read as "that half is
free", and BOTH were the probe comparing identical code against itself. The shipping split then
measured those exact two regions at −87% and −57%. The projection built on those probes (−9.5%) was
off by 3.5× *against* the change. Both units now paste `abl_defs()`; when a probe reports "no
effect", prove the define arrived before believing it. The one probe that DID reach its target
(`nogbuf` into `cs_leaf`, 1.486 → 1.075) was correct, as was the partial-store finding above.
THE TRAP FIRED TWICE MORE, found by audit 2026-08-01: (3) `nowave` never reached wavefront.hlsl —
the tile unit takes only `wavefront_ablation_defs`, so `gw_alloc`/`gw_min_bits` stayed
wave-cooperative in every "revert" arm ever run (the arm is DUAL-HOMED now, emitted by both defs
fns; see the wave-aggregated-atomics paragraph below for what that contaminated); and (4) the DXR
pipeline's own `feed_src` pasted no `abl_defs()` at all, so a `nopack` probe under `--dxr`
compared identical code against itself (now a CONDITIONAL paste — dxr.rs — so the unarmed source
stays byte-equal). Two structural guards shipped with the repairs: `FR_ABL (gpu):` announces the
raw value + the matched GPU arms once per process ("matched GPU arms: (none)" on a non-empty
FR_ABL IS the probe-reach alarm — trace.rs's `abl_announce`/`ABL_GPU_TAGS`, kept in lockstep with
every `abl_has` consumer), and FR_LEAF/FR_LGROUP now print loud on departure AND on an illegal
value instead of silently reverting to the shipping config (the FR_WIDE rule — a mistyped sweep
cell used to measure the default while believing it measured the lever).

**`compose` IS NOT DISPATCHED ON fb-OFF FRAMES (2026-07-24).** With no hemisphere pass there is
no ambient term to fold in, so `compose.hlsl` degenerated to `accum[i] = partial[i]` — a full-screen
12 B read + 12 B write, its own dispatch and its own barrier, moving data between two buffers for
nothing, while `leaf`/`sky` also wrote 12 B/px of `ambw` zeros that `compose.hlsl:22` already
declined to read. Now `leaf` (under its existing `LEAF_NO_FB` PSO) and `sky` (a uniform `fb_mode`
branch — one PSO, no compile arm to hang it on) splat straight into accum through
`queues.hlsli::accum_splat`, which is compose's store-or-add rule verbatim so the two cannot drift.
fb frames keep the old path untouched: the hemi wavefront runs after `leaf`, and compose runs over
EVERY pixel, so a sky pixel that skipped `partial` would have it read last frame's value.

Measured on THE WORLD (B70, after the pack split): frame span 2.359 → **2.235 (−5.3%)**,
`wavefront-replay` 1.558 → 1.416, and the `compose` region disappears from the table.
**BIT-IDENTICAL by construction** — an f32 store followed by an f32 load returns the same bits —
which the replay gate's zero-tolerance compare confirms (`accum-diff 0` at frame 0 and at warm
frame 1). Safe because leaf+sky rects partition the screen exactly (the exactly-once coverage
gate), so every pixel has one writer and the read-modify-write needs no atomic; `reference.hlsl`
has always written accum this way. Dropping the dispatch drops no synchronization either:
`record_terminal_fills` already ends with `args_to_uav`'s GLOBAL uav barrier.

**A dead candidate, measured so nobody rebuilds it: per-chunk BLAS opacity.** THE WORLD unions `any_alpha`/`any_transmissive` across islands (`scene.rs::finalize_scalars`) and `trace.rs` hands that one scalar to all ~890 chunk `geometry_desc` calls, so every ray runs `Proceed()`/`candidate_reject` against 34.4M triangles to serve San Miguel's foliage — and chunks are maximal BVH subtrees, so a per-chunk flag looked like a free de-pessimization. **The ceiling says no.** `FR_ABL=noalpha,notrans` (which now gates the three per-scene predicates, and through the derived `trace::non_opaque` the BLAS flag with them — so an ablation moves the shader arms and the AS together) makes the whole world OPAQUE: frame span 4.257 → 4.161 (**−2.3%**), `leaf` 2.029 → 1.914, and that is inside the run-to-run band. San Miguel alone reads −3.1%. A real per-chunk scheme recovers only a fraction of a ceiling that is already under 3%, so it cannot clear the gate bill. The lever is proven live by a positive control: `FR_ABL=noalpha` on san-miguel-lp fails `--check-gpu`'s cutout must-fire with `alpha-cutout rejections: 0`.

**The `cs_sky` load-balance bug (fixed; see `trace::SKY_SPLIT`).** The wavefront used to pay **+6.9 to +9.2 ms** for clouds while DXR paid **+0.2**, for the *same* pasted `trace_common.hlsli` march — a 30× discrepancy that can only be dispatch, never code. `cs_sky` ran ONE 64-lane group per SkyRec and grid-strode the whole rect inside it; a sky rect is not tile-sized (the quadtree emits it at whatever depth it proved empty, so a depth-2 rect at 1080p is 480×270 = 129,600 px), so one group ran ~2,025 serial laps of a volumetric march while the machine idled. Invisible for as long as a sky pixel was a dome+disc evaluation; dominant once clouds made each sky pixel ~100× dearer. The fix is dispatch-only (SKY_SPLIT groups share each record, `cs_prep_mul` multiplying instead of dividing): **B70 8.95 → 1.56 ms default and 13.07 → 2.81 stress; 4090 4.99 → 1.25 and 7.32 → 1.96** — 73–83%, cross-vendor. Every exact-zero gate stayed 0 and the same-seed image A/B stayed at its old magnitude, because the new indexing provably still partitions the rect. **The lesson generalizes: when one pipeline pays for a shared shader and the other does not, suspect the dispatch shape, not the shader** — and the instrument that exposes it is the per-pass `--gpu-timing` diff across two vendors, the same one that found the LEAF_GROUP wave64 bug.

**MEASUREMENT TRAP ON ARC — read before any config sweep.** Intel's driver compiles a NEW DXIL variant twice: PSO creation returns a fast-to-produce UNOPTIMIZED binary, and a background recompile replaces it on a WALL-CLOCK schedule — measured **~5–8 s after launch** — caching the optimized result in `%LOCALAPPDATA%\D3DSCache` (13.8 MB cap observed ⇒ LRU eviction is live; heavy variant churn can evict and re-expose the trap on a config that used to be warm). So the first run is not merely "slow at the start": a 140-frame `--spin` run (~4 s wall) ENDS before the optimize lands and is **100% fallback**. Measured on a fresh cs_leaf variant (1200 frames, per-120-frame windows): 7.20/6.40/5.82/5.62/2.84/1.62 → flat on the cached profile — the fallback is **~4.7× on the frame span, ~7.6× on the leaf kernel itself**, dead stable across the whole first run, and the next run reads 1.5 ms off the cache. Consequences, all shipped at least once: an inflated `f1` manufactures phantom regressions and flattering slopes in the `per-sample = (f16 − f1)/15` cost model (a "3.2× regression" and a "−39.9% win" were the same bad batch); and in a dense back-to-back sweep the compile queue (each config compiles pso_leaf AND pso_leaf_fb, plus wavefront/sky variants when LEAF_TILE moves) can push a variant's optimize past even its rep-2, which is how a fallback read can look "reproducible" — the retired "(16,128/256) 5× cliff" below was exactly this. Rules: **every point in a config sweep is a new variant** — discard a warm-up run per variant and never compare fresh against warm; and **an anomalous cell earns one re-run after a ≥10 s pause** (the optimize lands on wall-clock, not on run count) — a phantom collapses to baseline, a real effect survives. Beware that a fallback read REPEATS (~7 ms twice looks "reproducible"), so agreement between two back-to-back reps proves nothing by itself. Separately: the B70 repeats to ±0.002 ms while the 4090 spans 1.42–1.98 for one unchanged config, so a single sample is worthless on NVIDIA and merely dangerous on Intel — interleave and take medians (this is the same rule the `--spp` bench rows already carry, and it had to be re-learned here anyway).

**The leaf frontier was mis-tuned on EVERY vendor** (`render::LEAF_TILE` / `trace::LEAF_GROUP`, swept via `FR_LEAF`/`FR_LGROUP`; it read "on Intel" until the 2026-07-24 world re-measure showed the 4090 gaining as much or more). The two interact — a leaf rect is ~(rw·rh)/4^depth_full pixels, so shrinking the tile drops it below the group width and idles lanes — and neither axis can be read alone. Measured 2-D sweep on the B70 (gpu frame ms, default scene): today's (8, 32) = 1.652, and the optimum is **(32, 256) = 1.291 (−21.9%)**, with `--stress 5000` 2.009 → 1.457 (−27.5%). Three readings: **4×4 tiles lose at every group width** (+32% to +63% — finer subdivision buys sky proofs and a tighter bound, and the bound is worth ~nothing); **LEAF_GROUP=8 is bad everywhere**, which is the SIMD16 floor showing up directly (the constant is 32 from wave32/wave64 reasoning on the other two vendors — Intel wants 256, which you would not guess from them); and the "**reproducible 5× cliff**" this paragraph used to record at (16, 128) and (16, 256) — 7.13 ms, entirely inside `leaf+sky`, `overflow 0`, unexplained — was a PHANTOM, resolved 2026-07-22: 7.13 is the async-compile FALLBACK state (see the trap paragraph above — the fresh-variant band measures 7.09–7.17 with the excess entirely in the nested `leaf` sub-bracket, and the fallback cost is config-independent, which is why two different group widths "measured" the SAME number), while the same cells re-measured warm read 1.53/1.58 — flat with the whole re-swept TILE=16 row (g96–g256 all 1.51–1.58, so the row has no group-width structure at all, exactly as a grid-stride kernel should). The grid's surviving anchors re-verified warm on the committed tree: (8, 32) = 1.684, (32, 256) = 1.327 (−21.2%). **ADOPTED 2026-07-24 — (32, 256) IS THE DEFAULT** (`render::LEAF_TILE` = 32, `trace::LEAF_GROUP` = 256; the old pair is reachable as `FR_LEAF=8 FR_LGROUP=32`), after re-measuring on THE WORLD, which `--spin` never loads and which no earlier sweep had touched. It is a cross-vendor win, not an Intel one — and it is much bigger on MOVING frames than the original procedural-scene number suggested, because it halves the level ladder:

| | B70 | 4090 |
|---|---|---|
| THE WORLD, at rest (replay) | 4.257 → 3.615 (**−15.1%**) | 3.883 → 3.907 (+0.6%, inside a 0.25 ms IQR) |
| THE WORLD, moving (producing) | 5.590 → 4.383 (**−21.6%**) | — |
| `--spin path` default | 1.813 → 1.279 (−29.5%) | 0.929 → 0.666 (−28.3%) |
| `--spin path --stress 5000` | 2.017 → 1.416 (−29.8%) | 1.391 → 0.898 (−35.4%) |
| `--spin path` san-miguel-lp | 1.957 → 1.504 (−23.1%) | 1.247 → 0.722 (−42.1%) |

CPU tracer −2.2%/−3.3% (`--spin path`, default / san-miguel-lp) — no mode regresses. The mechanism is DISPATCH SHAPE, not the quadtree's product: a coarser frontier proves LESS space empty and still wins, because `depth_full` drops 8 → 6 (the ladder halves: levels 0..7 = 1.372 ms → levels 0..5 = 0.674 on the world) and a ~540-px leaf rect genuinely feeds 256 lanes where a ~32-px one only idles them. `sky` gains for the same reason (0.914 → 0.664 on the world): fewer, larger proven-empty rects amortize `cs_sky` better. The 4090's world-at-rest neutrality is not a counter-example — its resting frame is dominated by DLSS-RR at 2.4 ms, so a tracer change barely moves the span there, and the same GPU gains 28-42% the moment the ladder actually runs.

**The must-fire blocker was real but narrower than assumed, and the fix is a PIN, not a relaxation.** Exactly ONE gate goes vacuous at the coarse frontier — `verify temporal yaw: expected temporal sky-tiles > 0` — because a tile's query region is 4× wider per axis and so far less often lies WHOLLY inside the old sky region (static sky reuse is unaffected, and `--check-gpu`/`--check-dxr` pass at the shipping frontier unmodified). `--check` now pins the temporal family at `render::TEMPORAL_TILE` = 8 (`render::set_leaf_tile`, an atomic in place of the old `OnceLock` — the `set_aniso` knob idiom) and restores the shipping frontier after, printing `temporal gates: leaf frontier pinned at 8 (shipping 32)`. Gating the ALGORITHM and gating the SHIPPING CONFIG are two jobs; conflating them leaves only bad options (weaken a real guard, or freeze the constant). Every SOUNDNESS counter is frontier-independent and runs at both. **And the group axis ALONE is still worth nothing** — at TILE=8 the B70 reads g64/g128/g256 = +0.9/+6.3/+21.1% vs g32 (+/−0.4/+1.6/+9.5 stress), so the 256-group appetite belongs to the ~540-px TILE=32 leaves, not to the vendor: **the two constants must always move together**, and `main::vendor_defaults` stays mode-only.

**With that gone, the level ladder became the wavefront's dominant cost, exactly as originally claimed** — on `--stress 5000` the 8 level kernels were **1.78 ms of 2.81 (63%)** while `leaf+sky` was 0.83, and the cost did not track tile count. Two candidate explanations, and **the measurement killed the popular one**: nested timing regions around the two halves of each level put the `cs_prep` dispatch plus BOTH args transitions at **0.011 ms across all 8 levels** against the kernels' 1.817. Barriers and resource transitions are cheap here; the ~24 "pipeline drain points" that look so damning in the recording loop are worth eleven microseconds, and per-level counters + a static Dispatch (the obvious refactor) would buy exactly that. **Do not re-derive this** — the comment at the ladder in `record_wavefront` carries the numbers.

The ladder was **under-occupied**, not dispatch-bound: level d holds at most 4^d tiles, so under one-thread-per-tile levels 0-4 are ≤ 256 threads, while each of those tiles does the MOST work of any level (a shallow frustum covers a large fraction of the screen and its inherited cut has barely been refined). Level 0 is literally one lane descending a 1.8M-node BVH. **`cs_level_wide`** (`trace::WIDE_LEVELS`) gives the shallowest levels ONE GROUP PER TILE: 32 lanes share a breadth-first frontier that aliases frustum.hlsli's per-lane stack slab (zero extra groupshared — the phase and refine_cut's never overlap, the same argument the serial path already makes), `best` reduced by a min on the float bit pattern (exact for non-negative floats — `gw_min_if`, wave-reduced then one `InterlockedMin` per wave), with the serial path's stack-pressure fallback preserved verbatim on frontier overflow. It is a BFS, so the *pruning order* differs and node counts differ — but `best` is a min over the same candidate set and is order-independent, which is why the same-seed image A/B comes back at its old value to the digit. Deep levels keep the per-thread kernel: thousands of tiles with tight cuts and short descents, where a whole group per tile would waste 31 lanes (WIDE 8 measured WORSE than not doing it at all on Intel). Both frustum structures are implemented, so `--no-ftree` is gated too. Measured (interleaved, 3 reps, medians): **B70 −7.4% / −30.4% / −2.4%** and **4090 −1.5% / −11.1% / −23.1%** on default / `--stress 5000` / San Miguel. A naive single-shot sweep "showed" a 9-16% NVIDIA regression that the interleaved reps erase entirely — the 4090 spans 1.42-1.98 ms for one unchanged config while the B70 repeats within 0.002.

**The AMD (RDNA4) campaign, 2026-07-24 — the R9700 row every table was missing, and two constants re-swept before they shipped.** The box gained a **Radeon AI PRO R9700** (Navi 48, RDNA4) beside the 4070 Ti, which retires the standing "RDNA has no measurement here" excuse in `main::vendor_defaults` (now a measured paragraph: DXR beats the wavefront on RDNA4 on every scene at every spp, by a WIDER margin than Ada's, so there is still no AMD vendor rule to add). Two facts made the campaign cheap and are worth keeping. First, **the R9700 is a metrology-grade card**: 6 repeats of one config span **0.933-0.935 ms (±0.2%)**, B70-class determinism, so a single 600-frame run resolves anything above ~1% — the opposite of the 4070 Ti, whose spread reaches 15-22% on the same rows and where only the *ladder* (which is what these constants move) is readable at all. Second, **`--spin-frames` must be a multiple of `SPIN_LAP` = 600**: the pose is a pure function of the frame index, so a 200-frame run samples a third of the camera loop and is a *different workload*, not a noisier sample of the same one.

Two instruments came out of it and are IN. **`FR_DUMP_HLSL=<dir>`** (dxc.rs) writes every *assembled* kernel to disk, since kernels are built by string concatenation and no file on disk is what DXC sees; with the dump, **Radeon GPU Analyzer** (`SDKs/rga`, gitignored with the rest of `SDKs/*`) compiles them offline for `gfx1201`. Note the recipe needs the HLSL version passed through — `frustum.hlsli` uses `select()`, an HLSL 2021 intrinsic, so a default DXC front end rejects exactly the units worth profiling (the doc comment carries the full invocation). **`FR_WIDE`** and **`FR_LSTACK`** sweep `WIDE_LEVELS` and `LANE_STACK`, which `--no-wide-levels` could only turn off, never place. RGA finally answers the question `leaf.hlsl` has always asserted:

```text
  kernel        grp  VGPR   LDS   waves/SIMD (VGPR-limited, of 16)
  leaf-fb        32   240   2048    6      <- the arm LEAF_NO_FB compiles out
  hemi_cell      32   240  10240    6
  leaf           32   216   2048    7      <- 44% occupancy, the hot kernel
  reference       8   214   4096    7
  level          32    65   8192   16
  level_wide     32    54   8704   16      <- NOT VGPR-limited; LDS is its cap
  sky            64    40      0   16
```

(The LDS column is at the then-shipping `LANE_STACK` = 64 — `32 x 64 x 4 B` = the 8192 on `cs_level` — which is what this table then went on to change; at today's 16 those two rows read 2048/2560. Kept as measured, since it is the evidence for the change rather than a description of the result.)

So "VGPR count sets occupancy directly on RDNA" is measured, not asserted: `cs_leaf` runs at 7 of 16 waves and `leaf-fb`'s extra 24 VGPRs cost a whole wave slot — the mechanism behind `LEAF_NO_FB`'s documented -11%. It also **refutes** the natural follow-on guess: the level kernels are nowhere near register-bound, so anything throttling the ladder is the LDS slab, not registers.

**FIXED 2026-07-31: the AMD candidate-loop TMin defect — an AMD-only, textured-scene-only wavefront bug that was also costing 2x.** `san-miguel-low-poly.obj --check-gpu --prefer-amd` had failed for as long as the box had a discrete AMD adapter to run it on, always with the same signature (`tmin-overshoot 341393` = every hit pixel, `max rel t err 8.70e-1`, `mv_selftest` median 8.428e-1, 6 distinct GPU albedos), while the same command passed on NVIDIA and the GPU *reference* kernel agreed with the CPU perfectly. Diagnosis, in the order the evidence arrived: the culprit dump showed `wave_t − ref_t` constant at **6.4993** to five decimals while `ref_t` varied over five units, and the tile's inherited `t_start` was **6.49935** — arithmetic, not geometry, so the wavefront was reporting `t_true + t_start`. Arming `--heightfield`, which forces the hardware `TMin` to 0 and re-checks the interval logically, made it vanish outright (`max rel t err 0.00e0`). **The defect: on RDNA4, an inline RayQuery driven by a `Proceed()` candidate loop with a NONZERO `r.TMin` returns the hit distance offset by +TMin.** AMD re-origins the ray at TMin (long documented here, normally invisible); on this path the driver adds that offset back to a value that already carried it. Only the wavefront's leaf primaries pass a materially large TMin — the tile's inherited `t_start` — so only they show it; every other candidate-loop ray passes an eps-scale tmin, which is why `--check-dxr --prefer-amd` reads a clean `8.70e-6` on the same scene and the DXR pipeline is deliberately NOT armed.

The fix (`rt.hlsli::cand_tmin`, gated by `trace::cand_defs` on the PICKED adapter, `FR_ABL=nocandtmin` to reproduce on demand) is the mechanism this file already had for relief: hand the hardware the full positive ray and let the loop's own `ct > tmin && ct < tmax` re-check enforce the logical interval. With TMin = 0 there is no re-origining, hence no offset to get wrong — robust to whatever the driver does, unlike subtracting TMin back off, which would silently invert into the false-near-hit direction if AMD ever fixes it. **It is 2x FASTER, not slower** (san-miguel-lp GPU frame span 1.352 → 0.663, `leaf` 1.240 → 0.556, alpha-cutout rejections 9762 → 1005): committing an inflated `t` shrank `TMax` to a too-large value, so traversal kept enumerating candidates long past the real hit. The near-prune the workaround gives up is the one AMD was already measured not to have. Results: `tmin-overshoot` 341393 → **0**, `max rel t err` 8.70e-1 → **0.00e0**, image mean 2.56e-1 → 1.83e-9, `mv_selftest` FAIL → OK, distinct albedos 6 → **5272** (NVIDIA reads 5269). NVIDIA is untouched by construction (the define never arms) and re-verified bit-exact.

**THE CONTAMINATION IS THE LESSON.** Every AMD number ever taken on a TEXTURED scene before this fix is ~2x pessimistic in the wavefront column — including this campaign's own san-miguel rows and the `0.29x` DXR/wavefront outlier in `main::vendor_defaults`, which closes to `0.60x` once fixed and stops being an outlier at all. Untextured scenes never compiled a candidate loop and are unaffected. A gate that fails on one vendor is not a vendor curiosity to note and route around: it was silently taxing every measurement taken on that vendor, and the anomalous entry in a table was the bug telling us where it lived.

**A SECOND LESSON, from where the helper was first placed.** `rt.hlsli` has two INDEPENDENT per-scene arms — `#if defined(ALPHA_CUTOUT) || defined(HEIGHTFIELD)` (which wraps `trace_closest`/`occluded_q`) and `#ifdef TRANS_SHADOW` (which wraps `transmit_q`) — and `cand_tmin` is called from both. Defined inside the first, it compiles for every scene that carries both arms or neither, and fails with `undeclared identifier 'cand_tmin'` on the ordinary glass-without-cutout scene (an architectural OBJ: windows, opaque textures), taking out every ray-shooting kernel at `TraceGpu` init. **No gate could catch it**: san-miguel and THE WORLD have foliage, and procedural/`--stress`/powerplant have no transmissive material at all, so the committed scene set has no member in that quadrant — `FR_ABL=noalpha` on san-miguel is what reproduces it. A helper shared by two independent `#if` arms belongs OUTSIDE both, and the `cargo test` shader-source gate now pins that ordering. Generalizes: when adding a shader helper, check the preprocessor SCOPE of every caller, not just that the callers exist.

**THE TWO CONSTANT CHANGES WERE MEASURED ON THE OLD TREE, THEN RE-SWEPT ON THIS ONE (2026-07-31), AND THAT IS THE MOST USEFUL PART OF THIS ENTRY.** The original sweeps found `WIDE_LEVELS` 6 → 7 and `LANE_STACK` 64 → 32, both on a tree that predates the **coarse leaf frontier** (`LEAF_TILE` 8 → 32, `LEAF_GROUP` 32 → 256). That change moves the ground under both, so neither was adopted on sight; `FR_WIDE` and `FR_LSTACK` exist so the re-sweep was one command. The outcome differed per constant, and both outcomes are instructive.

**`WIDE_LEVELS` 6 → 7 SHIPPED, and the way it was measured is the reusable part.** `depth_full(1920, 1080)` is now **6**, not 8, so the levels are 0..5 and the shipped 6 ALREADY made every one of them wide — the justifying premise (level 6 being the first serial level and the ladder's most expensive) describes a level that no longer exists at 1080p. The re-sweep confirmed all-wide is right in range: on an R9700 the ladder falls monotonically as more levels go wide, default 0.429 → 0.318 → 0.291 → 0.213 → **0.197** and `--stress 5000` 1.127 → 0.733 → 0.545 → 0.343 → **0.307** at `FR_WIDE` 0/3/4/5/6. That left the constant **unfalsifiable at the resolution it was measured at**: every value ≥ 6 is one config at 1080p, and it only bites at 4K (`depth_full` 7) and 8K (8), where level 6 falls off the wide kernel.

**THE ROUTE — move the FRONTIER, not the resolution.** `--spin` cannot reach 4K (`W`/`H` are consts and `run_spin_gpu` clamps `--lock-res` to ≤ 1.0), and that looked like a hard blocker for a whole release. It is not: `depth_full` is the smallest D with `max(rw,rh)/2^D <= leaf_tile()`, so resolution and frontier enter **only** through that ratio, and a level-`d` tile's frustum is a pure function of (d, camera, aspect) — its rect spans `rw/2^d` and `ray_dir` divides by `rw` — with a tile count of ≤ 4^d. So **`FR_LEAF=16` at 1080p reproduces the 4K ladder exactly** (same levels, same tile counts, same frustums) and `FR_LEAF=8` the 8K one; only the leaf/sky pixel work below differs, and that is a separate timing region held fixed across the A/B. Confirmed rather than assumed: levels 0..4 measure **byte-identical** across all three frontiers, and levels 0..5 across the whole `FR_WIDE` sweep — the lever moves only the level it names, and `leaf` is unmoved to within 0.6%, which is what pins the win to the ladder. Ladder ms, 3 reps interleaved, medians:

| | WIDE 6 | WIDE 7 | |
|---|---|---|---|
| R9700 default | 0.147 | 0.133 | −9.5% |
| R9700 stress 5000 | 0.229 | 0.178 | −22.3% |
| R9700 san-miguel-lp | 0.119 | 0.104 | −12.6% |
| 4070 Ti default | 0.212 | 0.195 | −8.0% |
| 4070 Ti stress 5000 | 0.362 | 0.276 | −23.8% |
| 4070 Ti san-miguel-lp | 0.162 | 0.145 | −10.5% |

Level 6 alone goes 2–4× cheaper (R9700 0.029→0.015 / 0.067→0.016 / 0.029→0.014); frame span −1.7/−4.7/−2.3%. At 1080p the change is a **provable** no-op (`d < 6` ≡ `d < 7` for d ∈ 0..5, both consumers, measured identical ladder and span within 0.5%), so every shipping session today is untouched.

**WHY NOT 8:** level 7 is SCENE-DEPENDENT where level 6 is not — R9700 serial → wide reads 0.119 → 0.032 on `--stress` but 0.018 → **0.031** on default and 0.011 → **0.028** on san-miguel, i.e. wide LOSES on two of three. That is the crossover, and the mechanism is the documented one: level 7 holds up to 4^7 = 16384 tiles, which already fills the machine one-thread-per-tile, so a whole group each only pays where the per-tile descent stays long (`--stress`'s 5000 sparse objects). It is also exactly where the recorded B70 collapse at `FR_WIDE=8` lives — that row was necessarily taken at the old `LEAF_TILE` = 8 frontier, since 1080p/32 cannot tell 6 from 8 apart. **MEASURED ON INTEL 2026-08-01** (B70, `FR_LEAF=16`/`FR_LEAF=8` recreating the depth-7/8 ladders at 1080p, 2 reps ±0.002): the hole is filled and the B70 is MIXED where the other two vendors were uniform — at depth-7, wide-7 wins only `--stress` (span 0.910 vs 0.930; level 6 wide 0.041 vs serial 0.067) and LOSES default (0.809 vs 0.761 — level 6 wide 0.078 vs serial 0.030) and san-miguel-lp (0.922 vs 0.898 by span), the same scene-dependence the other vendors only show one level deeper; and at depth-8 the old `FR_WIDE=8` collapse REPRODUCES on the current tree (default level 7: wide 0.488 vs serial 0.018, 27×; stress 0.242 vs 0.146), so the shipping 7 is exactly right there. Verdict: WIDE_LEVELS stays 7 (cross-vendor uniform win at depth-7 on AMD/NVIDIA, stress win + depth-8 safety on Intel); a 4K Arc session leaves ~0.05 ms on the table on sky-heavy scenes, documented rather than vendor-keyed. Gated: `--check-gpu` at 800×600 has `depth_full` 5 and never creates a level 6, so the changed path is exercised by **`FR_LEAF=8 --check-gpu`**, which passes on both vendors (NVIDIA bit-exact: `max rel t err 0.00e0`, image `0.00e0`).

**The transferable lesson**, and it is the second time this campaign taught it: the old RDNA4 sweep that first proposed 7 was RIGHT and merely inapplicable — `FR_LEAF=8` recreates its exact tree and level 6 still halves there, as it reported. What changed was not the hardware's preference but which levels the shipping frontier creates. **A perf constant is only as good as the tree it was measured on, and the cheapest guard is to make the levers able to recreate the old tree**: `FR_LEAF` + `FR_WIDE` together span every ladder depth the renderer can produce, at one resolution, with no 4K path required.

**`LANE_STACK` 64 → 16 SHIPPED, a bigger cut than the original sweep proposed.** The LDS mechanism survived the coarse frontier even though the ladder it pays for is two levels shorter — R9700 −12.5% / −15.3% / −5.7% / −13.2% and 4070 Ti −9.3% / −21.1% / ~−5% / −16.0% on default / stress / san-miguel-lp / powerplant, with the `leaf` control flat on AMD at every setting (0.501/0.506/0.500/0.500), which is what pins the win to LDS occupancy rather than a second effect. The per-value table and the reason 16 beats 8 live at `lane_stack()` in trace.rs. Gated: `--check-gpu` passes at 8 and 16 on BOTH vendors, every exact-zero counter 0, NVIDIA bit-exact (`max rel t err 0.00e0`).

**The transferable lesson is that a perf constant is only as good as the tree it was measured on** — the code these were tuned against changed underneath them in the same month, and re-running turned one of them into a no-op and made the other twice as valuable as first thought.

Two hardware results, one of which needed correcting on the re-sweep. **The occupancy-vs-pruning trade is vendor-asymmetric, but WHERE the wall sits moves with the leaf frontier.** A shorter stack coarsens the bound, so `t_start` shrinks and leaf rays traverse more; on an R9700 that costs ~nothing (**AMD re-origins the ray at TMin, so the inherited bound is already "free and worth nothing" there**) while on a 4070 Ti TMin really prunes. On the pre-coarse-frontier tree that wall sat between 32 and 16, where 16 measured `leaf` +44% on a 4070 Ti and ate the whole ladder win. On THIS tree it sits one notch lower, between 16 and 8: 16 is free on both vendors, and 8 sends 4070 Ti `leaf` 0.579 → 0.637 (+10.0%) on powerplant — the 12.8M-tri scene where the bound has the most to prune — making it a net 3.6% regression there while still winning on the other three scenes. So the LESSON transfers and the NUMBER does not; sweep both vendors, include a heavy scene, and score `leaf` as well as the ladder. And **`LEAF_GROUP = 32` was confirmed optimal on RDNA4 at the then-shipping `LEAF_TILE` = 8** (g64/g128/g256 read 1.111/1.379/1.782 against g32's 0.879), while the Intel `(LEAF_TILE 32, LEAF_GROUP 256)` optimum **reproduced on AMD** at `(32, 128)` = 0.701 ms, −20.3% — that one is no longer a follow-on, it is what shipped. **`SKY_GROUP`/`SKY_SPLIT` were not swept** — `sky` measures 0.000-0.003 ms on all three scenes here, so there is nothing to win.

## Intel Arc / Xe2 (Battlemage): what the hardware actually offers

Researched 2026-07-26 against primary sources (Intel's *Arc Graphics Developer Guide for
Real-time Ray Tracing in Games* v4, the `igdext.h` public header, Microsoft's DXR 1.2 / SER /
OMM / work-graph specs, and 1438 real `d3d12info` capability dumps including two Arc Pro B70).
**The vendor-specific surface is narrower than it looks, and most of it this tree already
satisfies.** Recorded so it is not re-researched.

**Ruled out — do not spend time here again.**
- **Intel Extensions for DirectX (`igdext`) cannot control SIMD/wave width.** The only `SIMD`
  token in the 1326-line header is a read-only `SIMD16Required` query;
  `INTC_D3D12_CreateComputePipelineState` is a shader-BYPASS path (CM/SPIR-V/ESIMD instead of
  DXIL) with no width field. Its D3D12 HLSL surface is 9 functions, all 64-bit atomics. There is
  no public SDK repo any more (headers survive vendored in `intel/gits`, v4.20.5).
- **Its 64-bit typed-atomics extension is obsolete on Battlemage**:
  `AtomicInt64OnTypedResourceSupported` is false on A770 but **true** on B-series, so the
  standard SM 6.6 path covers it. (Groupshared 64-bit atomics remain unsupported on both.)
- **Opacity Micromaps: not supported on Intel** ("actively evaluating" — Microsoft). Our
  `ALPHA_CUTOUT` candidate loops stay the answer, and `FR_ABL=noalpha,notrans` already measured
  that whole ceiling at −2.3%, so this costs nothing.
- **SER gates on Shader Model 6.9, not `RaytracingTier`.** Measured here: the B70 reports SM 6.8
  and the 4090 SM 6.8, so neither can run it today — and SER accelerates *many divergent hit
  shaders*, which a one-shader-record SBT does not have.
- **XMX is dead weight** without a neural stage of our own (XeSS already uses it through Intel's
  DLL). `WaveMMA` never shipped in any Shader Model; the D3D12 door is Cooperative Vectors, which
  Microsoft has already deprecated in favour of an SM 6.10 redesign.

**Already satisfied — three facts that explain existing decisions, and one rule not to break.**
- **THE RULE: groupshared memory is allocated out of the same L1 that services the RT unit**
  (RT guide v4, p.26), so LDS in a ray-tracing kernel degrades ray throughput — "even if the
  groupshared memory is running on a different queue (e.g. an Async compute queue)". This tree is
  compliant by construction and that is worth keeping: the two kernels that trace rays
  (`cs_leaf`, `cs_hemi_leaf`) have **zero** groupshared — neither pastes `frustum.hlsli`, see
  `trace.rs`'s `leaf_of`/`hemi_leaf_src` — while the LDS-carrying kernels (`cs_level*`,
  `cs_hemi_root/cell`) trace no rays; and with one DIRECT queue and barriers between the ladder
  and the terminal fills, LDS-heavy and RT-heavy work never overlap. `leaf.hlsl`'s header carries
  the warning. Intel's prescribed alternative to LDS traffic — **wave intrinsics** — is what
  `ctr.hlsli`'s `ctr_add`/`ctr_bump` and `wavefront.hlsl`'s `gw_alloc`/`gw_min_if` now use.
  (Caveat: Intel's statement is Xe-HPG-era. Xe2 unified L1/SLM further, 192 → 256 KB, so the
  structural premise is if anything more true — but the penalty is not re-confirmed for Xe2.)
- **The Thread Sorting Unit sorts by shader RECORD, not shader function**, and it is disabled by
  RayQuery. That is why `--dxr-inline 1` beat recursive TraceRay 4–6× here rather than losing to
  it: our SBT has effectively one record ("identifier-only SBT records, no local root
  signatures"), so the DXR 1.0 path paid the full repacking cost — live state spilled to the ray
  stack, thread terminated, continuation dispatched, payload through memory, 256 B/ray minimum —
  for **zero** coherence benefit. Intel's own document predicts that outcome, and the guidance
  would only flip if we grew many materially different hit records.
- **Xe2 made `ExecuteIndirect` a hardware block** (Intel quote: up to 12.5× vs Alchemist's
  software emulation), which retroactively justifies the wavefront ladder on Intel and matches
  our own measured ~11 µs of ladder dispatch overhead across 8 levels. Intel's "avoid
  ExecuteIndirect on buffers of size 0 or 1" rule is **A-series-era guidance for the emulated
  path** — do not act on it for Xe2 without measuring.

**Measured caps on this box** (printed by `--check-gpu`; `query_caps` now walks the shader-model
seed down from 6.9, having previously reported 6.7 purely because 6.7 was what it asked about):

| | Arc Pro B70 | RTX 4090 |
|---|---|---|
| RT tier | 1.1 | **1.2** |
| shader model | 6.8 | 6.8 |
| wave lane count | **16..32** (A-series: 8..32) | 32..32 |
| total lanes | 8192 | 16384 |
| work graphs | **Tier 1.0** | Tier 1.0 |
| `WaveGetLaneCount()` @ group 32 / 64 / 256 | **32 / 32 / 32** | 32 / 32 / 32 |

Two things follow. The lane count is a **range the driver picks inside per shader**, so the caps
never predict it — `trace::wave_probe` asks a kernel of each shipping group width and
`--check-gpu` prints the table (it also FAILS on an inconsistent report, since aggregation that
reasoned about the wrong partition would be silently wrong). That consistency check is a
**ceiling**, `waves == ceil(group / lanes)`, never exact division: a group NARROWER than the wave
is one PARTIAL wave, which is exactly what a 32-thread group is on wave64 hardware — and 32 is a
shipping width (`cs_level`, `cs_level_wide`, `cs_hemi_*`), so an exact-division predicate would
fail the suite on AMD RDNA for no defect at all. And Xe2 dropped SIMD8, which is the
8 → 16 minimum above. **`LEAF_GROUP = 256` was never a wave-matching result** — at 32 lanes it is
8 waves — it is an occupancy/dispatch-shape result, exactly as its own doc comment concluded.

**Wave-aggregated atomics — SHIPPED, and MEASURED NEUTRAL. Do not re-derive this.** Intel's
guide prescribes wave intrinsics over shared-memory traffic, and the tree used ZERO of them, so
the three atomic hot spots were converted: `bound_query_wave`'s `gw_min_if`/`gw_alloc` (the FTREE
round was issuing up to 8 LDS atomics per lane per node — ~256 per group iteration to one
address; it is now one wave reduction and one atomic per node, via a per-lane bitmask compacted
by popcount rank), `level_finish`'s counter bumps through `ctr.hlsli`'s `ctr_add`/`ctr_bump`
(layered ON TOP of the HOMOGENEOUS-BATCH quadrant folding: 4 x 32 becomes 1), and `leaf.hlsl`'s
per-pixel `CTR_HEMI_PT`. Revert arm: `FR_ABL=nowave` (`nobatch,nowave` is the full pre-wave-pass
queue code).

Measured `--spin path` 1080p, interleaved, rep 1 discarded, median of 3, windowed `gpu frame
span`: **B70 default 0.780 vs 0.780 (0.0%), B70 stress 1.045 vs 1.045 (0.0%), 4090 default 0.246
vs 0.246 (0.0%), 4090 stress 0.505 vs 0.510 (-1.0%)**. The B70 repeats to ±0.001 ms, so those
zeros are real, not noise. **The atomics were simply never the bottleneck** — the same conclusion
the ladder's own comment already reached about prep dispatches and barriers (0.011 ms across 8
levels): the cost is the level KERNEL descending the BVH, not the bookkeeping around it. Kept
because it is strictly less shared-memory traffic, is what Intel documents, and costs nothing;
but do not expect it to buy anything, and do not spend further effort on atomic contention in
this ladder without new evidence. CAVEAT ON THAT TABLE (2026-08-01): those A/Bs were taken while
`nowave` reached only the ctr.hlsli half — wavefront.hlsl's `gw_*` frontier aggregation stayed
ARMED in the "revert" arm (the probe-reach trap, instance 3; the arm is dual-homed now). The
leaf/sky/counter halves really were neutral as measured; the LADDER half of the feature was never
actually A/B'd until the repair. RE-MEASURED 2026-08-01 with both halves armed (`--spin path`
1080p, current tree; B70 2 reps ±0.002, 4090 3 reps), and THE VERDICT INVERTS: `FR_ABL=nowave`
BEATS the shipping code on BOTH vendors — B70 span 0.637 → 0.628 default / 0.775 → 0.752 stress
(ladder 0.110 → 0.102 / 0.200 → 0.180, −7%/−10%), 4090 span 0.255 → 0.252 / 0.350 → 0.345
(ladder −5%). `nobatch,nowave` lands BETWEEN baseline and nowave (B70 stress 0.759), which
decomposes the pair cleanly: the HOMOGENEOUS-BATCH half is a keeper, and the `gw_*` frontier
aggregation is the whole regression — small (~1-3% span, 5-10% ladder), cross-vendor, and
invisible for a month because the revert arm never reached it. The "costs nothing" premise above
is retired; flipping the gw half back to plain atomics (keeping ctr.hlsli's, which the old
half-armed A/B measured correctly as neutral) is the open follow-on.

**THE 2026-08-01 PRESSURE/OCCUPANCY CAMPAIGN — the remaining questions answered behaviorally.**
(1) DEAD-ARM REGISTER PRESSURE (the LEAF_NO_FB class) is real but MARGINAL on Xe2:
`FR_ABL=noffcode,noelcode` (compile the firefly + emissive code OUT — a day frame executes
identically either way, so the A/B isolates pure allocation) reads leaf −2.1%/−3.2%
(default/stress spin — ~9 µs absolute) and **~0 on THE WORLD** (leaf 1.011 baseline vs 1.01-1.03
across arms — the world's leaf is ray-bound, not allocation-bound). So the 2×2 leaf-PSO ship —
plus the sky/reference/DXR variants the firefly axis would drag in, each paying Arc's
async-compile warm-up — is NOT taken; the probe arms stay as documented instruments. (2) The
IGC ISA route is BLOCKED on driver 8805: `IGC_ShaderDumpEnable`/`IGC_DumpToCustomDir` (plus the
EnableAll/PidDisable variants) produce ZERO files from the D3D12 UMD, no default dump dir
appears — and the registry route is NOW ALSO PROVEN DEAD (2026-08-04, elevation obtained): the
same value names verified present under BOTH `HKLM\SOFTWARE\INTEL\IGFX\IGC` and the adapter's
class key (`...\Control\Class\{4d36e968-*}\0002\IGC`), a GENUINELY FRESH kernel variant forced
past the D3DSCache (`FR_BALLAST=7` — the width report proved it compiled and ran), zero files
anywhere. ISA dumps are unavailable on 8805 by every documented route; the occupancy question
is answered behaviorally instead — see the FR_WIDTH paragraph below, which supersedes
"finding 1: cs_leaf is not allocation-crippled" with the direct width readings. (3) Reference
points on the current tree for future diffs
(2 reps, ±0.002): procedural spin span **0.636 ms** (leaf 0.419, ladder 0.110), stress **0.775**
(0.462/0.200); parked WORLD XeSS session span **3.23** = leaf 1.011 + sky/caches ~0.15 + feed
0.231 + xess-eval 0.522 inside the replay bracket — reconciling exactly with the recorded 3.30
baseline.

**THE 2026-08-04 REGISTER-CLIFF CAMPAIGN — the pressure story MEASURED, not inferred**
(`FR_WIDTH=1` + `FR_BALLAST=N`, both default-off, unarmed sessions untouched — the tzero
class; gates green armed and unarmed on B70 + 4090, check.png byte-identical). **FR_WIDTH**
arms a WIDTH_PROBE epilogue in every real kernel (counter slots ≥ CTR_COUNT — never zeroed,
never gated, by construction; DXR gets a dedicated `width_buf` at its otherwise-unbound u3):
each kernel reports its COMPILED `WaveGetLaneCount()` — the per-shader SIMD width IGC picks
from register pressure, printed at the spin accounting site, both check suites, and the C-key
verify. THE TABLE (B70, driver 8805): **leaf=16 hemi=16 reference=16 dxr-raygen=16
dxr-shade=16 vs sky=32 level=32**; the 4090 control reads 32 everywhere. Three readings:
(1) every RayQuery-carrying kernel compiles SIMD16 — ctr.hlsli's old "32 at every group
width" note was the TRIVIAL wave_probe talking (that probe measures group shape, not the real
kernels; note amended); (2) the DXR raygen reads 16 even THIN (mode 3's bare-hit raygen) —
the RT launch regime itself narrows, independent of footprint (the brief's hypothesis 1,
half-confirmed from inside); (3) reference=16 == dxr-shade=16 means the 1.9× deferred-kernel
penalty is NOT width — it is SPILL AT SIMD16. **FR_BALLAST=N** proves that directly: N
synthetic live floats (loop recurrence on the traced t — not dead, not rematerializable,
[unroll] register-resident; folded under a never-true `spp == 0xdead` branch so the image is
bit-identical) injected into cs_reference. THE KNEE: per-float cost runs ~1.5-2 µs to N=48
(occupancy dilution), breaks 3× between 56 and 60 (0.704 → 0.785 ms — a +0.08 step where
4-float steps cost ~0.01), and accelerates past it (N=160 = 1.613 ms, 2.6× baseline) — **the
reference kernel's own live state sits ~56-60 floats below IGC's spill edge**, and dxr-shade's
1.9× is bracketed by reference + O(100) ballast floats. THE STRIP SWEEP (FR_ABL × FR_WIDTH on
dxr-shade, base 1.238 ms): nosec 0.463 (−0.775), norefl 0.750, **noglass 0.749 — −0.489 on a
scene with ZERO transmissive geometry** — and the single-strip savings SUM to 1.22 vs nosec's
joint 0.78: **cost near the cliff is a THRESHOLD, not a per-feature sum** — removing EITHER
big arm's live state clears the same spill edge (the CHS campaign's sub-additivity, reproduced
in plain compute; noffcode −0.05 = lottery noise). No strip flips 16→32 — SIMD16 is sticky for
RayQuery kernels; the lever is spill traffic at 16, not width. CONSEQUENCE for the mode-3
follow-on: `dxr-shade < reference` is reachable by getting the deferred kernel's live state
under the knee — the norefl/noglass overlap says the reflection+glass DFS stash is the hog, so
splitting the reflection lap (or hit/sky) out of the kernel is the measured route, not a guess.
Sweep discipline: every N and every strip is a NEW kernel variant (maiden discard), width read
from the report line, ms from the LAST gputime table.

*Measurement trap this campaign re-learned the hard way:* `--gpu-timing` prints a table every 120
frames AND at exit, and a parser that takes the FIRST match reads frames 0-119 — the coldest
window there is, where `win ms` == `mean ms` by construction. That alone manufactured ±20%
"noise" and an apparent 19% NVIDIA win, in a region (`leaf+sky`) the change cannot even touch.
Always parse the LAST table.

**THE 2026-08-04 DISPATCHRAYS-REGIME CAMPAIGN — the launch tax's mechanism, closed.** The
register-cliff campaign left one anomaly: IDENTICAL code (mode-2 raygen == cs_reference, same
compiled SIMD16) paid 2-2.4× under DispatchRays on fat scenes and parity on thin ones. Three
instruments answered it the same day. (1) **`dxr stack:` line** at every DxrGpu construction —
`GetShaderStackSize` per export (hit-group members by the qualified `group::stage` spelling;
0xFFFFFFFF prints `-`) + the driver's default `GetPipelineStackSize`, off the SAME
`ID3D12StateObjectProperties` the SBT identifiers already cast; `FR_DXR_STACK=min|<bytes>`
overrides via `SetPipelineStackSize` (`min` = the call-graph bound from the driver's own
numbers — mode 2 has no TraceRay, so its true bound is the raygen frame alone; undershooting
real usage is device removal by spec, so never guess). VERDICT: **stack reservation REFUTED**
— B70 defaults are tiny and honest (mode 1 = 112 B, mode 2 = 64-192 B even on SM-lp; the
formula is visible: mode 0 = 80 + 2×1056 = 2192), so there is nothing to reclaim and
FR_DXR_STACK is a probe, not a lever. The one gem: mode-0's uber CHS reports **1056 B ≈ 264
floats of TraceRay-live state** (4090: 544 B) — the driver itself printing why mode 0 was
catastrophic; NVIDIA reports near-zeros everywhere else (mode 1: raygen=32, rest 0). (2)
**`FR_BALLAST=dxr:N`** (the reserved prefix, implemented — reference.hlsl's three ballast
blocks mirrored into dxr.hlsl's mode-2 arm under a compound `BALLAST_N && DXR_INLINE_SEC==2`
guard; dxr.rs pushes the define only at mode 2 and refuses other modes loudly, since a seed
whose update compiled out would "measure" a flat curve). THE KNEE-VS-KNEE (default procedural
`--spin path`, same binary same day, maiden discard, last-table 600-frame-lap mean, ms):

| N | B70 ref | B70 raygen | 4090 ref | 4090 raygen |
|---|---|---|---|---|
| 0 | 0.610 | 1.646 | 0.270 | 0.260 |
| 16 | 0.638 | 2.097 | 0.270 | 0.427 |
| 32 | 0.660 | 2.799 | 0.263 | 0.594 |
| 56/64 | 0.704/0.780 | 4.038/4.400 | — /0.433 | — /0.952 |
| 160 | 2.078 | 10.675 | ~1.2 | 5.122 |

THE READING: **the raygen has NO knee — it prices live state from float ZERO** at ~20-45
µs/float on the B70 (accelerating; compute pays 1.7 µs/float to its +56 knee), and the N=0
gap (1.646 − 0.610 ≈ 1.0 ms) ≈ the kernel's own ~50 live floats at that rate — **the entire
baseline DispatchRays tax IS live state × the RT launch regime's spill rate**. The 4090
control shows the same SHAPE at lower severity: raygen prices immediately (~10 µs/float,
compute free to +32) but its budget still covers the kernel's own state (baseline parity
0.260/0.270), with a second cliff at 128→160 (2.03 → 5.12). Width never flips (16/32
throughout) — spill traffic, not width; cross-vendor in shape, Intel-specific in severity.
(3) **The host×strip cross** (SM-lp, B70, `FR_ABL` × {compute reference, mode-2 raygen}):
base 0.710/1.908, nosec 0.534/**0.696** — stripping the secondaries brings DispatchRays to
COMPUTE PARITY, i.e. the whole real-scene host gap is the secondary machinery's live state —
norefl 0.687/1.131 (the same code removal saves 0.023 in compute and 0.777 in the raygen,
**34× amplification**), noglass 0.706/1.195 (glassless scene — the threshold again),
noalpha,notrans 0.547/1.136; and the DXR column repeats the non-additivity (singles sum
−2.26 vs joint nosec −1.21). CONSEQUENCES: the world's 2-2.4× tax and the ~65% hybrid margin
are fully mechanistic — fat-shader live state at the regime rate — so the only real levers
are the THIN-RECORD family (mode 3's bare-hit raygen, sbt-2 specialization: both already the
measured winners, now with their mechanism), and there is no stack/width/scheduling fix left
to find locally. The remaining WHY — what the RT launch regime does to the register budget
that compute hosting doesn't (128-vs-256 GRF mode? ray-state co-residency in the GRF?) — is
IGC/driver internals, i.e. the brief's territory; the knee-vs-knee table is the repro to
hand over. Discipline notes: the exit gputime table can be a DEGENERATE 1-frame window —
parse the LAST row match's `mean ms` column, which stays valid there; and an explicit
`--spin-frames` under 1600 exits 2 on Arc (the warmup guard) — pass `--spin-warmup 0` for
construction-only reads like the stack line.

**THE 2026-08-05 FINDING-1 AUDIT — the DispatchRays-tax claim adversarially re-measured, and
it survives as the FLOOR of a band.** Before the brief's spearhead claim ("DispatchRays costs
~2x on register-fat shaders with zero TraceRay") shipped to Intel, it was re-proven at every
layer, same driver 8805. (1) **Zero TraceRay is now an ARTIFACT proof**: FR_DUMP_HLSL the
mode-2 library, `dxc -T lib_6_5 -HV 2021 -O3 -Fc`, grep the disassembly — `dx.op.traceRay` =
0 on both the procedural and SM-lp configurations, `rayQuery_TraceRayInline` = 16 each (the
anti-vacuity; the reference-unit control reads 0/9). Grep the SOURCE and you get 6 hits, all
preprocessor-dead — the DXIL is the artifact; a cargo pin
(`mode2_raygen_tracerays_are_preprocessor_dead`, a DEPTH-TRACKED #else scan — the mode-2 arm
nests a SKY_LOD #if/#else, and anchoring on the first textual #else would false-pass a
mode-2-live TraceRay) guards the guards. (2) **THE BRACKET-ASYMMETRY TRAP, a reusable
class**: the `reference` gputime region contained bind_common + BOTH cloud-cache fills + the
PSO set + the trailing barrier, while `dxr-rays` wraps the bare DispatchRays — every recorded
reference-vs-dxr-rays compare was two DIFFERENT bracket shapes (~3% in the DXR arm's favor
here; the general lesson: two regions compared as flat must first prove the same nesting).
`record_reference` now carries a nested **`reference-kernel`** row mirroring `dxr-rays`'
shape exactly — the like-for-like column for every future compare. (3) **Re-measured
like-for-like** (min-of-2, spin last-table mean / world WinMs): B70 procedural 0.569 vs
1.655 = 2.91x, SM-lp 0.674 vs 2.221 = 3.30x, SM-lp --tile 2 0.649 vs 1.957 = 3.02x; THE
WORLD (FR_REF sessions, --tod 11, boot pose) 1.178 vs 2.808 = **2.38x** (mode 1 3.11 =
2.64x; the recorded 1.28/2.59/3.13 replicate as outer-bracket 1.215 / 2.81 / 3.11 — mode 1
within 1%). 4090: 1.12x/1.36x/1.24x — cross-vendor in shape, Intel in severity. (4) **The
build lottery is ONE-SIDED**: three comment-touch rebuilds of trace_common.hlsli moved the
mode-2 raygen 1.18-1.66 ms (procedural) / 1.81-2.22 (SM-lp) while the compute reference
repeated to ±0.001 across ALL variants — the codegen instability is entirely the DXR door's.
Ratio bands: procedural 2.07-2.91x, SM-lp 2.68-3.30x — **"~2x" is the BAND FLOOR, never
crossed below by any draw, scene, or arm**. (5) **FR_DXR_LEAN=1 — the dead-exports control**
(dxr.rs; mode-2-only by soundness: raygen-only export list on the SAME DXIL blob, zero hit
groups, MaxTraceRecursionDepth 0, null miss/hit SBT ranges; default-off byte-identical;
check-dxr green armed on both vendors): the tax PERSISTS raygen-only (world 1.93x, SM-lp
2.85x, procedural 2.92x) — finding 5's intercept attribution confirmed by its strongest
control — AND removing the dead-but-exported fat entries recovers a dose-responsive **0%
(procedural) / 12% (SM-lp, 2.193→1.922) / 18% (world, 2.81→2.28, 0.53 ms)** of dxr-rays on
the B70 (4090 nil): the driver pays real per-dispatch cost provisioning exports no ray can
reach. (6) Width parity at the exact configuration: FR_WIDTH on SM-lp reads reference=16 AND
mode-2 raygen=16 — the previously-unrecorded cell. **FR_REF=1** (main.rs) starts an
interactive session in the reference arm (first-entry default only; resize re-entries keep
Persist) — the world reference arm was previously reachable only by a manual R keypress no
scripted protocol can fire. Verdict shipped in the brief v1.7: heading re-scoped to "2-3x on
the same code", the "register-fat" qualifier retired (finding 5: the kernel's own ~50 live
floats price from float zero — slim scenes pay too), and the audit block added to finding 1.

**FR_WAVEVIZ (2026-08-05) — wave footprints made visible, and the launch-packing question
closed.** `FR_WAVEVIZ=1` arms the wave-ticket overlay: each covered kernel's wave mints ONE
ticket (`InterlockedAdd` + `WaveReadLaneFirst`, minted converged at kernel entry — per wave,
never per stride iteration), every pixel stores its wave's ticket as its LAST tbuf touch
(`asfloat` bit-cast — tbuf has no live-frame consumer), and the resolve stage hashes
ticket→color under the runtime `FLAG_WAVEVIZ` bit (32768). **I** toggles it live in GPU
arms (display-stage only, NO resets either way; plain presentation only; C-verify stands
down while live — tbuf holds tickets); headless `--spin` runs arm it for the whole run and
dump `waveviz-<arm>.png` + a compactness line (waves, px/wave, bbox stats — main.rs's
`waveviz_dump`, whose Rust hash mirrors resolve.hlsl's `wv_hash_color` term for term).
Covered: reference, leaf+sky (full wavefront coverage), the DXR raygen at inline 1/2
(mode 0 = lib_6_3 no wave ops, mode 3's thin raygen is pinned to write no tbuf — both
refuse loudly); `FR_WAVEVIZ=chs` (mode 1 only) tickets `chs_shade` instead, with the raygen
sentineling misses 0xFFFFFFFE (rendered dark). Compute units bump `counters[CTR_WV_TICKET]`
(= 30; CTR_TOTAL 31 — the never-zeroed ≥ CTR_COUNT class); the DXR pipeline uses
`width_buf` slot 2 (created for either probe now). Unarmed sessions byte-identical
(conditional defs pushes; `waveviz_blocks_are_guarded` + the widened dxr guard test pin it);
both check suites green unarmed on the 4090. THE ANSWER (60-frame parked spins, 1080p,
means over all waves): **launch packing is screen-tiled and FULL on BOTH vendors** —
reference / mode-2 raygen / mode-1 raygen-end all read bbox exactly 32 (4090, 8×4 at
SIMD32) or 16 (B70, 8×2 at SIMD16) at 100% compact with every lane live, so the
DispatchRays grid is packed exactly like the compute grid and the folklore is now a
measured row — **and the BTD-scatters-waves hypothesis is REFUTED, corroborating the
regime-pricing attribution** (the mode-2 tax cannot be packing). THE NEW FINDING: **Intel
fragments waves at TraceRay boundaries; NVIDIA does not.** B70 mode-1 raygen-end reads
195,203 wave executions for 129,600 waves' worth of pixels at **10.6/16 live lanes** (mean
bbox 132, 96.9% compact — mostly tiled, ~3% scattered shards), and the chs arm reads
163,284 hit waves at **8.9/16** — while the 4090's continuation and hit waves stay full
(31.9-32.0/32) and perfectly tiled. So mode 1 on Arc pays a SECOND mechanism beside the
live-state pricing: ~1/3 of post-TraceRay lanes are dead, i.e. TraceRay-heavy pipelines
lose wave occupancy at every stage boundary — consistent with finding 2's mode-0 column
and worth a brief row. The wavefront arm reads 1.5% compact BY DESIGN (our own grid-stride
spreads each wave across its whole ~540-px tile — the instrument documenting our dispatch
shape, not the driver's; do not read that row as a driver finding).

**Work graphs (`FR_WORKGRAPH=1`) — the ladder as a D3D12 work graph.** The one genuinely NEW
Xe2 capability, and the queue records were already "work-graph-shaped". `src/gpu/shaders/
workgraph.hlsl` replaces `cs_seed` + depth_full x (`cs_prep` + ExecuteIndirect) with ONE
`DispatchGraph`: a broadcasting node per shallow tile (the `bound_query_wave` frontier) handing
off to a coalescing node for deep levels (32 tiles per group — the `WIDE_LEVELS` split, which
work graphs express natively). Leaf and sky records keep their UAV queues, so every terminal
gate is untouched; only the ping-pong tile queues go, because they assume levels SERIALISE and
that is exactly what a graph breaks. `level_finish` is NOT forked — `#if defined(WORKGRAPH)`
swaps its child emission from `qout` to an `out TileRec[4] + mask` the node compacts by
popcount rank.

**Status: correct, and blocked on Intel's driver — re-confirmed on 32.0.101.8805 (2026-08-01):
the refusal arm was deleted locally per its own instruction and the IDENTICAL 0xC0000005 landed
at the first graph dispatch, backing ask still 517.62 MB, state object still building happily;
the arm now records both driver versions.** On the 4090 the whole `--check-gpu` suite
passes with the graph armed and the result is **bit-identical** to the ladder (`leaves 768 |
sky-tiles 4 | splits 257 | blocked 256 | cuts 65 | overflow 0`, same-seed image `mean |d| 0.00e0
max 0.00e0`). That took a fix worth recording, because the gate that caught it only fires at
half the resolutions: `cs_seed` used to enqueue a root TileRec unconditionally, and the graph
takes ITS root as CPU input, so `CTR_TILE_A` sat at 1 for the whole frame with no ladder level
to consume it. The depth-accounting gate reads A or B **by `depth_full` parity** (the last
level's INPUT counter is legitimately non-zero, which is why it cannot just check both), so at
800x600 with `LEAF_TILE` 32 — `depth_full` 5, odd — it read B and passed, while `FR_LEAF=16`
(`depth_full` 6) failed it outright with `1 tile records left`. `cs_seed` now takes `push0 = 1`
to skip the enqueue. **A parity-selected gate is only half a gate: prove an arm against BOTH
parities before believing it.** On the B70 the state object builds and reports `WorkGraphsTier 1.0`, then
`DispatchGraph` takes an access violation with the debug layer and GPU-based validation both
silent — so `FR_WORKGRAPH=1` is REFUSED on Intel with a loud line (`trace.rs`, the first real
caller of `adapter::picked_vendor()`; delete that arm and re-test on a driver newer than
32.0.101.8515). The corroborating tell is the backing-memory ask for the identical graph:
**517 MB on Arc against 82 MB on NVIDIA**.

**Performance: a WASH on NVIDIA, and scene-dependent** (`--spin path` 1080p, same discipline,
windowed span): **default 0.262 graph vs 0.245 ladder (+6.9%), `--stress 5000` 0.486 vs 0.505
(-3.8%)**; `leaf+sky` is unmoved (0.0% / +3.3%), so the delta really is the ladder. That fits
the mechanism: the graph deletes ~6 prep dispatches and lets levels OVERLAP, which pays on a
deep many-tile stress field and loses on the shallow sky-heavy default scene where the ladder's
own dispatch overhead was already only ~11 µs and Xe2/Ada both accelerate `ExecuteIndirect` in
hardware. **It is therefore an env lever and never a default** — and note it is worth exactly
ZERO on a resting frame either way, because structure replay skips the ladder entirely.

Three spec rules that shaped the design and are easy to get wrong:
- **Output allocation must be THREAD-GROUP uniform, and "varying includes threads exiting."**
  `cs_level_wide`'s `if (gtid.x != 0) return;` before `level_finish` would therefore be
  undefined behaviour in a node. The wide node keeps all 32 lanes alive: lane 0 computes the
  split into groupshared, a barrier publishes it, then every lane calls
  `GetThreadNodeOutputRecords` with a per-thread count of 0 or 1 (a varying COUNT is explicitly
  allowed; varying control flow is not). `OutputComplete()` is mandatory even for a 0-record thread.
- **Only self-recursion exists** (a node may not target an ancestor), the longest chain is 32
  with recursion levels counted individually, and `[NodeMaxRecursionDepth(0)]` means "not
  recursive" — which would make a self-output an illegal cycle, hence the `.max(1)` clamps.
- **Exceeding `[MaxRecords]` or the declared recursion depth is memory corruption or device
  removal, NOT a caught error.** So the deepest node counts any child it must drop into
  `CTR_OVERFLOW`, which every suite already gates at exactly 0 — a silent drop would be a hole
  in the image, i.e. the false-sky class.

**`[WaveSize]` is deliberately unused.** It is a *validated constraint* that fails PSO creation
out of range, it is compute-only (so it can never reach `dxr.hlsl`), and forcing 32 on a
register-heavy kernel converts a compiler-avoided spill into a mandatory one — Intel's compiler
picks the narrower width precisely to avoid spilling. If it is ever wanted, `[WaveSize(16,32,32)]`
(SM 6.8, range form) is the portable spelling; a bare `[WaveSize(32)]` fails on AMD wave64 parts.

## Cinematic capture (`--cinematic`): the offline beauty path

A first-class headless MEDIA mode, peer to `--check` and `--spin`: stills and camera-spline video sequences, rendered deterministically for the README/release. `src/cinematic.rs` is the PURE half (data model, spline, presets, HUD composite, `self_test` — no GPU, no platform API, covered by `--check` on every platform); the drivers live in main.rs beside `run_spin`/`run_spin_gpu`, whose frame contract they mirror. Presets: `hero` (one still; bare `--cinematic` = this, plus the catalogue), `islands` (one per world island), `tour` (THE LAP — closed-loop flight over the ring, the attractors sweeping dawn → moonlit night), `orbit`, `foliage`, `hud`, `list`; anything else is a JSON shot-list path. **`foliage` is the one preset whose SUBJECT IS MOTION** (`FOLIAGE_FRAMES`/`FOLIAGE_ISLAND`, default 120 frames on san-miguel): leaf sway is a per-frame displacement of real geometry, so a still shows one pose of it and reads as the rest pose — the media set had no asset that could show the feature at all. It reuses `island_shots`' authored pose VERBATIM (the clip is literally the `islands` still with the clock running) and is LOCKED OFF, expressed as a ONE-KEY `Sequence` — `pose_at` short-circuits at `len == 1` and returns key 0 with its pinned hour, so no spline runs and the pose is bit-identical every frame while the sway/cloud/firefly clocks advance underneath. A moving camera was rejected, not skipped: parallax over a static tree looks like sway under a static camera, which is the exact ambiguity the shot exists to remove. `self_test` pins BOTH silent-failure modes — `FOLIAGE_ISLAND` must be CURATED **and** have an `ISLAND_FRAMING` entry (an un-authored name errors nowhere and yields 120 frames of a roof), and the one-key pose must not move across `u` (a future spline that clamped into a 4-point window instead of short-circuiting would un-lock the camera silently). Sub-flags all carry the `--cinematic-` prefix precisely so ONE `starts_with` clause covers the family in `settings::headless_args` (the `--spin`/`--spin-frames` two-arm wart, avoided).

**WHY IT EXISTS, given `--spin` already walks a camera path**: `--spin` is a benchmark that writes no pixels and whose path amplitudes are benchmark-sized (≤ 0.12 diag). More importantly the INTERACTIVE renderer cannot produce these frames. Every cinematic output frame renders as N sub-frames of ONE pose, which buys what a live session cannot have: per-output-frame convergence (reconstruction warm-up, or accumulation), and the hemisphere bounce integrator (`Quality::fb`), which is still-frames-only by construction. **Cinematic is therefore the only path in the tree that can render a moving camera WITH GI.**

**THE GPU ARMS CAPTURE THROUGH THE UPSCALER CHAIN BY DEFAULT (2026-07-30)** — the "no upscaler exists headlessly" era is over; that constraint was policy, not architecture (only the availability probe and the swapchain were HWND-bound). `gpu::CineUp` (gpu/mod.rs) probes DLSS-RR → FSR4-RR → XeSS → FSR3 honoring the session chain flags (`--no-dlss`/`--fsr`/`--xess`/`--fsr3`; `--no-upscale` = the empty chain = accumulation) at **100% render scale — render res == output res, DLAA-grade** — deliberately bypassing `quantize_res` (its 36-px floor would shrink non-multiple heights; the per-engine `wire_*_feed` range checks still gate). The engine states and eval middles are the interactive session's OWN (`probe_native`, `record_xess_eval`, `record_fsr3_upscale`, `record_fsr_rr_sequence`, `rr_ngx_sequence` — refactored to take a bare command list), so the two paths cannot drift; DLSS rides the plain `HeadlessGpu` since the SL retirement (raw NGX needs no queue hook): `NgxRr::open` on the harness device, a per-shot-res `RrFeature` created with `dlaa = true`, and the evaluate is `rr_ngx_sequence` — the interactive session's own middle. The upscaled frame contract: every sub-frame is a fresh jittered frame (`accumulate: false`, `frame_jitter: Some(dlss::jitter_for(seq))`, `prev_cam` = the previous sub-frame's pose — static within an output frame, the spline step at boundaries, ONE pose feeding both the shader MVs and fc's prev matrices), `seq` free-running across the whole shot (Halton phase never restarts; history carries across output frames), reset only at seq 0, quality = `cine_quality` (preset 3, NOT `upscaler_1spp` — cleaner input only helps the model), `record_frame` → `record_feed` → the engine eval on one `hg.run` submit, and the frame written is `CineUp::read_output`'s linear-f32 readback of the engine's RECONSTRUCTED RGBA16F output. `--cinematic-samples` keeps its meaning as sub-frames per output frame (now warm/converge passes — stills at 256 are a converged DLAA still). **OUTPUT FRAME 0 GETS A WARM-UP, and it is not optional**: because `seq` free-runs and `reset` fires once per shot, frame f is reconstructed from (f+1)·samples of accumulated evidence while frame 0 has only its own — at the sequence default of 32 that is under HALF a jitter phase, so frame 0 is both history-starved AND sampled on a biased lattice, and it gets written to disk like every other frame (in a LOOPING clip that discontinuity shows once per lap). `run_cinematic_gpu` therefore runs `JITTER_PHASE.saturating_sub(shot.samples)` extra passes at the same pose on f == 0, emitting nothing — the loop body is unchanged and only the model state after the final pass is read back, so `reset` stays `seq == 0` and self-consistently lands on the first warm-up pass. Self-limiting by construction: a 256-sample still is already 3.5 phases so the sub is 0, and a 32-sample sequence frame 0 lands at 72 — between frames 1 and 2 in accumulated evidence, which is the continuity wanted. Deliberately NOT on the accumulation arm: N sub-frames in, one unweighted mean out, frame 0 statistically identical to frame 5. **Fallback ladder, all loud, never fatal**: chain exhausted / feed wiring rejected / `--no-upscale` → the accumulation loop below; a GI shot ALWAYS takes accumulation (the bounce integrator needs accumulating stills — this is what preserves the moving-camera-with-GI capability); `--cpu` keeps the CPU accumulation arm. `--check*`/`--spin` remain upscaler-free (gates and benchmarks must not move).

**THREE INVARIANTS, in the order they are easiest to get wrong.** (1) **The volumetric clock is per OUTPUT frame, never per sub-frame.** `Clouds::cine`/`Fireflies::cine` take (out_frame, fps) and are baked ONCE per output frame into `CineFrame`; keying them off the sub-frame index would average N different skies into one image and smear the clouds INSIDE a single frame. The clock is REAL SECONDS (`frame/fps`), deliberately not `spin`'s `CLOUD_SPIN_DT` (1/120 — a benchmark cadence that would run a 30 fps film's sky at quarter speed). (2) **The spline interpolates POSITIONS, never angles**: keyframes carry eye + look-at target, both splined as points, camera rebuilt by `Camera::look_at`. `spin_path_pose` interpolates yaw/pitch offsets, which is safe only because its amplitudes are tiny — a lap sweeps yaw through 2π, where interpolating the angle runs backwards across the wrap. (3) **Time of day is a pure function of the frame index**, never eased: `cinematic::path_hour` samples the world's attractor field along the path and band-limits it with a SYMMETRIC WINDOW (±1.5% of the lap). Easing (what `flycam.rs` does at `TOD_RATE`) carries hysteresis, so the same frame would render differently depending on how the camera got there; the window gives the same smoothing with no history, which is what keeps any single frame re-renderable in isolation. The window is not optional: the raw field's `1/(d²+r²)` weight spikes passing a SMALL island and the self-test measures 2.5 h between adjacent frames without it.

**Structure replay is a pure win here and is why high sample counts are affordable**: the pose is bit-identical across the sub-frames, so sub-frames 1..N-1 re-dispatch the persisted terminal queues and skip every frustum query while re-shading from a fresh ctx. `--check`'s replay family already gates replay-vs-trace bit-identity of tbuf/info/accum at frame 0 AND at a warm jittered frame 1 — exactly this configuration. DXR has no quadtree to persist (`replay: opts.replay && use_wave`).

**ONE presentation path for all three arms**: CPU, wavefront and DXR all hand a LINEAR f32 image to `cine_write_frame` → `render::resolve_hdr` → `tone::ToneParams::SDR` → `save_png`. Nothing goes through `record_resolve`/`hdr`/the tonemap PS, which would put the GPU arms on the shader's curve and the CPU arm on tone.rs's — which is exactly why `CineUp::read_output` hands back linear f32 instead of routing through `read_hdr_output` (that helper SDR-tonemaps on the way out). The GPU arms read back ONCE per OUTPUT frame (not per sub-frame): the engine output on the upscaled path, `accum` on the accumulation arms. Arm policy differs from `--spin` on purpose: cinematic takes the best available arm by DEFAULT (`--gpu` wavefront, else DXR, else CPU with a loud line) because its job is to finish; `--cinematic-gi` forces the wavefront (DXR has no hemi stage) with a loud line rather than silently dropping the feature.

**HDR** (`--cinematic-hdr`): sequences write 16-bit PQ/Rec.2020 frames (`cinematic::pq_rgb16` — `tone::ToneParams::hdr10`, the SAME curve an HDR10 swapchain presents through) and encode to HDR10 HEVC; stills additionally write a linear OpenEXR master and a PQ PNG. **THE TAGGING TRAP, measured**: the `-color_primaries`/`-color_trc` OUTPUT options that correctly tag an HEVC/mp4 do NOT reach the AVIF `colr` box — encoding that way writes primaries=2/transfer=2 ("unspecified") with only the matrix surviving, and a viewer then renders a PQ image as sRGB, washed out and silently so. `-aom-params` is worse (loses the matrix too). Stamping the frame with `setparams` inside the filter chain is what lands 9/16/9. Verify with the colr BYTES; `ffprobe` does not surface an AVIF container box at all and reports "unknown" for a correctly-tagged file.

**Framing lessons, each learned from a bad 4K frame.** The `islands` series shoots from INSIDE the ring looking OUT at 30° depression: a fixed world-space look direction points across the ring for some islands (a "portrait of Powerplant" containing Sponza, Rungholt and Vokselia on the horizon), and 30° drops the horizon — and therefore every other island — just outside the frame. Distance is a BOUNDING-SPHERE fit (`d = 2.5·fit`, `fit = max(0.75·radius, 0.5·height)`), not a multiple of the footprint: `Island::radius` is an x/z measure and says nothing about how tall a subject is, which cropped the Damaged Helmet at the neck — hence `Island::height` (WORLD_VERSION 1→2). The `hero` shot instead SHOOTS INTO THE LIGHT at a shallow 12° (eye placed opposite the sun's azimuth via `scene::sun_dir_for_tod`), because the series angle turns its back on a sunset and renders golden-hour Bistro as a grey plain. Re-placing that eye must PRESERVE the keyframe's pinned hour — a bare `Keyframe::new` drops it and the attractor field at the new position rendered Bistro at midnight.

**THE BOUNDING-SPHERE RULE IS FOR SUBJECTS AND WRONG FOR ENCLOSURES, and that cost the whole first media set.** Everything in the paragraph above is a correct refinement of one framing rule — and the rule itself only fits a subject you photograph from outside. The Damaged Helmet is such a subject and its shot is excellent; Rungholt is a landscape and reads from above. The other five are ENCLOSURES, and fitting an enclosure's bounding sphere from outside at 30° photographs its ROOF: the shipped set had Sponza — the most famous atrium in computer graphics — as a rectangle of tiles, San Miguel's patio as a grey box, Bistro's street as a smudge, and Vokselia as a near-black field. No formula over a bounding box recovers "stand in the courtyard", so `cinematic::ISLAND_FRAMING` authors eye/target/fov/exposure per island in units of its own radius/height, `island_shots` prefers it, and anything absent (a user's scene, a new island) falls through to the sphere fit unchanged. `hero --cinematic-island` ALSO honours an authored entry instead of relocating the eye sunward — that relocation is the outside-a-subject rule, and applying it to an interior pose puts the camera back outside the building looking at a wall. `cinematic::self_test` pins every table key against `world::CURATED`, because a key that matches nothing fails SILENTLY into the sphere fit — restoring roof photos with every gate still green. Two second-order findings: enclosures need GI (`--cinematic-gi`) since their light is almost entirely bounce, and they need +2 stops (see `-exposure`) since their sun is occluded. Finding the poses at all needs the island GEOMETRY, which only printed on a cache REBUILD — `run_cinematic` now prints the `isle` table (center/radius/height/hour) on every world run, and a near-plan-view probe per island is the cheap way to locate a courtyard before authoring an entry.

**The `tour` lap stays OUTSIDE, and the interior version was built before being rejected.** Threading the lap through the authored interiors sounds strictly better and measures worse: transits between enclosures must clear both rooflines so most frames are high anyway, the climbs in and out clip walls, and swinging between floor level and above the ring seven times makes the attractor clock lurch so the day sweep stops reading as one continuous sunrise (clearance drops to 0.12r, which is the gate telling you the camera is inside the geometry). The lap's own constants moved though: height comes off `Island::height` rather than `radius` (a footprint multiple flies a cathedral and a flat city at the same altitude), and the depression is ~15° rather than ~28° — a shallower angle puts sky and horizon in frame and makes an island read as a place instead of a diorama on a table.

`cinematic::self_test` (in `--check`) gates: spline determinism, `pose_at(0)` == key 0, closed-loop seam C1 **as a SCALING property** (a one-sided finite difference carries O(eps) truncation from the path's own curvature, so a fixed tolerance measures the probe, not the spline — uniform Catmull-Rom is analytically C1 at its knots and a naive limit rejected a provably-continuous seam), open-path endpoint clamping, the hour circle's short way round, HDR/PQ anchors, the ffmpeg command construction (HEVC + `hvc1` — QuickTime and Safari refuse the file without that tag — PQ tags, 10-bit, `setparams`), even-dimension rounding, and **the clearance gate**: every sampled pose on a generated lap must stay > 1.05× island radius from every island, tested over hostile radius spreads. That gate caught a real bug — a closed spline straight through the island EYES chords across the ring, and at n=2 degenerates to a line through both islands — which is why `world_lap` emits a TRANSIT key on the ring arc between each pair. Its clock continuity is tested by SCALING too, because an absolute per-frame bound would measure how hostile the synthetic config is rather than whether the function is continuous.

`--cinematic` is exclusive with `--spin`/`--check*` (they are benchmarks and gates on their own fixed scenes) but is deliberately ABSENT from the `--world` exclusivity list and from `world_wanted`'s conjunction — that absence IS how it gets the world by default while `--check*`/`--spin` still never load it, so every structural must-fire gate stays tuned to the procedural scene's topology and **no gate moves** (`check.png` verified byte-identical across the feature). It is also the first path that composites the HUD into a saved image (`Hud::composite_sdr` + `Hud::settle`, which pumps 5.2 s of wall clock so the FPS graph's 40 × 125 ms buckets are populated rather than empty) — amend the "P screenshots and `--check` PNGs contain NO HUD" known-accept accordingly. Touch `cinematic.rs`, the `run_cinematic*` drivers, `Clouds::cine`/`Fireflies::cine`/the `sway_time` clock, `Island::height`, or `Hud::composite_sdr`/`settle` → run `--check` (+ verify `check.png` is byte-identical), `--check-gpu`, `--check-dxr`, then `--cinematic hero`/`tour --cinematic-frames 60`/`foliage --cinematic-frames 30` and LOOK at the output. **A look-affecting change anywhere ages the committed media set** (`docs/media/`), and nothing gates that either: the 2026-07-30 re-shoot found the seven island stills stale from a matclass vocabulary change alone — Rungholt's Minecraft canopy had gained real leaf translucency and the committed portrait still showed the flat pre-foliage clumps.

## FSR (AMD ffx-api): Ray Regeneration + FSR4, and FSR 3.1 upscale-only

Levels 2 and 4 of the always-on upscaler chain (`--fsr` force-starts at FSR4-RR, `--fsr3` at FSR 3.1; K toggles — F belongs to the DXR pipeline). **`--fsr4` is `--fsr` with the level REQUIRED**: same force, same AMD adapter preference, but a fall-through is fatal — `run_window` compares `Opts::fsr4_required` against the session's ACTUAL wiring (`GpuContext::fsr_flavor`, so the check can never disagree with what got wired) and exits **2** with the adapter name plus the flags worth trying (`--fsr3` — cross-vendor; `--prefer-amd` — pick the other GPU on a multi-vendor box; `--fsr` — allow the fall-through), the probe's own `fsr4: level unavailable (…)` line having already printed the reason. A flag that removes FSR4 from the chain outright (`--no-fsr`, `--no-upscale`, a later `--xess`/`--fsr3`/`--nppd` force — `--nppd` force-starts at XeSS) also exits 2, at parse time, saying so instead: the level was never probed, so "unavailable" would be a lie. This is deliberately the ONE place where an unsupported feature is not a loud line + a working fallback — the flag exists to be told. **Two flavors of the same ffx-api effect** (`FFX_API_EFFECT_ID_UPSCALE`; the provider is a per-context version choice, not a different SDK), probed as two SEPARATE chain levels: `ffxQuery`'s version enumeration is the support probe and `fsr::pick_version` (pure, `--check-fsr`-gated) turns it into the flavor — level 2 (`chain.fsr4`) requires the Ray Regeneration provider (RDNA4) and errs without it (an RR probe *error* degrades identically), falling through to XeSS; level 4 (`chain.fsr3`) is **FSR 3.1 upscale-only** (cross-vendor, the committed provider DLL carries 3.1.5) via `ffxOverrideVersion` chained at create with the enumerated id, and is the chain's LAST level — its failure is what exhausts the chain (loud line + plain). The shim pins its compile-time API version (`FFX_UPSCALER_VERSION`) **only when not overriding** — the pin and the override are mutually exclusive pNext branches (a 3.1 provider may reject the 4.x pin desc). Only the MIT header subset is committed under `SDKs/fidelityfx-sdk` (vendored from GPUOpen-LibrariesAndSDKs/FidelityFX-SDK @ v2.3.0; see its README); the signed runtime DLLs ship in the `FidelityFX-Samples-prebuilt` drop that the default `--ffx-path` points into (`FRUSTRACER_FFX_PATH` overrides). The SDK is reached through a C++ shim (`shim/ffx_shim.cpp`, built by `build.rs` against the vendored headers — the pNext desc chains and `FfxApiResource` descriptions exist only there, never mirrored in Rust) that **preloads every `amd_fidelityfx_*_dx12.dll` next to the loader by absolute path (globbed) before `LoadLibraryExW`ing `amd_fidelityfx_loader_dx12.dll`** — DLL-dir search does NOT make the loader find its providers (`LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR` covers only static imports; the loader's runtime `LoadLibrary`-by-name lookups resolve from the module list, which is exactly what the preload populates — deleting it regresses every `--fsr` session to NO_PROVIDER); nothing links FFX, so every `--check*` stays DLL-free. **No proxies anywhere**: every ffx context is created on the NATIVE device and their dispatches record into our command list — the FSR path rides the XeSS-style native pipeline (the chain wires exactly ONE upscaler per session — except under `--quinlight`, which wires every supported level at once and fuses them; the ffx contexts coexist with a live raw-NGX DLSS-RR session and with XeSS, all on the one native device). It DOES compose with both GPU tracers: `--gpu` and the `--dxr` default feed FSR on-GPU (`FeedKind::FsrRr` = the eleven-plane `cs_feed_fsr_rr` kernel, `FeedKind::Fsr3` = the XeSS trio into FSR3's planes), with `present_trace_fsr_rr`/`present_dxr_fsr_rr`/`present_trace_fsr3`/`present_dxr_fsr3` wrapping the shared `record_fsr_rr_sequence`/`record_fsr3_upscale` middles. An explicit `--fsr`/`--fsr3` also flips the default adapter preference to AMD.

Unlike DLSS-RR (which denoises the composited radiance), Ray Regeneration is a **decoupled per-signal denoiser**, and the session subscribes to **every signal the renderer has a source for** — `ffx_sys::SIGNALS` = direct diffuse | direct specular | **ambient occlusion** | **indirect specular**, which is exactly the three stochastic terms a 1-spp frame produces (the shadow ray, the AO ray, the GGX reflection ray). Subscribing to only the two direct signals was the noise bug: AO and the whole reflection bounce landed in the residual, which is passed through UN-DENOISED by design — the clean signals were being denoised and the noisy ones were not. The set is ONE constant because the ffx header requires the chained per-signal descs to match the context's creation `signalFlags` **exactly** (each desc is required "if and only if" its flag is set); the shim keeps no per-context state, so the flag word travels on the dispatch desc and `ffxshim_denoise` stitches the pNext chain in flag order, rejecting any flag it has no desc for (dominant-light visibility, indirect diffuse and specular occlusion are unplumbed — no source; the chaining is generic so they drop in). `shade()` exports the four captures through `PrimarySurface` (assignment-only, ZERO rng draws — the combined-color path is bit-identical with capture on, gated): `direct_d` (albedo-free by construction — no division), `direct_s` and `ind_s` (the reflection subtree's whole contribution, `tput * rcol`) demodulated by the wire F0, and `ao` (the raw open fraction; the reflection ray's hit distance rides the indirect-specular plane's A channel, straight off the pack's `spec_hit_t` lane). What remains is the **exact-remainder residual** (`color − dd⊗kd − ds⊗f0 − ao·amb(n)⊗kd − is⊗f0`), which now holds only DETERMINISTIC terms: emissive, the glass transmission chain, quantization slop. `ao` is 0 under `fb.gi` (whose ambient is real RGB irradiance, not an AO-modulated sky) and the residual absorbs that term whole, as before. **The wire kd is the EFFECTIVE diffuse albedo `albedo·(1−metallic)·(1−transmission)`** (`PrimarySurface::diff_albedo` — the ONE derivation both CPU sites share; the GPU pack mirrors the multiply order bit-for-bit), 2026-08-01: it used to be the raw `albedo·(1−metallic)` with the residual absorbing the `(1−transmission)` factor, which meant every denoiser delta on `dd`/`ao` remodulated at `kd` instead of the physical `kd·(1−transmission)` — a **33× amplifier on water** (transmission 0.97) that presented as smeared terrain-colored blotches across rungholt's ocean in RDNA4 FSR4-RR sessions (NVIDIA/XeSS sessions never split signals, so only AMD showed it; the DXR tracer itself was exonerated first — every DXR-library ray passes literal TMin 0, so the RDNA4 candidate-loop +TMin defect structurally can't fire there, and the radiance A/B at the rungholt water pose reads the same 0.029% before and after). The change is assignment-only at the PrimarySurface capture (`trans` field), zero rng draws, `check.png`/`check_gi.png` byte-identical, every feed/composite gate green on both vendors, and it IMPROVED the water-pose CPU↔GPU albedo A/B (mean |d| 0.0025/0.0021/0.0060 → 0.0013/0.0010/0.0010). Sheen's `(1−0.157·sheen)` and translucency's `(1−tl)` (which does NOT scale ambient, so it cannot be folded into a kd shared by `dd` and `ao`) remain the residual-absorbed wire approximations — both zero on water. The REMAINING water known-accept is the specular half: `ind_s` is a ripple-swinging mirror signal (~28°/frame of reflected swing) handed to the temporal denoiser with zero MVs and a stochastic per-pixel hit-t guide — if smear persists after this fix, the levers to sweep are `--fsr-stability-bias` / `--fsr-kernel-relaxation` / `--fsr-normal-strength`, and the designed follow-up is a stabilized geometric-normal guide at ripple pixels via the free `g.sig.w` lane.

**The AO signal's remodulation factor is DIRECTIONAL, and that is what makes this subtle.** It used to be the flat `shade::AMBIENT` constant, which every composite site could simply be handed. The one sky (sky.rs) replaces it with `sky_sh.irradiance(n)` — a *per-pixel* value, because a surface facing the sky gets more of it than one facing the ground. So the factor cannot ride in as a root constant, and the composite pass grew a NORMALS-plane SRV (t7) plus the sky's 9 SH rows in its constants; `sh.hlsli` is the shared evaluator (one formula, two cbuffer bindings — the tracer's FrameCb and the composite's root constants), and `fsr_wire.hlsli` is the shared octahedral wire encoding.

**Every site must build that factor from the WIRE normal** (`fsr::wire_normal` — oct-encoded, 10-bit, decoded back), never from the full-precision shading normal, because the composite pass has *only the plane bytes* to rebuild it from. Use `n` on one side and `oct10(n)` on the other and the identity acquires a quantization-sized hole that no tolerance should be widened to absorb. The one wrinkle: the CPU-fed path's plane is oct-encoded from the **f16** G-buffer normal (`ffx_rr::record_upload`'s `ld16`), while the GPU feed encodes straight from the pack's f32 — so `render.rs::write_fsr` rounds through `q16v` first and the GPU kernel does not. Each path is self-consistent with the plane *it* produces, which is all the identity requires.

The composite identity `dd⊗kd + ds⊗f0 + ao·amb(n)⊗kd + is⊗f0 + residual == color` is the correctness spine, and it lives in THREE sites that must agree exactly — `fsr::split_signals`/`composite`, `feed.hlsl`'s `cs_feed_fsr_rr` residual, and `gpu/shaders/fsr_composite.hlsl`. **All three are gated**: the first by `--check-fsr`, the second by the feed gates, and the third — the kernel that runs nowhere else, and so for a while was tested by nothing — by the `fsr composite identity` gate in `--check-gpu`/`--check-dxr`. It makes the denoiser an IDENTITY (`FsrResources::record_signal_passthrough` copies each signal's input plane into its denoised-output UAV), runs the real `record_composite`, and checks the result against an oracle built from the PLANE BYTES the GPU stored — so what it pins is exactly that kernel: arithmetic, albedo decode, the SH evaluation against the normals plane, SRV table ORDER, and root constants, at ≤ 2 f16 ulp (measured max 1), with must-fires that a live AO term and a live reflection term exist (without them a wrong ambient factor is invisible). It was written because the kernel shipped BROKEN: `cbuffer C { uint rw; uint rh; float3 ambient; }` — HLSL never lets a float3 straddle a 16-byte boundary, so DXC bumped `ambient` to offset 16 (root DWORDs 4..6) while `record_composite` wrote it at DWORDs 2..4 of the 5 declared, making the pass read one shifted channel plus two undeclared DWORDs. The cbuffer therefore LEADS with its vector block — now `float4 sky_sh[9]`, so the constants are 38 contiguous DWORDs (`sky_sh[0..9] | rw | rh`); **never put a scalar in front of it**, and the gate fails at 16724 f16 ulp if you do. `--check-fsr` gates it per-pixel on rendered frames (worst rel err 1.2e-7 measured), plus the sky contract (every signal 0, residual == the sky color) and anti-vacuity must-fires (an occluded-AO pixel and a reflection pixel must exist — at the 1-spp preset AO is BINARY, so "fired" means ao == 0, not a fraction). Frame flow: `FsrBufs` signals + `GBufs` planes upload as sub-rects → one `ffxDispatch` denoises all four signals → composite CS remodulates → FSR4 upscales to the window → the existing tonemap presents. On the GPU tracers the upload half disappears: `cs_feed_fsr_rr` writes all eleven planes straight from the pack (the demodulation happens at the pack WRITE site, so the kernel only widens the sig lanes and computes the residual), and `record_fsr_rr_sequence` — the denoise+composite+upscale middle factored out of `present_fsr` — is shared verbatim by the CPU-fed, `--gpu`-fed, and DXR-fed presenters.

**Denoiser tuning levers** (`--fsr-max-radiance`, `--fsr-stability-bias`, `--fsr-radiance-clip-k`, `--fsr-disocclusion-threshold`, `--fsr-normal-strength`, `--fsr-kernel-relaxation`): the `FfxApiConfigureDenoiserKey` constants, applied once at denoiser creation through the long-plumbed-but-never-called `ffxshim_denoiser_kv`. Each defaults to None = **configure nothing**, so a flagless session runs the provider's own constants and behaves exactly as it did before these existed; a rejected key is a loud line, never a session failure. `max_radiance` is the firefly clamp — the one that matters most to a 1-spp path tracer.

The **FSR 3.1 flavor** (`gpu/ffx_up.rs::Fsr3Resources`, `gpu/mod.rs::present_fsr3`) has no denoiser anywhere: no `FsrBufs`/`PrimarySurface` signal capture (never allocated — `ctx.fsr_buf` stays `None`, sound either way since capture-on/off bit-identity is gated), no split, no composite CS, none of the RR planes; its `GBufs` is the **slim variant** (`GBufs::new_slim` — mvec + depth only; the guide planes are zero-length and the fill sites skip their encodes). Exactly three inputs upload — the frame's 1-spp HDR shade straight from `accum` (RGBA16F via `fsr::f16_sat`), a **RG16F bit copy of `GBufs::mvec`** (pixel-space MVs, so the dispatch's `motionVectorScale` is the bare `UPSCALE_MV_SIGN` — post-scale FSR sees the identical pixel-space MVs the RR flavor hands it via sign·rw × UV-delta; one polarity constant, two flavors), and the R32F reversed-Z clip depth (`xess::view_z_to_clip_depth` — depth deliberately stays 32-bit, the codebase-wide f32 depth wire contract) — then one FSR 3.1 upscale dispatch with the full temporal input set (jitter, frameTimeDelta, camera near/far/fovy, reset; reactive/T&C masks stay null — no transparency in this renderer) → the existing tonemap. Both flavors share `fsr_upscale_desc` (every field flavor-identical except `motionVectorScale`), the `SRV_SLOT_FSR` tonemap, and `read_fsr_output`.

Dynamic input resolution is first-class on every ffx context (`maxRenderSize` at creation = the window; every dispatch names its own `renderSize`), so both FSR flavors reuse the XeSS choreography verbatim: `xess::ScaleCtl` + `quantize_res` + `StepLimiter`, `GBufs::set_res`/`FsrBufs::set_res` in-place reinterpretation on a step (no reset, no prev drop — `fsr_prev` is a Camera whose basis re-derives at each frame's res), `tprev_res` drops the temporal cache, `--lock-res` pins. Range: seed = the Quality-mode query, floor = UltraPerformance, max = window (`fsr::fallback_render_res` stands in headlessly, gated). Conventions live in exactly two places: **`src/fsr.rs`'s constants** (`JITTER_SIGN` — start unnegated, unlike SL; `DEPTH_SIGN` for the "signed" view-Z; `UPSCALE_MV_SIGN`; all to be settled empirically on RDNA4 like the SL/XeSS signs were — wrong jitter sign = 2× wobble static, wrong MV polarity = directional smear) and **`gpu/mod.rs`'s desc builders** (`present_fsr`'s denoise desc + the shared `fsr_upscale_desc`). Notable: `FfxApiMatrix4x4` is row-major/row-vector, which per its own compatibility table means glam matrices **memcpy with NO transpose** (deliberate contrast with SL's `row_major()`); our pixel-space y-down current→previous MVs are exactly the denoiser's `PreviousUV − CurrentUV` after a pure (1/rw, 1/rh) scale (no sign flip), with the depth delta in B from the `prev_z` captured at the fill site; the upscaler shares that MV plane via `motionVectorScale` and takes the reversed-Z clip depth (`xess::view_z_to_clip_depth` — the upscaler context sets DEPTH_INVERTED to match); albedos are sqrt-encoded RGBA8 (`NON_GAMMA_ALBEDO` is the escape hatch); adaptive shading stays XeSS-only for now. `--check-fsr` (DLL- and GPU-free) gates: encoder roundtrips through their wire quantization (octahedral 10-bit, sqrt-albedo 8-bit + idempotence), the composite identity on random inputs and per-pixel on rendered frames at a native and an odd resolution, the sky contract (0, 0, sky, prev_z = far exactly), prev-depth agreement through the MVs vs frame A's own depth buffer, `FsrBufs::set_res`, the range fallbacks, the **provider pick** (`fsr::pick_version`'s FSR4-default / 3.1-fallback / --fsr3-force triangle — an available RR provider alone selects FSR4, deliberately not gated on parsing a 4.x display name; FSR2 never picked, forced-but-absent → None, plus `parse_provider_versions` robustness incl. multi-triple names), capture-on/off bit-identity, and `sqrt_wire` (the GPU pack's wire twin — no leading f16 hop, since the pack stores f32; idempotent, exact endpoints, and equal to `albedo_wire` on already-f16 values, which is what lets the GPU feed gates use it as their oracle). Run it (plus `--check` and `--check-dlss`) after touching `fsr.rs`, `shade.rs`'s PrimarySurface, the G-buffer fill sites, or `gpu/ffx*`; add `--check-gpu`/`--check-dxr` when the change touches `feed.hlsl`'s FSR kernel, the pack's sig lanes, or the `present_*_fsr*` chains.

## Registered consensus (--quinlight): fusing every upscaler at once

A port of **quinlight-player**'s `consensus_registered.comp` (`../quinlight-player`, GLSL/Vulkan) into `src/gpu/quin.rs` + `src/gpu/shaders/quin.hlsl`. The idea in one line: **run several upscalers over the SAME traced frame, register each one to an anchor with a per-pixel Lucas-Kanade solve, warp it into the anchor's frame, and reduce the stack with a per-channel winsorized mean.** It is **not** a temporal algorithm — no history, no motion vectors, no depth, no jitter, nothing carried between frames. Everything temporal already happened *inside* the engines. The fuse is spatial and stateless, which is exactly why it is small.

**The engines are the chain's own levels.** `--quinlight` suspends the "exactly one upscaler per session" policy: every level the box supports is wired (DLSS-RR, FSR4-RR, XeSS, FSR 3.1), all fed from the same G-buffer pack, all upscaling to window-res RGBA16F — N views of one frame. That coexistence is real and measured (see the chain section): since the SL retirement every context is native, so the XeSS/ffx contexts sit alongside a live raw-NGX DLSS-RR trivially. Measured on the dev 4090: `dlss-rr + fsr3 + xess`, three engines, winsorization live. The fuse is **N-generic** (`MAX_ENGINES = 4`), so a level that fails to probe is simply not an engine and the session degrades to whatever came up — with **N == 2 the winsorized mean provably degenerates to a plain mean** (two samples are symmetric about their own median, so the clamp cannot move either; `quin::self_test` gates that identity). Winsorization only starts REJECTING at N >= 3. GPU-fed only (`--dxr`/`--gpu`): there is deliberately no CPU-upload arm, since N engines would mean N window-res uploads per frame — the traffic the on-GPU feed exists to avoid.

**What was deliberately NOT ported**: the coarse-to-fine pyramid seed (3 more shaders + ~600 lines of host glue). Upstream needs it to capture LARGE inter-engine motion at 8K; inter-engine disagreement on one frame is sub-pixel to a few pixels, which the unseeded path's `MAX_DISP = 2.0` total clamp covers. The shader is upstream's `use_seed == 0` fallback, verbatim in structure. If engines ever visibly disagree by more, that is the signal to port it.

**The kernel** (16x16 groups, one dispatch, no intermediate flow/warp images — memory traffic stays at "N engine reads + 1 write", so the extra cost is purely compute): the anchor plane is cooperatively staged into groupshared **before** the divergent bounds return (a barrier in divergent control flow is UB — the load-bearing comment upstream); the structure tensor + bilateral log2-luma weights are **anchor-only**, so they are built ONCE and reused for every engine and every refine iteration, and only the residual RHS is per-engine; Levenberg damping (`det >= alpha^2 > 0`) makes EVERY pixel solve, and a guarded `det` forces `inv = 0` so a non-finite anchor folds its engines in **unwarped** — degrading to the unregistered mean, never to black. `finite1` is the exponent **bit test**, not `isnan` (DXC folds that away without strict IEEE, exactly as glslc does): a non-finite engine value is **MISSING**, not a 0 addend that would drag the mean toward black.

**The one substantive porting change is the clamp.** Upstream clamps samples to `[0, 100]` because its pixels are PQ display units (1.0 = 100 nits, so 100 IS its 10000-nit peak). Ours are **linear scene-referred** — paper white ~1.0, a physical sun disc ~44000 — so a 100 clamp would crush every highlight in the scene. The clamp's only job here is "a finite sample survives the RGBA16F store finite", so it sits at the output format's own ceiling (`65504.0`, f16 max). The winsor window (`TAU = 0.1` relative + `ABS = 0.02`) transfers **verbatim**: quinlight's 1.0 is paper white and so is ours (`scene::default_light`), so a relative tolerance plus a near-black absolute floor mean the same thing in both.

**The anchor defaults to a DENOISING engine** (`quin::Engines::default_anchor`: DLSS-RR if the box wired it, else FSR4-RR, else engine 0; `--quin-anchor N` overrides, and the startup line says which rule fired). The anchor is the engine that is never warped — it defines the spatial frame every other engine is registered INTO — and a ray-reconstruction image is the right reference for two compounding reasons: it is the CLEANEST image in the stack, and the LK structure tensor + brightness-constancy residual are built from the anchor ALONE, so anchoring on a noisy engine would have the solve chasing per-frame shading noise instead of geometry; and it is the image worth keeping, since the reduce can only pull the anchor toward the others. `quin::DENOISING` is the two-name list this keys on, and `self_test` pins the rule against every real engine set INCLUDING the RDNA4 one (`fsr4-rr + fsr3 + xess` -> anchor `fsr4-rr`) — which the dev NVIDIA box cannot run, so that gate is the only thing standing between that path and a silent regression. A mixed stack (denoising anchor + TAA engines) prints a NOTE at startup naming the caveat below, because a user who wired this expecting a free win should be told rather than left to find out.

**A STANDALONE pass with its own root signature** (4 root constants + one table of MAX_ENGINES SRVs + 1 UAV + one STATIC sampler, ~6 DWORDs): the tracer's root signature sits at 62/64 DWORDs and is never touched. The static sampler also buys the warp hardware bilinear, which is what deletes upstream's manual 4-tap. The fused image presents at `tonemap::SRV_SLOT_QUIN` (slot 8).

**The feed grew descriptor SETS, not registers.** The engines' plane sets OVERLAP in registers (DLSS-RR and FSR4-RR both claim u16..u22), and rewriting one set of descriptors between two dispatches recorded into the SAME command list is a bug — descriptors are read at execute time, so the last write would win for BOTH. So the heap holds `trace::FEED_SETS = 3` copies of the RP_TEX table (the u14 resolve target at each set's offset 0, then that set's NUM_FEED planes) and each engine's feed dispatch binds the table at ITS set. Zero root DWORDs (it is the same table), only heap slots; `TEX_HEAP_BASE` derives from it. `wire_feed` REPLACES (set 0 — the single-upscaler contract, and what lets `--check-gpu`/`--check-dxr` rewire one tracer from kind to kind), `wire_feed_add` APPENDS (`--quinlight`). 3 is the ceiling, not a guess: XeSS and FSR 3.1 take a **byte-identical** plane set (RGBA16F color / RG16F mvec / R32F reversed-Z clip depth, same encodings, same rest state), so one trio feeds both — FSR 3.1 upscales straight from the XeSS planes via `ffx_up::upscale_res_shared` (two read-only consumers of one texture in one state need no barrier between them) and only pays a feed of its own when XeSS is absent.

**In session, the FUSE is the upscaler.** G/X/K all toggle it against plain (`gpu_quin_avail` / `dxr_quin_key` in main.rs), and an F off/on restores it — there is deliberately no key that switches to a single engine, because the engines exist only as fuse inputs and nothing would bring the fuse back. `FRUSTRACER_STAB=1` reads the FUSED image in both arms (`read_quin_output`); an engine's own output would report that engine's stability and silently ignore the thing being measured. `--quinlight` does NOT compose with `--nppd` (exit 2, the `--fsr4` shape): the neural denoiser rides the XeSS present arm, whose frame SPLITS around the ORT run, and the fuse presents through its own — a quinlight session would build the ORT session and its ~340 MB of staging and then never dispatch it. Frame generation composes (fg is on by default here too): the session's one FG family rides the fused present — raw NGX pair-presents quin.output as the real half (SRV_SLOT_QUIN, `ngxfg_target`), ffx FI / XeSS-FG wrap the swapchain the fuse presents on and `quin_fg_tail` records their per-frame prepare; a fuse<->plain toggle crosses the same funnel-handshake / reset seams as every other arm.

**One trace, one list, one Present.** `present_trace_quin` / `present_dxr_quin`: trace -> one feed dispatch per engine -> every wired upscaler's own sequence (`rr_ngx_sequence` / `record_fsr_rr_sequence` / `record_fsr3_upscale` / `record_xess_eval` — all four already existed and are reused verbatim) -> the fuse -> tonemap(SRV_SLOT_QUIN). Single-queue FIFO order is the only synchronization the engines need: each reads the same read-only planes and writes its own output. Everything executes on the one native queue (the SL proxy queue died with the retirement). Because ONE frame is traced for all of them, the locked render res must be legal for EVERY engine — `GpuContext::quin_res_range` intersects their SDK ranges (max of the mins, min of the maxes) and the per-engine range checks in `wire_session_feed` enforce it.

**Gates.** `quin::self_test` (pure, DLL-free, in `--check`) pins the reduce's identities: the m==2 plain-mean degeneracy, m>=3 outlier rejection, median order-independence, and the non-finite-is-MISSING bit predicate. `--check-gpu` **M13** runs the REAL kernel through its REAL root signature and descriptor table over synthetic engines: **N==1 passthrough** (bit-exact — the degenerate arm + the SRV/UAV wiring), a two-identical-engine **IDENTITY fuse** (bit-exact: the LK must solve (0,0) — this is what catches the sampler's texel-centre convention, the groupshared HALO indexing, a residual sign flip, an inverted tensor solve), and a known **(+1,0) SHIFT** the registration must recover, measured against a `QUIN_ITERS 0` control because a solve that always returned zero would sail through the first two. Measured: **68x better than unregistered**. The oracle is the f16-ROUNDED source, not the f32 one — the engines live in RGBA16F, and charging the port for the wire's rounding would gate the upload, not the fuse (the `tonemap::selftest` precedent).

**MEASURED QUALITY — read this before assuming the fuse is a free win** (`FRUSTRACER_STAB=1`, still view, default scene, mean |d| /255):

| mode | stab |
|---|---|
| XeSS alone | ~0.40 |
| FSR 3.1 alone | ~0.35 |
| **fuse (XeSS + FSR3)** | **~0.32** |
| DLSS-RR alone | ~0.12 |
| fuse (DLSS-RR + XeSS + FSR3) | ~0.24 |

(Re-measured after the one-sky/glare merge, which quieted every arm; the ratios, and the conclusions, are the same ones the pre-sky numbers gave — 1.00/0.95/0.80/0.16/0.52.)

Two facts, and together they are the whole story. (1) Fusing **comparable** engines WORKS: registering + averaging two decorrelated noisy estimates is quieter than EITHER input — the consensus is doing real work, and the 2-engine case is a *provable* registered mean. (2) DLSS-RR **denoises** while XeSS and FSR 3.1 do not (they are TAA upscalers handed a raw 1-spp trace), so a stack mixing one clean image with two noisy ones pulls the clean anchor toward the noisier median: the 3-engine fuse is much worse than DLSS-RR alone. The anchor is a root constant and `--quin-anchor N` A/Bs it, but no anchor choice fixes a stack whose members have wildly different noise levels — that is the reduce's nature, not a bug. The honest configurations today are `--quinlight --no-dlss` (fuse the un-denoised peers) or a box whose wired engines are peers. The real fix is to make the engines peers: pre-denoise their SHARED input (OIDN/NPPD at render res) so every engine is fed a clean frame, which the on-GPU feed already makes possible.

Touch `quin.rs` / `quin.hlsl` / the feed sets / `wire_session_feed` / the `present_*_quin` chains -> run `--check` (the self-test), `--check-gpu` (M13 + every feed gate), and `--check-dxr` (the feed refactor touches both pipelines).

## Correctness invariants (the bug class to guard)

The whole design hinges on the inherited-distance bound in `src/frustum.rs::nearest_geometry_distance`:

- Distances are **Euclidean from the shared camera origin**, and all ray directions are **normalized**, so distance == ray parameter t. If either side of that breaks (e.g., unnormalized dirs), children start past real geometry and pixels silently show sky.
- The region proven empty by a parent is frustum ∩ **ball**(origin, t_start) — a spherical bound. Never reintroduce a planar near clip: a node may be skipped only when `max_dist(origin, aabb) <= t_start`, and candidates are clamped up to `t_start`.
- Tile frustums are built through the tile's **continuous pixel-grid edges** (`CamBasis::tile_frustum`), not pixel centers, so jittered accumulation samples stay inside their tile.
- Secondary rays (shadow/AO/reflection in `shade.rs`) must **never** see the tile's inherited tmin — it is a primary-frustum property only. The hemisphere integrator keeps this by construction: it runs its own tmin chains from t_start = 0 at its own apex (the shading point).
- Frustum-vs-AABB culling may only err toward "intersecting" (false positives cost efficiency, never correctness). Degenerate side planes become zero normals, which never cull.
- A "blocked" query (no distance progress, typically the huge ground-plane AABB) must still **subdivide** — children's smaller frustums can exclude the blocker; that is how sky tiles emerge. Stopping on blocked was tried and was 8× slower (kills parallelism and sky fills).
- `refine_cut` may drop a node from the cut only by the frustum test, the proven-empty-ball test (`max_dist <= tc` — safe because tmin only grows down the quadtree and hit acceptance is strictly `t > tmin`), or the **far-clamp test** (`dist >= t_far` — sound ONLY under the clamped-consumer contract: every bound query on that cut passes `t_limit <= t_far` and every ray uses `tmax <= t_far`; the primary path passes `t_far = INFINITY`, which never drops; hemisphere AO passes `ao_radius`). **Never add distance-to-best pruning to the cut**: a far node can be the nearest thing in a sibling's frustum; pruning it surfaces as false sky. `d >= best` belongs to the bound query (`visit`) only.
- Cut capacity: `refine_cut` is generic over the capacity N and keeps `out_len + work_len <= N` so a surviving node always has a slot — out of budget means an internal node is emitted coarsely, never dropped. The cut handed to children must only ever be `refine_cut` output.
- **Multi-sampling** (`--spp`, `render.rs::shade_pixel` + the `SampleId` enum; `leaf.hlsl`/`reference.hlsl`/`dxr.hlsl`'s sample loops) rides the leaf-tile argument below and adds no new one: sample k's offset is in [0,1) on pixel coords, so it lies inside the same pixel, hence inside every ancestor frustum, so it consumes the SAME inherited `t_start` and node cut. This is why the extra samples must take an in-pixel offset — never a wider filter footprint. Two contracts hold the rest together: (a) **one splat per pixel per frame** — the N colors average locally and splat once (`do_splat = false` on the extras, the HOT top-up mechanism generalized), so `accum`'s store-vs-add semantics and `resolve`'s frame-count divisor are untouched, and on the GPU the divide happens BEFORE `accum` (in the leaf/raygen kernels), since `feed.hlsl` reads `accum` as one sample's radiance in absolute units; (b) **sample 0 owns every side channel** — tbuf/info/G-buffers/MVs are written by the `primary_sample` only, so the guides stay tied to the jitter the upscaler was told about. `spp` is pinned to 1 on fb (H) frames (`FrameCtx::spp()` and the CB): the bounce integrator converges by frame accumulation, and N samples would mean N hemispheres per pixel (and N hemi points, which would blow `cap_hemi_pt` on the GPU). `--check`'s per-sample verify sweep is the gate; do NOT let an extra sample write meta, and do not "fix" a gate by pinning spp inside `verify` — the sweep is the proof.
- **ONE SAMPLE-POSITION CONVENTION.** Every temporal consumer is told ONE sub-pixel jitter per frame, so every path that shades a pixel must place sample 0 where that offset says — `render::first_sample_offset` (CPU) and `sample_pos` (HLSL) are the single rule, used by leaf tiles, the sky-tile flood (`fill_sky_rows` / `cs_sky`), `reference.hlsl` and `dxr.hlsl` alike. A path that shades somewhere else while the frame declares an offset is a REGISTRATION defect, and the trap is that it leaves motion vectors CORRECT (they subtract whatever position they are handed) — under pure translation a sky ray's direction reprojection is the identity, so a buggy and a correct renderer both write exactly zero. It is observable only in colour; the `sky registration` gate is the one that sees it, and `mv_selftest`'s sky arm deliberately claims only the weaker `dir`-is-the-preimage-of-`(fx,fy)` pairing (which is what catches `sparse_fill` reusing a cell sample's direction). Corollary for any new shading path: whatever `(fx, fy)` you pass a `write_gbuf_*` helper, `dir` must be `ray_dir` of exactly that.
- Leaf tiles use the inherited cut without re-culling; this is sound because jitter is in [0,1) on pixel coords and quadrant splits exactly partition the parent rect, so every leaf ray stays inside every ancestor frustum. `intersect_multi`/`occluded_multi` are for rays that own a cut built for THEIR apex — primary rays in quadtree leaves, hemisphere leaf rays. Plain mode, verify's reference rays, and any ray without a matching-apex cut stay on `intersect`/`occluded`.
- **Hemisphere bounce integrator** (`src/hemi.rs`, `src/sphcell.rs`, opt-in `Quality::fb`): the hemisphere above a shading point is a spherical-triangle quadtree — root = tangent half-space (`TileFrustum::half_space`), level 1 = 4 octants from the right-handed `onb(n)` (`t1×t2 = n`, asserted by `sphcell::self_test`), deeper = great-circle midpoint splits (`tri_cell`, 3 planes). Midpoint children exactly partition the parent, so inherited (tc, cut) is sound by the pixel-quadrant argument; leaf samples (Arvo uniform-in-triangle) stay strictly inside their cell (fp ≪ the plane test's inclusive slack). The apex is `hit + n·eps` and the root `t_start` is 0, NOT eps — a ball(o, eps) claim is false at concave corners (the false-sky shape); the tangent plane excludes the own surface geometrically (it lies at −eps below). The root cut is `[0]` — a primary tile's cut is INVALID at a different apex. AO clamps every query to `ao_radius` via `nearest_geometry_distance_within`: `None` then means "open within the radius" and must never be consumed as sky. Empty cells integrate analytically (Lambert PSA; GI: `sky::dome()` × PSA — **the DOME, never the disc**, see the one-sky bullet — with `sky_cell` refinement to ~12°, since a coarse centroid measurably over-brightens, plus ~6° inside a 30° cone of the sun where the Mie aureole lives); unresolved cells at the `LEAF_LEVELS` cutoff distribute one stratified ray per sub-cell (one query amortizes 4 rays — per-ray queries were measured to cost more than they save), shaded at the depth-1 `BOUNCE_Q` policy. **`BOUNCE_Q.ao_samples` MUST STAY > 0 and is the whole difference between GI and a flat ambient.** A bounce surface's own ambient is `sky_sh.irradiance(n) * ao`, so at `ao_samples: 0` the factor is 1.0 and every bounce surface is lit as though it stood in an open field — including a wall deep under an arcade. Each occluded direction then hands the integral a full-sky-lit surface and `gi()` collapses toward the unoccluded sky value EVERYWHERE: a uniform lift with no structure, i.e. exactly the flat ambient constant this tier exists to replace, and BRIGHTER than no GI at all while looking visibly worse. It shipped that way and **every gate passed**, because the `--check`/`--check-gpu` GI A/Bs score against a reference running the same `BOUNCE_Q` — both sides were flat together, which is the cost of sharing a policy constant between estimator and oracle (the sharing is still right; the blind spot is inherent and the instrument for it is looking at an ENCLOSURE). Measured on San Miguel's patio at 15:30, 1280x720/96 spp, luminance: `ao_samples` 0 → mean 46.30 / shadowed 35.68 / contrast 2.34, vs 1 → 26.15 / 14.45 / 4.70, against fb-OFF 22.45 / 6.17 / 10.01 — so the fix keeps shadows 2.3× above the no-GI tier (that IS the bounce) and doubles contrast. 1 ray suffices (2 measured 0.19% different, +21% time). Open scenes barely move (`check_gi.png` 0.8% of pixels; powerplant/rungholt/helmet/vokselia < 2% mean) because their hemispheres are mostly sky — the tier only matters where bounces dominate. `hemi_leaf.hlsl`'s `n_ao` argument is the GPU mirror and moves in lockstep. PSA per point must account to π; `--check` re-validates every empty-cell claim and leaf-ray tmin with reference rays (`false-empty` / `tmin-overshoot` — the transplanted bug-class gates). **Hemi sharing** (`hemi::share_capture`/`HemiShare`, default on, `--no-hemi-share` kills): fb leaf tiles shade in 2×2 cells; when all four pixels hit the SAME triangle with BIT-EQUAL shading normals (⇒ bit-identical `onb(n)` partitions ⇒ per-member PSA still accounts to π exactly) the rep captures ONE tree whose every plane is padded by `pad_k = δ_∥·|in-plane(n_k)| + η·|n_k·n|`, where δ/η are the apex spreads MEASURED from the fp hit points — never assume coplanarity: möller-trumbore fp puts hit points off the tri plane, so the root pad is η (~ulps), and the `η ≤ eps/4` qualifier is what keeps the own surface (at −eps) geometrically excluded (uniform padding would flood every claim with the phantom own surface — the blocked-everywhere collapse). A `δ ≤ ao_radius/8` cap rejects grazing-angle groups whose pads would degrade the capture to all-blocked. Empty cells fold ONCE into the record (their PSA/sky values are pure functions of the shared n/sun — bit-identical per member); query-leaves store (cell, tc, cut) and every member — rep included — shoots its OWN fresh rays with `tmin = max(0, tc − δ)` (a member point at ray param s ≤ tc − δ lies within tc of the rep — inside the proven ball; AO captures at `t_limit = ao_radius + δ` so members keep exactly `ao_radius`). Cut drops stay member-valid because member rays live inside the padded cones (|n_k·Δ| ≤ pad_k) and ball/far-dropped regions are proven empty / beyond every member's clamp — comment discipline: the ball region is EMPTY, not unreachable. Capture consumes no rng; a capture over `FB_DEPTH_CAP` or out of record slots poisons → the group shades per-pixel, coarser never wrong (`--check` must-fires a depth-`FB_DEPTH_CAP+1` poison so the guard can't rot into dead code). Sharing amortizes bound queries only — never samples, never view-dependent terms. There is deliberately NO ray-count-parity gate between the share arms: the padded frustums legitimately reclassify borderline cells (empty ↔ query-leaf), so `hemi_leaf_rays` differs between arms — the paired same-seed A/B is the estimator gate; don't "restore" parity. Measured (default scene, 8-frame bench): hemi-ao −25%, hemi-gi −29% ms/frame, hemi queries −48%, with ~65% of fb points shared (bit-equal n restricts sharing to flat-shaded geometry — the ground plane, where the cost lives).
- **The one sky, and the disc-exactly-once rule** (`src/sky.rs`, `src/sh.rs`): there is ONE light — a sky sphere at infinity — stored in two representations split by **frequency**, because the two bands need different sampling strategies. (The dusk/night **fireflies** — `src/fireflies.rs`, see the command block — are the one documented extension: N point lights in the DIRECT tier only — display paths + `shade()`'s direct loop, never any gather, never the SH/dome — so this table is unchanged by them. NOTE that this is a true GATHER EXCLUSION and is therefore *not* what the star field does: fireflies are local emitters with a direct-lighting strategy that already delivers them, so a gather must not see them at all; the stars have no such strategy and are delivered to gathers in a different representation instead. Historical comments calling the firefly rule "the stars rule" predate the star row below.) The smooth **scattering dome** (single-scatter Rayleigh + Mie, `sky::dome`) is projected once at load into **order-2 SH** (9 RGB coefficients, `Scene::sky_sh`) and evaluated as analytic irradiance per shading normal (`sh::Sh9::irradiance`, Ramamoorthi & Hanrahan 2001) — **zero rays**. The sharp **sun disc** (`sky::Sun`: direction + `cos_radius` + `e_over_pi` + derived `radiance`) stays an explicit light: cone-sampled and shadow-rayed. This split is forced, not stylistic — SH has no notion of visibility (you cannot shadow-ray a coefficient) and a 2° sun is ~1e-4 of the hemisphere, so gathering it by cosine sampling is pure noise; conversely irradiance is a cosine convolution that annihilates everything above l=2, so 9 coefficients are near-lossless for the dome. **THE INVARIANT: the sun disc is delivered exactly once per light path.** A ray sees the disc only if no light-sampling strategy already covers the sun along that path — camera and glass misses call `sky::radiance` (dome + disc); the specular reflection miss takes the dome plus a **MIS-weighted** disc (balance heuristic, `sky::mis_weight`, since `direct_s` also delivers it — zero extra rays, zero extra rng draws); and **every gather path — `hemi::sky_cell`, the GI leaf miss, the SH projection, and both `--check` GI reference estimators — calls `sky::gather`, which carries NO DISC.** Break that last one and you get a double count of light `direct_d` already delivered with its own shadow ray, *and* a ~1e3-radiance firefly into hemi's 2^18 fixed-point accumulator, which saturates outright. It is also why `sky_cell`'s centroid point-sampling still works: excluding the disc removes the sharp feature entirely (a cell coarser than the sun would alias it catastrophically), leaving only the Mie aureole, which its `near_aureole` branch refines to ~6° inside a conservative 30° cone. **MIS is a partition of ONE integral between TWO strategies, so it may only down-weight the light-sampled specular when the BSDF-sampled half is actually going to run.** The VNDF reflection ray is gated (`shade.rs::refl_ray` = `q.reflections && depth == 0 && (metallic > 0.04 || roughness < 0.45)`, hoisted so the direct loop and the reflection block read the SAME predicate; `shade.hlsli` mirrors it), and where that gate fails — the low preset, which sets `reflections: false`, and EVERY `depth > 0` surface, i.e. anything seen in a mirror or through glass — light sampling is the only estimator of the sun's specular and must carry `w_l = 1`. Weighting it down there deletes energy nobody delivers: a mirror under the low preset measured `w_l ≈ 0.005` (p_bsdf ≈ 5e4 vs p_light ≈ 261), a ~200× too-dark highlight. This shipped once; don't re-introduce it by computing `w_l` unconditionally because "MIS is always safe". `sky::Sun::sample_dir` consumes **exactly the two rng draws** the old rect sampling did, in the same order — that is what preserves every same-seed / replay / `VisCtl`-burn bit-identity contract, and `--check-xess` is the proof. `DOME_SCALE` is the one brightness knob (it sets both the sky you look at and the ambient it casts, which is the whole point); `sky::self_test` gates the resulting irradiance into a physically sane, blue-dominant band. Disc radiance is **derived** (`e_over_pi·π/Ω`), so narrowing `SUN_ANGULAR_RADIUS` sharpens shadows and brightens the disc **without moving scene exposure** — the knob an HDR pipeline wants (caveat: the upscaler input planes are f16, so a physical 0.27° sun at ~44,000 radiance leaves only ~1.5× headroom; 2° leaves ~80×). The disc is **antialiased against the ray's angular footprint** (`sky::disc`'s `half_angle`, from the `pixel_cone` we already carry for texture LOD): the limb is a step function of DIRECTION, but a ray has a footprint, and a pixel that half-covers the sun gets half its radiance — without it the edge is a binary per-ray test that crawls under motion at 1 spp. `half_angle = 0` reproduces the hard step exactly, which is what `sky::self_test` scores the AA against. **Do not soften the limb itself to make the sun look better** — it is hard in reality, and the softness you want is glare (`--no-bloom` / `src/bloom.rs`), which lives at the display stage where it belongs.
  - **THE STAR ROW — starlight is a real, moon-independent night ambient** (`sky::star_glow` / `sky::gather`, `--tod` sessions only). The same once-per-path rule as the disc, with the polarity REVERSED: nothing importance-samples a star, so instead of being *excluded* from gathers the field is delivered to them in a different REPRESENTATION — **points to the eye** (`sky::stars`, inside `radiance()`), **the field's smooth analytic mean to the gathers** (`sky::star_glow`, inside `gather()`), carrying **identical total energy** either way. That equality is the whole design and it is PROVEN, not asserted: `STAR_FLUX` is a literal (mirrored in `trace_common.hlsli`, the clouds-wind idiom) that `sky::self_test` pins by **enumerating all 6·64² = 24,576 hash cells** through the same occupancy/tier/tint logic `stars()` uses, then re-integrating `star_glow` back over the upper hemisphere (measured 0.9% short — that is the ±0.05 blend band handing energy to the ground bounce, exactly as `dome`'s own does; hence a 2% tolerance, not an ulp count). **The field cannot simply be projected**: `Sh9::project` is a 16,384-point quadrature and the stars cover ~0.067% of the sphere, so a direct projection lands ~11 random hits — sampling noise that would shift with `PROJ_SAMPLES`. The mean is not a convenience approximation but the EXACT order-2 content, since SH carries only DC + linear + quadratic and a near-uniform point field has nothing else down there. Shape reuses `dome`'s own horizon blend + `GROUND_ALBEDO` bounce (a hard step rings under order-2 truncation — the Gibbs guard `GROUND_ALBEDO` already exists for). Gated by `Scene::night` through a **BRANCH** in both `star_glow` and `gather` (`gather` returns `dome` BITWISE at `night == 0`), so **every day session is bit-identical by construction** — verified byte-for-byte: `check.png` AND `check_gi.png` are unchanged against the pre-feature build, and the day ambient still reads `(0.11950045, 0.1763439, 0.2472961)`. Magnitude at the energy-honest `STAR_AMBIENT_K = 1.0`: **0.63× the moonlit dome's ambient** (so night ambient is ~1.63× brighter overall) — deliberately the same ORDER as moonlight rather than the ~0.2% real starlight is, on the `MOON_E_OVER_PI`-is-ARTISTIC precedent, since `STAR_E` was authored to make points read at interactive resolutions. `STAR_AMBIENT_K` is the one knob; `self_test` bands the ratio at [0.1, 3.0] so a retune cannot silently move night exposure. GPU: `sky_gather` in `trace_common.hlsli` and **exactly two** call sites (`hemi.hlsli`'s empty-cell term, `hemi_leaf.hlsl`'s leaf miss) — the SH rides `sky_sh` which is CPU-computed and uploaded, so `shade.hlsli`, `feed.hlsl`, and the FSR composite's AO remodulation pick the floor up free, and `night` was already in the CB (no `CB_STRIDE` move). No `CACHE_VERSION` bump — `sky_sh`/`night` are derived, never serialized. Zero rng draws. **The strongest twin gate is `--check-gpu --tod 2`'s hemi GI A/B** (GPU hemi vs the CPU reference, both now carrying the glow): it reads **1.70%**, where a missing or mismatched HLSL `STAR_FLUX` would land near 40%, because the moonlit dome it is measured against is itself only ~1e-3. Known-accepts: the VISIBLE night sky is unchanged (the glow is gather-only, so the space between stars still shows the bare dome — deliberate, it is what keeps the energy delivered once); and because the moon is ANTIPODAL (`scene.rs`'s `apply_tod`) it is up whenever the sun is down, so today the floor's payoff is the dusk/dawn handoff band and moon-elevation independence rather than a genuinely moonless night. Touch `star_glow`/`gather`/`STAR_FLUX` or their HLSL twins → run `--check`, `--check --tod 2`, `--check --stress 5000 --tod 2`, `--check-gpu --tod 2`, `--check-dxr --tod 2`.
  - *Removed with the rect light:* **`src/shaft.rs` and `fb.shadows`**. Light shafts built a 4-corner frustum from the shading point through the light RECT and `classify(su,sv)` partitioned that rect — both premises die with a disc at infinity. They were opt-in and measured **~3× slower** than the rays they saved (75% of shadow rays skipped, but a shaft query costs more than the rays it removes). The same economics killed a specular-bounce cone accelerator: one query > one ray's traversal. The H key now cycles off → AO → GI (three tiers, not four).
  - *Also removed:* the flat `shade::AMBIENT` constant `(0.14, 0.17, 0.23)`. It was a hand-tuned fudge ~2.7× darker than the sky it stood in for, which is exactly why `fb.gi` frames (which integrate the real sky with rays) came out brighter than the sampled path — the tiers disagreed about what the sky *is*. They now agree. Pleasingly, the physically-derived Rayleigh irradiance lands at **(0.120, 0.176, 0.247)** — the old constant, reproduced to a few percent, but *derived* instead of guessed.
  - *Also removed:* the `1/d²` falloff, and with it the `k`/`k²` light-rescaling hacks in `stress_scene`/`tile_scene`. A sun does not get closer to one end of a tiled field.
- **What the frustum quadtree is actually worth, decomposed** (measured, and narrower than the algorithm's framing suggests). The often-quoted **0.87–0.93 Intel / 1.31–1.37 NVIDIA** marginal ratios and ~spp-16 Intel crossover are HISTORICAL: they came from the old `(LEAF_TILE, LEAF_GROUP) = (8, 32)` frontier, before the shipping `(32, 256)` dispatch-shape change, and the old timing stream included uneven asynchronous-compilation bias. Do not publish or use those ratios to choose a default until the current frontier is rerun cross-vendor. The durable result is the ablation: setting `t_start` to 0 while keeping the quadtree cost only 1.1–1.7% on the measured Arc runs and straddled zero on a 4090. Two independent repairs of the bound both made it near-perfect and both moved ray traversal by ~0: subdividing the ground quad (measured and rejected — bound quality soared, and it cost time everywhere, because more, smaller boxes means more frustum-tree nodes), and clipping the box to the frustum before ranging (`frustum::frustum_aabb_dist`, the `FR_RANGE=1` lever, **default OFF**). `slab_t` culls a node only when it lies ENTIRELY before tmin, and that region is empty almost by definition. **Optimize tiles proven empty (zero rays) and cut-seeded custom traversal, not physical ray length.**
  - **THE FR_RANGE NUMBERS, so nobody re-derives this in either direction** (2 interleaved reps of `--spin path --spin-frames 300`, 7950X3D; `--spin` is deterministic, so the counters are bit-identical across reps). Default scene: **41.03/41.48 ms OFF vs 41.33/41.35 ON** — neutral. `--stress 5000`: **44.37/44.75 OFF vs 54.26/54.43 ON — the clipped range is +22.0%**. Ray traversal is untouched either way: 7 node visits differ out of 4.88 BILLION on the default scene, 35,546 out of 8.07 billion on stress (0.0004%). The lever therefore ships OFF, which also puts the CPU on the same ranging the GPU has always used (`frustum.hlsli` only ever had `point_aabb_dist`), so the two quadtrees are comparable by construction. **And here is the part that is easy to get backwards:** the clipped range DID buy a much better empty-space proof — on stress it cut blocked queries 41,059 → 12,011 (3.4×), proved 14% more sky pixels, and eliminated 14% of primary rays — and was still 22% slower. So the maxim above needs its second half: the empty-space proof must also be CHEAP. A proof that costs more than the rays it eliminates is a net loss, and bound quality (`mean t_start/t_hit`, `blocked`) is a DIAGNOSTIC, never a target.
  - **NODE VISITS ARE NOT MILLISECONDS, and the frustum ladder is 0.22% of them.** Three read-only instruments settle the whole "make the frustum query better" direction, all default OFF and bit-identical off (`check.png`/`check_gi.png` byte-equal with them in the tree): **`src/oracle.rs`** (`FR_ORACLE=1` — probes in `tile_step` and the leaf arm; touches no `LocalStats`, so bench counters stay oracle-clean while armed), **`stats::ray_nodes_prim`** (the PRIMARY share; `ray_nodes` stays the TOTAL because the builder bake-off and the adopt bench read it that way), and **`shade::abl`** (`FR_ABL=noshadow|noao|norefl|noglass|nosec`, the CPU twin of `gpu::trace::abl_has` — each arm neutralizes one secondary-ray consumer while KEEPING its rng draws and its `secondary_rays` increment, so only the traversal disappears). Measured on default / san-miguel-lp / `--stress 5000`: **(a)** the ladder is **0.213/0.220/0.222% of CPU node visits** and **0% of a RESTING GPU frame** (structure replay deletes it; ~25% of a PRODUCING wavefront frame, and the DXR pipeline has no ladder at all) — so every idea that makes the ladder cheaper is capped there however good its hit rate looks, and the oracle's 39–46% straddle-mask rate is 46% of 0.22%; **(b)** the quadtree has already taken most primary traversal — primary rays are 26/24/35% of ray nodes in plain and **7/13/14% in hybrid**; **(c)** the counters do not predict the clock — secondary rays are 86–89% of ray NODES but `FR_ABL=nosec` saves only **29.0%/15.6%** of frame time, so CPU traversal is ~⅓ of the frame (~⅕ on San Miguel) and **shading is the rest** (per arm vs off: noshadow −8.9/−7.2%, noao −11.8/−9.7%, norefl −4.6%). **(c) is the answer to the standing paradox** that the quadtree removes 26% of ray-node visits with no wall-clock gain — those visits were ~8% of the frame — and its methodological consequence is that **`ray_nodes`, the currency the builder bake-off above scores in, tracks at most a third of the CPU frame**: confirm a node-count verdict in ms before acting on it. WHAT THIS KILLS, so nobody rebuilds them: Teller's covering test (fires on 19.7% of San Miguel's pixels, ceiling **~0.7% of time** once (b) and (c) are applied), his straddle-mask inheritance, MLRTA's Kay–Kajiya interval reject and adaptive tile termination, and the kd-tree/occupancy-grid idea of feeding the BVH a second structure — all ladder-bound (`docs/papers/`). Teller's frustum ADVANCE is separately unavailable: it needs cell adjacency, which a BVH does not carry. Also measured and free: **`--aniso 16` vs `--aniso 1` is 32.84 vs 32.85 ms** on san-miguel-lp — the Mip-mapping section's Intel Sponza +8.7% is pose- and scene-specific, not a general cost. **THE ESTIMATOR HERE IS NOT THE ONE THE PROFILING SECTION PRESCRIBES**: sustained 32-thread load thermally destabilizes this box (one `norefl` rep read 192 ms against a 38 ms min, and a first ladder pass was incoherent enough to be unusable), and since `--spin` is deterministic — counters bit-identical across reps — ALL of that variance is machine state, so use **min-of-N with a cooldown between runs and short 120-frame runs**, never the median. "Interleave and take medians" is tuned for the GPU async-compile trap and is the wrong estimator for this one.

- **Volumetric clouds** (`src/clouds.rs`, default-on, `--no-clouds` — the cloud rules extend the one-sky table without changing it): a drifting slab of 2D COVERAGE (`cloud_cover`, 2 octaves of integer-hash value noise over `sky::pcg_mix`, u32-exact CPU↔GPU — where clouds ARE) carved by genuinely 3D EROSION noise (`erosion3`/`vnoise3` — what shape they are) inside a coverage-driven vertical window (`cloud_prof` — taller where denser), marched TWO-PHASE with a per-(pixel, frame, sample) DITHERED phase, the whole field sampled through a static low-frequency **3D curl-noise wind warp** (`curl_offset` — v = ∇ψ₁×∇ψ₂ of two 3D noise potentials, soft-normalized to |v| < 1; sampled at raw world coordinates and applied OUTSIDE `advect`, so the advection-identity gate G6 survives verbatim; shared bitwise by density and density_lo — the shadow must track the cloud, and G8 needs the one domain; the march folds ONE warp per ray into its origin — per-coarse-step warps measured +21 ms CPU / ~2× GPU per-sample, and the fold keeps the interval-skip exact in field space). Three lessons are load-bearing here, each shipped wrong once: (1) **the march phase must be dithered per (pixel, frame, SAMPLE)** (`dither_jk` — a pure integer hash of (pixel, frame) + k/spp stratification: still zero rng draws, still same-seed/replay/VisCtl-safe, bit-equal across both GPU kernels; with a FIXED phase, `dt = thick/(N·d.y)` makes sample altitudes ray-independent, and any smooth field sampled on N fixed planes renders as N nested step-entry contours — the wedding-cake bug; and with ONE j shared by all spp samples, `--spp` averages N copies of the same phase and softens NOTHING — the dithered-look complaint. The frame term + stratification make --spp, accumulation, and RR/XeSS converge the grain; the night spp-stability gate PASSES with it (measured — stratified spp=4 means are phase-invariant while spp=1 carries full phase noise, so the gap widens; the SKY_J lesson is about the sky-tile fill's DIRECTION set and still stands separately); `CLOUD_TEMPORAL_DITHER`, CPU+HLSL lockstep, drops the frame term alone if a future gate objects); (2) **the field must be genuinely 3D** (a 2D field's occupied sets at different altitudes are nested level sets of one function — no amount of vertical shaping, shear, or height-blending de-nests them); (3) **unresolvable octaves must collapse to their means** (`oct_t` per-octave anti-alias to the sampling footprint — point-sampled unresolved detail beads at grazing angles; the mip philosophy). `density_f(p,w) ≤ density_lo_f(p,w)` holds POINTWISE BITWISE (erosion only subtracts before the shared prof multiply) — self-test G8, and the soundness of phase A's cover-only occupancy probe. All three look bugs were invisible to every gate — the gates prove soundness, never looks; the three-pose screenshot check (up `--cam 0,0.5,14,0,12,0`, grazing `--cam 0,0.5,14,0,5,-8`, default) is the only instrument for this class. Rules, in order of how expensive they are to re-learn: (1) **`sky::dome()` never sees clouds** — the SH projection, `hemi::sky_cell`, the GI leaf miss, and both GI reference estimators keep integrating exactly the function the ambient was built from; a drifting occluder cannot live in a load-time SH projection, so the static ambient under an overcast patch is a documented known-accept. (2) **Display paths see the whole backdrop (dome + disc + STARS) through the layer**: `radiance(o, d, …)` gained the ray ORIGIN (a finite-altitude slab has parallax) and returns `backdrop·T + scatter`; the reflection-miss MIS site applies the same T along `rdir` from the hit point. (3) **The direct sun is scaled by `clouds::sun_transmittance`** (2 density evals, ¼/¾ slab heights) once per `shade()`, after the `/n_shadow` normalization and **before** the `prim.direct_d/direct_s` export — FSR-RR's dd/ds carry the cloud shadow, so the composite identity closes untouched. The MIS pair therefore delivers the sun through two slightly different transmittances of one field (light strategy: the 2-eval T; BSDF strategy: the march's T) — bracketed, never a hole or double count; do NOT "fix" it by forcing one T on both sides (cheap-T at the miss punches the sun through a visibly occluding cloud; march-T in the direct loop is the CPU cost you were avoiding). On the GPU (wavefront AND DXR) that 2-eval `sun_transmittance` is served from the slab-space **cloud-shadow cache** at the shipped `--cloud-shadow N` default (`clouds::shadow_grid_row` is the ONE grid; the domain reduction to F(M.x,M.z) is EXACT, only the bilinear fetch approximates — `--no-cloud-shadow` restores the per-pixel 2-eval path bitwise; the CPU tracer is always the uncached exact path). (4) **Off is bit-identical, per-ray and per-flag**: `--no-clouds` takes guarded branches; even enabled, a cloud-free ray gets `None`/an exact 1.0 and the backdrop passes through bitwise (the conservative octave-partial-sum cutoffs inside `density()` are the clear-sky fast path AND exact — value-continuous across their branches). (5) **The clock is main.rs's** (`Persist::cloud_time`, f64): upscaler/denoiser frames advance every frame, plain accumulation only at `frame == 0` (a converging still frame — and its replays — integrates ONE sky), `--spin` uses `idx·CLOUD_SPIN_DT`, every `--check*` pins `CLOUD_CHECK_TIME` so every gate pair compares one sky. Clouds have no MVs (drift = shading change to the upscalers, accepted) and are modeled from below only. At night `scene.sun` IS the moon, so moonlit clouds and moon shadows need no special casing; `sun_col` reuses the dome's own `t_sun_path`, so clouds redden at sunset in lockstep (under a TOD scrub `e_over_pi` already carries `sun_fade` — a bounded double-tint, accepted as artistic). GPU: one port in `trace_common.hlsli` (wavefront + DXR + hemi all paste it); `FLAG_CLOUDS` = 256; state rides the cam rows' free w lanes (`SCENE_DIAG`/`CLOUD_TIME`). `clouds::self_test` (in `--check`) pins: the off bit-identity sweep, T∈[0,1]/finiteness over direction×time, Beer monotonicity in optical depth, the per-ray bit-passthrough, cloudy/clear/ground-shadow must-fires, the exact advection identity (one shared `advect` expression — the altitude shear term is time-independent, so it cancels), and horizon-fade continuity. Two field resolutions, deliberate: `density` (4 octaves, bottom-heavy amps — octaves 0-1 place the clouds, 2-3 only erode edges) is the VISIBLE surface; `density_lo` (2 octaves + the dropped octaves' mean) carries everything that is lighting rather than surface — the shadow T, the march's sun probe, and the whole reflection-miss march (`along_rough`: 2 steps — a reflected sky is seen through the GGX lobe, the bounce-cone philosophy). **The sky-tile fill is spp-aware under clouds**: a proven-empty tile still traces zero rays, but at spp > 1 it averages spp sample positions (`fill_sky_rows` / `cs_sky`), because the old one-center-direction rule was premised on a sky smooth at sub-pixel scale and a cloud cover-ramp isn't — the spp wavefront-vs-reference image A/B is what caught it (5% divergences at cloud edges). The extra offsets are the PHASE-0 Halton set (`SKY_J`, injected by `trace::spp_defs`), deliberately frame-INDEPENDENT **and anchored to the pixel CENTER** — see the sample-position rule below for why both halves matter. **SAMPLE 0 IS A DIFFERENT MATTER AND TAKES THE FRAME'S DECLARED JITTER** (`render::first_sample_offset` / `sample_pos` — the same rule leaf tiles, `reference.hlsl` and `dxr.hlsl` use; there is now exactly ONE sample-position convention in the tree). It used to hard-code the pixel center while the frame declared a Halton offset to RR/XeSS/FSR/NGX anyway — a REGISTRATION lie, and a subtle one because the motion vector stays correct throughout (it subtracts whatever position it is handed): the reconstructor simply places sky content at center+jitter when it was sampled at center, and the jitter moves every frame. NO MOTION-VECTOR GATE CAN SEE THIS — under pure translation a sky ray's direction reprojection is the identity, so both the buggy and the correct renderer write exactly zero — which is why it survived; it is observable only in COLOR, via the `sky registration` gate (hybrid vs the sky-tile-free plain path at a nonzero declared jitter, bit-exact; teeth measured at 120000/138954 px, worst rel 2.49e-1). The old note here claimed per-frame sky offsets are rejected by the night spp stability gate outright; that is TRUE OF THE EXTRAS ONLY and was over-generalized to the center. Writing δ for the per-frame offset and ∇f for the sky's screen gradient, the first-order inter-frame difference is `(δ₀−δ₁)·∇f` for spp=1 and, at spp=4, `(δ₀−δ₁)·∇f` if only the extras move, the SAME if the whole set translates rigidly, and `¼(δ₀−δ₁)·∇f` only when sample 0 moves and the extras stay anchored — averaging over a rigidly translated stratified set does not attenuate a translation, only the aliasing residual. So the shipping arrangement is the one that WIDENS the gate's margin, measured `--check --tod 2` 3.09× → 3.19× quieter. `check.png` is byte-identical across the change (its context declares no jitter). Touch clouds.rs / the cloud block in trace_common.hlsli / the radiance signature / `shadow_grid_row` / the sky-lod or cloud-shadow fill kernels → run `--check` (clouds G13/G14 + settings self-test), `--check --stress`, `--check --tod 2`, `--check-xess` (rng alignment), `--check-fsr`, and `--check-gpu` + `--check-dxr` (default-ON caches + the fill-vs-oracle/on-off A/B gates; `--no-sky-lod --no-cloud-shadow` for the off arms).
- **Temporal cache** (`src/temporal.rs`, static scenes only): every entry is a standalone world-space claim "frustum ∩ ball(origin, tc) empty" — `+INF` claims the *whole* cone, valid only as the composition of a `None` query with the inherited ball claim. Entries built on temporal seeds stay standalone because every cross-frame hop only shrinks (δ subtraction, 1e-4); δ = 0 transfers exactly, no shrink. Reuse under motion goes through the **segment decomposition** in `temporal::lookup`: `[0, t_start]` is the tile's own inherited claim; `[t_start, seed]` is covered by a **region-min query** — the old quadtree partitions the direction sphere into frontier cells, and the seed is the min claim over every old cell the tile's tilt-widened cone (`{corners} ∪ {corners + (δ_safe/t_start)·t̂}`; t̂ itself when t_start = 0) overlaps. There is deliberately no apex condition and no single-cell containment test — both structurally never fire under motion. The query projects only the extreme dirs: `CamBasis::project` of a positive combination is an exact *convex* combination of the projections (weights `αᵢ(dᵢ·forward)`) — this gnomonic identity is what makes the padded bbox a conservative cover; `project` must stay a pure ratio of dots, and the extremes need no normalization. **Two pads, opposite polarity**: `PAD_Q` inflates the query box (erring inclusive is sound — extra cells only lower the min); `ACCEPT_PAD` tolerates projections just off the old screen and must stay well below `aabb_outside`'s producer slack (~6e-3 px) — accepting more admits directions no old claim covers. The NaN-child = parent-frontier rule is valid only because entries are stored **before** recursing, sky stores return without recursing, and the cache is cleared every producing frame (NaN ⇒ whole subtree NaN; `+INF` ⇒ all-NaN subtree). A NaN root, a blown visit budget, or any off-screen extreme must yield **None, never Sky** (an unfinished or empty min surfacing as `+INF` is the false-sky shape). Capping the descent depth is always sound (tc is monotone nondecreasing down the old chain); never shrink the query box as a cost fix (drops cells from the min — unsound). The recursion replays `trace_tile`'s exact midpoint splits and quadrant numbering (TL=0 TR=1 BL=2 BR=3). The consumer holds a **ring** of the last `TRING = 3` producing frames' caches (newest first, each paired with the exact basis it traced with; `TRING + 1` buffers so a victim always exists outside the ring): claims never go stale in a static scene, so an older entry can answer a region that panned off the newest screen and back. `temporal::lookup` retries older entries only on pose-specific misses (off-screen, behind-plane, blown budget); a completed **not-useful** min stops the scan — it is pinned by real nearest geometry that no older cache can claim past (`Miss` in temporal.rs). **Temporal query skip** (`Seed::Skip`/`Seed::TCut`, newest entry only): a completed FINITE min also predicts the tile's bound query is pinned (the blocked regime — the dominant tile outcome), so the tile SKIPS the query and runs only `refine_cut` on its own inherited cut at max(seed, t_start) — nothing old feeds the refine, so the skip is **unconditionally sound**; the old min merely chooses when it is profitable (an emptied refine is then a sky *proof*: frustum∖ball had no surviving subtree and the ball is empty by the claim). Under an identical basis the node's OWN old cut (stored per Split node in the double-buffered `CutStore`, paired with the ring's newest entry) is refined instead — same cone, already node-tight. Decay control: skipping stores no advance, so claims shrink by δ per hop; a 3-bit age in the CutStore's packed word (`MAX_ADOPT_AGE = 3`, keyed by the tile's own (depth, path) — both frames share the rect layout) forces a real query on the 4th consecutive hop. Do NOT hand an old cut from a DIFFERENT pose to the refine: it was tried via hull-containment against ancestor cells and measured SLOWER (an ancestor's cut covers 4× the cone — refining it costs more than the skipped query; the `--check` A/B rows are the regression guard, ray nodes bit-identical, ~35% fewer frustum nodes, measured win on the 1-spp motion workload). The ring's newest entry must be the last **producing** full-res hybrid frame with the basis it traced with — the whole ring drops on resolution change or any non-participating frame; structure-replay frames (below) are NOT producers and freeze the ring (no clear, no rotation) rather than invalidating it, which is sound because a replay re-presents that exact quadtree and asserts nothing new. Temporal data flows only through the primary-path `t_start`; secondary rays and the cut rules are untouched. Sky reuse under motion fires only when the query region lies wholly inside old sky — structural under pure rotation (the yaw check gate), typically 0 under dolly because the λ_max tilt drags the region toward the focus of expansion across the sky boundary; don't "fix" either.

## Architecture notes

- **The command line is `src/cli.rs`, and `cli::parse_from` is PURE** — that purity is the feature, not tidiness. `Opts`, the 144-arm parse loop and the `--help` text moved out of a 17.4k-line `main.rs` (which lost 1,284 lines to the extraction); `Cli` carries what were `main()` locals (the positional scene arg, `--stress`/`--tile`/`--cam`/`--spin`, every `check_*`/`*_dump`, `--cinematic`, `world_flag`). **Nineteen flag arms used to store straight into the "knob before scene load" process globals from inside the parse loop** (`texture::set_mips`, `texture::set_aniso`, `bvh::set_height_armed`, `scene::set_spray`, `clouds::set_enabled`, `gpu::trace::set_cloud_shadow`, `gpu::dxr::set_inline_mode`, …), and `settings::apply_globals` wrote those SAME statics moments earlier — two writers whose layering held **by call ordering alone**. They are ordinary `Opts` fields now (`mips`, `aniso`, `h2n`, `n2h`, `tinted_shadows`, `spray`, `depth_tint`, `water`, `heightfield`, `bloom`, `clouds`, `cloud_shadow`, `sky_lod`, `fireflies`, `fireflies_count`, `dxr_inline`), applied to the globals **exactly once** by `main`'s lever block — the block that already prints the departure lines, and which must stay ahead of any `Bvh::build`, the scene-cache probe, and every `--check*` dispatch (all of which read the statics directly). `apply_globals` is gone; `apply_to_opts` seeds the fields instead. **So: a new flag adds a field in `cli.rs` plus one line in that block. A setter call in the parse loop silently un-gates the parser** — which is the whole point, because purity is what lets `cli::self_test` run INSIDE `--check` without corrupting the texture/BVH/effect state that same run is using. Two contracts inside the block are load-bearing: **mips before aniso** (`set_mips(false)` forces aniso to 1 and `set_aniso` re-reads the mips switch, so this is the only order in which `--no-mips` still implies `--no-aniso`) and **one `heightfield` field storing BOTH `set_height_armed` and `set_height_on`** (what keeps `--no-heightfield --heightfield` a true arm, and what the headless paths reading `height_on()` depend on). Diagnostics accumulate into `Cli::notes` and `main` prints them, so a gate can parse an argv silently; hard errors still `exit(2)` in place (an invalid command line has nowhere to return to), which is why `self_test` only feeds it valid input. `--help` sets `Cli::helped` and BREAKS rather than printing — `main` calls `usage()` — so the throwaway parse `main` runs cannot emit the text twice. `cli::self_test` (in `--check`) pins the purity gate itself (parse an argv that moves every lever; the globals must come back untouched AND the fields must have moved — without that second half it passes vacuously), later-flags-win on the paired arms, the SDR|Sdr10|HDR10 three-way, the settings-seeded precedence seam, `--blas-split`'s optional value not swallowing a scene path, and `--help` stopping the parse. `settings::headless_args` deliberately SURVIVES as the pre-parse scan (it decides whether to load the file at all): replacing it with a throwaway `parse_from` was tried and rejected — `settings::self_test` probes it with partial command lines like `["--check-gpu", "--stress"]`, and a real parse of a valueless `--stress` `exit(2)`s the process mid-gate.
- `render.rs` has one depth-first driver: `trace_tile` (per-tile step `tile_step`: bound query over the inherited cut, then `refine_cut` for the children) recurses via nested `rayon::join` (tiles ≤ 32 px go sequential; ≤ 8×8 = `LEAF_TILE` shade per-pixel) with a `max_depth` cap checked **after** the leaf check, so a cap at or past the leaf depth is bit-identical to uncapped. `render_frame_capped` is dynamic resolution: at the cap, unresolved tiles sparse-fill (`sparse_fill` — "a pixel is not a little square"): one real `shade_pixel` point sample per 16×16 cell (`SAMPLE_CELL`) at a per-frame random pixel, stored `KIND_LEAF` with exact t/G-buffer — sound because every pixel of a capped tile lies inside the tile frustum, so the inherited cut/`t_start` (temporal seed included) apply, the leaf-tile argument — and the rest of each cell flooded with its sample's color/depth as the `KIND_COARSE` fallback; `MIN_BUDGET_DEPTH` floors the cap. The cap is uniform across the screen — that is what makes depth-first safe here (a wall-clock deadline would refine one corner and starve the rest, which is why the old driver was breadth-first); it also keeps cuts on the recursion stack, hot in cache. No clock is read in the driver: `main.rs` estimates next frame's cap from `last_ms` with a log4-proportional controller on a fractional accumulator (`depth_est`, clamped `[MIN_BUDGET_DEPTH, depth_full]`, slow-up/fast-down, deadband above 60% budget), updated only after budget frames. Trade-off: a hard cut to dense geometry can blow one frame before the controller reacts.
- Buffers (`accum`, `tbuf`, `info`) are `&[AtomicU32]` with relaxed ops — safe because each pixel is written by exactly one task per frame. `frame == 0` **stores** instead of adds; that implicit clear is how accumulation resets (camera move, mode/quality change) work — there is no explicit buffer clear anywhere. `main.rs` also resets `frame` on the budget↔normal transition so accumulation never adds onto a flat-quad frame.
- `accum` holds f32 bit-patterns of linear RGB; `resolve()` averages by sample count, tonemaps, upscales (the fixed half-res moving mode renders into a prefix of the full-res buffer; dynamic-res mode stays full-size), and blends the overlay.
- `tbuf` (per-pixel primary-hit t) exists solely for `render::verify`; `info` (depth|kind) feeds the overlay and verify's sky classification.
- Stats are batched into `stats::LocalStats` per tile and flushed once — never `fetch_add` per node visit or per ray. `shade()` takes `&mut LocalStats` directly; the hemisphere counters (`hemi_*`) ride the same struct and only print when nonzero.
- Bounce modules: `sphcell.rs` is pure direction-sphere math (van Oosterom–Strackee solid angle — NOT Girard, which cancels on small cells; Lambert PSA; Arvo triangle sampling with the +π parameterization — its self-test caught the sign mirror once already). `hemi.rs` mirrors `trace_tile`'s step (query → advance → refine → 4-way split, blocked-must-subdivide included) sequentially per shading point inside the existing rayon leaf task — no new parallelism. `sky.rs` is the one sky (scattering dome + sun disc; read its header before touching a sky call site), `sh.rs` is the order-2 SH the dome's irradiance rides on, and `clouds.rs` is the volumetric cloud layer `radiance()` composes over the backdrop (read its header for the known-accepts) — all pure math, zero rng draws, which is what keeps every same-seed/replay contract intact.
- **The two-tree split** (`src/ftree.rs`): the BVH's two consumers want opposite trees — ray traversal wants fat leaves and slab tests; the frustum bound query never touches a triangle (a leaf sets `best` to the BOX distance) and wants many small tight boxes.
  (The old "shafts deliberately stay binary" carve-out is moot — shafts are gone.) So hemi bound queries run on an **8-wide frustum tree** collapsed from the binary BVH (largest-area-first expansion, deterministic; every slot box IS a binary node's AABB, so bounds come out **bit-identical** — `ftree::self_test`, run by `--check`, pins a 512-probe equivalence sweep plus the slot audit and cut translation), 256 B/node SoA so the 8 slot tests compile to lane math. Cut entries are slot-refs (`(node << 3) | slot`); `Accel::of(bvh)` is the dispatch handle (rays ALWAYS on the binary BVH; `Accel::ray_roots` translates a cut iff a ray seeds from it). Built lazily on the first hemi query (only fb sessions pay; ~26-42 ms + ~256 B per 7 binary internals) — `--no-ftree` is the kill switch. **The lazy cache is a `OnceLock<FTree>` FIELD ON `Bvh`, not a process-global**, and that is a correctness property rather than a style choice: a wide slot id and its slot→binary-node map are only meaningful against the exact hierarchy they were collapsed from, so the cache must die with it. It used to be a `static INSTALLED: OnceLock<&'static FTree>` justified by "exactly one BVH is live per process" — which is FALSE the moment a scene is edited live: `main.rs`'s Y/Z frustum-snapshot path does `*bvh = Bvh::build(scene)` twice (clear, then append+rebuild), and the global would have handed the OLD tree's slot boxes and `bnode` ids to the NEW hierarchy — wrong bounds, and cut translation into a stale node array. Ownership now makes that unrepresentable (`Bvh::from_parts` is the one constructor, so every builder, the alt builders, and both scene-cache loaders get an empty cache by construction), and `ftree::self_test` pins it with two simultaneously-live BVHs whose trees must be distinct objects carrying their own slot boxes. An empty scene still builds ONE physically-present wide node whose occupancy mask quantizes to zero: the GPU's `ROOT_CUT_SLOT` path expands entries 0..7 and reads `ft_nodes[0]` BEFORE `ft_slot` tests occupancy, so a zero-length upload is an out-of-bounds read rather than an empty query (`quantized()`'s non-finite-extent arm has always been written for exactly this node). Measured: hemi-ao −15/−17%, hemi-gi −4/−8% ms/frame (default + San Miguel). **Never tune either tree against classic SAH**: measured on `--stress 5000`, SAH's predicted node visits ANTI-correlate with this renderer's measured `ray_nodes` (3-axis binning: SAH +20%, measured −33% — shared-origin rays with inherited `t_start` violate the uniform-random-line derivation); score builders on the measured counters. The CPU tile recursion is WIRED but default-off (`--ftree-tiles`): `tile_step`/`adopt_step` dispatch through `Accel::for_tiles`, the whole-screen root becomes the wide root slots, and leaf tiles translate their slot-ref cut to binary ray roots ONCE per tile (`shade_tile`/`sparse_fill`) — measured wall-neutral on San Miguel and ~10% slower on `--stress` no-temporal (fat singleton-entry cuts + short descents = the short-query regime; counted frustum nodes still drop −21..45%, so the quantized-box layout re-measures this), with the adopt off/on ray-node bit-identity surviving under slot-ref cuts; `--check`'s `wide-tiles` gate forces the lever on for one full verify pass so the wiring can't rot. Shafts deliberately stay binary (off-by-default feature; its `classify` fallback hands consumers a literal binary `[0]`). **On the GPU the split is finer-grained and live** (`gpu/shaders/ftree.hlsli`, compiled in by `#define FTREE` through the kernel-assembly defs — the ALPHA_CUTOUT pattern): the TILE kernels bind the wide tree at t0 (`SwAccel::Both` uploads both structures; `bind_common` binds wide, `record_hemi` rebinds binary) and measured **−23% on the `gpu hybrid` bench** (interleaved medians 3.41 vs 4.45 ms — that row is warm-clock noisy, never trust single samples) with the same-seed image BIT-IDENTICAL to the reference; the HEMI kernels deliberately stay on the binary tree, because hemi bound queries terminate in ~10 visits and a wide pop's unconditional 8 slot tests lose to the binary pop's 1 (the wide tree measured +35% ms there — after already fixing a worse version: `ft_expand` orders survivors with a selection scan over a live-bitmask precisely so every local-array index is compile-time; an insertion sort's dynamic shuffle indices demote the arrays to GPU scratch memory, which alone was +58%). GPU cuts never seed rays in the default configuration, so the quantized upload drops `bnode`; under `--sw-rays` (the software-ray lever) a parallel flat `ft_bnode` map uploads and `level_finish` translates each leaf-emitting split's slot-ref cut to binary node ids at emission — the GPU flavor of `Accel::ray_roots`. **The GPU upload is u8-quantized** (`ftree::QFNode`, 112 B vs the CPU FNode's 256 — `FTree::quantized()` at upload, ftree.hlsli decodes `org + q·sca` with `precise` so an fma can't round a face inward): every face rounds OUTWARD in a per-node frame, verify-adjusted against the exact decode expression, so decoded boxes CONTAIN the true ones — all three prunes weaken only conservatively (the plan-doc proof; `self_test` audits containment + per-face quantum slack with an ulp term for sub-ulp-sca flat axes, and the same-seed image stays bit-identical). Split-format verdict (measured): the CPU keeps f32 nodes — decoding cost +9–15% on the hemi bench with node counts unchanged; the GPU takes quantized — San Miguel `gpu hybrid` medians 2.86 vs 3.03 ms (bandwidth pays once the tree exceeds cache; the L2-resident default scene reads ~+6%, inside that row's noise) and the tree upload drops −56% (SM 157 → 69 MB; ~1.5 GB → 0.65 GB at the 90M-tri scale).
- **Structure replay** (`src/replay.rs`): a full-depth uncapped hybrid frame records its terminal quadtree — every leaf (rect, inherited `t_start`, cut) and sky rect, bump-allocated with relaxed atomics — and the next frame **replays** it (`render::render_frame_replay`: flat par_iter to `shade_tile`/banded `fill_sky`, zero frustum queries) when its `CamBasis` is bit-equal at the same res (`replay_key` in main.rs). Sound because the structure is a function of (scene, BVH, basis, rw, rh) only; shading params (quality, frame, jitter, G-buffers) come from the fresh ctx, so quality/denoiser toggles deliberately do NOT invalidate — only motion, res steps, budget frames, and plain mode do. Every accumulation still frame and every DLSS/XeSS/OIDN-temporal still frame after the first replays. Replay frames record nothing, write no temporal cache, and freeze `tprev_*` (see the temporal bullet). `--check` gates: exact terminal pixel accounting, replay-vs-trace bit-identity of tbuf/info/accum at frame 0 AND at a warm jittered frame 1 (which is the proof that a warm identical-basis re-trace has the identical terminal structure), and a post-replay dolly verify on the frozen cache. Overflow/capped-arm contact poisons the recording — the frame just isn't replayable, never wrong.
- **Input** (`src/input.rs` + `src/flycam.rs`): input.rs is the main-thread SDL event drain only (toggles/quit/resize edges). Camera flight/look — and **Xbox-controller (XInput) support**: left stick = analog flight (deflection = speed, full tilt == key speed), right stick = look rate (`LOOK_RATE` rad/s × deflection), triggers = up/down, bumpers = the Ctrl(/16)/Shift(/8) divisors (smoothstep-eased over `SLOW_EASE_S` = 0.25 s in log2 space, so engaging/releasing glides instead of stepping 8-16× in one tick; rest states exact), D-pad left/right = time-of-day scrub (with `,`/`.`, integrated at `TOD_RATE` = 1 h/s into the same one-lock `FlyState` snapshot the pose rides) — live on the **flycam thread**, a ~500 Hz wall-clock integrator (high-res waitable timer; `timeBeginPeriod` fallback) sampling `GetAsyncKeyState`/`GetCursorPos`/`XInputGetState`, which read live OS state from any thread while the main thread is blocked in a trace (SDL state only updates at pump time — useless off-thread; this is why per-frame integration made a tap mean "a whole frame of motion" or nothing). Each tick integrates with the MEASURED dt (clamped 0.1 s), so displacement is an exact function of wall-clock hold time at any framerate; the thread runs at `THREAD_PRIORITY_ABOVE_NORMAL` (the rayon pool saturates every core at normal priority for the whole trace — a starved tick still can't lose displacement, dt being measured, but it coarsens the sampling the 500 Hz rate is there to buy). Focus-gated via `GetForegroundWindow`; drag-look latches only on client-area presses and reads the PHYSICAL primary button (`SM_SWAPBUTTON` picks `VK_RBUTTON` — Windows swaps buttons at the message layer, which is where SDL's `mouse_state().left()` used to pick it up, but `GetAsyncKeyState` is physical). A **pause gate** covers session rebuilds: a long FRAME must integrate (the entire point of the feature), but a resize/F11 re-entry presents nothing for seconds (kernel compile, scene upload, BLAS build), and flying through that with W held is flying blind — so the thread spawns PAUSED, `session` resumes it once its frame loop is live, and `run_window` re-pauses on the resize path. Paused ticks still advance the dt clock, so resuming costs one tick, never the whole span in one step. The render loop consumes exactly ONE `FlyCam::snapshot()` per iteration (trace pose == MV pose == prev-capture pose — the temporal/replay/upscaler bit-equality contracts) and `moved` is the snapshot bit-compare (`Camera: PartialEq`); a session-local `Camera` write would be overwritten — teleports must go through `FlyCam::set`. The thread is spawned once in `run_window` and owns the pose across resize re-entries (`Persist` no longer carries `cam`). Headless paths (`--check*`, `--spin`) never touch it.
- Epsilons are scale-relative to `Scene::diag` (set in `SceneBuilder::finish`); OBJ scenes are auto-fitted to diagonal 10 so the same camera/light/epsilon constants work for any model.
