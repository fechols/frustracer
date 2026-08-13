# Sky, lighting, and clouds

`--tod`, fireflies, emissive lights, the volumetric cloud layer, `--cloud-shadow`, `--sky-lod`. Also holds `--no-audio`, which sits in this run for contiguity rather than topic.

Extracted verbatim from `CLAUDE_Historical.md`, which keeps a stub pointing here. Nothing in this file was rewritten.

```
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
                                      # --no-emissive-lights spells the default). OFF by default
                                      # on the user's THIRD-round call (2026-08-06 — a one-day
                                      # default-ON flip was feel-tested "beautiful" and reverted
                                      # same-day): the CPU shadow-ray cost is real (the MEASURED
                                      # block below: ~3.3 of the +5.5 ms is rays — irreducible,
                                      # they ARE the lights; EL_BOOST below ~doubles r_infl2
                                      # pre-cap, so in-range counts sit ABOVE those pre-boost
                                      # figures — every MEASURED number in this entry is
                                      # pre-boost) and only ONE world island — bistro — carries
                                      # emissive maps, so a default-ON session paid derivation
                                      # everywhere for one island's benefit. The old rationale's
                                      # other half — faint pools — IS resolved: EL_BOOST (the
                                      # LOOK FINDING below) stays, so an armed session now earns
                                      # its ms. THE XeSS/FSR3 AUTO-ARM IS NARROWED TO
                                      # DENOISER-LESS SESSIONS (2026-08-10, the user's call —
                                      # the emissive-integration campaign; the arm's 2026-08-08
                                      # three-round history: introduced fourth round, retired
                                      # fifth on the premise that NRD integrates the bounce's
                                      # emissive ahead of the TAA clamp, RE-INSTATED sixth when
                                      # the user's feel-test found NRD NOT sufficient — a
                                      # verdict that PREDATED FLAG_NRD_GI, when the bounce rode
                                      # the un-denoised residual): main::upscaler_defaults — the
                                      # vendor_defaults sibling — arms emissive lights in
                                      # sessions whose WIRED upscaler is TAA-class (XeSS/FSR
                                      # 3.1) AND that will run NO pre-upscale denoiser fold
                                      # (dn_fold = dn_kind && rtgi at the call site), because a
                                      # TAA neighborhood clamp rejects the RTGI bounce's sparse
                                      # stochastic emissive and armed cluster NEE is the
                                      # deterministic delivery THOSE sessions need — while a
                                      # default XeSS/FSR3+FRD session integrates the bounce's
                                      # emissive itself (the GI-fold firefly relax +
                                      # stabilization, measured 42% of DLSS-RR's pool delivery
                                      # at 0.47 stability; NRD 68%) and keeps the compiled OFF.
                                      # --no-frd/--no-rtgi/--nppd sessions keep the arm;
                                      # known-accept: a mid-session denoiser SHED leaves
                                      # neither fold nor NEE (the shed line is the signal). Its
                                      # cinematic twin applies the policy once per
                                      # run_cinematic_gpu run at the first wiring that took,
                                      # with the REAL dn_fold since the capture pipeline gained
                                      # its own denoiser slot (gpu::CineDn, same day — see the
                                      # cinematic section); quin/plain/RR-class sessions
                                      # keep the compiled OFF. Opts::emissive_lights_explicit is
                                      # the veto (BOTH CLI spellings set it — presence, not
                                      # value, the dxr_inline_explicit doctrine, so
                                      # --no-emissive-lights is the spelled opt-out in XeSS/FSR3
                                      # sessions; a settings-file value vetoes too — the menu
                                      # writes effects.emissive_lights). Headless paths never
                                      # run the policy, so the armed must-fires still need an
                                      # explicit --emissive-lights and every gate stays a pure
                                      # function of the command line. The NEE-keep rule below is
                                      # untouched (an armed session runs NEE with the bounce
                                      # suppressing its display-add).
                                      # At load, finalize_scalars enumerates emissive triangles
                                      # (per-tri power = area × Ke × a 4-tap map_Ke mean) and
                                      # clusters them — deterministic grid seed at EL_GRID_K ×
                                      # content diag, then min-power-into-nearest agglomerative
                                      # merge to the budget; SERIAL and index-ordered, so
                                      # byte-deterministic like the BVH build; derived-never-
                                      # serialized (the sky_sh precedent — warm loads re-derive,
                                      # no CACHE_VERSION move, and neither lever keys the
                                      # .fcache). PLACEMENT has its own A/B lever, `--el-cluster
                                      # grid|som` (2026-08-06, the --bvh-builder bake-off
                                      # pattern — dev lever, no settings row, illegal value
                                      # exits 2 at the lever block): grid = the shipped clusterer
                                      # bit-identically (the ONE mode conditional sits between
                                      # the merge and the finalize); som = emissive::som_refine,
                                      # a power-weighted BATCH-SOM refinement of the merged
                                      # centers — radius-0 neighborhood ⇒ exactly weighted
                                      # Lloyd's, deliberate: merged centers carry no lattice
                                      # topology (the bvh som_codes lattice learned a
                                      # space-filling CURVE; that purpose doesn't transfer) —
                                      # EL_SOM_EPOCHS=8 fixed serial epochs, ties to lowest
                                      # index, fixed-order f32 sums ⇒ byte-deterministic; power
                                      # conserves BY CONSTRUCTION (final pass assigns every tri
                                      # exactly once), count can only shrink, zero rng.
                                      # self_test gate 8 runs the som arm unconditionally
                                      # (determinism, conservation × EL_BOOST, budget cap,
                                      # influence band, and a must-differ-from-grid
                                      # anti-vacuity); the judge for a real A/B is the
                                      # feature's own calibration instrument — a GI (H) still
                                      # frame at the same pose is ground truth, so `--el-cluster
                                      # grid` vs `som` screenshots + the check counters decide.
                                      # Each cluster is a Lambertian DISC
                                      # light: irradiance C/π/(d²+rc²) — the +rc² denominator IS
                                      # the near-field softening (no hot spot beside a large
                                      # panel) — × an EMISSION LOBE (2026-08-11, see below) and
                                      # windowed by the fireflies' (1−d²/r²)² exact-zero
                                      # falloff at an influence radius derived from EL_MIN_E
                                      # (the ONE cost-vs-reach knob: the per-pixel scan pays a
                                      # shadow ray per in-range light), floored at 2·rc, capped
                                      # at EL_RMAX_K·diag_c, lum clamped under EL_E_MAX (f16
                                      # headroom, the sun-disc lesson).
                                      # THE EMISSION LOBE (2026-08-11): until then every cluster
                                      # was ISOTROPIC — `irradiance(l, d2)` took no direction, so
                                      # the C/(d²+rc²) the module documents as the ON-AXIS value
                                      # of a Lambertian disc was delivered in EVERY direction, and
                                      # a one-sided panel lit points behind it exactly as brightly
                                      # as points in front (shadow rays hide much of it; what
                                      # survives is over-lighting to the side and edge-on). The
                                      # triangle normal was ANNIHILATED on the first line that
                                      # touched it — `0.5*(b-a).cross(c-a).length()` — so nothing
                                      # downstream could have known. Now `Cluster` accumulates
                                      # nacc = Σ w·n beside the existing Σ w·centroid at the SAME
                                      # weight (linear, so the agglomerative merge stays
                                      # associative/index-ordered/byte-deterministic), and the
                                      # mean resultant length R = |Σw·n|/Σw falls out free.
                                      # MEASURED FIRST, and the measurement is what justified
                                      # building it: helmet min 0.820 mean 0.986 max 1.000 (32 of
                                      # 32 panel-like), bistro Exterior min 0.139 mean 0.620 max
                                      # 1.000 (18 of 32, with 4 genuine bulbs under R=0.2) — real
                                      # emissive content is strongly ORIENTED, and bistro is
                                      # ideally MIXED so it exercises both ends.
                                      # THE AXIS IS ORIENTED AGAINST THE AUTHORED VERTEX NORMALS,
                                      # never by winding alone (derive_parts took a `normals`
                                      # param for exactly this): the cross product's only
                                      # orientation is index order, and this renderer does not
                                      # otherwise trust it — surface_point keeps an unconditional
                                      # face flip precisely for "a mesh whose winding disagrees
                                      # with its authored vertex normals", a modeling error the
                                      # loader PRESERVES. Harmless while the normal was discarded
                                      # a line later; not harmless once the lobe points where it
                                      # points, because every other emissive path is TWO-SIDED
                                      # (the display add has no facing test, moller_trumbore is
                                      # two-sided, the fb.gi gather takes emitters from either
                                      # side) — so a reversed panel would keep glowing on screen
                                      # AND keep lighting the room under GI while its NEE pool
                                      # silently vanished, i.e. light removed where it belongs,
                                      # which is strictly worse than the isotropic over-lighting
                                      # this replaces. High R cannot detect it (R says the winding
                                      # is CONSISTENT across a cluster, never that it points
                                      # outward). The rule is surface_point's own: sum the three
                                      # authored normals (one direction per face, robust to a
                                      # single bad vertex) and flip when they disagree; no normal
                                      # array, or a cancelling/NaN authored set, keeps winding
                                      # BITWISE. Gated by self_test 9(f) with teeth BOTH ways —
                                      # the quad whose winding normal is −Y, authored +Y, must
                                      # read +Y with the array and −Y without it (the second arm
                                      # is what stops the probe passing vacuously on geometry that
                                      # never disagreed; the pre-fix first arm reads −1.0).
                                      # RESIDUAL known-accept: a genuinely two-sided emissive
                                      # plane (one-sided winding, no back face) now lights only
                                      # its front — correct for a Lambertian emitter, and the
                                      # display add showing it lit from behind is the pre-existing
                                      # inconsistency, not this.
                                      # The profile is `f = 1 − R + saturate(dot(v, w))` for v =
                                      # R·n_c (stored UNNORMALIZED so |v| IS R) and w the unit
                                      # direction FROM the light TO the receiver. R = 0 BRANCHES
                                      # to exactly 1.0 (never a computed ×1.0), so a bulb — and
                                      # every emissive-free scene — runs the pre-lobe stream
                                      # bitwise; R = 1 is saturate(cos), exactly 0 at 90° and
                                      # behind. ATTENUATION-ONLY, and that bound is the whole
                                      # design rather than a taste call: saturate(dot(v,w)) ≤
                                      # |v| = R gives f ≤ 1, so r_infl2 (a closed-form solve of
                                      # the ISOTROPIC falloff against EL_MIN_E), the exact-zero
                                      # window, and cull_tile's EXACTNESS argument — which is what
                                      # lets CPU and GPU cull independently with no bit-parity
                                      # contract — all stand completely unchanged, and the feature
                                      # costs strictly LESS work than before. The energy-conserving
                                      # variant `1 + R·(4·saturate(cos) − 1)` (sphere-mean exactly
                                      # 1) is the physically complete form and peaks at 4×, so it
                                      # needs r_infl re-derived from the profile MAXIMUM and the
                                      # cull proof re-validated — deliberately not the first ship.
                                      # KNOWN TRADE: this attenuates rather than redistributes, so
                                      # directional clusters read dimmer overall and EL_BOOST (an
                                      # artistic constant, already moved once for exactly this)
                                      # is the retune knob — OWED: the feel-test, since the look
                                      # is not gate-visible. Transport: el_c[64] appended LAST
                                      # after `split` (xyz = v, w = R — R carried explicitly so
                                      # neither shader needs a normalize or a sqrt), CB_STRIDE
                                      # 4608 → 5632 and the FrameCb size assert 32 →
                                      # 48·MAX_EMISSIVE_LIGHTS (the MAX_SPP-lockstep class; this
                                      # does NOT touch the 64/64-full root signature, which
                                      # governs raising the light CAP, not adding a row).
                                      # Gates: emissive::self_test gate 9 — R=0 bit-exact 1.0 over
                                      # a direction sweep, f ∈ [0,1] swept over R × direction (the
                                      # bound the cull rests on), R=1 equals saturate(cos) with
                                      # exact zeros behind, monotone in angle with the 1−R floor,
                                      # and DERIVATION teeth BOTH ways: a flat quad must read
                                      # R ≥ 0.99 and a CLOSED BOX must disarm to R ≈ 0 and land on
                                      # the exact-1.0 arm (the bulb case — measured 0.000), plus
                                      # 9(f), the winding arm above. NOTE the isotropic exact-1.0
                                      # branch is near-VACUOUS on real content — bistro's dimmest
                                      # cluster reads R = 0.139, not 0, so real bulbs take the
                                      # computed arm at a mild ≤14% back-side attenuation; the
                                      # branch is a structural guarantee for emissive-free and
                                      # exactly-cancelling scenes, not the common path.
                                      # Liveness proven by image A/B, not assumed (the N9 "the
                                      # flag reached the pack but changed nothing" lesson):
                                      # forcing lobe → 1.0 moves the armed helmet check frame
                                      # (884AF885 → D42676D2) AND fails gate 9.
                                      # Sampled in shade()'s
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
                                      # determinism, power conservation through the merges —
                                      # scaled by EL_BOOST, which lands once at the C_c fill —
                                      # budget cap, falloff zeros/monotone/clamp, the 4-tap map
                                      # identity, the som gate-8 family — runs regardless of
                                      # arming, pure math);
                                      # --check-gpu/--check-dxr must-fire emissive_rays > 0 on
                                      # emissive scenes WHEN ARMED (run them WITH
                                      # --emissive-lights on helmet/bistro — the --heightfield
                                      # checks-follow-the-session-flags pattern; NEE stays live
                                      # under default-ON RTGI per the NEE-keep rule in the
                                      # --no-rtgi entry, so the must-fires need no extra flag;
                                      # CPU-side
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
                                      # RESOLVED 2026-08-06: the PHYSICAL calibration (C = area
                                      # × map_Ke radiance) read as glowing bulbs with FAINT
                                      # pools — bistro's lamp clusters total power 0.164, so
                                      # pools only beat the dome after true nightfall at close
                                      # range; the ARTISTIC per-cluster boost landed as
                                      # EL_BOOST = 2 at the C_c fill in emissive::derive_parts
                                      # (the MOON_E_OVER_PI/STAR_E precedent — one constant,
                                      # self_test power-conservation scales with it, the loud
                                      # derivation line prints the boosted total annotated).
                                      # NOTE the boost also ~doubles r_infl2 pre-cap, so reach
                                      # AND per-pixel scan cost rise with it — the feel-test
                                      # knob if 2 overshoots is the same one constant
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
                                      # src/shaders/skylod.hlsli): -9.8% frame at spp=16, -1.0%
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
```
