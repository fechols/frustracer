# Global illumination and acceleration structures

The `--rtgi-bounces` GI ladder and its Russian-roulette half-rungs, coincident-face culling, the BVH builder bake-off, `--blas-split`, and `--dual-gpu`.

Extracted verbatim from `CLAUDE_Historical.md`, which keeps a stub pointing here. Nothing in this file was rewritten.

```
cargo run --release -- --rtgi-bounces 1.5  # THE GI LADDER (2026-08-12) — real-time GI as a BOUNCE
                                      # BUDGET rather than a switch, so quality scales per GPU and
                                      # resolution. Rungs 0 | 0.5 | 1 | 1.5 | 2 (any N in [0,2]
                                      # parses — the implementation reads floor and fract, never a
                                      # table; out of range EXITS 2 rather than clamping, the
                                      # loud-lever rule, since a silently clamped quality knob
                                      # reports the wrong arm in an A/B). DEFAULT 2.0 since
                                      # 2026-08-12 — TWO REAL BOUNCES, the user's feel-test call
                                      # ("a pretty big perf hit, but it looks AMAZING"), and the
                                      # cost is accepted rather than discovered: 1 -> 2 is CPU
                                      # 51.61 -> 57.41 ms and GPU leaf 0.292 -> 0.374 (4090) /
                                      # 0.423 -> 0.579 (san-miguel-lp), i.e. +28-37% of the leaf
                                      # region and ~4% of GPU frame span on the light scene.
                                      # `--rtgi-bounces 1` is the lever for the time back and is
                                      # the arm every pre-ladder number in this file describes;
                                      # 1.5 buys most of the look for 62% of the increment on a
                                      # heavy scene. THE DEFAULT LIVES IN ONE PLACE,
                                      # `shade::DEFAULT_BOUNCES` — it was reaching four sites
                                      # (that static, cli::defaults, main's lever-line comparand
                                      # and the self-test's expectation) and a half-landed flip is
                                      # SILENT there: the lever line would either stop announcing
                                      # a departure or start announcing the default. settings.rs's
                                      # Cycle row keeps its own default_ix (it indexes strings, not
                                      # floats) and cli::self_test pins the two against each other.
                                      # --no-rtgi is an alias for 0 and --rtgi for 1, all three
                                      # composing under ONE later-flags-win rule (cli::self_test
                                      # pins the 11-case table). Settings row:
                                      # Renderer/rtgi_bounces — beside `bounce`, the still-frame
                                      # hemi tier that TAKES PRECEDENCE over it, because the two
                                      # are one decision and the precedence is invisible with the
                                      # pair on different pages (and a bounce budget is not an
                                      # "effect" in the sense bloom and fireflies are) — a
                                      # StepF { 0, 2, step 0.5 } whose
                                      # `default` READS `shade::DEFAULT_BOUNCES` (restart tier —
                                      # both GPU blocks are compile defines). A stepper rather
                                      # than a Cycle because the budget genuinely is a continuum
                                      # the parser takes anywhere in [0,2] and the step quantizes
                                      # it to the rungs; five unrelated strings was the tail
                                      # wagging the dog. STEP 0.5 IS LOAD-BEARING: a power of two,
                                      # so every stop is exactly representable in f32 and the
                                      # stepper lands on them BITWISE — main's lever line
                                      # (`!= DEFAULT_BOUNCES`) and rtgi_corr_p's rung split are
                                      # both float-equality tests, and a 0.1-style step would
                                      # accumulate to 0.30000001 and announce a departure from a
                                      # value the user had just selected as the default.
                                      # cli::self_test WALKS the (min, max, step) tuple and
                                      # requires exactly [0, 0.5, 1, 1.5, 2], each parseable by
                                      # the CLI (teeth: step 0.25 fails with the walk printed).
                                      # The old `rtgi` bool key, and the String this field briefly
                                      # was, are both simply unknown to the schema and ignored —
                                      # deliberately unmigrated, the hdr10 precedent.
                                      # THE FLIP'S OWN PROOF, and the shape to reuse for any
                                      # default move: `--check --rtgi-bounces 1` reproduced the
                                      # PRE-FLIP check.png and check_gi.png BIT-FOR-BIT, so the
                                      # change provably moved the default and not the arm — a
                                      # golden re-baseline that also demonstrates what it did not
                                      # touch. Both goldens are now the rung-2 frame.
                                      #
                                      # THE HALF RUNGS ARE RUSSIAN ROULETTE ON THE DELTA OVER THE
                                      # SH×AO TAIL, and that is the whole design. Every rung already
                                      # ends in a tail approximating the transport a real gather
                                      # would compute, so continuing with probability p and
                                      # weighting the DIFFERENCE by 1/p gives
                                      #     E[tail + (G−tail)/p·[continue]] = G
                                      # — unbiased for the DETERMINISTIC rung above, with the
                                      # variance riding on (G−tail)² instead of G² because the tail
                                      # is a control variate (small in the open, largest in
                                      # enclosures, which is exactly where the samples are worth
                                      # spending). A plain coin flip between the two rungs delivers
                                      # only p of the correction — BIASED, measured −0.0194 signed
                                      # at rung 0.5 — and that trap is what the gate's teeth
                                      # (FR_RTGI_NOWEIGHT=1) reproduce on demand. NEVER "simplify"
                                      # the weighted delta into an average of two images.
                                      #
                                      # ONE FIELD CARRIES IT: `Quality::rtgi_bounces: f32`,
                                      # DECREMENTED per level, so the budget IS the recursion bound
                                      # — it replaced a `depth == 0` gate that could only ever
                                      # express "one" and said nothing about the reflection/glass
                                      # children (which set 0.0 explicitly and can never inherit
                                      # one). Not a bool beside a probability, because
                                      # `Quality { rtgi_bounces: 0.0, .. }` then pins the whole tier
                                      # in a single token and the ~dozen composition gates that need
                                      # a deterministic AO ambient cannot be left half-pinned.
                                      # `shade::rtgi_gather` is the ONE gather both arms call, which
                                      # is what makes the roulette's expectation land ON the
                                      # deterministic rung rather than merely near it (the
                                      # unbiasedness gate self-oracles against it).
                                      #
                                      # TRANSPORT — TWO compile defines AND TWO runtime bits, and
                                      # the runtime half is NOT optional: `RTGI` (n ≥ 1, the
                                      # deterministic gather) + `RTGI_CORR` = p (the correction) and
                                      # `RTGI_CORR_L0` (n < 1, i.e. the correction is at level 0),
                                      # beside FLAG_RTGI and FLAG_RTGI_CORR (bit 27). A
                                      # COMPILE-ONLY correction shipped first and fired inside gates
                                      # that had pinned the tier off — drawing rng, tracing, and
                                      # folding into prim.direct_d in frames that asked for none of
                                      # it (NRD n3 byte-diff 853, FRD F3, N7's sky-ext-skip
                                      # hit-bad/hit-px landing on EXACTLY p). ANY new shading feature
                                      # needs both halves. `gfx::frame::rtgi_corr_p` is the ONE
                                      # derivation the define and the bit share: p = n for n < 1,
                                      # else min(n−1, 1) — a plain fract(n) reads 0 at exactly 2.0
                                      # and SILENTLY COMPILED the top rung's second bounce out
                                      # (--check-gpu caught it as `level-1 0`); the CPU expresses
                                      # rung 2 by recursing on a decremented budget, which HLSL
                                      # cannot do, and this is that shape flattened to the two levels
                                      # the GPU carries. The correction's gather-shade is SEQUENTIAL
                                      # with the bounce's in shade_full, never nested, so peak
                                      # register pressure is max() of the shade_split instances and
                                      # not their sum — FR_WIDTH=1 reads leaf/sky/level 32/32/32
                                      # unchanged at every rung.
                                      #
                                      # THE TAIL COSTS AN AO RAY, which is the one structural cost
                                      # worth knowing: a correction needs `tail`, `tail` needs `ao`,
                                      # so any surface running one traces an AO ray. Hence rung 0.5
                                      # is DOMINATED by rung 1 (measured level on CPU 51.28 vs 51.61
                                      # ms and on GPU — same image in expectation, more variance, no
                                      # cheaper: it keeps the primary AO ray rung 1 elides), and
                                      # hence GPU rung 2 traces an AO ray whose result is
                                      # ALGEBRAICALLY CANCELLED (p = 1, so the correction removes
                                      # exactly the tail the ray produced) where the CPU's
                                      # deterministic depth-1 arm never enters that block at all —
                                      # same value, one ray apart, absorbed by the statistical
                                      # CPU-vs-GPU gates. Dropping rung 0.5's AO ray for a coarser
                                      # control variate is the documented rescue; it changes
                                      # prim.ao semantics, so it is not free.
                                      #
                                      # MEASURED (min-of-N, maiden discarded). CPU --spin path 120f
                                      # procedural: 0 = 42.25, 0.5 = 51.28, 1 = 51.61, 1.5 = 55.58,
                                      # 2 = 57.41 ms. GPU leaf region, 4090, procedural 3000f: 0.263
                                      # / 0.373 / 0.292 / 0.361 / 0.374 ms; san-miguel-low-poly
                                      # 2400f: 0.362 / — / 0.423 / 0.520 / 0.579. Two readings: on
                                      # a heavy scene rung 1.5 buys rung 2's image for 62% of the
                                      # 1→2 increment, and 1→2 is only ~4% of GPU frame span on the
                                      # procedural scene — so FULL 2-bounce may simply be affordable
                                      # without the trick. GPU WAVE DIVERGENCE blunts stochastic
                                      # depth generally: a 32-lane wave pays for any lane that
                                      # continues, and at p = 0.5 essentially every wave has one, so
                                      # ray counts halve while wall clock moves much less.
                                      #
                                      # Everything below describes rung ≥ 1 and is unchanged by the
                                      # ladder: ONE cosine-sampled bounce ray per pixel
                                      # per frame IS the ambient term — hit shades at hemi's
                                      # BOUNCE_Q leaf policy (1 shadow + 1 AO + SH×AO ambient, the
                                      # tail standing in for deeper bounces; bounce hits never
                                      # re-bounce), miss returns sky::gather (NO sun disc — the
                                      # once-per-path rule). Cosine importance sampling makes the
                                      # sampled radiance the irradiance-convention estimate
                                      # directly (no π — the L-in-L-out pin), so accumulation
                                      # converges it on stills and the temporal denoisers launder
                                      # it on the 1-spp upscaler contract — real one-bounce GI +
                                      # REAL EMISSIVE TRANSPORT (bounce hits carry the display
                                      # emissive add) in every session, all three render modes:
                                      # CPU inline at the ambient tier (shade.rs), wavefront +
                                      # DXR through ONE implementation in shade_full (the second
                                      # shade_split call at hemi_leaf.hlsl's literal args — DXC
                                      # constant-folds it thin, the Q4 rung-2 mechanism at source
                                      # level). PRECEDENCE: the still-frame hemi tiers (H) win —
                                      # FLAG_RTGI (bit 19, the runtime half; derivation guarantees
                                      # fb_mode == 0 whenever set) clears on fb frames, and the
                                      # CPU arm gates !fb.ao/fb.gi-first + depth == 0. EMISSIVE —
                                      # THE NEE-KEEP RULE (2026-08-08, the XeSS feel-test: a
                                      # TAA-class upscaler's neighborhood clamp rejects sparse
                                      # stochastic emissive, so bounce-only pools vanished under
                                      # XeSS/FSR3 while DLSS-RR reconstructed them): armed
                                      # cluster NEE STAYS LIVE under RTGI — the bounce's
                                      # emissive display-add suppresses instead, per frame, so
                                      # exactly ONE mechanism delivers emitter-as-emitter
                                      # transport. CPU: Quality::emissive_display (true
                                      # everywhere; the RTGI arm's bounce Quality sets it
                                      # el.is_none(), propagating through the bounce's glass
                                      # chain via ..*q); GPU: shade.hlsli's emissive block gates
                                      # on `cam_lights || !(flags & FLAG_EMISSIVE)` — no new
                                      # shader argument (camera laps always add; hemi's fb.gi
                                      # bounce keeps the add because FLAG_EMISSIVE clears at
                                      # fb_mode==2, the gather delivering; an unarmed frame's
                                      # clear flag keeps the RTGI bounce as the only delivery).
                                      # The emissive/el-cull must-fires therefore work unchanged
                                      # on armed sessions. Known-accept: UNARMED stochastic
                                      # emissive is TAA-clamped under XeSS/FSR3 (RR integrates
                                      # it) — RESOLVED TWICE, by different mechanisms per
                                      # session class (see the --emissive-lights entry's
                                      # auto-arm paragraph): default XeSS/FSR3+FRD sessions
                                      # INTEGRATE the bounce's emissive through the denoiser
                                      # fold since 2026-08-10 (FRD's GI-fold firefly relax +
                                      # stabilization — 42% of DLSS-RR's pool delivery, the
                                      # emissive-integration campaign; the 2026-08-08
                                      # NRD-not-sufficient feel-test predated FLAG_NRD_GI),
                                      # while denoiser-less TAA sessions (--no-frd/--no-rtgi/
                                      # --nppd + every cinematic XeSS/FSR3 capture) still get
                                      # the cluster-NEE auto-arm. The residual known-accept:
                                      # sessions that veto BOTH (--no-emissive-lights with no
                                      # fold, or a mid-session denoiser shed).
                                      # CONTRACTS:
                                      # the arm is Quality-gated (`Quality::rtgi_bounces`,
                                      # constructors read shade::rtgi_bounces — check harnesses pin
                                      # the FIELD per pass, never a global), draws 2 rng only when
                                      # armed (+1 for a rouletted rung's own continue/terminate
                                      # decision, so a continued pixel's stream is LONGER than a
                                      # terminated one's — sound because pixels have independent
                                      # streams and nothing downstream reads a POSITION in one)
                                      # (off path textually skipped, never burned — the fb
                                      # precedent); shading-only, so every exact-zero soundness
                                      # gate and the temporal/replay machinery are untouched
                                      # (spin counters bit-identical across arms, measured); runs
                                      # identically under VisCtl Off/Capture/Apply (no VisRecord
                                      # contact — adaptive streams stay aligned; the AO-reuse
                                      # savings are gone, shadow reuse remains); prim.ao stays
                                      # 0.0 (the fb.gi export precedent — GI rides FSR-RR's
                                      # un-denoised residual, the accepted v1 known-accept FOR
                                      # FSR-RR; NRD sessions fold the bounce into the dd sig
                                      # capture instead since 2026-08-08 — FLAG_NRD_GI, see the
                                      # --nrd entry — because riding the residual there meant
                                      # ReBLUR denoised only direct light, the "NRD isn't on"
                                      # user-report class; the
                                      # composition-gate family in check-gpu/check-dxr pins
                                      # rtgi_bounces: 0.0 on its uq so the AO-term must-fires keep
                                      # teeth). GPU: rtgi_defs() in every SHADE_HLSLI unit (both
                                      # pipelines — the probe-reach rule), CTR_RTGI_RAYS = 25
                                      # (level 0) + CTR_RTGI_RAYS2 = 26 (level 1 and deeper), so
                                      # CTR_COUNT 26→27, width slots 27..31, CTR_TOTAL 32 — the
                                      # source-text asserts are DERIVED from the consts now, since
                                      # a hand-written literal pins the cross-language agreement
                                      # PLUS a renumbering nobody promised not to do; the
                                      # wavefront-vs-reference same-seed A/B stays EXACT 0.00e0
                                      # armed at every rung. Gates: --check's `rtgi` must-fire, read
                                      # as the RUNG SIGNATURE — the PAIR (level-0, level-1) must be
                                      # (0,0) at rung 0, (>0,0) at 0.5 and 1, (>0,>0) at 1.5 and 2
                                      # (the single-counter form read 0.5 and 1 as the same session;
                                      # the zero arms are the teeth, since each rung's blocks are
                                      # compile defines and a count on a rung that omits them means
                                      # a define escaped its guard); the `rtgi-rr` UNBIASEDNESS gate
                                      # (below); + the `rtgi-ab` estimator/wiring A/B
                                      # (32-frame RTGI accumulation vs 4-frame fb.gi, trimmed
                                      # mean rel measured 0.0124 / signed −0.0010 vs 0.08/0.02
                                      # limits — the probe-level GI gate already pins the bounce
                                      # MATH, its cosine reference IS this estimator; the A/B
                                      # pins bq.rtgi_bounces per-Quality so it keeps teeth under
                                      # --no-rtgi); check-gpu's + check-vk V7's rung-signature
                                      # must-fires.
                                      # THE `rtgi-rr` GATE is the ladder's own correctness property
                                      # and the only thing in the suite that can see whether the 1/p
                                      # weight is RIGHT rather than merely present. SELF-ORACLING,
                                      # which is what makes it strong: a stochastic rung estimates
                                      # EXACTLY the deterministic rung above it, so the oracle is
                                      # another SHIPPING setting rather than a synthetic reference
                                      # that could drift. BOTH arms (0.5→1 and 1.5→2), because the
                                      # level-1 failure modes are invisible at level 0 — the
                                      # emissive-display double count (`gather_q`'s INHERITED flag:
                                      # at rung 2 the deterministic arm runs at depth 1 where `el`
                                      # is already None, so a bare el.is_none() would re-enable the
                                      # add while NEE is live) and a budget that failed to
                                      # decrement both need a budget left to spend. MEASURED
                                      # (deterministic — primary_seed is a pure function of
                                      # pixel/frame/sample, so these repeat exactly): signed
                                      # +0.0004 on BOTH arms; the FR_RTGI_NOWEIGHT teeth −0.0194
                                      # and −0.0056.
                                      # ITS GATE-DESIGN LESSON, which cost a wrong bound first: the
                                      # SIGNED mean is the verdict and `mean_abs` is a loose sanity
                                      # bound only. Both arms are 1-spp, so the absolute statistic
                                      # is dominated by per-pixel VARIANCE and is the same size as
                                      # the bias being hunted — it reads 0.0149 unbiased against
                                      # 0.0248 biased and CANNOT separate them, while the signed
                                      # form reads +0.0004 against −0.0194, fifty times apart. Bias
                                      # cancels nothing in a signed mean and everything in an
                                      # absolute one; pick the statistic from the failure you are
                                      # hunting (the --spp gate's own lesson from the other side).
                                      # The gate costs 448 CPU frames on EVERY --check (2 arms ×
                                      # 96 weighted + 96 naive + 32 oracle); whole suite 32 s.
                                      # NEGATIVE RADIANCE IS CORRECT ON A STOCHASTIC RUNG and
                                      # --check-dxr's `non-finite || negative` counter conflated it
                                      # with NaN: an unbiased estimator of a non-negative quantity
                                      # samples below zero exactly when G < tail·(1−p), measured
                                      # 131/12/2 of 1.44M at p = 0.25/0.5/0.75 while the radiance
                                      # mean stays flat at 0.044-0.053%. Clamping would reintroduce
                                      # the bias the ladder exists to avoid, so the counter is SPLIT
                                      # — non-finite always fails, negative is allowed only on a
                                      # fractional rung and bounded at 0.1% of samples.
                                      # FR_ABL=nogi is the cost probe (drop ray + bounce shade,
                                      # the norefl shape; in nosec; dual-homed CPU+GPU).
                                      # MEASURED (--spin path 1080p procedural, on vs off): CPU
                                      # 46.2 vs 42.0 ms (+10%, the budget controller absorbs it
                                      # as resolution); 4090 wavefront span 0.323 vs 0.277,
                                      # DXR 0.310 vs 0.256 (~+0.05 ms — under the FR_BALLAST
                                      # regime-pricing prediction); B70 wavefront 0.834 vs 0.713
                                      # (+0.121 ms, leaf +26%, still SIMD16 — the spill cliff
                                      # did NOT trip, so the documented two-kernel wavefront
                                      # fallback stays unbuilt). check.png RE-BASELINED
                                      # 2026-08-08 (armed default); check_gi.png byte-unchanged
                                      # (the fb.gi arm precedes the RTGI arm). Known-accepts:
                                      # 1-spp bounce noise under plain presentation (upscaler
                                      # toggled off); slight RR ghost risk on GI (no dedicated
                                      # guide); AMB_BUMP doesn't apply
                                      # on the RTGI arm (the bounce is drawn about n_g — the
                                      # fb.gi shape); the helmet cross-vendor two-device gate was
                                      # already red pre-RTGI (1902 hot ch vs 720 at --no-rtgi —
                                      # helmet is outside the documented two-device scene set)
                                      # and RTGI nudges it (2102) — pre-existing, not this
                                      # feature. LADDER known-accepts: rung 0.5 is dominated by
                                      # rung 1 (above); a stochastic rung's correction rides FSR's
                                      # un-denoised residual, so the deterministic rungs are the
                                      # recommendation for FSR4-RR sessions; rung 0.5 re-arms the
                                      # documented amb_irradiance-vs-sh_irradiance approximation
                                      # (prim.ao is REAL there, so the bridge's ao·amb term is back
                                      # — the same approximation every non-RTGI NRD session already
                                      # carries); ReBLUR's own anti-firefly WILL clamp 1/p spikes,
                                      # so --nrd-no-anti-firefly is the A/B when measuring (FRD is
                                      # already correct by argument — its GI-fold exempts the
                                      # diffuse lane for the K ≥ 1/p reason frd.rs states).
                                      # Follow-ons documented here: the FSR-RR indirect-diffuse
                                      # signal (shim desc + 12th plane + pack lane) if RDNA4
                                      # sessions show residual noise; STRATIFYING the roulette over
                                      # frames (a golden-ratio-rotated per-pixel sequence, the
                                      # clouds::dither_jk shape) instead of an independent
                                      # Bernoulli per frame — worth real noise at the ~10-30
                                      # effective frames a denoiser integrates; throughput-driven p
                                      # (a natural --rtgi-bounces auto); rungs above 2, which
                                      # generalize as written on the CPU and want an outer loop over
                                      # the (trace → shade_split → correct) block on the GPU.
                                      # Touch shade.rs's ambient tier / rtgi_gather / gather_q /
                                      # shade.hlsli's correction + amb_tail export / rtgi_corr_p /
                                      # the counter pair → run --check at ALL FIVE rungs (+ the
                                      # goldens byte-compared at 0 and 1), --check-gpu and
                                      # --check-dxr at all five, --check-gpu on san-miguel-lp at 0.5
                                      # and 1.5, cargo test, and the enclosure feel-test below.
                                      # THE HONEST QUALITY READ IS PERCENTILES, NOT MEANS (the
                                      # still-camera-darkening lesson): each real gather replaces an
                                      # over-bright approximation with the truth, so expect the
                                      # shadowed p10 to DROP and contrast to RISE as the ladder
                                      # climbs — enclosures read darker and better-structured, not
                                      # brighter. Open scenes barely move.
                                      # (The NEE-keep follow-on SHIPPED same-day — the XeSS
                                      # feel-test objected on schedule; see the EMISSIVE
                                      # paragraph above.)
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
cargo run --release -- --dual-gpu 4      # SPLIT THE FRAME ACROSS TWO ADAPTERS (src/gpu/dual.rs;
                                        # DEFAULT OFF, --no-dual-gpu spells it). N = the
                                        # SECONDARY's share of the 8 level-`--dual-gpu-depth`
                                        # tile ROWS, 1..=7 (bare --dual-gpu = 2); the primary
                                        # keeps the top rows. Eighths rather than a boolean
                                        # because the optimal share is NOT the compute-balanced
                                        # one and cannot be guessed: it minimises
                                        # max(T(1-s), r*T*s + s*K) over payload, link speed AND
                                        # tracer cost. `--dual-gpu-auto` hands it to
                                        # dual::Balancer (N is then the STARTING share) which
                                        # converges to that optimum and SAYS SO — a silent zero
                                        # is indistinguishable from a feature that never armed,
                                        # which is why the settle verdict prints.
                                        # THE SPLIT IS A CONTIGUOUS ROW BAND on both sides, and
                                        # that shape is what makes MIXED MODE work: the
                                        # wavefront expresses it as a `TileSplit` tile mask
                                        # (level_finish's ownership test + cs_compose's
                                        # split_owns_px), DXR as a shrunken DispatchRays grid
                                        # (`DxrGpu::set_band`; rw/rh stay full-screen in the CB
                                        # and `band_id` lifts every DispatchRaysIndex back to
                                        # absolute, so ray_dir, sample_pos, the sky-LOD lattice
                                        # index and the cloud dither hash are unchanged BY
                                        # CONSTRUCTION). `TileSplit::row_range` converts one to
                                        # the other and answers None — never a bounding box —
                                        # for any mask that is not a band.
                                        # EITHER DEVICE RUNS EITHER PIPELINE. The primary's arm
                                        # is the session's (--gpu/--dxr/SPACE); the SECONDARY's
                                        # follows ITS OWN adapter's vendor (dual::arm_for —
                                        # Intel -> wavefront, NVIDIA/AMD -> DXR, the
                                        # vendor_defaults discipline keyed off AdapterPick::
                                        # vendor, a FACT, not adapter::picked_vendor(), which
                                        # names the primary). `--dual-gpu-arm wave|dxr` forces
                                        # it (and ARMS --dual-gpu at its default if it wasn't
                                        # already — the --dual-gpu-auto rule: an arm forced on a
                                        # device that was never opened is a silent no-op); a
                                        # secondary without RT tier 1.0 degrades to the
                                        # wavefront loudly whatever is asked, which is
                                        # soundness rather than policy and is gated separately
                                        # from the vendor table for exactly that reason.
                                        # ONE SITE PER PIPELINE: GpuContext::record_split is the
                                        # schedule (secondary submit -> primary record ->
                                        # d3d.split_frame -> wait -> band out -> hop -> band in),
                                        # reached by record_trace's six wavefront presenters and
                                        # record_dxr_trace's six DXR ones, so the feature cannot
                                        # be half-wired. `split_frame` is what makes it
                                        # CONCURRENT rather than serial. ORDERING RULES, each of
                                        # which has shipped wrong once: arm the SECONDARY first
                                        # and narrow the primary only once it took (a refused
                                        # primary beside an armed secondary renders the band
                                        # twice); both or neither, restoring to TileSplit::ALL,
                                        # which maps to the one region neither arm can refuse;
                                        # freeze the split while `p.accumulate && p.frame > 0`
                                        # (accum is per-device — a row that moves mid-
                                        # accumulation holds fewer samples than record_resolve
                                        # divides by: a dark band, darker the later the move)
                                        # and NEVER on p.replay, which would freeze a parked
                                        # session forever; every error DEFERRED past the wait
                                        # (returning with the secondary's fence in flight leaves
                                        # the next submit resetting an executing allocator,
                                        # which removes the adapter); and `record_feed`/
                                        # `record_resolve` are FULL-SCREEN, so the band must
                                        # land on the list before them.
                                        # WHAT CROSSES: accum 12 B/px always, + GBufCore 16 and
                                        # GBufExt 72 when the primary's feed reads them
                                        # (dual::fed_strides — 28 B/px for XeSS/FSR3, 100 for
                                        # RR/FSR4-RR, paid EVERY frame on an interactive path
                                        # against a capture's once per OUTPUT frame). `pack_full`
                                        # is the AND of both sides: copying a band into a
                                        # stride-sized dummy is an out-of-bounds
                                        # CopyBufferRegion, which does NOT fault — the list
                                        # fails to Close and the allocator is permanently broken.
                                        # AND THE ONE-SIDED CASE DENIES THE FRAME rather than
                                        # just narrowing the payload: a primary that HAS a pack
                                        # beside a secondary that has none would still SPLIT, so
                                        # the primary never writes gbuf/gbuf_ext for the band
                                        # while record_feed reads them FULL-SCREEN — stale
                                        # MVs/depth/albedo handed to the upscaler as if current,
                                        # the quieter half of the same hole (the reverse pairing
                                        # is fine: a primary with no pack has no feed to read
                                        # one). Same answer as the mixed-arm stand-down, its own
                                        # once-per-episode line.
                                        # The secondary has no feed of its own, so
                                        # force_gbuf_ext/force_fsr_sig are mirrored onto it per
                                        # frame (without the latter an FSR4-RR session gets zeros
                                        # for dd/ds/ao/ind_s in the secondary's rows — a black
                                        # band, present only under --fsr4).
                                        # MIXED ARMS STAND DOWN ON fb FRAMES (dual::mixed_denies):
                                        # DXR has no hemisphere stage at all, so a DXR partner
                                        # would draw its band with the bounce tier absent — a
                                        # flatter-lit band appearing the instant H is pressed.
                                        # --cinematic additionally forces a wavefront secondary
                                        # for GI shots (same reason) and for overlay shots (the
                                        # band carries `info`, and DXR's info is the CONSTANT
                                        # pack_info(0, KIND_LEAF), not a quadtree depth).
                                        # EVERY DENIED FRAME DEMOTES ITS TICK (Balancer::
                                        # mark_unsplit, called from record_split's ONE
                                        # single-GPU exit — a deny, a declined region, or a
                                        # zero share all land there): `tick` has already
                                        # committed kind=Split, and observe's Split arm would
                                        # then feed ctl.update the whole frame as prim_ms and
                                        # ZERO as sec_ms, read err=1.0, and step the share UP on
                                        # a frame the secondary sat out — so held H under a
                                        # mixed pair ratchets to the ceiling while dual_ms fills
                                        # with single-GPU times and split_loses never fires.
                                        # Tick::Idle is the state observe already ignores. rows
                                        # and held are deliberately LEFT ALONE: zeroing rows
                                        # strands a PINNED balancer at zero forever (nothing
                                        # outside the auto arms restores it) and zeroing held
                                        # reaches the same place through the freeze path. Only
                                        # the FRAME is idle, not the decision. --spin and
                                        # --cinematic never had this: they pass literal zeros to
                                        # observe, which ShareCtl::update early-returns on.
                                        # MEASURED on this box (4090 primary + Arc Pro B70
                                        # secondary, the second x16-length slot electrically x4):
                                        # IT LOSES, and the transfer is the whole cost.
                                        # --spin path --dxr 1080p: single 0.37 ms/frame, pinned
                                        # 4/8 11.61 (band-out 2.57 + hop 0.51 + band-in 3.81 =
                                        # 6.89 of it, prim-wait 0.00 at every share — the primary
                                        # always finishes first and waits on the wire),
                                        # --dual-gpu-auto 1.01 and settles at 0 of 8 with its
                                        # verdict. The wavefront arm reads the same shape. THE
                                        # ONE CONFIGURATION THAT WINS is a --cinematic GI capture
                                        # (2/8, 4-8%), where the band crosses once per OUTPUT
                                        # frame and amortises over shot.samples. A phase table
                                        # summing to ~0.01 ms with every share within noise of
                                        # the baseline is the a28953c signature of a run that
                                        # measured single-GPU and reported it as a free split.
                                        # --spin-warmup: derives from picked_vendor() = the
                                        # PRIMARY, so a NVIDIA-primary run defaults to 20 frames
                                        # while an Intel SECONDARY needs 1600 — pass it
                                        # explicitly or every number is the Arc async-compile
                                        # fallback (recorded in abf7b28; no code guards it).
                                        # Gates: dual::self_test + gfx::frame::split_self_test in
                                        # --check (payload/ring arithmetic, convergence to the
                                        # analytic optimum s* = T/(T(1+r)+K), the interfering-box
                                        # family, the PINNED-balancer sweep, the freeze family,
                                        # the vendor-arm policy table with its caps invariant
                                        # stated SEPARATELY, the band-vs-owns_px agreement
                                        # sweep — the last is the one nothing checked and its
                                        # teeth are that an owns_px midpoint flip fires 1920 px
                                        # at 1080p while split_self_test stays green);
                                        # --check-dxr's band sweep (11 complementary pairs over
                                        # depth 1..=3, dirtied by a YAW because a dolly leaves
                                        # sky bit-identical and left 83 of 600 rows blind,
                                        # accum + tbuf scored both directions, the partition
                                        # proof, the mask-vs-band cross-check, dg.band() vs what
                                        # was asked, and per-ROW anti-vacuity); --check-gpu's
                                        # wavefront split + two-device families; and a
                                        # cargo-test source gate that every DispatchRaysIndex()
                                        # in dxr.hlsl is band_id-lifted (comments MUST be
                                        # stripped first — two of the seven occurrences are
                                        # prose); and --check-dxr's TWO-DEVICE gate, which runs
                                        # BOTH arms (DXR+DXR, skipped loudly when the secondary
                                        # lacks RT tier 1.0, and DXR+wavefront, which is what
                                        # arm_for actually picks here) over every share at
                                        # depth 1..=2 and does the REAL transfer.
                                        # THAT GATE CLOSES THE HOLE THE WAVEFRONT TWO-DEVICE
                                        # FAMILY DOCUMENTS IN ITSELF ("sensitive to a MISPLACED
                                        # copy, not to an absent one"): rendering A's reference
                                        # just before the split leaves A's out-of-band rows
                                        # stale-CORRECT, so a transfer that never happened
                                        # compares clean. Here A's plane is dirtied with a
                                        # different pose first and read TWICE — PRE, over the
                                        # BAND ROWS ONLY (whole-frame would dilute by the share
                                        # and make the teeth a function of the split), must be
                                        # visibly wrong, ASSERTED at >= 0.1; TRANSFERRED,
                                        # whole-frame, must be right. MEASURED: clean 6e-7..2e-6
                                        # against a 1e-3 limit, absent copy 1.47e-1 (147x over,
                                        # 343398 hot channels vs 720), one-row-misplaced copy
                                        # 4.36e-2 — matching the wavefront gate's own recorded
                                        # 4.3e-2 for that revert. Structure (the pair tiles the
                                        # frame exactly, neither device empty) stays EXACT on
                                        # every pairing; radiance relaxes when the arms or the
                                        # vendors differ, since either means two intersectors on
                                        # one HLSL source. Both adapter orientations pass
                                        # (--prefer-intel flips which card is A)
```
