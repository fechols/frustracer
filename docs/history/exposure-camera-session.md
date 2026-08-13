# Exposure, camera feel, and session levers

Temporal-reuse A/B levers, bloom, auto-exposure and its spike guard, `--move-ease` keyboard flight, `--gpu-debug` and the crash handler. `--move-ease`/`--no-move-ease` sit here for contiguity.

Extracted verbatim from `CLAUDE_Historical.md`, which keeps a stub pointing here. Nothing in this file was rewritten.

```
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
                                      # by tens of percent). Never widen those limits to pass a port.
                                      # M13 STOPS AT THE PYRAMID'S OUTPUT, and the last step — the
                                      # tonemap PS sampling that pyramid and blending it into the
                                      # frame — was covered by NOTHING until M13b (2026-08-11),
                                      # which is how it carried an off-by-half-texel from the
                                      # beginning: `uv = (pos.xy + 0.5)/dims` where SV_Position
                                      # ALREADY points at the pixel centre (the `src.Load(int3(
                                      # pos.xy, 0))` two lines up relies on exactly that, since Load
                                      # truncates the .5 away). The glare is half-res, so the error
                                      # was half a full-res texel = a QUARTER of a glare texel, and
                                      # it DISAGREED with the CPU twin (bloom.rs's textbook
                                      # `u = (x + 0.5)*gw/w - 0.5` lands on 7.75 where the shader
                                      # landed on 8.0). Fixed to `pos.xy / dims`. Visually tiny —
                                      # it is a quarter texel on a deliberate blur — which is
                                      # exactly why nothing noticed for so long; it surfaced only
                                      # because it DEFEATED a test being written for something else
                                      # (the M12b pre-glare plant relied on that sample landing
                                      # squarely on a bright texel, and the slip split it across
                                      # two). M13b's assertion is SYMMETRY and needs no CPU tent
                                      # mirrored: one bright texel in the glare must paint a halo
                                      # CENTRED on it, so the pixels either side come back equal.
                                      # A half-texel slip does not erase the halo, it SPLITS it
                                      # across two pixels — so the gate asserts the peak is a peak
                                      # AND both axes are symmetric, and reports all five numbers in
                                      # either failure (a bare anti-vacuity message there blames the
                                      # test for what the shader did). Measured: correct reads peak
                                      # 0.502 / sides 0.376 x4; the old code reads peak 0.396 | left
                                      # 0.396 right 0.247 | up 0.396 down 0.247 — teeth verified by
                                      # planting the original expression back (exit 1, both arms)
cargo run --release -- --no-auto-exposure  # KILL the adaptive aperture (src/autoexp.rs +
                                      # gpu/autoexp.rs + shaders/autoexp.hlsl — DEFAULT ON
                                      # since 2026-08-10, the user's call; THREE MOVES: ON for
                                      # one day, OFF on 2026-08-08 (with RTGI on by default the
                                      # enclosures that motivated it light themselves, so the
                                      # aperture was holding near 1.0 and paying for nothing),
                                      # ON again now — AFTER the same-day EV_MAX rein-in to
                                      # +2.0, which is what makes an armed session a bounded 4x
                                      # rather than the 32x that washed genuinely dark scenes
                                      # into dusk. --no-auto-exposure holds exposure at exactly
                                      # 1.0 plus any --exposure-bias (the pre-feature look),
                                      # --auto-exposure spells the default, and the compiled
                                      # default is DUPLICATED in autoexp.rs's ENABLED
                                      # initializer + cli.rs defaults() + settings.rs's menu-row
                                      # Toggle default — flip all three in lockstep; only a
                                      # DEPARTURE prints a lever line).
                                      # Armed, auto-exposure is a DISPLAY-STAGE controller in the bloom
                                      # discipline: meters read the PRE-glare, PRE-exposure
                                      # linear tonemap source (GPU-tonemapped arms: a two-level
                                      # fxc cs_5_0 mean-log2-luminance reduction recorded in
                                      # fullscreen_to_backbuffer, read back on the gputime
                                      # frame ring — FRAMES_IN_FLIGHT frames of latency, never a
                                      # stall; CPU-presented arms: autoexp::meter_accum/meter_hdr
                                      # inside CpuPresent::resolve[_hdr], same-frame), and
                                      # session()'s ONE controller tick per frame eases a clamped
                                      # EV (EV_MIN −2 .. EV_MAX +2, deadband 0.15 EV, tau 1 s;
                                      # ask = TARGET_LOG2 − measured, mid-grey log2 0.18 —
                                      # EV_MAX went 3 → 5 → 3 → 2 across 2026-08-10: +5 is a 32×
                                      # aperture that washes a genuinely dark scene into dusk, and
                                      # PARKING ON THE CLAMP IS THE DESIGN — night is meant to read
                                      # dark, with --exposure-bias the per-session escape. The last
                                      # move answers a FIREFLY report (XeSS+NRD, dark emissive
                                      # scenes) and is a PARTIAL lever by construction: the aperture
                                      # decides where the frame sits on the tonemap, and 32× sat in
                                      # the SATURATING region where an outlier and its surround
                                      # compress together (noise laundered by the rolloff) while
                                      # lower apertures sit in the near-LINEAR region and preserve
                                      # contrast — so narrowing damps MODERATE outliers and makes
                                      # SATURATING ones MORE salient (the dot pins white while the
                                      # background darkens). The fireflies are 1-spp RTGI-bounce
                                      # variance on small emitters; the source-level fix is
                                      # upscaler_defaults' dn_fold early-out, which leaves an
                                      # XeSS+NRD session without the deterministic cluster NEE) into
                                      # tone::ToneParams::exposure — a linear PRE-CURVE multiply
                                      # that the tonemap PS and tone::shape (and the CPU
                                      # present_px family, which threads it) BRANCH around at
                                      # exactly 1.0, so the off arm and EVERY headless path are
                                      # bit-identical BY CONSTRUCTION: the controller lives in
                                      # the interactive frame loop only — --check*/--spin/
                                      # --cinematic never adapt (cinematic keeps its own
                                      # -exposure, applied to LINEAR radiance upstream — the
                                      # one-write-site rule, never through the curve). Open-loop
                                      # by design: the meter reads pre-exposure input and
                                      # exposure exists only downstream, so there is no servo
                                      # feedback mode to oscillate. It exists because the
                                      # tonemap anchors at a fixed paper white while enclosures
                                      # (San Miguel's patio, Sponza's atrium) sit 2-3 stops
                                      # under a lit exterior — the interactive counterpart of
                                      # --cinematic's -exposure. The eased EV survives resize/
                                      # F11 (Persist::autoexp_ev — a rebuild must not flash the
                                      # frame); a display move retunes the CURVE, never the
                                      # aperture (refresh_display preserves exposure); P
                                      # screenshots carry the live exposure (P captures what the
                                      # screen shows — both readback and CPU re-resolve paths);
                                      # the HUD/menu overlay deliberately ignores it (the UI is
                                      # authored in display space — exposing it would dim the
                                      # menu with the scene). Menu holds stay live (present_again
                                      # re-records the meter). FR_AEXP_TRACE=1 prints
                                      # measured/ev/scale ~1/s — the calibration probe for
                                      # TARGET_LOG2, the one look knob (the EL_BOOST class).
                                      # Settings rows: Effects/autoexp + exposure_bias (both
                                      # LIVE, no reset — the bloom shape; a menu edit persists,
                                      # the CLI flags seed). Gates: autoexp::self_test in
                                      # --check (structural-off exactly-1.0 for any EV, exact EV
                                      # anchors, bias-composes-with-auto-off, convergence with
                                      # never-overshoot, clamp parking, deadband bitwise hold,
                                      # dt clamp, determinism, the mid-grey meter identity +
                                      # accum/hdr agreement — save/restore levers, so the
                                      # default-OFF --check still proves the on arm) and
                                      # tone::self_test's exposure family (pre-curve-scaling
                                      # bit-equality vs scaled input, 1.0 bitwise inert);
                                      # check.png untouched (headless exposure IS 1.0, and the
                                      # branch makes that structural, not fp luck)
cargo run --release -- --exposure-bias 1.5  # manual aperture offset in STOPS (-8..=8 — the
                                      # cinematic -exposure range; default 0): composes
                                      # additively with the controller's EV and still applies
                                      # under --no-auto-exposure (the manual exposure lever).
                                      # Interactive-only by construction — the bias reaches the
                                      # screen through the session controller's set_exposure,
                                      # which headless paths never tick
cargo run --release -- --move-ease 0.18  # KEYBOARD FLIGHT EASE-IN/EASE-OUT (seconds, 0.0..=1.0,
                                      # DEFAULT 0.18 = camera::MOVE_EASE_S, 2026-08-12).
                                      # `--no-move-ease` is 0 = the pre-ease HARD STEP, and the
                                      # integrator BRANCHES around the eased arm entirely, so
                                      # the off arm is textually and bitwise today's code (the
                                      # "the skip is a BRANCH, never a computed 1.0" discipline).
                                      # Keyboard flight used to be a hard step function of key
                                      # state — WASD/QE/arrows drove velocity 0 -> full and full
                                      # -> 0 in one 2 ms tick — while the analog stick was
                                      # already smooth (deflection IS the throttle, shaped by
                                      # stick()'s magnitude^STICK_CURVE), so only the keyboard
                                      # read as abrupt. THE CONTROLLER IS NOT TOUCHED: left
                                      # stick / triggers / the QA `drive` verb keep their verbatim
                                      # analog math, which is what keeps every recorded --frd-lab
                                      # / frqa strafe metric comparable.
                                      # INTEGRATED IN THE 500 Hz FLYCAM THREAD with the measured
                                      # dt — the same wall-clock discipline displacement, the TOD
                                      # scrub and the Ctrl/Shift slow-modifier ramp already use.
                                      # That is what makes the inertia DETERMINISTIC: total
                                      # displacement stays a pure function of how long a key was
                                      # held, independent of frame rate, timer jitter, or a main
                                      # thread blocked in a 100 ms trace.
                                      # ONE eased vector, CAMERA-RELATIVE (x = right, y = up,
                                      # z = forward — the analog stick's own basis, which is what
                                      # lets the ramp advance BEFORE the pose mutex and keeps the
                                      # idle early-out a pure input test). A vector rather than a
                                      # scalar plus a latched direction is what makes a direction
                                      # change SLEW: W->S passes through the origin (a momentary
                                      # stop — the inertia) instead of flipping at full speed, and
                                      # a full reversal takes 2x the ease; W->D arcs. A coast
                                      # follows the camera, so swinging the mouse mid-coast curves
                                      # the path instead of skating.
                                      # THE DIRECTION IS STILL NORMALIZED IN WORLD SPACE, exactly
                                      # as the pre-ease code did, and that is not incidental: `f`
                                      # carries a Y component when pitched, so the camera basis is
                                      # NOT orthonormal and normalizing the coefficients instead
                                      # would run a forward+vertical combination (W+E while
                                      # pitched) up to ~31% fast. The ease therefore moves only
                                      # the MAGNITUDE, through camera::move_scale = smoothstep of
                                      # the ramp length — exactly 0.0 at rest and exactly 1.0 at
                                      # full deflection, so full speed is still exactly
                                      # diag*0.1875/s and the ease is invisible at both ends.
                                      # THE ONE INVARIANT: the ramp must reach EXACT rest.
                                      # flycam.rs's idle early-out is what keeps an idle session's
                                      # shared state bit-untouched, which is what makes
                                      # `moved = cam != prev_snap` a correct signal and therefore
                                      # what lets plain accumulation, structure replay and every
                                      # upscaler history converge on a still camera. So
                                      # camera::move_ease is a fixed-length LINEAR slew that SNAPS
                                      # to the target when within one step (the auto-TOD
                                      # snap-within-one-step idiom, and the `ramp` closure's hard
                                      # saturation) — never an exponential smoother, which is
                                      # asymptotic and would re-write cam.pos forever after key
                                      # release, killing convergence on every idle session with no
                                      # error anywhere. |ramp| <= 1 falls out by construction (a
                                      # point stepped along the segment toward a unit target never
                                      # leaves the unit ball), so no clamp exists to audit.
                                      # The early-out additionally tests the ramp AFTER this
                                      # tick's update — skipping a tick would freeze a coast
                                      # mid-glide and the camera would resume from it on the next
                                      # key press. The pause/focus gate PARKS the ramp beside the
                                      # drag (a gated span presents no frames, so resuming must
                                      # not dump a coast into it: closing the menu with W held
                                      # re-eases from rest), and FlyCam::set raises a
                                      # `motion_reset` the integrator consumes, so a teleport
                                      # ARRIVES STOPPED — without it a scripted `tp` -> `sync` ->
                                      # `screenshot` would drift through the settle.
                                      # `speed` already carries dt and the eased Ctrl/Shift slow
                                      # factor, so the two eases COMPOSE and the fine-scrub chords
                                      # are unchanged. The audio wind improves for free (its
                                      # speed atomic now ramps instead of stepping). Zero rng
                                      # draws; interactive-only by construction — headless paths
                                      # never build a FlyCam, so --check*/--spin/--cinematic
                                      # cannot reach it and check.png/check_gi.png are
                                      # byte-identical (verified, not assumed).
                                      # Settings row: Advanced/move_ease (Live — the row writes
                                      # through FlyCam::set_move_ease, the MenuFx::SetTod
                                      # precedent; no frame or history reset, this is input
                                      # response rather than shading). The default is TRIPLICATED
                                      # in camera::MOVE_EASE_S, cli::defaults() and that row's
                                      # StepF default — flip all three in lockstep; cli::self_test
                                      # pins the first two against each other and
                                      # settings::self_test pins the third (plus the shared
                                      # 0.0..=1.0 range).
                                      # Gates: camera::self_test in --check (DLL-free and
                                      # non-cfg'd, unlike flycam itself, so it runs on every
                                      # platform) — EXACT REST from anywhere at any tick rate with
                                      # TEETH (an exponential smoother run through the same probe
                                      # must provably FAIL, else the arm passes on anything that
                                      # merely gets small — exercised: it reads
                                      # v = 0.3577418 and exits 1), exact saturation, the unit
                                      # ball over 20k randomized target switches, the smoothstep
                                      # shape + monotonicity + exact endpoints, dt-INVARIANCE (the
                                      # determinism claim: the ease takes MOVE_EASE_S of wall
                                      # clock in BOTH directions at 30/60/240/500 Hz to within one
                                      # tick, and the displacement integral agrees across rates —
                                      # bounded rather than exact, since the sum is a left-Riemann
                                      # quadrature of a smoothstep), the off arm incl. NaN, and
                                      # the REVERSAL anti-vacuity (a W->S flip must pass through a
                                      # stop and take ~2x the ease — without it the whole gate
                                      # would pass on a no-op).
                                      # FEEL-TESTED AND KEPT at 0.18 (2026-08-12, the user's call:
                                      # "works amazingly well"). The verdict named a payoff the
                                      # feature was not built for and which is worth stating,
                                      # because it is the SHAPE and not the smoothing: FINE
                                      # POSITIONING from a digital key. Under the hard step,
                                      # displacement from a tap is LINEAR in hold time, so the
                                      # shortest human tap (~50 ms) always moved 0.05x full speed
                                      # and there was no way to ask for less. Under the ease a tap
                                      # shorter than the ramp never reaches full speed at all — the
                                      # ramp only gets to tau/T and smoothstep is QUADRATIC near
                                      # zero, so the integral makes displacement CUBIC in tau
                                      # (~2*tau^3/T^2 counting the symmetric decay). The
                                      # attenuation vs the hard step is therefore 2*tau^2/T^2:
                                      # 1.6x at a 100 ms tap, 6.5x at 50 ms, 40x at 20 ms (derived
                                      # from the shipped functions, not measured). That is the same
                                      # trick a stick's magnitude^STICK_CURVE response plays, so
                                      # the keyboard now has an analog-feeling low end — and it is
                                      # the reason to be suspicious of "just make T smaller if it
                                      # feels laggy": T shrinks the fine-control range as T^2 while
                                      # only shrinking the lag as T
cargo run --release -- --no-move-ease  # the hard-step A/B arm, spelled explicitly (later flags
                                      # win: `--no-move-ease --move-ease 0.3` = 0.3)
cargo run --release -- --no-autoexp-spike-guard  # A/B lever: let the aperture boost NOISE SPIKES
                                      # along with the scene. DEFAULT ON (2026-08-11) — the
                                      # per-pixel half of auto-exposure, and the answer to
                                      # "I still see fireflies with NRD + auto-exposure".
                                      # THE MECHANISM, which bounds what it can buy: the
                                      # tonemap SATURATES, so a spike already deep in the
                                      # rolloff (radiance >> paper white) pins white at 1x and
                                      # at 4x alike and the aperture never made it worse. The
                                      # population auto-exposure genuinely CREATES is the
                                      # near-LINEAR one — a dot at ~0.3 over a ~0.02 surround,
                                      # where a 4x aperture roughly DOUBLES the absolute
                                      # display gap. That is what this removes, and nothing
                                      # else; EV_MAX's own note (autoexp.rs:63-77) is still
                                      # right that the fireflies' SOURCE is upstream
                                      # (1-spp RTGI-bounce variance on small emitters, left
                                      # without deterministic cluster NEE by the
                                      # upscaler_defaults narrowing — `--emissive-lights`
                                      # removes them at the source, measured +61% pool
                                      # brightness; this is the cosmetic complement, not a
                                      # rival).
                                      # THE INVARIANT, and the whole reason it is safe to arm
                                      # by default: auto-exposure may never make an outlier
                                      # BRIGHTER than the auto-exposure-off image would have.
                                      # The exemption only ever relaxes toward 1.0 and never
                                      # past it (`1 + (E-1)(1-w)`, both factors non-negative ⇒
                                      # >= 1.0 by CONSTRUCTION, no clamp to audit), so the
                                      # worst case of a false positive is "that glint looks
                                      # like it did before auto-exposure" — a look that already
                                      # ships. Contrast nrd::oracle::rclamp_scale, whose
                                      # correction can pull a real emissive texel BELOW truth
                                      # and which is therefore default-OFF.
                                      # THE RULE (autoexp::guard_scale, the ONE source of
                                      # truth; tonemap.hlsl is its twin): over an 8-tap DONUT
                                      # at radius R, cap = max(K_MEAN*ring_mean,
                                      # K_MAX*ring_max) — rclamp's conjunction, same rationale:
                                      # a lone stochastic dot has mean ~= max so the mean term
                                      # binds, while a resolved feature (a lamp, the sun's
                                      # limb, a fireflies.rs GLOW SPLAT) has a bright
                                      # neighbour, giving max >> mean, and is shielded; both
                                      # reductions come out of one loop so the discrimination
                                      # is free. Centre EXCLUDED (frd's lesson: with it in, a
                                      # lone outlier dominates its own cap). Out-of-range taps
                                      # EXCLUDED, never replicated, and under GUARD_MIN_RING=4
                                      # in-range taps the pixel is left unguarded — so the
                                      # frame border is un-guarded rather than guarded against
                                      # copies of itself. w = smoothstep over luma/cap in
                                      # [1, RAMP], scaled by --autoexp-spike-strength: a SMOOTH
                                      # ramp, not a binary pin, so a pixel near the threshold
                                      # cannot pop between two brightnesses frame to frame.
                                      # THE SATURATED END OF THE RAMP IS TESTED DIRECTLY
                                      # (`luma >= cap * RAMP`) rather than reached through
                                      # luma/cap, and that is not a micro-optimization: an
                                      # exactly-black ring makes cap exactly 0, which is a lit
                                      # dot on absolute black — the MOST extreme outlier the
                                      # guard can ever see — and the obvious
                                      # `if !(cap > 0) { return exposure }` guard hands precisely
                                      # that pixel the full boost. It shipped that way for a day.
                                      # Testing the product exempts it AND keeps x/0 out of the
                                      # expression, so neither compiler has to agree about inf
                                      # (the division is only ever evaluated where cap > 0). NaN
                                      # in either reduction still takes the off arm, one line
                                      # earlier, via `!(luma > cap)`. Gated by self_test arm 13b.
                                      # A DONUT AT RADIUS 4, not the adjacent 3x3 ring both
                                      # sibling clamps use, and the constants are MEASURED
                                      # rather than inherited — deliberately NOT rclamp/frd's
                                      # K_MEAN = 8.0 ("the two clamps speak one value"), because
                                      # those run on raw 1-spp radiance over an adjacent ring
                                      # while this runs on DISPLAY luma AFTER a temporal
                                      # upscaler has already integrated the dot toward its
                                      # surround. MEASURED (FR_AEXP_GUARD_TRACE=1, bistro
                                      # Exterior --tod 22, XeSS+NRD 1080p, the recorded lamp
                                      # poses): luma/ring_mean p99.9 is only 2.39 (terrace) /
                                      # 2.92 (street) — i.e. at K = 8 this would have fired on
                                      # NOTHING, the "it gated nothing" failure RCLAMP_K_HARD
                                      # records, and a guard that cannot fire cannot be gated.
                                      # The radius is the other measured half: r 2 -> 4 takes
                                      # fired px 84 -> 455 (terrace) and 255 -> 1287 (street),
                                      # and the street pose's max luma/ring_max goes 43.6 ->
                                      # 919.1 — the tell that at radius 2 the ring is still
                                      # INSIDE the spike (so the K_MAX shield protects the very
                                      # thing being hunted) and at radius 4 it clears the blob.
                                      # Still only 0.02-0.06% of frame.
                                      # DETECTION READS PRE-GLARE (the ring taps AND the centre
                                      # luma come off `src`, not the post-glare `c`): bloom's
                                      # whole job is spreading an outlier into its
                                      # neighbourhood, so measuring after it would let a
                                      # firefly's own halo raise the ring mean and shield it.
                                      # Costs nothing — the taps are `src` loads either way.
                                      # OFF IS STRUCTURAL: lever off / strength 0 / exposure
                                      # <= 1 / nv < MIN_RING each BRANCH around the block and
                                      # return the exposure BITWISE (never a computed
                                      # `lerp(E,E,0)`, never a `* 1.0` — frd_temporal.hlsl's
                                      # "the skip is a BRANCH"). exposure <= 1.0 is an off arm
                                      # for a REASON, not as thrift: when the aperture is
                                      # stopping DOWN, exempting an outlier would make it
                                      # brighter relative to its surround, the exact inverse of
                                      # the intent. And headless paths never tick the
                                      # controller, so E is exactly 1.0 there and the guard is
                                      # structurally UNREACHABLE in --check*/--spin/--cinematic
                                      # (check.png + check_gi.png verified byte-identical
                                      # across the whole feature, including after all six CPU
                                      # resolvers were rewired).
                                      # BOTH present families carry it: the GPU tonemap PS (8
                                      # extra point loads, behind a group-UNIFORM cbuffer
                                      # branch, so a disarmed frame executes the pre-feature
                                      # instruction stream) and the six CPU resolvers, which
                                      # take a per-pixel exposure PLANE computed once per frame
                                      # from the pre-glare source and handed to the existing
                                      # `exposure` parameter — present_px* and tone::shape are
                                      # UNCHANGED. The CPU half is not optional: the
                                      # P-screenshot path for a GPU session is a CPU re-resolve
                                      # of an HDR readback (read_hdr_output), so without it
                                      # captures would not match the screen — and QA-socket
                                      # screenshots are the instrument this feature is measured
                                      # with.
                                      # NO LDS, no group-shape pin: the GPU site is a PIXEL
                                      # shader, so none of cs_nrd_out's apparatus (groupshared
                                      # halo, barriers hoisted above early returns,
                                      # nrd_out_group_shape_is_derived) applies.
                                      # NAMING: nothing here is spelled "firefly" on purpose —
                                      # `--fireflies`/`--no-fireflies` are the glowing INSECTS
                                      # (src/fireflies.rs) and a user reaching for the noise
                                      # lever must not kill the bugs. Their glow splats are
                                      # Gaussian and several px wide, so the K_MAX arm is what
                                      # shields them.
                                      # WHERE the aperture is spent is itself a lever —
                                      # `--autoexp-mode`, below — and since `lights` became the
                                      # DEFAULT this guard is INERT unless a session passes
                                      # `--autoexp-mode tonemap`. It is display-stage BY NATURE (it
                                      # relaxes a per-PIXEL display exposure, which the lights arm
                                      # holds at exactly 1.0), so this is inherent rather than a
                                      # wiring gap: keep that in mind before reading a firefly
                                      # report as evidence about the guard.
                                      # Levers: --autoexp-spike-guard spells the default;
                                      # --autoexp-spike-strength K (0..=1, default 1 = the
                                      # outlier is presented exactly as if auto-exposure were
                                      # off); FR_AEXP_GUARD_K=<mean>,<max> and
                                      # FR_AEXP_GUARD_R=<n> are the sweep probes (loud on
                                      # departure, loud+default on an illegal value — the
                                      # FR_LEAF rule) and ride INJECTED DEFINES
                                      # (autoexp::guard_defs, the spp_defs idiom) so they
                                      # provably REACH the shader; FR_AEXP_GUARD_TRACE=1 dumps
                                      # the luma/ring_mean + luma/ring_max histograms the K
                                      # constants are meant to be PICKED from (deliberately NOT
                                      # a luma/cap histogram, which would be circular — the cap
                                      # is a function of the very constants being chosen).
                                      # Reached in a GPU session through the screenshot path,
                                      # so the workflow is the lever plus a `frqa screenshot`
                                      # per pose. Settings rows: Effects/autoexp_spike_guard
                                      # (Toggle, LIVE, default true — the default-true fact is
                                      # duplicated in autoexp.rs's GUARD static, cli.rs's
                                      # defaults() and here, flip all three in lockstep) +
                                      # autoexp_spike_strength (StepF 0..1 by 0.1).
                                      # Gates: autoexp::self_test arms 10-14 in --check (the
                                      # bitwise off arms incl. NaN, the safety invariant swept,
                                      # ramp monotonicity + both endpoints, the K_MAX shield
                                      # with TEETH BOTH WAYS — a lone spike must be exempted
                                      # AND the same pixel with one bright neighbour must NOT
                                      # be, since a guard that never fires and one that fires
                                      # everywhere both pass a one-sided test — and
                                      # guard_plane's None fast paths / determinism / unguarded
                                      # border); --check-gpu M12b drives the REAL PSO over a
                                      # synthetic frame carrying a lone spike, a PAIR exactly
                                      # the ring radius apart, and flat field, scoring the
                                      # shader against autoexp::guard_plane per pixel, plus
                                      # strength-0 inertness, the safety invariant ON THE IMAGE
                                      # (no channel brighter than the boosted frame, none
                                      # darker than the unboosted one), and strength NESTING.
                                      # Its fixture DERIVES the pair separation from the live
                                      # radius, which makes it a probe-REACH test too: it
                                      # passes at FR_AEXP_GUARD_R=1|2|3, and a shader stuck at
                                      # the compiled default would ring at the wrong radius and
                                      # blow the per-pixel arm. Teeth verified live:
                                      # FR_AEXP_GUARD_K=4,0.0001 collapses the shield and M12b
                                      # FAILS with the paired spike at e 1.454.
                                      # M12b'S PRE-GLARE ARM is the one that gates the
                                      # detection-reads-`src` decision, and it exists because
                                      # every OTHER arm runs with bloom off, where `c0 == c`
                                      # BITWISE — so a shader that measured the ring after the
                                      # halo passes all of them. It renders two bloom-ON arms
                                      # (strength 1, tent step == the ring radius) and RECOVERS
                                      # the chosen per-pixel exposure by inverting the SDR curve
                                      # (x = -ln(1 - v^2.2)) across them: both present the same
                                      # post-glare colour, one at e_px and one at e_hi, so their
                                      # ratio IS e_px/e_hi and NO tent has to be mirrored on the
                                      # CPU (that is M13's job). The post-glare HYPOTHETICAL is
                                      # then run through the shipped guard_plane on the recovered
                                      # image, so the gate reports which of the two readings the
                                      # shader actually took: measured `recovered e 1.01 tracks
                                      # the pre-glare 1.00, not the post-glare 4.00`. Teeth
                                      # verified live by planting the post-glare read (both the
                                      # centre AND per-tap ring — a centre-only plant does NOT
                                      # fail, since 0.265 still clears a pre-glare cap of 0.08):
                                      # exit 1 with the recovered exposure landing exactly on
                                      # 4.00. It also carries its own VACUITY note, which fired
                                      # on the first run at bloom strength 0.6 — the halo raised
                                      # the ring but not enough to flip the verdict (gap 0.18),
                                      # so the constants are 1.0/ring-radius by measurement.
                                      # COST, measured (4090, 1080p, aperture verified BOOSTING
                                      # in both arms via FR_AEXP_TRACE — a null result at
                                      # exposure <= 1 gates nothing, the block being structurally
                                      # unreachable there): GPU present +0.031 ms/frame (+0.4%,
                                      # bistro --tod 22 XeSS+NRD parked at 3.61x aperture, 140.36
                                      # vs 140.98 fps median) — INSIDE the run-to-run band; CPU
                                      # present +0.8/+2.3 ms on a ~59 ms frame (~2.5%, procedural
                                      # `--cpu --no-upscale --exposure-bias 2`, two interleaved
                                      # reps), which is the arm that materializes two planes and
                                      # is why the guard is a prepass there rather than 8 taps
                                      # inlined into resolve's atomic loop. Do NOT re-measure the
                                      # CPU arm on a heavy scene — CPU bistro traces at ~1000
                                      # ms/frame and swamps the present stage 1000:1 (that
                                      # attempt read "no effect" and was measuring the tracer).
                                      # FRUSTRACER_STAB parked: median 0.230/255 in BOTH arms —
                                      # the ramp is smooth enough that no pixel oscillates across
                                      # it between frames.
                                      # MEASUREMENT TRAP, learned here: a guard on/off A/B
                                      # across TWO PROCESSES is not an instrument — the
                                      # exposure controller eases to its own value per session
                                      # and XeSS/NRD temporal state is not reproducible across
                                      # them, so the cross-session diff showed tens of
                                      # thousands of pixels BRIGHTENED by a feature that can
                                      # only darken. Difference against the same run's own
                                      # baseline (the memory-sampler lesson, in another
                                      # currency); a same-session toggle is the missing
                                      # instrument, and the 16x-local-5x5-median firefly count
                                      # is ALSO blind here (a median sits inside a blob and
                                      # read 0 in every arm, the off one included). The
                                      # within-frame `fired` count and the histogram are the
                                      # valid instruments today.
                                      # Known-accepts: SATURATING spikes are unaffected (they
                                      # pin white at any aperture); a spike's BLOOM HALO is not
                                      # an outlier so it still takes the full boost while the
                                      # core does not (bloom strength is small, so this softens
                                      # rather than inverts; guarding bloom's input would
                                      # change what glare means); the METER still sees the
                                      # spikes, so a sparkly frame still stops the aperture
                                      # down slightly (an outlier-robust meter is a separate,
                                      # additive mechanism); a genuine isolated small specular
                                      # glint over a dark surround is a false positive, bounded
                                      # by the invariant above; frame-border pixels are never
                                      # guarded. OWED: the user's feel-test — the constants are
                                      # measured but the LOOK is not gate-visible
cargo run --release -- --autoexp-mode tonemap  # WHERE the aperture is spent. DEFAULT
                                      # `lights` since 2026-08-11 (the user's call, after a
                                      # feel-test): the controller's EV becomes a gain on the
                                      # SCENE'S LIGHTS (src/autoexp.rs's `Mode`,
                                      # scene::apply_light_gain) and the presentation curve holds
                                      # at exactly 1.0. `--autoexp-mode tonemap` is the pre-feature
                                      # presentation-curve multiply, the opt-out AND the A/B
                                      # partner. The default is TRIPLICATED in autoexp.rs's MODE
                                      # initializer, cli.rs's defaults() and settings.rs's
                                      # `Cycle { default_ix }` — flip all three in lockstep;
                                      # cli::self_test pins that they AGREE (it reads the menu row
                                      # back, so a flip that forgot settings.rs — which would let a
                                      # settings FILE silently re-select the other arm on every
                                      # launch that has one — fails loudly). Settings row:
                                      # Effects/autoexp_mode (Live). NOTE the default makes the
                                      # SPIKE GUARD inert: its arming test is `exposure > 1.0` and
                                      # this arm leaves display exposure at exactly 1.0, so
                                      # `--autoexp-spike-guard` (itself default-ON) only does
                                      # anything under `--autoexp-mode tonemap`. Inherent, not an
                                      # oversight — a global light gain has no per-pixel lever.
                                      # THE PHYSICS, which is why it is cheap AND why it buys less
                                      # than it looks: the rendering equation is LINEAR in emitted
                                      # radiance, so scaling every emitter by g scales every path's
                                      # radiance by exactly g — the same image the tonemap arm's
                                      # pre-curve multiply produces. That is not asserted, it is
                                      # GATED (`light-gain-ab` below) and MEASURED live: the same
                                      # pose, the same +2 EV, one arm each, 1920x1080 DXR on
                                      # DamagedHelmet at --tod 22 with --emissive-lights, reads
                                      # mean |d| 0.0000/255, worst 1 LSB, mean level 24.24 vs 24.24.
                                      # FIVE of the six emitter families take the gain as DATA, so
                                      # they reach BOTH renderers with ZERO shader edits (the
                                      # EL_BOOST parity-by-data precedent — the cbuffer already
                                      # transports every one of them): sun/moon (`e_over_pi` AND the
                                      # cached disc `radiance` — a fix that moved only the first
                                      # would leave the disc behind and no arithmetic gate would
                                      # notice), the dome (`Scene::sky_scale`), the SH ambient
                                      # (`Sh9::scaled` — projection is LINEAR, so scaling nine
                                      # coefficients IS the projection of the scaled dome; NEVER
                                      # re-project, that is the measured 235 -> 17 fps stall),
                                      # fireflies (the baked `w` lane, which both the point light
                                      # and the glow multiply by), and the emissive NEE clusters
                                      # (`EmissiveLight::color`). The two that CANNOT ride data take
                                      # one CB float (`FrameCb::light_gain`, riding el_meta.y's free
                                      # lane rather than moving CB_STRIDE — the ff_count/_pad3
                                      # idiom; HLSL reads it through `scene_light_gain()`): the
                                      # emissive DISPLAY add, whose magnitude lives in the
                                      # SERIALIZED material stream — and which is exactly the term
                                      # a first draft would forget, leaving emitters reading as
                                      # DIMMING as the aperture opens — and the star field, whose
                                      # constants are mirrored HLSL literals.
                                      # THE THREE ABSOLUTE CEILINGS SCALE WITH IT (EL_E_MAX 1000,
                                      # FF_GLOW_L_MAX 512, STAR_L_MAX 4096) and that is what keeps
                                      # linearity EXACT rather than approximate: each is f16
                                      # HEADROOM, i.e. a property of the presented magnitude, so a
                                      # fixed bound would clip a gained value against an ungained
                                      # one and go sub-linear precisely where an emitter is
                                      # brightest. Scaling them also keeps `min` selecting the same
                                      # side, so NO branch, clamp selection or rng draw depends on
                                      # g — which is why the gate reads EXACTLY 0.000e0 at a
                                      # power-of-two gain rather than merely small (x4 is an
                                      # exponent bump, so the reassociation `(e*g)*ndl` vs
                                      # `(e*ndl)*g` is exact too).
                                      # REACH AND COST ARE INVARIANT, for free: `r_infl2` is derived
                                      # ONCE at load from EL_MIN_E, and the gain multiplies
                                      # `color` PER FRAME without re-running `derive_parts`, so every
                                      # cluster keeps its influence radius, its per-pixel scan cost,
                                      # its shadow-ray count and its `emissive::cull_tile` sets. The
                                      # deliberate opposite of EL_BOOST, which lands INSIDE the
                                      # derivation and doubles r_infl2 with the power.
                                      # NON-COMPOUNDING: `Scene::light_canon` holds the ungained
                                      # values and `apply_light_gain` is ABSOLUTE, always writing
                                      # FROM it — a ratio against the live values would compound its
                                      # own rounding over a session's thousands of controller steps
                                      # and make "the same EV" depend on how the session got there.
                                      # Each producer captures what IT produces (`finalize_scalars`
                                      # the sun + dome scale, `refresh_sky_sh` the SH — separately,
                                      # since the interactive scrub throttles re-projection at
                                      # SH_TOD_STEP and the two legitimately disagree mid-scrub,
                                      # `apply_tod_lit` the sun + dome scale again after a TOD
                                      # write). Derived, never serialized (the sky_sh precedent — no
                                      # CACHE_VERSION move, no lever word).
                                      # THE BUG THIS SHIPPED WITH, because the shape recurs: a first
                                      # draft ended `refresh_sky_sh` with a whole
                                      # `apply_light_gain`, which writes the sun FROM the canon —
                                      # and on a FLAGLESS session nothing had captured the canon yet
                                      # (`apply_tod_lit` is structurally unreachable without --tod),
                                      # so every session silently traded its real sun for
                                      # LightCanon's straight-up placeholder. `check.png` caught it;
                                      # the light-gain A/B could not, because it renders BOTH arms
                                      # from the same already-clobbered scene. TWO fixes, and the
                                      # second is the transferable one: capture in
                                      # `finalize_scalars` (the one funnel every load path runs),
                                      # and SCOPE `refresh_sky_sh`'s re-gain to the coefficients it
                                      # just produced — which makes the clobber unrepresentable AND
                                      # leaves an uncaptured canon VISIBLE to the canon-agreement
                                      # gate instead of hiding it by making canon and scene agree.
                                      # A guard that the bug itself satisfies is not a guard.
                                      # THE CONTROLLER STAYS OPEN-LOOP, which is the one thing this
                                      # arm genuinely had to add. Today's meter reads a frame the
                                      # aperture never touched; here the gain feeds the RENDERER, so
                                      # a raw measurement would close a servo loop around a
                                      # FRAMES_IN_FLIGHT-delayed readback. `autoexp::degain`
                                      # subtracts the stops the measured frame was RENDERED with
                                      # (gpu/autoexp.rs carries it PER RING SLOT beside `pending`,
                                      # since the readback lands frames later; the CPU meter is
                                      # same-frame and uses the value still applied), which recovers
                                      # log2(L) exactly and makes the controller mathematically
                                      # identical to the tonemap arm's — so its convergence,
                                      # deadband and clamp gates transfer unchanged. VERIFIED LIVE:
                                      # both arms settle on `measured -10.004 ev +1.850 scale 3.605`
                                      # to three decimals (FR_AEXP_TRACE=1, same pose); without the
                                      # subtraction the lights arm would have read -8.154.
                                      # An EV move is a LIGHTING change, so it takes the TOD block's
                                      # semantics VERBATIM and sits beside it in session(): plain
                                      # accumulation resets (frame = 0) while every upscaler/
                                      # denoiser history, the temporal frustum cache, the claim ring
                                      # and structure replay are KEPT. The controller's DEADBAND is
                                      # what makes that affordable — a converged session leaves
                                      # `light_gain` bitwise unchanged and never enters the block,
                                      # so a still frame keeps accumulating.
                                      # WHAT IT COSTS — accepted when it became the default,
                                      # so read this as the standing known-accept rather than an
                                      # argument against:
                                      # (a) NO SPIKE GUARD. `autoexp::guard_scale` works by relaxing
                                      # a per-PIXEL DISPLAY exposure toward 1.0, and this arm leaves
                                      # display exposure at exactly 1.0, so the guard's own arming
                                      # test (`exposure > 1.0`) is false by construction. A global
                                      # light gain has no per-pixel lever, so this is inherent, not
                                      # an omission — the arm re-opens the firefly issue the guard
                                      # closed. (b) an EV move is a per-frame lighting change the
                                      # temporal histories must absorb (they do — the TOD-scrub
                                      # class — but the tonemap arm has nothing to absorb).
                                      # WHAT IT BUYS: the upscalers and denoisers integrate the
                                      # BRIGHTENED signal, so their absolute clamps
                                      # (--fsr-max-radiance, NRD/FRD's firefly caps) act at the
                                      # PRESENTED brightness rather than at a fixed scene
                                      # brightness — arguably where they belong. That is the
                                      # thing worth A/B-ing, and it is the reason both arms ship
                                      # (and, on the user's feel-test, why this one became the
                                      # default).
                                      # `--exposure-bias` composes into either arm through the same
                                      # `autoexp::ev_total`, so `--no-auto-exposure --exposure-bias 2
                                      # --autoexp-mode lights` is a fixed "make the scene physically
                                      # 4x brighter" lever — and is exactly how the image A/B above
                                      # is taken (no controller, no easing, no guard).
                                      # HEADLESS IS UNREACHABLE: the controller only ticks in
                                      # session(), so --check*/--spin/--cinematic hold the gain at
                                      # exactly 1.0 and every multiply is branched around it.
                                      # check.png/check_gi.png byte-identical — verified, and it is
                                      # the check that earned its keep here.
                                      # Gates: `light-gain` (scene::light_gain_self_test in --check
                                      # — pure, scene-independent: bitwise inertness at 1.0 from
                                      # BOTH directions, non-compounding along several paths to the
                                      # same g, every magnitude scaling EXACTLY at powers of two,
                                      # both of the sun's, REACH invariance, and the
                                      # scale-vs-re-project SH identity); `light-gain-ab` (the
                                      # end-to-end renderer gate — canon agreement on the real
                                      # loaded scene, a g==1.0 render bitwise equal to the ungained
                                      # one, a bitwise RESTORE after a gain, and the linearity
                                      # itself at mean rel < 1e-4 relative to the image's own
                                      # magnitude (the --spp A/B's shape — an absolute bound is a
                                      # different bound on every scene), with teeth both ways: a
                                      # 1.5x-wrong gain must blow the bound, and the row PRINTS
                                      # which families the scene could exercise, since a green run
                                      # on a scene with no stars/fireflies/emissive proves the gain
                                      # HARMLESS there, not correct — the N6b INERT note). TEETH
                                      # EXERCISED, not claimed: un-gaining the emissive DISPLAY add
                                      # takes the helmet arm to 6.594e-2 (660x over) and dropping
                                      # the finalize_scalars capture fails four assertions naming
                                      # `sun false`. A bad canon EARLY-OUTS rather than gaining:
                                      # apply_light_gain writes FROM the canon, so acting on an
                                      # uncaptured one would replace the real lights with
                                      # placeholders for every gate below — including the TRACKED
                                      # golden, which the user would then have to `git checkout`
                                      # back. One red line beats a cascade plus a rewritten
                                      # check.png.
                                      # AND THE GPU HALF HAS ITS OWN GATE (`light-gain` in
                                      # --check-gpu, 2026-08-11): the CPU arm proves the RENDERER
                                      # is linear, which is a different claim from the gain
                                      # REACHING the GPU — five families ride cbuffer rows the
                                      # scene already transports, but the emissive DISPLAY add and
                                      # the star field read el_meta.y through scene_light_gain(),
                                      # three absolute ceilings scale with it, and
                                      # FrameCb::refresh_sky_rows' el_a/el_b copy is the most
                                      # fragile line of the lot (without it a gain move brightens
                                      # every emitter's display add while its cluster NEE stays
                                      # dark — a HALF-applied gain, exactly what a loose bound
                                      # swallows). GPU vs GPU, one refresh_sky apart, never GPU vs
                                      # CPU (the two intersectors legitimately disagree at grazing
                                      # edges — T1/T2's statistical bars exist for that, and
                                      # folding it in would blunt the one comparison this gate is
                                      # for). DXR gets NO twin by argument, not omission:
                                      # DxrGpu::refresh_sky is textually TraceGpu::refresh_sky,
                                      # both delegating to the ONE refresh_sky_rows, and
                                      # shade.hlsli/trace_common.hlsli are one TEXT compiled
                                      # against two root signatures (the N8 argument). MEASURED
                                      # exactly 0.000e0 on every arm; TEETH exercised BOTH ways —
                                      # dropping the el_a/el_b copy reads 3.126e-4 with worst
                                      # EXACTLY 0.75 (= 1 − 1/4, the NEE contribution missing its
                                      # whole 4x, the signature that names the bug), and pinning
                                      # scene_light_gain() to 1.0 reads 1.423e-1 on the helmet and
                                      # 5.274e-2 on --tod 2 alone (so the night procedural scene
                                      # covers the star half without needing an emissive scene).
                                      # NOTE the el_a/el_b plant is only 3.1x over the bound —
                                      # emissive NEE is a small share of frame energy — which is
                                      # precisely why the bound stays at 1e-4 and why the default
                                      # procedural run, whose families line reads [sun dome+SH]
                                      # ALONE, proves the GPU wiring harmless rather than correct:
                                      # `--check-gpu --tod 2` and an emissive scene are what reach
                                      # the other four. `cli::self_test` pins the vocabulary, both
                                      # later-flags-win orders, and that the parse never touches the
                                      # live mode (lever_snapshot's `amode`) — which matters here
                                      # more than usual, since --check runs autoexp::self_test and a
                                      # parse that moved the mode would have that gate scoring the
                                      # wrong arm. autoexp::self_test arms 15/16 pin the
                                      # exactly-one-destination split (display x light == exposure,
                                      # BITWISE) and the de-gain identity with teeth.
                                      # Run after touching autoexp.rs / scene.rs's canon+gain /
                                      # sky.rs's star gain / the shade.hlsli + trace_common.hlsli
                                      # twins: --check (+ byte-compare BOTH goldens), --check
                                      # --stress 5000, --check --tod 2 (stars + fireflies live),
                                      # --check-gpu, --check-gpu --tod 2, --check-dxr, and a scene
                                      # with real emissive (helmet or bistro --tod 22
                                      # --emissive-lights) so BOTH families lines read all six.
                                      # Known-red on that last one and NOT this feature: the helmet
                                      # fails --check-gpu's two-device gates on a plain DAY run
                                      # too, where the gain is structurally inert — the documented
                                      # helmet-is-outside-the-two-device-scene-set caveat.
                                      # finalize_scalars debug_asserts light_gain == 1.0 (it
                                      # CAPTURES the canon, so it must not run on a gained scene;
                                      # true at all eleven call sites today — all load-time, and
                                      # the one live scene edit, the Y/Z frustum snapshot,
                                      # rebuilds only the BVH — but capturing a gained value would
                                      # compound the gain permanently and silently, the same class
                                      # as the sun clobber). apply_tod_lit is the other capture
                                      # site and legitimately runs gained: it recomputes the sun
                                      # from the hour first, so what it captures is canonical by
                                      # construction. Follow-on documented in
                                      # autoexp.rs: a per-family weighting would make the arm a look
                                      # tool rather than an equivalent — deliberately NOT built,
                                      # since the uniform arm's whole value is being provably the
                                      # same image
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
cargo run --release -- --no-crash-handler  # kill lever: don't install the crash handler
                                      # (src/crash.rs, DEFAULT ON — see the Crash handling
                                      # section). On any fault it prints a symbolized stack that
                                      # crosses the Rust/C++ boundary and writes
                                      # frustracer-crash-<pid>.txt + .dmp next to the exe.
                                      # FR_NO_CRASH=1 is the env spelling; FR_CRASH_FULLDUMP=1
                                      # dumps full memory (~10 GB on THE WORLD);
                                      # FR_CRASH_VERIFY=1 reports after main returns whether the
                                      # filter is still ours;
                                      # FR_CRASH_TEST=deref|cpp|panic|overflow|atexit faults on
                                      # purpose
```
