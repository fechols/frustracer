# Materials, textures, and surface detail

Mip-mapping and anisotropy, heightfield relief, normal-map conversions, slope mips, spec-AA, detail texturing and detail AO, tinted shadows, spray, depth tint, and the ambient bump response.

Extracted verbatim from `CLAUDE_Historical.md`, which keeps a stub pointing here. Nothing in this file was rewritten.

```
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
cargo run --release -- --no-slope-mips  # A/B lever (2026-08-05): normal-map mip chains back on the
                                      # raw-byte box filter — the "normal maps flatten with
                                      # distance" behavior. DEFAULT ON: `scene::finalize_normal_
                                      # mips` (called once from load_scene's post-match site, the
                                      # foliage::attach slot — material wiring is identical on
                                      # cold/warm/glTF/world/tile loads, so warm and cold chains
                                      # can't diverge) marks every texture some material's
                                      # normal_tex points at (`Texture::normal_role`, in-memory
                                      # only) and rebuilds its chain with `build_mips`' SLOPE arm:
                                      # decode each 2x2 tap to a tangent vector (the exact
                                      # perturb_normal decode incl. z >= 0.05), average the SLOPES
                                      # (x/z, y/z — the linear quantity, so the mean IS the mean
                                      # tilt; averaging unit VECTORS under-tilts and sample-time
                                      # renormalization restores length, not tilt: a steep/flat
                                      # quad's true mean slope −2.02 read −0.77 off the byte
                                      # average, 2.6x flat), re-encode normalize(sx,sy,1). Alpha
                                      # (the n2h/h2n height) keeps the raw box average in every
                                      # arm — height IS linear, and the planned cavity tap wants
                                      # those mips correct — so the BVH height sweep/relief march
                                      # see identical data either way. A texture id shared with a
                                      # rough/metal/emissive/albedo role is skipped with one loud
                                      # line (slope-encoded mips would corrupt the other role's
                                      # samples — coarser, never wrong). Mips are derived-only
                                      # (never persisted), so NO CACHE_VERSION move and the lever
                                      # does NOT key the .fcache (the --no-mips class); GPU parity
                                      # is free (the chain uploads verbatim, CPU+GPU move
                                      # together — smlp/bistro albedo A/Bs unchanged at 0.0001).
                                      # BC7 corner: under --no-n2h plain normal maps compress and
                                      # slope mips encode fine (measured worst 30.8 dB vs the 25
                                      # limit). Gates: texture::self_test's slope family (mean-
                                      # slope preservation with ANTI-VACUITY teeth — the legacy
                                      # average must provably fail the bound — y-lane twin,
                                      # alpha-stays-box, constant-map bit-exact roundtrip at
                                      # every level, and the normal_role=false legacy bit-pin =
                                      # the off arm's unit-level identity); liveness proven by the
                                      # smlp --check frame moving between the on/off arms with the
                                      # suite green in both (that frame is check-san-miguel-low-
                                      # poly.png now, not check.png — see the golden-routing rule
                                      # in the test-suite paragraph; it is gitignored, so the
                                      # compare is a HASH between two runs, not a git status),
                                      # world cold/warm smoke rebuilds identical chain counts
cargo run --release -- --no-spec-aa   # A/B lever (2026-08-08): no slope-variance → roughness fold —
                                      # the pre-feature behavior, where mip-averaged normal-map
                                      # detail and faded detail-field octaves VANISH with distance
                                      # (distant bumpy surfaces collapse to a mirror-flat mean
                                      # normal at authored roughness — the missing-Toksvig
                                      # signature the slope mips above deliberately left open:
                                      # their renormalize keeps the mean tilt, discards the
                                      # SPREAD). DEFAULT ON ("spec-AA"): detail maps stay in the
                                      # rendering equation at every distance in the statistical
                                      # sense — what a footprint can no longer show as normal
                                      # perturbation shades as a wider GGX lobe instead,
                                      # α′² = α² + 2σ² (`shade::spec_aa_fold`, the LEAN/Kaplanyan
                                      # fold; σ² = mean per-axis slope variance), applied to
                                      # rough_eff between the ripple and the PrimarySurface
                                      # capture so ggx_alphas/sheen/denoiser guides all see it,
                                      # while `refl_ray` keeps reading the FLAT roughness (the
                                      # rng-schedule rule — the fold moves the lobe, never the
                                      # draw schedule; shadeclass strips stay valid verbatim).
                                      # TWO SOURCES, each exactly 0.0 where nothing was resolved
                                      # away (identity BY BRANCH `s2 > 0` — sqrt(sqrt(r⁴)) is not
                                      # an f32 identity): (1) normal maps — `build_mips`' slope
                                      # arm now carries an exact-f32 law-of-total-variance
                                      # side-chain (`Texture::var_mips`, the σ² of the BASE
                                      # slopes inside each footprint, sqrt-domain u8 vs
                                      # SPEC_AA_S2_CAP=0.5 — lossless through the fold's
                                      # saturation; the RGB/alpha byte paths untouched so the
                                      # slope-mip gates cannot move), wrapped by
                                      # finalize_normal_mips into a grayscale COMPANION texture
                                      # appended at the END of scene.textures (no id shift — the
                                      # cache-v7 argument; every store site runs before the pass,
                                      # so companions never reach a sidecar — derived-only, NO
                                      # CACHE_VERSION move, no lever-word bit), level 0 ALL-ZERO
                                      # so the lod ≤ 0 bilinear escape reads exact 0.0 and the
                                      # fold self-disables at magnification structurally; sampled
                                      # through the SAME TexFilter as the map itself
                                      # (Scene::tex_var → Mat.normal_var_tex on GPU, GpuMat/Mat
                                      # 104→108 B lockstep), ×normal_scale² (the decode scales
                                      # slopes linearly); (2) the detail field — `shade::
                                      # detail_var(dlod)`: each octave's applied tilt scales with
                                      # its window wk, so applied variance goes wk² and the
                                      # discarded share (1−wk²) transfers, ×bw²
                                      # (detail_bump_weight — applied + transferred = bw²·full at
                                      # EVERY distance, the invariant; a polished visor is never
                                      # frosted by detail it would never have shown), plateauing
                                      # past DETAIL_AO_RANGE at the field's whole variance
                                      # (VNOISE_GRAD_VAR = 0.1104, measured by deterministic
                                      # lattice MC over vnoise3_vg, mirrored literal, self-test
                                      # re-measures ±10%). GPU: FLAG_SPEC_AA=1048576 (runtime
                                      # lever, the FLAG_DETAIL shape; rides FrameCb::with_frame so
                                      # DXR inherits it), shade.hlsli term-for-term twin,
                                      # companions ride texs[] verbatim (BC7 like any linear map).
                                      # `--no-slope-mips`/`--no-mips` kill the map half
                                      # automatically (no normal_role ⇒ no planes); the detail
                                      # half is independent. Zero rng draws anywhere. Gates:
                                      # texture::self_test's variance family (constant-map exact
                                      # 0 every level, LTV vs a direct 16-slope oracle ±1 byte,
                                      # uniform-block exact 0 + mixed-block >0, cap saturation,
                                      # decode(enc(0)) bitwise 0, role-off empty) +
                                      # shade::spec_aa_self_test (fold anchor (2σ²)^¼, monotone
                                      # bounds, open-window bitwise-0 incl. −1e30/−∞, plateau vs
                                      # an independently assembled closed form, AO-lever share,
                                      # the MC pin) in --check; off arm proven byte-identical
                                      # (check.png/check_gi.png == pre-feature under
                                      # --no-spec-aa); the ON arm moves both goldens (the detail
                                      # transfer fires on the procedural scenes' mid/far pixels —
                                      # re-blessed). Far-field magnitude at defaults: σ² ≈ 0.044
                                      # ⇒ rough 0.5 → ~0.62, gentle by construction (STR=0.5 and
                                      # AO_STR=0.125 govern it; DETAIL_STR² scaling built in)
cargo run --release -- --normal-strength 1.5  # session multiplier on every material's normal-map
                                      # strength (0.0..=8.0; default 1.0 = bit-identical off arm,
                                      # gated k != 1.0 — smlp --check frame hash-equal proven
                                      # (check-san-miguel-low-poly.png today, the golden-routing
                                      # rule); 0 =
                                      # normals fully off, the A/B floor). Applied POST-CACHE in
                                      # load_scene at the apply_tod site (never bakes into a
                                      # sidecar) and post-derive_heights, so relief's height_amp
                                      # (derived from the UNSCALED normal_scale) stays put — the
                                      # decode slopes and --heightfield relief deliberately
                                      # decouple at k != 1 (a feel lever). GPU plumbing free:
                                      # GpuMat.normal_scale packs from the material at upload and
                                      # shade.hlsli's perturb_normal already multiplies — zero
                                      # shader edits, zero CB lanes. Per-load data (the --tod
                                      # class): an Opts field read off SceneRequest, NO process
                                      # global, no lever-block line. Settings row: Effects/
                                      # normal_strength (restart tier, StepF 0..8 by 0.5).
                                      # cli::self_test pins default/last-wins/purity. Both flags
                                      # exist because normal-mapped surfaces read FLAT in
                                      # daylight: the mip flattening above (fixed) plus the
                                      # order-2 SH ambient being a cosine convolution (inherent —
                                      # a ±15° bump tilt moves irradiance a few percent; the
                                      # documented follow-on is a height-driven cavity/local-AO
                                      # term folded into prim.ao so the FSR composite identity
                                      # holds for free, best as h_level0 − h_levelK off the alpha
                                      # mips this feature keeps box-correct)
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
cargo run --release -- --no-detail-tex  # A/B lever: no Unreal-1 style DETAIL TEXTURING (default
                                      # ON, 2026-08-05 — shade::detail_field/detail_bump + the
                                      # shade.hlsli twins; the depth-tint runtime-lever class, no
                                      # cache contact). On MAGNIFIED textured hits — the completed
                                      # isotropic ray-cone lod dlod < 0, i.e. texels blur under
                                      # bilinear, exactly the regime Unreal 1 faded its detail
                                      # texture in over — 3 octaves of WORLD-SPACE 3D value noise
                                      # multiply the albedo (grayscale, DETAIL_AMP 0.18 halving,
                                      # mean ≈ 1 — energy-neutral, self-test-pinned) AND tilt the
                                      # SHADING normal by the same field's analytic gradient's
                                      # tangential projection (micro-bump, DETAIL_BUMP_K, composed
                                      # normal-map -> detail
                                      # -> ripple; n_g untouched — the n_g/n_s split).
                                      # THE DOMAIN IS WORLD-SPACE (2026-08-06 — the rungholt
                                      # anti-tiling rework): q3 = the hit's barycentric REST-POSE
                                      # position (shade::tri_rest_point — Scene::positions IS the
                                      # rest pose on both paths, sway shears the ray / rides TLAS
                                      # instance transforms, so grain never crawls on a swaying
                                      # leaf) over the PER-MATERIAL texel scale s
                                      # (Scene::detail_scales — a sampled per-material MEDIAN of
                                      # the tri_uv_basis texel-size formula, derived in
                                      # finalize_scalars, never serialized — the sky_sh
                                      # precedent; rides GpuMat.detail_scale, stride 100→104 B),
                                      # so octave 0 stays one noise cell per texel-EQUIVALENT
                                      # and the field still self-scales per surface. It moved
                                      # OFF UV texel space because atlas meshes repeat the same
                                      # UV rect per block face (rungholt: 704 distinct vt coords
                                      # for 6.7M tris) and any UV-domain noise tiles in lockstep
                                      # — the user-visible repeated-blotch walls; world position
                                      # decorrelates by construction. s is deliberately NOT
                                      # per-face (the first draft's own seam bug, user-caught):
                                      # greedy-meshed exports make per-face texel density wildly
                                      # non-uniform — vokselia's Grass spans s 0.11..215 with
                                      # 9103 distinct values across merged runs — so a per-face
                                      # s made q3 jump at every face boundary (X-shaped seams on
                                      # flat grass); one s per material is continuous across
                                      # every face by construction. s == 0 (no valid basis
                                      # anywhere) skips the whole field (structural off — and
                                      # the bump/march no longer need the basis at all: the
                                      # tangential projection replaced the (t,b) frame, which
                                      # was orthonormal, so the tilt is identical where both
                                      # exist). Known-accepts: grain frequency uniform per
                                      # material (no within-chart density adaptation — coarser,
                                      # never wrong); f32 fract granularity
                                      # ~1/32 cell on the finest octave at THE WORLD's
                                      # coordinate scale (subvisible); relief-marched hits
                                      # sample the base-plane barycentric point. TWO
                                      # material guards from the first feel-test (the
                                      # frosted-visor finding): TRANSMISSIVE materials skip the
                                      # whole field (their "albedo" is the transmission tint —
                                      # graining it mottles glass/water), and the BUMP is damped
                                      # by the PER-PIXEL map-driven rough_eff over
                                      # DETAIL_ROUGH_LO..HI = 0.2..0.45 (detail_bump_weight; HI =
                                      # the reflection gate's own glossy threshold) — a slope
                                      # that reads as grain on a matte wall FROSTS a tight
                                      # specular lobe, and DamagedHelmet is ONE material whose
                                      # roughness MAP makes the visor smooth and the shell rough,
                                      # so the damping must be per-pixel (safe: the bump draws no
                                      # rng, unlike the reflection gate, which must stay on the
                                      # flat factor). Octave k
                                      # lives in its own lod window saturate(−dlod−k): fades in
                                      # only once resolvable — the window IS the anti-alias (the
                                      # clouds oct_t lesson) and the progressive detail ladder;
                                      # salts 40..42 (ripple owns 16..19, clouds 0..7). ONE
                                      # clouds::vnoise3_vg / cloud_vnoise3_vg eval per octave
                                      # yields value + gradient in one 8-corner fetch (u32-exact
                                      # CPU↔GPU — trace_common.hlsli precedes shade.hlsli in all
                                      # five units; G10d pins the fused eval bit-equal to the
                                      # standalone vnoise3/vnoise3_grad), so
                                      # grain and bump are one coherent surface. Zero rng draws;
                                      # detail flows into the albedo local BEFORE the
                                      # PrimarySurface capture, so guides/f0/diff_albedo stay
                                      # coherent (the normal-map-content precedent).
                                      # UNTEXTURED MATERIALS GRAIN TOO (2026-08-06 — the
                                      # powerplant ask; the block hoisted OUT of the Textured
                                      # albedo arm on both CPU+HLSL): the field never needed
                                      # UVs — its domain is rest-pose position over s — only a
                                      # texel-equivalent SIZE and a fade window, so materials
                                      # with no albedo map take a SYNTHETIC s =
                                      # scene::DETAIL_UNTEX_K (1.5e-6 — the user's two-round
                                      # calibration off the 3e-4 draft: "100x smaller", then
                                      # "2x smaller" again) × CONTENT diag ×
                                      # --detail-untex-scale (derive_detail_scales' kind-keyed
                                      # arm — a Textured material with degenerate UVs keeps its
                                      # bitwise-0.0 off, the self-test pin) and a world-space
                                      # window dlod = log2(cone_w / s) — the cone footprint in
                                      # q-units, MINOR-axis by the same convention as the
                                      # textured aniso arm, exactly 0 at cone_w == s; NEVER
                                      # filt's untextured −∞ base, which would saturate every
                                      # octave window open. Textured materials keep their
                                      # texture-lod window verbatim (bitwise). The synthetic s
                                      # rides GpuMat.detail_scale unchanged (kind-agnostic
                                      # pack); shadeclass strips keep soundness (only the
                                      # textured-window arm folds with TexKind — stripped
                                      # lambert/gloss records legitimately run the untextured
                                      # arm, so their register win shrinks by the detail
                                      # block). --detail-untex-scale 0 is the bitwise off arm
                                      # (the pre-untextured-arm renderer); it reads at
                                      # DERIVATION (restart tier, no cache contact — scales are
                                      # derived-never-serialized). CONSEQUENCE: procedural/
                                      # stress scenes now grain close up — check.png MOVED at
                                      # this change (the liveness proof; the old byte-identical
                                      # claim retired), and the off arms are lever off /
                                      # window closed / s == 0 / transmissive. NO
                                      # NORMAL_MAP_Y_SIGN in the bump (that flip compensates the
                                      # loader V-flip; this field is authored in +u/+v directly).
                                      # FLAG_DETAIL = 32768 (the CB bit, depth-tint shape); the
                                      # iso arm reuses filt's lod base, the aniso arm keys the
                                      # window off the footprint's MINOR axis (shade::detail_
                                      # aniso_base + the shade.hlsli twin, 2026-08-05 — the
                                      # Minecraft-tops finding: the old isotropic recompute
                                      # carried the major axis's -log2|n·d| view-tilt stretch,
                                      # so the window CLOSED on every grazing-VIEWED face while
                                      # SampleGrad kept its albedo texel-sharp — block sides
                                      # detailed, tops flat, a binary flip between adjacent
                                      # faces of one cube. The window must key off what the
                                      # sampler leaves unresolved, and aniso resolves down to
                                      # the short axis. Deliberately UNCAPPED by MaxAnisotropy:
                                      # past the cap it opens ~log2(ratio/max) too far — at the
                                      # default 16 vs tri_grads' 0.05 nd floor that is 0.32 lod
                                      # of amplitude-0.18 grain at near-silhouette grazing, the
                                      # accepted price of a max-free HLSL twin. Self-test gates:
                                      # conformal-equals-log2 unit pin, major-stretch invariance
                                      # with log2(16) teeth, gu/gv symmetry, degenerate-finite).
                                      # Hemi bounce cones' octant-scale lod keeps detail ~off in
                                      # gathers (not a contract — a very close bounce hit can
                                      # fire, bounded + rng-free). Gates: shade::detail_self_test
                                      # in --check (off anchors bit-exact, window endpoint, fade
                                      # continuity, bounds, energy mean, INTEGRABILITY — gradient
                                      # vs central difference at cell-INTERIOR probes: value
                                      # noise is only C¹ at lattice boundaries, so a straddling
                                      # difference is O(h) and measures the probe, the gate's own
                                      # first-draft bug — bump guards + sign pin, the rough-window
                                      # endpoint/monotonicity pins incl. the smooth-pixel
                                      # verbatim no-op, determinism, the ANTI-TILING teeth
                                      # (grain/pools/shadow field bitwise-differ across a
                                      # 16-q-unit block advance per axis), and the per-material
                                      # scale DERIVATION gate — a two-quad one-material scene
                                      # with a 4x density spread must derive the single
                                      # hand-computed median, degenerate UVs exactly 0.0);
                                      # san-miguel-lp --check/--check-gpu/--check-dxr all green
                                      # with the field live (same-seed wavefront-vs-reference
                                      # 0.00e0, albedo A/B 0.0001). The --cam caveat class:
                                      # default poses have few dlod<0 px — feel-test close to a
                                      # textured wall vs --no-detail-tex. Settings rows: Effects/
                                      # detail_tex (restart tier) + detail_strength (StepF 0..4
                                      # by 0.25).
                                      # STRENGTH KNOBS (2026-08-06, the --normal-strength class):
                                      # --detail-strength K (0..=4, DEFAULT 0.5 — the same-day
                                      # feel-test calibration; 1.0 spells the ORIGINAL
                                      # full-strength field and is the ×1.0 bit-identical arm)
                                      # scales the GRAIN family's amplitudes
                                      # (albedo grain + micro-bump, which consumes the field's
                                      # gradient — linear in amp); --detail-ao-strength K (see
                                      # the --no-detail-ao entry) scales the AO family. Process-
                                      # global statics (scene::set_detail_strength/_ao_strength,
                                      # main's lever block, loud on departure from 1.0); the GPU
                                      # twins are the injected DETAIL_STR/DETAIL_AO_STR defines
                                      # (trace::detail_defs, pasted beside spp_defs into every
                                      # SHADE_HLSLI unit on BOTH pipelines — restart tier, the
                                      # probe-reach rule). detail_self_test PINS both knobs to
                                      # 1.0 via an RAII guard (a --detail-strength 2 --check
                                      # still proves the math) and gates them: K=0 exactly
                                      # inert, ×2 doubles the deviation BITWISE (power-of-two fp
                                      # exactness — teeth, not tolerance). Follow-ons:
                                      # per-material-class amplitude if one look doesn't
                                      # fit stone and cloth alike
cargo run --release -- --no-detail-ao # A/B lever (2026-08-05): no detail cavity AO — the
                                      # detail-tex runtime-lever class (depth-tint shape, no cache
                                      # contact). DEFAULT ON, three coupled terms off one height:
                                      # the detail field's PITS darken the AMBIENT + DIRECT
                                      # SPECULAR terms by shade::detail_cavity
                                      # (exp(DETAIL_AO_K·min(h,0)), K=3.0 = the feel knob, HLSL
                                      # literal twin — EXP, not linear: strictly positive at any
                                      # strength, no clamp to audit; the linear K=1 first draft
                                      # measured 0.6/255 mean on a sunlit rungholt plaza —
                                      # provably live, provably invisible, don't re-timid it
                                      # without an image A/B), where h = the field's value − 1 — the
                                      # field's mean is 1.0 BY CONSTRUCTION, so h IS signed
                                      # depth-below-neighborhood and the "local mean" is a
                                      # compile-time constant: a ZERO-lookup cavity term that
                                      # auto-fades with the field's dlod windows — PLUS (feel-test
                                      # round: "grain yes, AO no") two COARSE OCTAVES
                                      # (shade::detail_ao_field — 8/4-texel-EQUIVALENT cells on
                                      # the world-space q3 domain (see --no-detail-tex, the
                                      # 2026-08-06 anti-tiling rework), salts 43/44,
                                      # amps 0.5/0.35, windows log2(cell) − dlod, so they fire out
                                      # to dlod < DETAIL_AO_RANGE = 3: mid-distance POOLS of
                                      # occlusion at a lower frequency than the grain, which is
                                      # what makes the cavity read as AO instead of darker
                                      # speckle — and since round 5 their GRADIENT feeds the
                                      # micro-bump too, × DETAIL_AO_BUMP_K = 1.5: directional
                                      # relief RIMS at the pool scale the eye actually sees at
                                      # mid-distance) and REAL HORIZON-MARCHED SUN SHADOWS on the
                                      # direct diffuse (shade::detail_sun_shadow, round 5 — the
                                      # 2026-08-05 flat-tops campaign; REPLACED the statistical
                                      # detail_micro_shadow, which darkened pits by depth with no
                                      # direction): a closed-form occlusion trace of the field
                                      # toward the sun — 8 taps (linear 1-4 texel-equivalents for
                                      # contact coverage, geometric 6-20 where a 2° sun is
                                      # penumbra-soft anyway) step q3 along the sun's UNNORMALIZED
                                      # tangent projection l_t = l − n(n·l) (the same tangential
                                      # projection the bump applies — azimuths agree by
                                      # construction, the old (t,b) frame is deleted), testing
                                      # upstream terrain against the sun ray rising
                                      # (n·l)/(|l_t|·DETAIL_SHADOW_HT) field-units/texel, early-
                                      # exiting on the closed-form HMAX bound (the clouds
                                      # interval-skip shape — high-sun pixels exit in 2-3 taps),
                                      # shadow = exp(−DETAIL_SHADOW_K·max penetration), soft
                                      # contact = the penumbra + the 1-spp anti-alias (K→∞ is
                                      # binary). The shadow field = grain octave 0 + both pools
                                      # (~3 noise evals/tap; sub-texel octaves are speckle-scale).
                                      # The shadow height scale is INCIDENCE-ADAPTIVE (round 6):
                                      # HT_eff = lerp(DETAIL_SHADOW_HT_LO=1.5, HI=6.0,
                                      # saturate(n·l)). HI is the deliberate ARTISTIC OVERRIDE
                                      # of the bump-coherence value (= DETAIL_BUMP_K): at the
                                      # coherent 1.2 the ray cleared HMAX in a fraction of a
                                      # texel for any sun above ~15°, measured invisible at the
                                      # world's own hours (the MOON_E_OVER_PI precedent — a
                                      # shadow that vanishes whenever the scene is lit reads as
                                      # absent); HI puts shadows in play at steep incidence
                                      # (noon tops) while at GRAZING incidence — noon SIDES,
                                      # sunset tops — rise ∝ n·l already makes shadows maximal
                                      # at the physical steepness, so the override fades to LO
                                      # instead of multiplying 5× onto an already-strong
                                      # response (the round-6 overdone-sides fix; rise stays
                                      # strictly increasing in ndl — d/dndl = LO/(…)² > 0 — so
                                      # higher-sun-never-darker survives, sweep-pinned). The
                                      # DIRECT N·L rides a second round-6 ceiling,
                                      # shade::detail_ndl_cap (DETAIL_NDL_CAP = 0.5, HLSL twin):
                                      # the detail tilt may move a light's diffuse by at most
                                      # ±CAP of the PRE-detail N·L (n_pre, retained iff
                                      # detail_bump ran — sound pre-ripple: ripple and detail
                                      # are structurally disjoint, water is transmissive).
                                      # Under-cap pixels (tops) keep the raw value BITWISE;
                                      # grazing-lit faces compress to the ceiling; p <= 0 kills
                                      # bright terminator speckle (detail may not light a
                                      # pre-detail-unlit facet), and the lower bound is the
                                      # anti-extinguish floor. EVERY direct-tier light rides it
                                      # (shade::capped_ndl — sun, fireflies, emissive clusters;
                                      # round 6b's one-rule-per-light-family uniformity; the
                                      # moon is the sun struct, covered by construction). THE RELIEF-WINDOW
                                      # LAW the
                                      # campaign settled (user-verified both ways — tops relieve
                                      # at sunset, walls at noon): a heightfield only
                                      # self-shadows/contrasts when light arrives shallower than
                                      # its slopes, and contrast ∝ tan(incidence)·tilt, so the
                                      # old 5° tilts gave every flat surface a 5°-wide relief
                                      # window pinned at ITS grazing configuration — which is why
                                      # DETAIL_BUMP_K also rose 0.35 → 2.0 (~20-30° facets, real
                                      # dirt; round-1's 0.35 was tuned on close-up walls, never
                                      # on tops). THE LAW CUTS BOTH WAYS (round 6, the
                                      # overdone-sides feel-test): tan(incidence) is 0.27 on a
                                      # noon-lit top vs 3.7 on a noon-lit side — a 14×
                                      # structural gap — so ONE global gain tuned for the
                                      # minimal-response faces overdrives the maximal ones by
                                      # the same factor; the fix is the CEILINGS above (a
                                      # compressor), never a smaller crank (which un-fixes the
                                      # tops). Round-5 A/B at the same rungholt pose: tod 15
                                      # shaded plaza mean 5.7/255, 14.1% px > 16 (flat wash →
                                      # cobblestone); tod 11 sunlit 3.5/255, 3.6% px > 16 (vs
                                      # round-4b's 3.4 with the whole march structurally dead at
                                      # that sun — the win is directionality + the shaded case);
                                      # known-accept: the march scales the whole direct-diffuse
                                      # bundle keyed on the PRIMARY light's azimuth, so
                                      # firefly/emissive diffuse damps with it — negligible
                                      # amplitude). It exists
                                      # because a flat sky-lit surface cannot get texel contrast
                                      # from normals (order-2 SH ambient is smooth, N·L sits at
                                      # the cosine max under a high sun — the Minecraft-tops
                                      # finding, round 2): the only texel-scale signal there is
                                      # sky VISIBILITY, i.e. occlusion. Pits-only (peaks exactly
                                      # 1.0, structurally — the call sites branch on h < 0, never
                                      # `* 1.0`), deliberately NOT energy-neutral (it is
                                      # occlusion), bounded > 0 by the field's 0.05 floor, zero
                                      # rng draws. THE LOAD-BEARING PLACEMENT: applied AFTER the
                                      # PrimarySurface captures at the color assembly (CPU
                                      # shade.rs + the shade.hlsli twin's both ambient arms —
                                      # split arm scales amb_w, sampled arm a hoisted amb_t;
                                      # prim.ao/direct_s exports stay UN-cavitied), so the
                                      # deterministic delta lands in FSR's exact-remainder
                                      # residual — texel-crisp under FSR-RR (the residual is
                                      # un-denoised by design; folding into prim.ao would hand
                                      # texel-frequency detail to the denoiser to blur) and the
                                      # composite identity closes untouched, zero wire-format
                                      # contact. Known-accepts: a reflection LAP's cavity rides
                                      # the denoised ind_s instead (identity closes either way);
                                      # hemi bounce hits inherit through BOUNCE_Q's sampled arm
                                      # (bounded, rng-free — the detail accept class); mild
                                      # double-darkening where fb.gi/hemi-AO already integrates
                                      # macro occlusion (different scales, multiplicative —
                                      # lower DETAIL_AO_K if a GI feel-test objects); direct
                                      # sun/translucency/emissive/transmission untouched, so
                                      # sunlit tops show mostly grain, sky-lit tops the cavity.
                                      # FLAG_DETAIL_AO = 65536 (runtime CB bit, both pipelines'
                                      # one flags site). Gates: detail_self_test's cavity family
                                      # (off/peak anchors bitwise, pit monotonicity, bound
                                      # anchors, 0⁻ continuity, determinism, and the end-to-end
                                      # TEETH — a real field pit must darken, the same q3 at a
                                      # closed window must be bitwise 1.0, and finding NO pit
                                      # fails loudly), the ao-field gradient family (closed-
                                      # window bitwise-inert incl. the gradient, integrability
                                      # vs central difference at cell-interior probes of BOTH
                                      # pool lattices in ALL THREE axes at step 0.05 q-units —
                                      # 0.25 measured the
                                      # pools' own third derivative at 0.014 and failed a
                                      # correct gradient, the detail_field lesson squared), and
                                      # the march family (closed-window/zenith/sub-horizon
                                      # bitwise 1.0, occluder teeth with DIRECTIONALITY — the
                                      # opposite azimuth at the found shadow point must differ —
                                      # plus a THIRD-AXIS liveness scan along +z (a port that
                                      # dropped an axis would pass the x-scan while shadowing
                                      # nothing on half the walls),
                                      # low-sun-shadows-more, an ndl SWEEP pinning that the
                                      # adaptive HT keeps shadow monotonicity, LO/HI endpoint
                                      # pins, strict positivity, determinism), the round-6
                                      # ndl-cap family (under-cap bitwise raw, over-cap lands
                                      # exactly on p·(1±CAP), p<=0 forces <=0, continuity at
                                      # p=0), and the ANTI-TILING teeth (grain/pools/shadow
                                      # field must each bitwise-differ across a 16-q-unit — one
                                      # Minecraft block — advance on every axis, the exact
                                      # offset the old UV domain aliased);
                                      # procedural/stress scenes carry the field via the
                                      # untextured arm (see --no-detail-tex — check.png moved
                                      # with it); the same-seed wavefront-
                                      # vs-reference A/B stays exact (shared shade.hlsli source).
                                      # --detail-ao-strength K (0..=4, DEFAULT 0.125 — the
                                      # 2026-08-06 feel-test calibration, 1.0 = the original
                                      # amplitudes; the
                                      # --normal-strength class, see the --no-detail-tex entry's
                                      # STRENGTH KNOBS block): scales the pool amplitudes (height
                                      # + rims + cavity input + their shadow-field share) with
                                      # the march's HMAX early-exit bound scaling in LOCKSTEP
                                      # (computed in detail_sun_shadow since the knobs — an
                                      # unscaled bound would clip exactly the cranked shadows the
                                      # knob asked for; the kao=4 must-find-shadow gate pins it).
                                      # Settings rows: Effects/detail_ao (restart tier) +
                                      # detail_ao_strength (StepF 0..4 by 0.125)
cargo run --release -- --no-amb-bump  # A/B lever (2026-08-05, the flat-tops campaign's
                                      # centerpiece): no AMBIENT BUMP RESPONSE — the sampled/SH
                                      # ambient tiers go back to the plain order-2 irradiance.
                                      # DEFAULT ON: shade::amb_irradiance = irr(n_g) +
                                      # AMB_BUMP_K·(irr(n_s) − irr(n_g)), clamped ≥ 0, K = 6.0
                                      # (HLSL literal twin, FLAG_AMB_BUMP = 131072) — the
                                      # HL2-radiosity-basis / bent-normal / SH-dominant-light
                                      # class: irradiance is a cosine convolution, so even the
                                      # EXACT sky response to a bump tilt is a few percent —
                                      # structurally too smooth to show texel relief at any
                                      # tilt; the SH linear band already carries the dome's
                                      # bright direction, so amplifying the deviation response
                                      # is the cleanest member of the trick family (no second
                                      # light, no new data, one extra 9-madd SH eval).
                                      # First-order and DIRECTIONAL: facets tilted toward the
                                      # bright horizon/sun azimuth brighten, away darken — the
                                      # sky itself lights the bumps, which is what the user's
                                      # "indirect lighting needs relief" framing asked for.
                                      # Applies to the FULL n_g→n_s deviation (normal maps +
                                      # detail bump + ripple), so bistro's real normal maps gain
                                      # daylight relief too. THE RESPONSE IS CEILINGED
                                      # (AMB_BUMP_CAP = 0.25 — round 6's 0.5 halved in 6b: at
                                      # HIGH NOON a vertical block side gets ~no direct sun, so
                                      # its light is almost entirely this ambient term and a
                                      # ±50% cap was a ±50% swing of its TOTAL brightness —
                                      # the up-close side contrast the feel-test kept
                                      # reporting; noon tops are direct-dominated, unaffected):
                                      # the SH irradiance derivative is maximal when n ⊥ the
                                      # dome's dominant direction, i.e. on the SIDES the K was
                                      # not tuned for (tops sit near the dominant direction,
                                      # derivative minimal), so the amplified delta is capped
                                      # at ±CAP of the base by a SCALAR rescale (hue never
                                      # shifts); under-cap pixels — the tops, ~10-30% relative
                                      # on the tod-15 plaza — return the uncapped formula
                                      # BITWISE. n_s == n_g (flat-shaded geometry —
                                      # checked FIRST, skipping the lever atomic on the
                                      # majority) and lever-off return the old expression
                                      # VERBATIM — procedural/stress scenes bit-identical by
                                      # construction (verified: check.png hash unchanged).
                                      # Deliberately NOT energy-conserving (artistic
                                      # exaggeration, the MOON_E_OVER_PI class); the FSR
                                      # composite recomputes plain irr(wire n_s), so the
                                      # amplified delta rides the exact-remainder residual —
                                      # identity closes untouched, zero wire contact; hemi
                                      # fb.gi untouched (real gathered irradiance). Zero rng.
                                      # Gates: detail_self_test's amb family (n_s==n_g bitwise
                                      # identity, lever pinned on for the amplification teeth
                                      # then the off arm re-verified — the wide-tiles
                                      # save/restore pattern, since a `--no-amb-bump --check`
                                      # would otherwise fail its own anti-vacuity; sign teeth
                                      # vs a synthetic +x linear band, amplified delta > 1.5×
                                      # raw delta, clamp floor, determinism; round-6 cap teeth —
                                      # over-cap delta lands ON the ceiling with hue ratios
                                      # preserved on a CHROMATIC SH, under-cap bitwise the
                                      # uncapped formula). Settings row:
                                      # Effects/amb_bump (restart tier). Knobs: AMB_BUMP_K,
                                      # AMB_BUMP_CAP
```
