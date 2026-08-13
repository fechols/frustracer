# The Vulkan backend

The `--check-vk` gate suite, stages V0 through V19 — device pick, the derived descriptor map, the reference kernel, the wavefront quadtree, hemisphere tiers, structure replay, FSR3, NRD, and the display and present paths. The single largest entry in the notebook.

Extracted verbatim from `CLAUDE_Historical.md`, which keeps a stub pointing here. Nothing in this file was rewritten.

```
cargo run --release -- --check-vk     # THE VULKAN BACKEND ACTUALLY RUNNING SOMETHING (unix; src/vk/device.rs
                                      # + src/vk/headless.rs, 2026-08-10 — the Vulkan port's M2b+M2c,
                                      # M3a's V5, M3b/M3d's V6, M3c's V7, M3e's V8, M3f's V9; --sw-rays covered
                                      # in M3g, staging uploads in M3h, the real --blas-split in M3i,
                                      # BC7 in M3j).
                                      # --check-spirv proves the corpus COMPILES; this proves a DEVICE
                                      # CONSUMES it, and those are different claims: spirv-val validates a
                                      # MODULE and knows nothing about the pipeline layout we bind it
                                      # against, so a module can be perfectly valid and read the wrong
                                      # resource. Ten stages: V0 the pure pick/memory logic (runs with no
                                      # Vulkan on the box at all) | V1 loader + instance + device | V2
                                      # compile smoke.hlsl to SPIR-V and build the pipelines | V3 run the
                                      # seed -> prep -> indirect-fill chain and read the results back |
                                      # V4 the wave-width table and the subgroup-size decision | V5 the
                                      # TRACER's own register map, derived from the corpus, with every one
                                      # of its kernels bound against it | V6 the reference kernel actually
                                      # RENDERING a frame, scored against the CPU plain reference | V7 the
                                      # WAVEFRONT QUADTREE, scored against V6 — the first EXACT-ZERO gates
                                      # on Vulkan, because two kernels on one device have no excuse |
                                      # V8 the HEMISPHERE BOUNCE TIERS (the H key's AO and GI), the last
                                      # render-path arm the Vulkan tracer did not have | V9 STRUCTURE
                                      # REPLAY, the only gate here scored against ITSELF rather than
                                      # against a reference.
                                      # V3 IS THE POINT: constants reaching a kernel, storage buffers
                                      # bound and written, a GPU-WRITTEN counter turned into dispatch
                                      # arguments by a second kernel, and a third kernel launched from
                                      # those arguments with the CPU never seeing the count. That is the
                                      # whole wavefront design, and it is the SAME smoke.hlsl the D3D12
                                      # suite runs, unmodified — the shader is shared, only the recording
                                      # differs. MEASURED (Radeon 8060S / RADV STRIX_HALO, Mesa 26.0.3,
                                      # Vulkan 1.4): PASSED, and PASSED again forced onto llvmpipe, so
                                      # the chain is proven on two independent ICDs.
                                      # V0 EXISTS BECAUSE OF THE TWO-ICD HAZARD, and it is the most
                                      # transferable part: this box exposes a real RDNA3.5 iGPU AND
                                      # llvmpipe, and a ranking bug that prefers the software device does
                                      # not fail — it renders, correctly, a hundred times slower, and
                                      # every measurement taken afterwards describes llvmpipe. So `pick`
                                      # and `mem_type_index` are ordinary functions over plain data with
                                      # teeth both ways: iGPU must beat CPU-class AND must still beat it
                                      # with the list REVERSED (else the answer is a property of
                                      # enumeration order, not of the ranking); a REJECTED device is never
                                      # chosen however high it ranks; ties break by enumeration index; and
                                      # the requirement mask in mem_type_index is a hardware constraint,
                                      # so a type carrying every wanted flag but masked out must NOT be
                                      # picked.
                                      # THE absent/told SPLIT, found by exercising the lever rather than
                                      # by reading the code: `device::VkError` carries `absent`, and the
                                      # gate SKIPs (exit 0) only on it. A box with no loader, no device,
                                      # or no device meeting the floor is an environment fact — the
                                      # bare-checkout degrade every SDK gate here follows. A mistyped
                                      # FR_VK_DEVICE is NOT: the box may be full of working GPUs and the
                                      # lever named none of them, so that exits 2 (the --check-gpu
                                      # convention: 2 = environment, 1 = a gate failed). The first draft
                                      # skipped-and-exited-0 on both, which turns being-told into
                                      # passing — the --fsr4 doctrine violated in the one place nothing
                                      # would have noticed.
                                      # FIVE HARD DEVICE REQUIREMENTS, none of them chosen here — each is
                                      # something the corpus already needs. Vulkan 1.3 (the SPIR-V target
                                      # env; a 1.2 device could not consume the modules) and
                                      # scalarBlockLayout (what -fvk-use-dx-layout costs, and therefore
                                      # what ONE Rust CB packer costs). The second is not theoretical
                                      # here — smoke.hlsl's `RWStructuredBuffer<uint3> args` is a 12-byte
                                      # stride under DX rules against std430's 16, so the smoke test
                                      # exercises the relaxation it requires. REQUIRED, never preferred:
                                      # a device without it would validate every module and then read the
                                      # wrong bytes. M3a added the descriptor-indexing three, and they are
                                      # requirements in the same sense: runtimeDescriptorArray (texs[] is
                                      # an unbounded Texture2D array, i.e. OpTypeRuntimeArray on a
                                      # descriptor), shaderSampledImageArrayNonUniformIndexing (every
                                      # texture fetch goes through NonUniformResourceIndex), and
                                      # descriptorBindingPartiallyBound (ONE layout serves every kernel,
                                      # so most sets are bound with slots the dispatched kernel never
                                      # touches — without it, binding a set for the sky fill would demand
                                      # valid descriptors for the hemi queues, the G-buffer pack and the
                                      # whole texture table). Probed as three separate bits so a device
                                      # missing one says WHICH.
                                      # RAY QUERY IS ENABLED-WHEN-PRESENT, not required, and the
                                      # distinction is --sw-rays: that arm assembles rt_sw.hlsli and
                                      # traverses our own BVH in the shader, so a device with no ray
                                      # tracing can still run the tracer. What must not happen is a
                                      # session silently running the HARDWARE arm with the feature off,
                                      # which is exactly what shipped in M3a's first draft — the device
                                      # was created without VK_KHR_ray_query, RADV accepted every module
                                      # anyway, and 25 validation errors ("SPIR-V Capability RayQueryKHR
                                      # was declared, but ... rayQuery") is how the enable list was found
                                      # rather than guessed. The chain is not optional and each link
                                      # earns its place: ray query needs acceleration structures,
                                      # acceleration structures need bufferDeviceAddress (their geometry
                                      # is addressed by device address, not by descriptor) and
                                      # VK_KHR_deferred_host_operations (a hard dependency of the
                                      # extension, not of the feature). V5 says so up front on a device
                                      # without it instead of blaming the corpus with a pile of
                                      # capability errors. The no-ray-query BRANCH is untested — both
                                      # ICDs on this box have RT.
                                      # BINDINGS ARE COMPUTED, NEVER WRITTEN DOWN: the descriptor-set
                                      # layout reads vk::spirv::binding_of, the same rule the -fvk-*-shift
                                      # flags are generated from, because a literal is exactly how a
                                      # layout drifts away from the flags its modules were compiled with.
                                      # reg_of_binding is the INVERSE, and self_test pins the round trip
                                      # over the whole range in both directions plus the stride between
                                      # shifts — two independent matches over four constants is precisely
                                      # the shape that stays correct until someone edits one of them.
                                      # TEETH, all three exercised rather than asserted: a binding
                                      # shifted by ONE fails V3 with `outbuf[0] = 0xeeeeeeee ... (never
                                      # written)` — the buffer is pre-poisoned with a sentinel so a
                                      # never-dispatched fill cannot read back as zeros and look like a
                                      # shader that ran; under FR_VK_VALIDATION the same bug is named
                                      # exactly ("uses descriptor [Set 0, Binding 2002, variable
                                      # \"outbuf\"] but the binding was not declared"). Losing the
                                      # roundup in prep ((n+63)/64 -> n/64) fails on the args themselves
                                      # ([8,1,1] vs [9,1,1]). Removing the shader's own bounds guard
                                      # fails on the TAIL check — 64 sentinel words past the fill count
                                      # that the last group provably dispatched over and must not have
                                      # written (a tooth the D3D12 twin does not have). A second pass at
                                      # count 0 pins the empty-queue case every ladder level hits: args
                                      # [0,0,1] and not one word written.
                                      # V4 TEETH, all four exercised the same way: the PROBE_GROUP define
                                      # NOT REACHING the shader (the probe-reach class this codebase has
                                      # shipped repeatedly — every width above 32 then silently compiles
                                      # at the default 32 and the table answers CONFIDENTLY; caught by an
                                      # echoed-width check, "kernel echoed group width 32 but was
                                      # compiled for 64"); a descriptor binding shifted by one; the
                                      # shader's own PLAUSIBLE-LIE detector (drop WaveIsFirstLane so every
                                      # lane counts and the reported width stops matching the counted
                                      # waves — "reports 64 lanes but counted 64 waves, not 1"); and a pin
                                      # that is silently not applied ("asked for subgroup size 32 and got
                                      # 64 — the pin was ignored", which correctly does NOT fire at g32,
                                      # where the pin is unobservable because 32 is what the driver picks
                                      # anyway). The wave-count check is a CEILING, never exact division:
                                      # a group NARROWER than the wave is one PARTIAL wave, the very case
                                      # the probe exists to find.
                                      # VALIDATION IS ON BY DEFAULT, and it changed BECAUSE of that second
                                      # tooth: V4 binds a ONE-binding descriptor set, so a layout that
                                      # disagrees with the module has nothing to desynchronize against and
                                      # RADV resolved the mismatch to the only slot — the planted shift
                                      # PASSED unvalidated (V3's four bindings catch the identical bug
                                      # structurally; one binding cannot). The layer names it exactly, so
                                      # a correctness gate with that instrument installed and unused is
                                      # the --gpu-debug lesson repeated. FR_VK_VALIDATION=0 opts out and
                                      # says so; a missing layer is a loud line and an honestly
                                      # unvalidated run, reported off the FACT (Vk::validated) rather than
                                      # the request, since otherwise a weak run and a strong one log
                                      # identically.
                                      # V5 — THE TRACER'S REGISTER MAP, DERIVED (M3a, 2026-08-10;
                                      # src/vk/reflect.rs + src/vk/layout.rs). The first stage that is
                                      # about the RENDERER rather than about Vulkan. D3D12 gets
                                      # module-vs-layout agreement from create_root_signature — ~150
                                      # hand-written lines kept in step with src/shaders/ by care — and
                                      # what makes that survivable is that a disagreement fails PSO
                                      # creation LOUDLY. Vulkan gives no such guarantee (M2c's planted
                                      # shift PASSED unvalidated), so writing that table a second time
                                      # would be writing the same liability twice in the API where it
                                      # fails quietly. THE LAYOUT IS THEREFORE NOT WRITTEN AT ALL:
                                      # vk::reflect walks the compiled SPIR-V for OpVariables carrying
                                      # DescriptorSet+Binding, classifies each from its pointee type
                                      # (image Sampled=1/2, sampler, AS, Block vs BufferBlock), unions
                                      # every unit into a Map, and vk::layout turns THAT into the
                                      # VkDescriptorSetLayouts. Consequence worth stating: adding a
                                      # resource to a shader adds it to the layout automatically, and a
                                      # binding two units disagree about is a hard error at build time
                                      # instead of a wrong-resource read at run time.
                                      # THE FAMILY IS THE UNIT, and that is a finding rather than a filing
                                      # choice: D3D12 has several root signatures and their register maps
                                      # genuinely CONTRADICT — t0 is `bvh_nodes`, a structured buffer, in
                                      # the tracer and `b_src_diff`, a Texture2D, in FRD — so one map over
                                      # the whole corpus would report a conflict and be RIGHT to. V5
                                      # builds the TRACER family (reference/resolve/wavefront/sky/leaf/
                                      # leaf_fb/hemi_wave/hemi_leaf/compose/feed/nrd_bridge, both vendor
                                      # arms x both sway arms, deduped by source); each further family is
                                      # its own layout, and the conflict detector is what will say so if
                                      # one is mixed in.
                                      # THE MAP IS SCENE-DEPENDENT, which is the strongest argument for
                                      # deriving it: DXC drops resource declarations no entry point
                                      # references, so the procedural scene yields 46 slots while
                                      # san-miguel-low-poly yields 49 — uv_tri_mat/mat_cutout/mat_shadow
                                      # appear only once ALPHA_CUTOUT and TRANS_SHADOW arm. A layout
                                      # written down from one scene would silently omit bindings another
                                      # scene's modules use, and M3b's session layout must therefore be
                                      # derived from the modules THAT session compiled. The slot count in
                                      # the summary line is what makes the difference visible (the
                                      # --check-spirv assembled-bytes lesson, in another currency).
                                      # MEASURED: 26 units -> 46 slots in 2 sets -> 45 compute pipelines
                                      # bound, validation clean, on RADV AND on llvmpipe — so the whole
                                      # tracer's kernel set binds on a software Vulkan device too, which
                                      # is what makes this gateable in CI without a GPU.
                                      # V5 TEETH: FR_VK_DROP_BINDING=<set>:<binding> OMITS one slot from
                                      # the derived layout — the only way to test a layout that is derived
                                      # FROM the shaders, since nothing else can make one wrong — and the
                                      # run must fail. Exercised on two very different descriptors:
                                      # 0:1007 (the TLAS) and 1:1010 (the unbounded texs[] array), both
                                      # exit 1 with the layer naming the variable ("uses descriptor [Set 0,
                                      # Binding 1007, variable \"tlas\"] but the binding was not declared").
                                      # THE HONEST LIMIT, measured: with FR_VK_VALIDATION=0 the SAME drop
                                      # exits 0 — vkCreateComputePipelines accepts it and the driver
                                      # resolves the missing slot — so V5's teeth are validation-layer
                                      # teeth, which is the M2c lesson holding at 46 bindings rather than
                                      # at one, and the reason validation is armed by default. Anti-vacuity
                                      # beside them: the derived map must contain an acceleration
                                      # structure, a sampled image, a sampler, a uniform buffer, a storage
                                      # buffer, a storage image, at least one UNBOUNDED array (a layout
                                      # that sized texs[] to 1 would truncate every scene's texture table)
                                      # and >= 30 slots, and at least one pipeline must have been created.
                                      # LEVERS: FR_VK_DEVICE=<index|name-substring> forces the adapter
                                      # (loud; an ambiguous substring is an ERROR rather than first-match,
                                      # since "amd" matching two adapters and quietly taking one is how a
                                      # measurement ends up describing the other device);
                                      # FR_VK_VALIDATION=0 disarms VK_LAYER_KHRONOS_validation +
                                      # VK_EXT_debug_utils (armed otherwise), and the gate FAILS on any
                                      # ERROR-severity message; FR_VK_MAP=1 prints the derived register map
                                      # (the D3D12 root signature's Vulkan twin — "what does the tracer
                                      # actually bind" answerable without reading the shaders);
                                      # FR_VK_DROP_BINDING=<set>:<binding> is V5's teeth above;
                                      # FR_VK_DROP_STREAM=<name> and FR_VK_AB_FRAMES=<n> are V6's (below);
                                      # FR_SPLIT_AUDIT=1 and FR_SPLIT_NOREBASE=1 are M3i's, and are the D3D12
                                      # levers of the same names rather than new ones;
                                      # FR_VK_AB_DUMP=1 writes the two converged images + the CPU t buffer
                                      # as raw f32 (vk-ab-{cpu,gpu,t}.f32) for spatial attribution — the
                                      # FR_CHECK_AB_DUMP idiom, and what turned "the shading is 11% dark"
                                      # into "every pixel is shading as triangle 0" in one scanline print;
                                      # FR_VK_RES=WxH moves the V6/V7 gate frame (default 800x600) — and
                                      # V7's drained-queue check is PARITY-SELECTED, so
                                      # **FR_VK_RES=400x300 is what covers its other arm** (see V7);
                                      # FR_VK_CTRS=1 dumps V7's raw counter block, which is how the b1
                                      # write-after-read hazard below was found; FR_VK_STAGE=<bytes> (a k or m
                                      # suffix scales) is the staging ring's cap and is M3h's teeth — see below.
                                      # V6 — THE REFERENCE KERNEL RENDERING (M3b, 2026-08-10;
                                      # src/vk/scene.rs + src/vk/tracer.rs). V5 proved every tracer kernel
                                      # BINDS; this is the first stage that proves one of them is RIGHT,
                                      # and it is where the port stops being about Vulkan and starts being
                                      # about the renderer: a stream at the wrong slot, a GpuMat stride
                                      # skewed by -fvk-use-dx-layout, a BLAS built from the wrong device
                                      # address, a cbuffer field landing one dword over — none of those
                                      # fail a pipeline creation, and ALL of them make a picture that
                                      # disagrees. The comparison is --check-gpu's own M2, transplanted:
                                      # one unjittered frame each for primary VISIBILITY (t + hit/sky
                                      # class, which scores the acceleration structure), then an
                                      # accumulated pair for RADIANCE (which scores the shading, the
                                      # materials, the sky and the cloud caches), then the RESOLVE LINK
                                      # (hdr == accum/samples — the one thing here that touches an IMAGE,
                                      # so also the proof that storage-image creation, the GENERAL
                                      # transition and the image descriptor are wired). Statistical, for
                                      # the reason recorded there: hardware watertight intersection is not
                                      # moller_trumbore, and the RNG streams differ by design.
                                      # MEASURED (RADV STRIX_HALO, 400x300, procedural scene):
                                      # class-mismatch 0, rel-t violations 0, max rel t err 1.19e-5;
                                      # radiance per-channel mean rel diff 0.030% against a 2% bar (sky
                                      # -0.02%, geometry +0.04%); resolve link 0 channels past one f16
                                      # step. --stress 200 reads 0.009%. AND ON llvmpipe: 0.206% at 2
                                      # frames — a second, wholly independent ray-tracing implementation
                                      # agreeing with the CPU tracer, which is what makes this gateable in
                                      # CI without a GPU.
                                      # THE BUG IT CAUGHT ON ITS FIRST RUN, because it is the whole reason
                                      # the stage exists: --blas-split is ON BY DEFAULT, so BLAS_SPLIT is
                                      # compiled in and every intersector site reads
                                      # tri_of(InstanceID(), PrimitiveIndex()) = blas_tri[chunk_base[inst]
                                      # + prim]. Those two were bound to the zero dummy, so tri_of
                                      # returned 0 for every hit and the ENTIRE FRAME shaded as triangle
                                      # 0's material — and the visibility gate could not see it at all,
                                      # because `t` comes from the ray query while the triangle id is what
                                      # indexes positions/normals/tri_mat. It presented as an 11% darkening
                                      # that survived every feature lever; what identified it was the
                                      # per-primary-hit SPLIT (sky matched to -0.02% while geometry was
                                      # -11%, so the CB, the sun rows, the whole 4608-byte -fvk-use-dx-
                                      # layout packing and both cloud caches were already proven right) and
                                      # then a scanline print showing the GPU returning ONE colour for
                                      # every material. The fix is the identity remap, which is not a
                                      # stand-in: one BLAS over the index stream in original order makes
                                      # blas_tri[i] = i and chunk_base = [0] the CORRECT single-chunk
                                      # values.
                                      # TWO METHOD LESSONS, both cheap and both re-learnable the hard way.
                                      # (1) THE METRIC IS --check-gpu's, TO THE LINE, and picking a
                                      # different one was this stage's first bug: a mean of PER-PIXEL
                                      # relative errors reads 16% on a CORRECT image here, because it is
                                      # dominated by dark pixels where a 1-spp path tracer's own noise
                                      # dwarfs the value and the two sides draw independent samples by
                                      # design. The ratio of channel SUMS asks the question that
                                      # distinguishes a wired-wrong renderer from a noisy one, and its 2%
                                      # bar is calibrated against exactly this disagreement on D3D12.
                                      # (2) FR_VK_AB_FRAMES is what tells a BIAS from a slowly-converging
                                      # one: the residual read +3.44/+3.45/+3.48% at 16/64/256 frames —
                                      # FLAT — which is what proved it was a defect before any of the
                                      # bisection started.
                                      # TEETH: FR_VK_DROP_STREAM=<name> binds one named stream to the zero
                                      # dummy (a layout DERIVED from the shaders cannot be tested by
                                      # writing a wrong one). MEASURED, and THE TWO SCENES DISAGREE —
                                      # which is why both are worth running. Procedural: blas_tri,
                                      # tri_mat, materials, indices each FAIL. san-miguel-low-poly:
                                      # materials 84.749% | texs 6.014% | tri_mat 1.614% + 653 moved |
                                      # blas_tri 1.080% + 0 moved | indices 0.485% pass | positions
                                      # 0.129% pass | normals 0.022% pass. positions/normals passing is
                                      # a fact about the shading path rather than a weak gate —
                                      # surface_point takes the hit point from ro + rd*t, reads positions
                                      # only for a degenerate-normal fallback and the texture-LOD term,
                                      # and falls back to the face normal when the interpolated one is
                                      # zero; textures NARROWED it (positions went ~0 -> 0.129% once the
                                      # LOD term stopped being dead) without crossing the bar.
                                      # TEXTURES (M3d, 2026-08-11) — src/vk/textures.rs: one VkImage per
                                      # Scene::textures entry, full mip chain, _SRGB vs _UNORM by the
                                      # texture's own role flag (SceneGpu::new_uploaded's contract in the
                                      # other API; texture.rs still owns the texels, the chain, the slope/
                                      # variance arms and the sRGB role). Uploads batch through one
                                      # reusable host-visible staging buffer — one submit per ~64 MB of
                                      # whole chains, not one per (texture, mip), since 313 textures is
                                      # ~3000 subresources and a fence each would dominate the load.
                                      # texs[] is written by KIND, not by register: it is the corpus's one
                                      # unbounded sampled-image array, which V5's own anti-vacuity already
                                      # asserts exists — so no TEX_TABLE_BUFS literal, which matters
                                      # because that const lives in the Windows-only gpu/trace.rs and a
                                      # second copy here would be exactly the transcription M3a exists to
                                      # avoid. samplerAnisotropy is the one CORE feature the backend
                                      # enables, ENABLED-WHEN-PRESENT: a device without it runs --aniso 1,
                                      # which is the isotropic ray-cone lod path VERBATIM.
                                      # BC7 WAS DEFERRED HERE AND LANDED IN M3j (below): M3d uploaded every
                                      # texture as RGBA8, i.e. the --no-bc7 arm, which is a MEMORY question
                                      # that moves the CPU-vs-GPU comparison in the SAFE direction — the CPU
                                      # samplers keep exact RGBA8 either way — and therefore a legitimate
                                      # place to stop. It inherited bc7::should_compress verbatim when it
                                      # did land, including the carve-out that alpha-masked and
                                      # height-carrying textures never compress (a VISIBILITY contract).
                                      # THE ANTI-VACUITY PROBE, and it earns its keep immediately: every
                                      # metric above passes identically whether 313 images reached the
                                      # shader or were uploaded and never read, so V6 re-renders the SAME
                                      # pose with texs[] flattened to 1x1 white and requires the two GPU
                                      # images to DIFFER (GPU-vs-GPU, one descriptor write apart — the N8
                                      # shape). THE COUNT IS THE TEST AND THE MEAN IS ONLY REPORTED,
                                      # because a signed mean CANCELS: Sponza moves 29.6% of its channels
                                      # for a 0.26% mean, which a mean-based bar would have read as
                                      # "textures barely matter". And the probe is STRICTLY STRONGER than
                                      # the radiance bar on a textured scene: FR_VK_DROP_STREAM=blas_tri
                                      # is M3b's own bug (every hit resolves to triangle 0) and on
                                      # san-miguel it reads 1.080%, i.e. it PASSES the bar the stage
                                      # shipped with — only the probe's 0-channels-moved catches it.
                                      # COUNTERS are read back and REPORTED, never asserted: rt.hlsli's
                                      # count_alpha_rej/count_height_rej and the tinted-shadow tally are
                                      # all #ifdef HAVE_COUNTERS, which ctr.hlsli defines for the
                                      # WAVEFRONT kernels and deliberately not for the reference kernel,
                                      # so a "> 0" must-fire here would assert against an instrument that
                                      # structurally cannot reach its target (the confidently-wrong class,
                                      # caught by writing the must-fire and watching it read 0 on a scene
                                      # famous for its foliage). It becomes a real must-fire in M3c; the
                                      # readback and the per-frame vkCmdFillBuffer zero exist now so that
                                      # gate inherits a working instrument.
                                      # MEASURED (RADV, the 800x600 gate frame, 16 frames): san-miguel-low-poly
                                      # 5.6M tris / 313 textures / 2973 subresources / 511.5 MB -> 0.006% with
                                      # class-mismatch 0; Sponza.gltf 0.007%, rungholt 0.007%, DamagedHelmet
                                      # 0.020%, stress + procedural green. (The M3d readings were taken at the
                                      # then-default 400x300 and are superseded, not contradicted: smlp read
                                      # 0.007% there, the helmet 0.081%.)
                                      # WHAT V6 STILL SKIPS, LOUDLY: a device with no ray query. M3b/M3d's other
                                      # declared boundaries — staging uploads, the real --blas-split, BC7 — are
                                      # M3h, M3i and M3j, all landed.
                                      # V7 — THE WAVEFRONT QUADTREE, and the first EXACT-ZERO gates on
                                      # Vulkan (M3c, 2026-08-11; the ladder half of src/vk/tracer.rs). V6's
                                      # bars are all statistical, and unavoidably so: it is scored against the
                                      # CPU, hardware watertight intersection is not moller_trumbore, and the
                                      # RNG streams differ by design. V7 is scored against the VULKAN
                                      # REFERENCE KERNEL — which V6 just proved renders the CPU's picture — and
                                      # two kernels on one device running the same rays through the same
                                      # shade.hlsli have no such excuse. That is the whole reason the ladder
                                      # lands AFTER the reference kernel rather than instead of it.
                                      # MEASURED (RADV STRIX_HALO, the 800x600 gate frame): claim-violation 0 |
                                      # false-sky 0 | tmin-overshoot 0 | hybrid-extra 0 | max rel t err 0.00e0,
                                      # and the same-seed image A/B reads **EXACT 0.00e0 with 0 hot channels**
                                      # on procedural, DamagedHelmet, Sponza and vokselia — i.e. the wavefront
                                      # and the reference produce a BIT-IDENTICAL accum buffer. The two scenes
                                      # that arm the candidate loops read 7.47e-9 (san-miguel-lp, max 4.65e-3)
                                      # and 3.25e-9 (rungholt, 1 hw-edge px), both with 0 hot channels: an
                                      # alpha/tint candidate loop legitimately sees a grazing edge differently
                                      # at TMin=t_start than at TMin=0, which is the documented two-intersector
                                      # class. Worth stating plainly because the D3D12 note predicts otherwise:
                                      # NVIDIA reads bit-exact there and **AMD was recorded as re-origining the
                                      # ray at TMin, landing 1-2 ulp away — that does NOT reproduce on RADV via
                                      # Vulkan on opaque scenes.** The gate is still written as a bounded HOT
                                      # COUNT with the edge pixels excluded from the max, because it must hold
                                      # on the hardware that does.
                                      # THE STRUCTURE MATCHES D3D12 EXACTLY, which is a stronger statement than
                                      # any of the tolerances: at 800x600 both suites report `leaves 768 |
                                      # sky-tiles 4 | splits 257 | blocked 256 | cuts 65 | overflow 0`. Same
                                      # quadtree, two backends, two intersectors. That agreement is why the
                                      # gate frame moved to --check-gpu's own resolution (M3b/M3d ran at
                                      # 400x300) — it also costs nothing, since the wall clock here is DXC
                                      # compiling the corpus and not the CPU reference (smlp 22.1 s at 400x300
                                      # vs 21.7 s at 800x600), and it is what makes the transmissive must-fire
                                      # fire at all: at 400x300 ZERO shadow rays cross San Miguel's glass.
                                      # THE GATES, each transplanted from --check-gpu at its own strength:
                                      # claim-violation (THE soundness contract, asserted DIRECTLY rather than
                                      # by proxy — the leaf queue's inherited t_start against the EARLIEST t
                                      # either intersector reports, the most pessimistic ground truth
                                      # available); exactly-once coverage + queue accounting (leaf and sky
                                      # rects must PARTITION the screen, no pixel may keep the 0xffffffff
                                      # sentinel cs_clear_info flooded, both tile queues must have drained, and
                                      # CTR_OVERFLOW must be 0 — the queues are sized to the structural worst
                                      # case, so an overflow is a sizing bug and never a scene); false-sky /
                                      # tmin-overshoot / hybrid-extra (the overshoot bucket is SPLIT by
                                      # whether the inherited bound could explain the miss at all — see the
                                      # M3k block below for the decomposition and its teeth); the LeafRec
                                      # frontier-handle cookie/token
                                      # ABI audit; and anti-vacuity both ways (a ladder that emitted no sky
                                      # tile proved no empty space, which is the quadtree's entire product).
                                      # THE DRAINED-QUEUE CHECK IS PARITY-SELECTED and that is not incidental:
                                      # cs_prep zeroes only the OUT counter, so the last level's IN counter
                                      # legitimately still holds the tiles it consumed, and WHICH queue must be
                                      # empty follows depth_full % 2. The default 800x600 gives depth 5 (odd);
                                      # **FR_VK_RES=400x300 gives 4 and is what covers the other arm.** The
                                      # D3D12 twin has the identical expression and its own note about a bug an
                                      # odd depth hid — a parity-selected gate is half a gate until both
                                      # parities have run, so run both. RUN THE EVEN ARM ON THE PROCEDURAL
                                      # SCENE: at 400x300 San Miguel's glass is crossed by ZERO shadow rays, so
                                      # that pairing FAILS on the transmissive must-fire rather than on
                                      # anything about parity — which is the caveat below, seen from the
                                      # resolution side.
                                      # THE COUNTER MUST-FIRES M3d COULD ONLY REPORT ARE REAL HERE, which was
                                      # half the point of the stage: cs_leaf pastes ctr.hlsli, so HAVE_COUNTERS
                                      # is finally defined in a kernel that runs, and CTR_ALPHA_REJ /
                                      # CTR_TRANS_PASS / CTR_HEIGHT_REJ / CTR_RTGI_RAYS become assertions in
                                      # BOTH directions (a masked scene that rejects nothing has dead cutout
                                      # code; an opaque scene that rejects anything means ALPHA_CUTOUT did not
                                      # compile out). Measured: rungholt 5427 cutout / 14752 tint, smlp 1353 /
                                      # 13, Sponza 4, procedural 0/0 with 341393 RTGI bounce rays. Same --cam
                                      # caveat as --check-gpu's twins — a pose containing no masked geometry
                                      # trips the must-fire, and San Miguel's 13 tint crossings out of 480000
                                      # px is exactly how close that is.
                                      # THREE THINGS THE LADDER SPELLS DIFFERENTLY FROM D3D12, and only three.
                                      # (1) THE PING-PONG IS A DESCRIPTOR SET. qin/qout (u5/u6) swap every
                                      # level and Vulkan has no rebound root UAV, so set 0 is allocated
                                      # SEVERAL times off ONE layout — LADDER A (u5=qa, u6=qb), LADDER B (the
                                      # swap), and TERMINAL (u5=cloud_lod, u6=cloud_shadow, the registers'
                                      # second meaning once the ladder has drained) — and a pass binds the
                                      # variant its parity names. V8 adds two more of the same shape (see
                                      # below), for five. Set 1 is allocated once and shared by all of them:
                                      # every variant is a set-0 property. A handful of writes at init, one
                                      # vkCmdBindDescriptorSets per dispatch group, zero per-dispatch
                                      # descriptor traffic. (2) PER-DISPATCH PUSH CONSTANTS
                                      # BECOME vkCmdUpdateBuffer. b1 is a uniform buffer here (DXC has no flag
                                      # to promote a cbuffer to push constants, and [[vk::push_constant]] would
                                      # be an HLSL edit) and the ladder rewrites it twice per level, which a
                                      # host write cannot do — every host write inside a run() closure happens
                                      # before the submit. An inline transfer update lands at the right point
                                      # in the stream and costs nothing extra, since a barrier already sits
                                      # between every pair of dispatches. (The dynamic-offset UBO ring is the
                                      # other shape and was rejected: it would make the DERIVED layout
                                      # special-case one binding.) (3) THERE ARE NO RESOURCE STATES — one
                                      # global barrier covering COMPUTE|TRANSFER -> COMPUTE|DRAW_INDIRECT
                                      # replaces D3D12's UAV barrier AND the args buffer's
                                      # UNORDERED_ACCESS<->INDIRECT_ARGUMENT transition pair.
                                      # THE BUG THAT COST THE LADDER, and it is (2)'s fault: a per-dispatch
                                      # constant block needs a WRITE-AFTER-READ edge as well as the obvious
                                      # read-after-write one, and the WAR edge is the one that is easy to omit
                                      # and invisible when you do. The transfer is free to execute ahead of the
                                      # dispatch it textually FOLLOWS, so cs_prep read the NEXT level's push3,
                                      # wrote its indirect args to the wrong slot, and every level after the
                                      # first dispatched zero groups. Nothing faulted, validation was clean,
                                      # and the frame came back with one split and no terminals — found by
                                      # FR_VK_CTRS=1 showing BOTH tile counters at 0 beside a split count of 1.
                                      # The push helper carries both barriers itself now, which is why it is
                                      # one function rather than three lines at each of its two dozen call
                                      # sites.
                                      # --sw-rays IS COVERED (M3g) and wanted NO new set variant, which is the
                                      # part worth keeping: the lever swaps every RayQuery body for
                                      # rt_sw.hlsli's traversal of OUR binary BVH, reading the tree at t0 and
                                      # tri_idx at t1 — two registers that already MEAN different things per
                                      # phase here, so it is two more OVERRIDES on variants that exist. The
                                      # ladder keeps the WIDE tree at t0 (a frustum query cannot descend the
                                      # binary one) and takes ft_bnode at t1 for level_finish's leaf-cut
                                      # translation; the TERMINAL variant — bound by the leaf, sky AND
                                      # reference passes alike — takes the binary tree at t0 and the real
                                      # tri_idx at t1. That is D3D12's own two rebinds, spelled as the
                                      # difference between two variants instead of as root-descriptor writes.
                                      # It found ONE porting defect and the gate did NOT: cap_cut is DOUBLED
                                      # under sw_rays_leaf + FTREE (level_finish allocates a second slot per
                                      # split for the translated binary ids) and the doubling had not been
                                      # mirrored, so the pool exhausted 107 times at 800x600 and those tiles
                                      # degraded to ROOT seeding — sound, and a different structure, against
                                      # D3D12's 0 on the identical frame. The frontier counters stayed inside
                                      # their bounds throughout, so nothing but a side-by-side read noticed;
                                      # CTR_CUT_FALLBACK is now GATED at 0 here (where --check-gpu only counts
                                      # it) precisely because the pool is sized to a STRUCTURAL bound in both
                                      # arms, which makes a nonzero count a sizing transcription error and
                                      # nothing else — the class a second backend introduces. V7 also gains
                                      # the lever's OWN anti-vacuity, --check-gpu's verbatim: the three
                                      # CTR_FRONTIER counters fire once per CONSUMED non-root LeafRec (never
                                      # per ray — a per-ray atomic would tax the very path the lever exists
                                      # to measure), so rays > handles IS the reuse claim stated as an
                                      # inequality, and the OFF arm demands exact 0 because
                                      # frontier_record_reuse zeroes its flag on !SW_RAYS_LEAF while still
                                      # executing all three atomics. Fixed, the two backends agree to the
                                      # digit: 768/768 non-root handles, 468.8 rays/handle, 0 fallbacks,
                                      # cuts 449 (vs 65 unlevered — the terminal-cut skip that SW_RAYS_LEAF
                                      # disables), llvmpipe byte-for-byte the same. AND V6 gets STRICTLY
                                      # BETTER under the lever, which is the tell that the port is right:
                                      # max rel t err 2.33e-1 with one disagreeing pixel through the
                                      # hardware intersector, 2.10e-5 with zero through ours — both sides
                                      # of that comparison now run the SAME traversal as the CPU tracer, so
                                      # the watertight-vs-moller edge class disappears exactly as the D3D12
                                      # note predicts for TMin re-origining.
                                      # ONE THING THE LEVER DOES NOT BUY HERE, said
                                      # rather than implied: a device with no ray tracing. Its corpus declares
                                      # no acceleration structure (which is why the TLAS descriptor write is
                                      # GUARDED on the map — the samplers have carried that guard since M3a,
                                      # and this one was simply never reachable before), but VkScene still
                                      # builds a BLAS/TLAS nothing reads, so V6 still requires
                                      # VK_KHR_ray_query. COMPOSE is
                                      # dispatched under fb ONLY, and that is D3D12's rule verbatim rather
                                      # than an omission: with fb off the leaf and sky passes splat straight
                                      # into accum through accum_splat, so a compose would be a
                                      # buffer-to-buffer copy of a full screen.
                                      # M3h — STAGING UPLOADS (2026-08-11), the last of vk/scene.rs's three
                                      # declared M3b boundaries. Every immutable stream — the scene's eleven, the
                                      # software trees, and the texture chains that already had their own ring —
                                      # is DEVICE_LOCAL now and written through src/vk/stage.rs's one reusable
                                      # host-visible chunk (64 MB cap, sized DOWN to what a scene actually
                                      # carries, freed before each constructor returns, so peak commit is
                                      # steady-state + one chunk rather than twice steady-state — the
                                      # SceneGpu::new_uploaded discipline in the other API). Two costs it
                                      # removes, only one of which is measurable here: PEAK HOST MEMORY (a
                                      # mapped upload is a second full copy and the repack feeding it was a
                                      # THIRD — at 34.4M tris the wide tree alone is ~480 MB and the binary tree
                                      # ~960, so the collect-then-map shape cost ~2.9 GB of RAM to move ~1.4 GB)
                                      # and READ BANDWIDTH FOREVER (a host-visible buffer a shader reads is host
                                      # memory a shader reads — invisible on this UMA box and a per-access PCIe
                                      # round trip on a discrete GPU, i.e. the half that cannot be measured here
                                      # and so the half worth getting structurally right rather than
                                      # empirically). The generator form is what deletes the repack outright:
                                      # stream_gen takes a COUNT and a closure, so positions/normals/uv convert
                                      # INSIDE the ring and blas_tri's identity remap (138 MB of `i` at 34.4M
                                      # tris) is never built at all. WHAT STAYS MAPPED, deliberately: frame_cb,
                                      # push, and the hemi probe points — the dynamic class, where staging would
                                      # trade a host write for a host write plus a copy plus a barrier.
                                      # THE BARRIER IS EMITTED ON THE LAST CHUNK ONLY, and that is a property of
                                      # Vulkan's synchronization scopes rather than an optimization: a pipeline
                                      # barrier's first scope is "all commands earlier in SUBMISSION ORDER",
                                      # which spans earlier submits on the same queue, so one barrier after the
                                      # final write covers every chunk in whatever command buffer it landed. Its
                                      # dst mask carries ACCELERATION_STRUCTURE_READ as well as SHADER_READ,
                                      # because positions/indices are read by the BLAS build through a device
                                      # address and not by a shader — a shader-only mask would be right for nine
                                      # of the eleven streams and silently wrong for the two whose corruption is
                                      # hardest to attribute.
                                      # THE COVERAGE PROBLEM AND ITS LEVER, which is the transferable part: the
                                      # multi-chunk path is otherwise reachable only on a scene big enough to
                                      # need it, so the gate's reach would be a property of which scene somebody
                                      # happened to run — at 64 MB the procedural check scene is one chunk per
                                      # stream and every off-by-one is invisible, while san-miguel's 67 MB index
                                      # stream chunks by ACCIDENT. FR_VK_STAGE=<bytes> caps the ring, so
                                      # FR_VK_STAGE=64k puts every stream on any scene through the chunked path;
                                      # measured procedural 16 -> 88 submits with every gate's number IDENTICAL
                                      # (radiance 0.045%, same-seed image 0.00e0, hemi 0.0067/3.02%, replay all
                                      # zero). BYTES rather than the MiB the first draft took — the procedural
                                      # scene's largest stream is under 1 MB, so a 1 MiB floor armed the lever
                                      # and chunked nothing (13 submits before and after): an instrument at the
                                      # wrong RESOLUTION cannot see what it was built for, the v1.5.3 lab lesson
                                      # in a third currency.
                                      # GATED IN TWO PLACES because neither reaches the other's failures.
                                      # V0 runs vk::stage::self_test (pure, device-free, beside the device pick)
                                      # over the CHUNK PLAN — coverage, byte-vs-element offsets, the ragged
                                      # tail, the empty stream, a ring that is not a multiple of the element
                                      # size (must round DOWN), and the oversized-element case where the loop's
                                      # own `.max(1)` keeps it terminating and would overrun the mapping, which
                                      # is why stream_gen refuses that configuration outright rather than
                                      # trusting the plan. The loop CONSUMES `plan` rather than repeating it, so
                                      # the gate scores the shipping code and not a parallel description of it.
                                      # TEETH, both exercised: an offset in elements instead of bytes fails V0
                                      # on every scene AND loses the device under FR_VK_STAGE=64k (a copy past
                                      # the destination) while passing every device gate at the default ring —
                                      # the coverage argument, demonstrated; a dropped tail chunk fails V0 plus
                                      # four V6/V7 gates (class-mismatch 341393, radiance 85.194%, rtgi rays 0).
                                      # The V6 line reports staged BYTES and SUBMITS split three ways
                                      # (scene/textures/trees) for the reason the texture line reports bytes: a
                                      # mapped upload and a staged one allocate the same buffers and render the
                                      # same frame, so only the byte total tells them apart — and the chunk
                                      # count is additionally what says whether the multi-chunk path ran at all.
                                      # TWO SHARED-CORE MOVES FELL OUT, both the gpu_bvh_node shape: gfx::scene
                                      # gains stream_bytes (the ring's sizing rule, beside the wire formats it
                                      # counts — GpuMat's own size is a term, so a copy next to one backend's
                                      # ring goes stale when the material struct grows; gpu/trace.rs imports it
                                      # under its old local name) and ftree gains quantize_node/bnode_at, so the
                                      # wide tree streams a node at a time while D3D12 keeps materializing —
                                      # one per-node CONVERSION, two iterations. `quantized()` is now that
                                      # function in a map, so the two cannot diverge.
                                      # MEASURED (RADV, 800x600): procedural staged 4.7 MB in 16 submits,
                                      # san-miguel-low-poly 929.6 MB in 29 (scene 327.2/16 — three of those
                                      # chunked — textures 511.5/10, trees 90.8/3), rungholt 702.6/24,
                                      # Sponza 507.2/24, vokselia 203.7/17, the helmet 107.7/18. Six scenes,
                                      # both FR_VK_RES parities, --sw-rays (cuts 449, fallback 0, 768/768
                                      # frontiers at 468.8 rays/handle — M3g's numbers to the digit),
                                      # --no-ftree, --no-wide-levels and llvmpipe all unmoved; and the same
                                      # 929.6 MB through a 64 KB ring is 6735 submits with every number
                                      # IDENTICAL (radiance 0.006%, image 7.47e-9, hemi 0.97%) — the
                                      # chunking proof at a scale no default configuration reaches.
                                      # M3i — THE REAL --blas-split (2026-08-11), the last of vk/scene.rs's M3b
                                      # boundaries and the one with a known hazard behind it: one BLAS per maximal BVH
                                      # subtree under the cap, each instanced identity with InstanceID = the chunk
                                      # index, built worst-case with ALLOW_COMPACTION and compacted into an exact-size
                                      # arena. NO SHADER CHANGES AT ALL — `BLAS_SPLIT` was already compiled in (the
                                      # lever is on by default), `tri_of(inst, prim) = blas_tri[chunk_base[inst] +
                                      # prim]` was already the intersector's rule on both backends, and the tracer
                                      # already bound both streams; M3b simply filled them with the single-chunk
                                      # values. So this is one file, and the planner (`blas_split::plan`), the vertex
                                      # WINDOWING (`plan_windows` / SPLIT_INDEX_CEILING — the RDNA4 index-value
                                      # workaround, which ships here rather than staying D3D12-only because it is a
                                      # property of the geometry desc, and because the defect was found on AMD
                                      # hardware this backend also runs on) and every structural contract they satisfy
                                      # are the SHARED pure code `--check` already gates.
                                      # THE MEASUREMENT IS THE WHOLE POINT, and it reproduces the D3D12 rationale
                                      # from the other API: BLAS scratch is sized by the LARGEST SINGLE GEOMETRY, so
                                      # san-miguel-low-poly (5.6M tris) asks **665 MB of transient scratch as one
                                      # BLAS and 7 MB as 198 chunks** — a 95x cut — while the COMPACTED result is the
                                      # same size either way (650 vs 648 MB). That is the number the Intel
                                      # device-removal note predicts (one 34.4M-tri BLAS asked 1891 MB and removed the
                                      # device); nothing here has been run at that scale, but the shape is confirmed.
                                      # BOTH ARMS COMPACT, and that is deliberate rather than incidental: the unarmed
                                      # `--no-blas-split` arm goes through the SAME `build_blas_set`, because with it
                                      # on a separate uncompacted path its size report would be a worst-case number
                                      # sitting beside the split's compacted one, and the A/B those two lines invite
                                      # would read a COMPACTION win as a SPLIT win (measured on the procedural scene
                                      # before the refactor: 8 MB split vs 12 MB single, of which the entire
                                      # difference was compaction — both arms read `compacted from 12`).
                                      # THREE VULKAN SPELLINGS, all of them the same shape as D3D12's: the arena is
                                      # one buffer with each structure created as a VIEW at a 256-byte-aligned offset
                                      # (`VkAccelerationStructureCreateInfoKHR::offset`, the same number D3D12 spells
                                      # `..._BYTE_ALIGNMENT`); the compacted sizes come from a
                                      # `ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR` query pool
                                      # (`cmd_write_acceleration_structures_properties` after the builds, read with
                                      # `TYPE_64 | WAIT`) rather than D3D12's postbuild-info buffer; and builds run
                                      # SERIALLY through one shared scratch with an AS_WRITE -> AS_READ|WRITE barrier
                                      # between them, which IS the serialization the sharing requires and not an
                                      # optimization to remove. ONE DELIBERATE ABSENCE: no VRAM pre-flight. That check
                                      # exists on D3D12 because WDDM silently DEMOTES an over-budget commit to system
                                      # memory and renders at a tenth the speed; here `vkAllocateMemory` returns
                                      # OUT_OF_DEVICE_MEMORY and `Vk::buffer` makes it a loud failure, so the
                                      # quiet-wrong outcome the pre-flight guards against does not exist
                                      # (VK_EXT_memory_budget is the instrument if a predictive one is ever wanted).
                                      # STILL NOT DONE, both because their consumer does not exist here:
                                      # `--dxr-sbt`'s `refine_by_class` (the class index means something only to a
                                      # ray-tracing PIPELINE's SBT, and this backend has RayQuery and nothing else)
                                      # and `--foliage-sway`'s animated TLAS ring (D3D12's STATIC TLAS holds every
                                      # chunk at identity, which is the rest pose every headless gate traces, so this
                                      # builds exactly that and never calls `foliage::split_plan` — the two backends'
                                      # PARTITIONS therefore differ chunk for chunk on a swaying scene and their
                                      # IMAGES do not, because `tri_of` follows whatever partition it was handed).
                                      # THE COVERAGE LEVER ALREADY EXISTED, which is the one place this was easier
                                      # than M3h: `--blas-split N` is a CLI flag, so any scene reaches the multi-chunk
                                      # path (procedural 80k tris reads 2 chunks at the 64k default, 230 at 512, 1811
                                      # at 64). V6 gains the `--check-gpu`/`--check-dxr` anti-vacuity twin — an armed
                                      # cap that produced ONE chunk on an OVER-cap scene FAILS (every hit lands in
                                      # instance 0 and `chunk_base[0]` is 0, so a wrong arena offset, a wrong window
                                      # and a wrong instance id all still read right), while an under-cap scene is a
                                      # NOTE — plus a `V6 blas-split:` line carrying chunks, prims min/mean/max, and
                                      # the compacted/uncompacted/scratch sizes.
                                      # FR_SPLIT_AUDIT=1 IS PORTED (the bistro-dusk shard hunt's instrument): read
                                      # blas_tri, chunk_base and the reordered index stream straight back and memcmp
                                      # against the CPU plan, so "the GPU sees wrong remap DATA" is answerable without
                                      # touching a shader — on both backends, because the question is about the
                                      # STREAM and the two backends stream differently. It cost one bit in
                                      # `vk::stage`: every staged buffer now carries TRANSFER_SRC, UNCONDITIONALLY
                                      # rather than under the lever, because a diagnostic that only runs against a
                                      # specially-built buffer set has a subject that is not the shipping
                                      # configuration. Found by running it — the validation layer named the missing
                                      # usage bit exactly, the `--gpu-debug` lesson yet again.
                                      # MEASURED (RADV, 800x600, 18 arms, 0 failures): chunks at the 64k default read
                                      # procedural 2, --stress 200 3, Sponza 10, vokselia 52, rungholt 161, smlp 198,
                                      # the helmet 1 (under cap — the NOTE); compaction is 850 -> 648 MB on smlp,
                                      # 1014 -> 674 on rungholt, 283 -> 185 on vokselia. EVERY SCENE'S RADIANCE A/B
                                      # IS UNMOVED TO THE DIGIT from the pre-split recording (0.045 / 0.008 / 0.020 /
                                      # 0.007 / 0.007 / 0.022 / 0.006%), and unmoved ACROSS caps (smlp reads 0.006%
                                      # at 65536, at 4096, and at --no-blas-split). Both FR_VK_RES parities,
                                      # --sw-rays, --no-ftree, --no-wide-levels and llvmpipe all pass; llvmpipe
                                      # additionally EXERCISES the degenerate-compaction branch for free (it reports
                                      # no size reduction, so every chunk takes the CLONE arm — `compacted from 7` at
                                      # 7 MB).
                                      # ONE REAL EFFECT, and it attributes cleanly: the split perturbs V7's same-seed
                                      # image ONLY on CANDIDATE-LOOP scenes. vokselia at 52 chunks reads EXACT 0.00e0
                                      # with 0 hot channels, and so do the helmet and the procedural scene at every
                                      # cap — an opaque scene has no candidate loop, so re-partitioning changes
                                      # nothing at all. The cutout/tint scenes move a handful of channels and the
                                      # count grows with the chunk count: smlp 7.47e-9 / 0 hot at --no-blas-split ->
                                      # 2.76e-8 / 2 hot at 198 chunks -> 6.32e-8 / 4 hot at 2053; rungholt 3.25e-9 / 0
                                      # -> 5.99e-8 / 3. That is the DOCUMENTED two-intersector class with a second
                                      # source added: an alpha/tint candidate loop's enumeration order depends on the
                                      # acceleration structure's partition exactly as it already depended on TMin, so
                                      # a triangle at a chunk boundary can be surfaced in a different order. It is why
                                      # the gate is a bounded hot COUNT rather than an absolute limit (4 channels of
                                      # 1.44M), and why M3c's recorded 7.47e-9/3.25e-9 are now the --no-blas-split
                                      # numbers rather than the default ones.
                                      # TEETH, three planted and exercised: every instance reporting InstanceID 0 (the
                                      # M3b bug's exact shape) takes radiance 0.045% -> **17.932%** while the
                                      # VISIBILITY gate stays perfectly clean at class-mismatch 0 — the reason that
                                      # bug survived its first run, reproduced deliberately; unaligned arena offsets
                                      # are named by validation (`offset (20288) must be a multiple of 256`); and
                                      # rebasing a chunk's indices WITHOUT sliding its vertex window reads 341148 of
                                      # 341393 hit pixels wrong at radiance 60.520%. A fourth (every chunk pointed at
                                      # the first chunk's index slice) HUNG the build rather than failing, which is
                                      # its own answer: garbage vertex ids make absurd AABBs.
                                      # M3j — BC7, IN BOTH ARMS (2026-08-11), and the reason it is not
                                      # "purely a memory question" the way the plan filed it. The
                                      # session default is `Gpu(Fast)`, so a backend with only the ispc
                                      # arm would have to choose, for its DEFAULT session, between a
                                      # ~20-second encode stall and silently ignoring the flag — and
                                      # D3D12's own rule is that neither is acceptable (an encoder that
                                      # fails to construct falls back LOUDLY to RGBA8, never to an
                                      # implicit CPU encode). So the GPU encoder is the port, and the
                                      # CPU arm rides along as the independent cross-check it is there.
                                      # THE KERNEL IS NOT PORTED. `shaders/bc7enc.hlsl` compiles to
                                      # SPIR-V exactly as it compiles to DXIL, and the ONE thing a
                                      # second host cost it is a byte-offset WINDOW (`src_off`/
                                      # `dst_off`, ex-`_pad` plus one dword): D3D12 slides root SRV/UAV
                                      # virtual addresses per band, while Vulkan binds one descriptor
                                      # per buffer for a whole batch of levels and moves the window in
                                      # the constants — a per-dispatch descriptor rebind would be a
                                      # descriptor-set allocation per mip, and a dynamic-offset binding
                                      # would make the DERIVED layout special-case one slot. D3D12
                                      # passes 0 for both, which is exact integer arithmetic, so both
                                      # hosts encode the same blocks (`Num32BitValues` 8 -> 9 is the
                                      # whole D3D12-side change). `Quality::effort` moved to src/bc7.rs
                                      # for the same reason: one kernel, two hosts, one table.
                                      # THREE DIFFERENCES FROM THE D3D12 HOST, and only three.
                                      # (a) NO BANDING — that path exists because an upload heap's rows
                                      # are 256-byte-pitch-aligned and one 4K mip does not fit a ring;
                                      # here the texture batch loop already guarantees a whole chain
                                      # fits, so one dispatch covers a whole level at a TIGHT `mw * 4`
                                      # source pitch and the kernel's own clamp edge-replicates the
                                      # bottom and right edges (which is what makes a non-4-aligned mip
                                      # legal). (b) THE CONSTANTS ARE A UNIFORM BUFFER — DXC has no flag
                                      # to promote a cbuffer to push constants and `[[vk::push_constant]]`
                                      # would be an HLSL edit — rewritten between dispatches with
                                      # `vkCmdUpdateBuffer` and carrying the ladder's own lesson: BOTH
                                      # barriers, because a per-dispatch constant block needs a
                                      # WRITE-AFTER-READ edge as well as the read-after-write one. The
                                      # price is that a batch's dispatches SERIALIZE, which is fine (a
                                      # 4K level is a million blocks and fills the machine alone) and
                                      # cheaper than a descriptor set per mip. (c) TIGHT BLOCK ROWS —
                                      # `row_blocks` is `bw`, not the 256-byte-aligned `block_pitch`,
                                      # because `vkCmdCopyBufferToImage` takes `bufferRowLength` in
                                      # TEXELS with no pitch to honour, so the shear trap the kernel's
                                      # header names cannot arise here. MEASURED, and worth not
                                      # re-deriving: an explicit `blocks(w) * 4` row length is EXACTLY
                                      # equivalent to 0 for a compressed image (a zero row length is
                                      # "tightly packed per imageExtent", and that already rounds up to
                                      # whole blocks) — the explicit form is documentation, not a fix.
                                      # What is NOT free is the OFFSET: every staged level is padded to
                                      # 16 B, and a 4-aligned one is a validation error naming the copy.
                                      # NOTHING ABOUT THE DECISION IS RE-MADE: `bc7::should_compress` is
                                      # called verbatim, carve-out included — alpha-masked cutout masks
                                      # and height-carrying relief fields stay exact RGBA8 because the
                                      # intersector `.Load()`s a hard threshold out of them, which is a
                                      # VISIBILITY contract rather than a per-backend quality choice —
                                      # and `Texture::srgb` picks `BC7_SRGB_BLOCK` vs `BC7_UNORM_BLOCK`
                                      # exactly as it picks `_SRGB` vs `_UNORM`. `textureCompressionBC`
                                      # is the second ENABLED-WHEN-PRESENT core feature beside
                                      # `samplerAnisotropy`, and for the same reason: its absence lands
                                      # on `--no-bc7`, a shipping arm, rather than on a degradation
                                      # invented here. (On D3D12 neither is a feature bit at all —
                                      # anisotropy is a static sampler and BC7 is a format — so this is
                                      # one of the few places the two APIs' SHAPES differ rather than
                                      # their spelling.) The staging ring gained `STORAGE_BUFFER` usage
                                      # unconditionally, since the encoder reads it IN PLACE the way
                                      # D3D12's arm reads its upload heap as a root SRV; found by
                                      # running it and reading the validation layer, the FR_SPLIT_AUDIT
                                      # lesson repeated one commit later.
                                      # THE BATCH LOOP GAINED A SECOND BUDGET, and it is not the obvious
                                      # one: a batch is bounded by the ring in SOURCE bytes and by the
                                      # block buffer in ENCODED bytes, and encoded is NOT simply a
                                      # quarter of source — a 1x1 mip is 4 raw bytes and a whole 16-byte
                                      # block, so a table of tiny textures encodes to FOUR TIMES what it
                                      # staged and would overrun a block buffer sized off the ring.
                                      # V10 IS THE GATE, in two halves, and it exists because every
                                      # other number in the suite is blind to the encoder. V6's radiance
                                      # A/B is a 2% bar on a frame where textures are one term of the
                                      # shading, and its texture anti-vacuity probe asks whether the
                                      # table reached the shader, not whether its CONTENT survived —
                                      # MEASURED: a kernel that ignores `src_off` moves that A/B to
                                      # 1.132% and PASSES, while V10 reads 20.7 dB and fails. The
                                      # STRUCTURAL half is synthetic (`--check-gpu`'s `bc7-gpu`
                                      # transplanted, teeth for teeth: an all-even flat colour must
                                      # round-trip BIT-EXACT through the hardware decoder, every block
                                      # of a flat texture must be byte-identical to block 0, a gradient
                                      # ramp must clear 30 dB at all four effort tiers, and a
                                      # two-CLUSTER block must land within 6 LSB — which is what proves
                                      # the mode-1 arm FIRED and that its partition/anchor tables agree
                                      # with THIS hardware decoder), so it fires on every scene
                                      # INCLUDING the untextured procedural default where the fidelity
                                      # half has nothing to score. The FIDELITY half is M11's twin: the
                                      # session's own arm over the scene's own textures, decoded back
                                      # through `gfx::shaders::BC7_READ_HLSL` (moved out of gpu/trace.rs
                                      # — two gates scoring one encoder against two hardware decoders
                                      # must be reading the same text) into a plain BC7_UNORM image,
                                      # never `_SRGB`: the kernel must see raw code values or the
                                      # transfer function is charged to the encoder. RGB only, for M11's
                                      # reason. The 25 dB bar is a WIRING bar — pitch/footprint/format
                                      # errors land at 10-20 dB — and BOTH GATE PROBES ENCODE AT A
                                      # NON-ZERO AND DIFFERENT src/dst WINDOW (16 and 32), because a
                                      # gate that always passed 0 would score the kernel's arithmetic
                                      # and never the one thing a second host added to it, and equal
                                      # values would not separate a kernel that applied one offset to
                                      # both.
                                      # MEASURED (RADV, the 800x600 gate frame): DamagedHelmet 106.7 ->
                                      # 42.7 MB (4 of 5 — the 5th is its n2h normal map, whose ALPHA is
                                      # the relief field), Sponza 490.7 -> 230.7 (66 of 93),
                                      # rungholt 42.7 -> 26.7 (1 of 2 — the other is its cutout
                                      # atlas), vokselia 21.3 -> 21.3 (0 of 1: its ONE texture is the
                                      # alpha-masked Minecraft atlas, so the carve-out excludes the
                                      # whole table — a scene where BC7 is armed and correctly does
                                      # nothing), san-miguel-low-poly 511.5 ->
                                      # 405.6 (77 of 313 — the odd-dim carve-out, which is San Miguel's
                                      # arbitrary-dim research scans and the same 63%-of-opaque-MB
                                      # population the D3D12 note records). V10 structural reads flat
                                      # bit-exact / ramp >= 39.6 dB at all four efforts / two-cluster
                                      # 1 LSB on every scene and on llvmpipe; V10 fidelity reads worst
                                      # 42.6 dB on the helmet and **32.0 dB on san-miguel** — the SAME
                                      # worst-texture number D3D12's M11 records for its GPU arm at
                                      # `fast`, one kernel scored through two vendors' decoders. The
                                      # radiance A/B moves and stays tiny (helmet 0.020% unarmed ->
                                      # 0.016% armed, both arms), which is the liveness signal: an
                                      # encoder that produced nothing would leave it EXACTLY where
                                      # `--no-bc7` does. AND llvmpipe READS THE SAME FIDELITY NUMBERS
                                      # AS RADV TO THE DIGIT (helmet mean 0.414 LSB, max 82, worst 42.6
                                      # dB): BC7 DECODE is spec-bit-exact, so identical numbers across
                                      # two independent implementations say the ENCODER produced
                                      # identical blocks — the M3e integrator result in another
                                      # currency. Fifteen arms green (every scene, `--no-bc7`,
                                      # `--bc7-cpu` on two scenes, `--bc7-quality slow`, `--sw-rays`,
                                      # `FR_VK_STAGE=64k`, llvmpipe), zero validation errors.
                                      # TEETH, three planted and exercised: a kernel ignoring `src_off`
                                      # (V6 1.132% PASSES, V10 20.7 dB FAILS — the whole argument for
                                      # the stage in one line), one ignoring `dst_off` (V6 2.604%, V10
                                      # 18.4 dB), and staging offsets aligned to RGBA8's 4 instead of
                                      # BC7's 16, which validation names exactly
                                      # (VUID-vkCmdCopyBufferToImage-dstImage-07975). A fourth was
                                      # planted and turned out to be a mathematical identity
                                      # (`next_multiple_of(4)` IS `blocks(w) * 4` for every w), which is
                                      # its own reminder that a tooth must be shown to bite.
                                      # V8 — THE HEMISPHERE BOUNCE TIERS, the last render-path arm the Vulkan
                                      # tracer did not have (M3e, 2026-08-11). The H key's AO and GI: the batched
                                      # hemi wavefront (root -> cell levels -> leaf rays) and the one cs_compose splat
                                      # it feeds. Two halves, answering different questions.
                                      # THE PROBE HALF runs ONLY the hemisphere passes over a CPU-generated point set
                                      # (run_hemi_probes, the D3D12 peer's twin), so both sides integrate at the EXACT
                                      # same (o, n) and a statistical comparison against a 4096-sample CPU reference
                                      # means something. Its exact-zero gates: **psa-viol** — every point's projected
                                      # solid angle must account to pi, which is the integrator's own
                                      # partition-of-unity (empty cells contribute analytically, leaf cells by
                                      # sampling, and if the two do not tile the hemisphere the estimate is wrong by
                                      # whatever is missing, silently and in a way no image comparison localizes);
                                      # **false-empty** — an empty-cell claim re-validated with six real RayQuery rays,
                                      # i.e. the hemisphere's spelling of the false-sky bug; **tmin-overshoot** — a
                                      # leaf ray inherits tc from its OWN apex's tmin chain (never the primary tile's)
                                      # and a tmin=0 reference ray must not hit strictly inside the claimed ball.
                                      # THE FRAME HALF then runs one real wavefront frame with GI on, which is the only
                                      # thing that exercises what the probe path bypasses: cs_leaf's fb arm appending
                                      # one shading point per hit pixel, the batch loop draining them, and cs_compose
                                      # turning partial + ambw*ambient(H) into finite radiance. `pts == hits` is an
                                      # exact ACCOUNTING identity, not a tolerance — it catches an append that drops or
                                      # doubles.
                                      # MEASURED (RADV STRIX_HALO, 800x600, 157 probes x 8 seeds): psa-viol 0 with max
                                      # err 2.42e-5 | false-empty 0 | tmin-overshoot 0 | overflow 0, on every scene;
                                      # AO vs a 4096-sample cosine reference mean |d| 0.0067 procedural / 0.0018 smlp
                                      # / 0.0018 rungholt / 0.0017 vokselia (limit 0.02) with the SIGNED mean inside
                                      # +/-0.0005 (limit 0.005 — that estimator is unbiased, and a bias is the failure
                                      # a mean-absolute bar cannot see); GI vs the same depth-1 BOUNCE_Q policy on both
                                      # sides 3.02% / 0.97% / 1.13% / 1.11% (limit 5%). The GI frame reads
                                      # hit-px == hemi-points EXACTLY on every scene (341393 procedural, 341388 smlp).
                                      # AND ON llvmpipe THE FIXED-POINT ACCUMULATOR IS BIT-IDENTICAL TO RADV's:
                                      # DamagedHelmet reads the same psa max err 2.42e-5, the same leaf-rays 80384,
                                      # the same hemi-rays 5466520, the same AO 0.0069/+0.0005 and GI 2.15%. That is
                                      # hemi.hlsli's own design claim — integer atomics are order-independent, so a
                                      # queue-driven integrator is reproducible — measured across two wholly
                                      # independent Vulkan implementations rather than asserted.
                                      # TEETH, exercised rather than claimed: flipping the ROOT pass onto the wrong
                                      # descriptor parity (the one mistake the parity dance invites — the root writes
                                      # hqout, so it runs under the ODD variant and level 0 reads hq_a as hqin under
                                      # the EVEN one) fails FIVE gates at once — psa-viol 157/157 at max err 3.14
                                      # (the whole hemisphere unaccounted, because leaf cells contribute the PSA and
                                      # there were none), leaf-rays 0, AO mean |d| 0.586, GI 100%, and the frame
                                      # must-fire.
                                      # THE ANTI-VACUITY IS SPLIT, and the split is a measurement rather than a
                                      # preference. `leaf-rays > 0` is UNCONDITIONAL: the sampled tier fires on any
                                      # scene, and it is exactly what a mis-parity'd batch loses. `empty-cells > 0` —
                                      # the ANALYTIC tier — rides the suite's own `structural` predicate, because a
                                      # real scene can legitimately have no cell it can prove empty: DamagedHelmet at
                                      # its default pose reads empty-cells 0 HERE, and --check's CPU estimator reads
                                      # "0.0 cells empty, 64.0 rays" per point at the same pose, while this run's
                                      # 80384 rays over 8 seeds and 157 probes is 64.0 per probe EXACTLY. That
                                      # per-point agreement with the CPU is a stronger statement than the must-fire
                                      # would have been, which is why the gate is skipped there rather than relaxed.
                                      # NOT COVERED, and it is a real gap rather than an oversight: the CPU suite's
                                      # `cut-miss` counter, which re-traverses from the root to prove a cut-seeded
                                      # query saw everything the full tree would. That instrument lives in
                                      # hemi::VerifyCounters and has no GPU counterpart on EITHER backend.
                                      # WHAT M3e SPELLS DIFFERENTLY: two more set-0 variants, and they move FIVE slots
                                      # rather than the ladder's two — the cell ping-pong at u5/u6, the hemi leaf queue
                                      # at u7, the hemi cut pool at u9, and t0 back to the BINARY BVH, because the hemi
                                      # kernels compile frustum.hlsli's binary bound_query deliberately (short queries
                                      # lose on the wide tree: measured +35% there against -54% on the tile path). So
                                      # the binary tree is uploaded IN ADDITION to the wide one and both are live in
                                      # one frame. THE HEMI UNITS RE-DECLARE u5/u6/u7/u9 AS DIFFERENT STRUCTS —
                                      # HemiCellRec queues where the ladder has TileRec ones — and that is NOT a
                                      # conflict for the derived map, which keys on descriptor KIND (both are storage
                                      # buffers) rather than on the HLSL type: which is precisely why the variants are
                                      # a per-(set, register) OVERRIDE table over one shared base rather than a second
                                      # layout. The queues are sized to ONE BATCH of the worst-case fan-out
                                      # (HEMI_BATCH * 4^(HEMI_MAX_DEPTH-1) = 1,048,576 records, ~290 MB), and that
                                      # batch reset IS the memory bound — the reason the wavefront is batched at all.
                                      # THE LEVER-REACH BUG M3e FOUND, and it is this codebase's own recurring class:
                                      # --sw-rays, --no-sky-lod, --no-cloud-shadow and --no-wide-levels printed their
                                      # loud lines on Linux and ARMED NOTHING. M1 moved SW_RAYS / WIDE_LEVELS_ON /
                                      # set_cloud_shadow / set_sky_lod / set_waveviz into gfx::shaders with the shader
                                      # assembly, but their stores in main's lever block stayed behind #[cfg(windows)]
                                      # — so both Vulkan gates assembled the DEFAULT corpus while announcing otherwise.
                                      # Found by running --check-vk --sw-rays and watching the ladder it had just said
                                      # it would skip. The stores are portable now, and the arms they unlock are real
                                      # new coverage: --no-wide-levels (the SERIAL ladder — same structure, same
                                      # exact-zero image A/B, so the BFS/DFS order-independence claim holds on Vulkan
                                      # too) and --no-ftree (the binary-tree ladder, the other side of the t0 upload's
                                      # if/else) both PASS at 0.00e0. The one consequence worth stating: --sw-rays'
                                      # corpus genuinely declares NO acceleration structure, so V5's anti-vacuity
                                      # stands that one shape down under the lever — the expectation follows what the
                                      # units were compiled from, like every other scene-keyed must-fire here.
                                      # V9 — STRUCTURE REPLAY, and the only gate in this suite scored against
                                      # ITSELF (M3f, 2026-08-11). A frame whose basis bit-equals the previous
                                      # producing frame's skips the seed and the WHOLE level ladder and
                                      # re-dispatches the persisted terminal queues (qleaf/qsky/cut_pool +
                                      # CTR_LEAF/CTR_SKY/CTR_CUT). Every other stage here is scored against
                                      # something — the CPU, the reference kernel, a 4096-sample cosine
                                      # reference — so every bar is statistical; this one's claim is that the
                                      # terminal structure is a PURE FUNCTION of (scene, BVH, basis, rw, rh)
                                      # while spp/jitter/frame/fb/quality/clouds all ride the cbuffer, which
                                      # makes a replay not an approximation of a fresh trace but BIT-IDENTICAL
                                      # to one. MEASURED (RADV, 800x600): tbuf-diff 0 | info-diff 0 |
                                      # accum-diff 0 | sentinels 0 | leaf 768/768 sky 4/4 cut 65/65 | split
                                      # produce 257 replay 0 tiles 0/0, on procedural, san-miguel-low-poly,
                                      # DamagedHelmet, Sponza, rungholt, vokselia, --stress 200, both
                                      # FR_VK_RES parities, --no-ftree, --no-wide-levels, and llvmpipe.
                                      # ONE NEW KERNEL, and it is a different kernel rather than a flag on
                                      # cs_seed: cs_seed_replay zeroes every counter EXCEPT the three the
                                      # fills consume (plus CTR_SKY_PX, which is part of the same terminal
                                      # record). Everything else is a factoring — record_terminal_fills and
                                      # record_hemi_tail are now SHARED by the full path and the replay, so
                                      # "replay re-runs only the terminal fills" is a fact about the code
                                      # rather than a claim about two similar-looking blocks.
                                      # BOTH TEETH EXERCISED, and each catches what the others cannot.
                                      # (a) A FULL TRACE in the replay slot passes bit-identity, coverage AND
                                      # the terminal counts — all four quality gates — and ONLY the
                                      # ladder-did-not-run check fires (measured split 257 tiles 192/0). That
                                      # is the anti-vacuity half: a "replay" that quietly re-traced everything
                                      # is bit-identical BY CONSTRUCTION. (b) Dropping cs_seed_replay's
                                      # keep-set fails four gates, and the interesting half is which two it
                                      # does NOT: tbuf-diff and accum-diff stay 0, because a replay that
                                      # dispatched no terminal work leaves those buffers STALE-CORRECT from
                                      # the producing frame — the M3d lesson in another currency (an
                                      # operation that never happened compares clean), and the whole reason
                                      # the sentinel is re-flooded here rather than reusing V7's.
                                      # THE BUG IT FOUND ON THE WAY, in the code M3e had just shipped: the
                                      # Vulkan frame path never dispatched cs_clear_h, so an fb frame
                                      # integrated on top of whatever the previous fb frame (or probe run)
                                      # left in the fixed-point H accumulator — hbuf is written by ATOMIC ADD,
                                      # which is what makes the integrator order-independent and also what
                                      # makes an unzeroed frame double-count. Invisible to every gate that
                                      # existed: V8's frame half scores ACCOUNTING (hit-px == hemi-points) and
                                      # its A/Bs run through run_hemi_probes, which does clear. Found by
                                      # putting the two paths next to each other to factor them. The fb arm
                                      # added here is the gate that would have caught it (996888 channels on
                                      # procedural, 1014795 on smlp).
                                      # THE fb ARM'S BAR IS CHOSEN BY A CONTROL, and the distinction between
                                      # choosing the TIER and setting the BAR is the whole design. It renders
                                      # TWO fresh fb traces first: if they agree bitwise the integrator is
                                      # reproducible on this scene and the replay must agree bitwise too — no
                                      # tolerance at all, which is what every scene but one gets. If they do
                                      # not, the replay cannot be held to better. MEASURED:
                                      # san-miguel-low-poly reads a ~190-channel control out of 1.44M, and
                                      # FR_ABL=noalpha returns BOTH that and the replay diff to EXACTLY 0
                                      # while notrans does not — so the source is CUTOUT GEOMETRY INSIDE THE
                                      # GI HEMISPHERE (rungholt has cutout too and reads 0; every non-fb arm
                                      # reads 0 on every scene and every lever).
                                      # A CONTROL OF ZERO IS ONE BERNOULLI DRAW, NOT A PROOF — the defect
                                      # this arm SHIPPED with (M3f), found in M3g by a lever that made the
                                      # noise SMALLER rather than one that made it bigger. Through the
                                      # hardware intersector smlp's control never reads 0, so the tier is
                                      # never in doubt; under --sw-rays the same scene falls to 3-9 channels
                                      # and the control reads 0 about a QUARTER of the time — at which point
                                      # one sample declares a noisy frame reproducible and holds the replay
                                      # to an exact bar it cannot meet (measured 0/6/6/3 over four runs, the
                                      # replay drawn from the same distribution). The fix spends more traces
                                      # only where the answer matters: a nonzero control has already chosen
                                      # the tier and an exact match has already passed, so ONLY the ambiguous
                                      # corner — control 0 AND a differing replay — re-draws (up to 3 more,
                                      # any nonzero one demoting to the relaxed tier), which costs nothing on
                                      # the common path and drops the flake rate as p0^k. Observed live: the
                                      # re-draw fired in 2 of 6 smlp runs and rescued both.
                                      # AND THE M3f ATTRIBUTION IS CORRECTED BY THAT SAME MEASUREMENT: it is
                                      # NOT the driver's RayQuery candidate enumeration order. --sw-rays
                                      # replaces every candidate loop with our own fixed-order walk of our
                                      # own BVH and the effect SURVIVES, same noalpha attribution; the two
                                      # fresh traces there additionally do identical WORK (points, rays,
                                      # empty cells, overflow, AND cut fallbacks all equal — the last added
                                      # to the tuple to rule out a per-batch hemi cut-pool exhaustion, the
                                      # one way two traces could legitimately traverse differently), so it is
                                      # the same rays shaded to slightly different values. Mechanism OPEN, in
                                      # the shared HLSL rather than either backend, and naming it needs an
                                      # instrument neither suite has. The relaxed tier's bar is an ABSOLUTE
                                      # fraction of channels and deliberately NOT "no worse than the control",
                                      # because a real defect inflates BOTH: with clear_h removed the smlp
                                      # arm reads fd 1014795 against a control of 1014789, so a
                                      # relative-to-control bar would have PASSED the exact bug the arm exists
                                      # to catch, while the absolute bar fails it 705x over. What LICENSES
                                      # the relaxation is a separate assertion: the two fresh traces must have
                                      # done the same WORK (identical point/ray/empty-cell/overflow counts —
                                      # measured 341388/2130500/832792/0 both times), i.e. the difference is
                                      # fp ordering inside one traversal rather than a different traversal.
                                      # STILL NOT COVERED: the AUTO predicate. D3D12 keeps a last_struct key
                                      # and selects inside record_frame; this backend has no per-frame driver
                                      # yet, so that lands with the presenter rather than being written now as
                                      # a field nothing reads — the caller proves the bit-equality, and V9 is
                                      # that caller.
                                      # V11 — FSR3 UPSCALING OVER THE STOCK ffx_vk BACKEND (B1,
                                      # 2026-08-11): the first VENDOR upscaler on this backend, and
                                      # the first thing here that is not our own code running. B0
                                      # compiled FidelityFX 1.1.4's seven backend-neutral TUs; B1
                                      # adds the eighth (`src/backends/vk/ffx_vk.cpp`, one 233 KB
                                      # unit — the frame-interpolation half stays uncompiled),
                                      # `shim/ffx_fsr3_vk.cpp`, `src/vk/fsr3.rs`, and this gate. It
                                      # is CPU-FED from `accum` + `GBufs::new_slim` — an upscaler is
                                      # a function of the G-buffer and the CPU renderer produces one
                                      # on every platform — so it needs no G-buffer pack from the
                                      # Vulkan tracer and no presentation stage.
                                      # PORTED FROM ../quinlight-player/cpp/fsr3_shim.cpp, the
                                      # working reference (the same tree the DLSS/FG/XeSS shims and
                                      # `ffx_msvc_compat.h` came from). THREE FACTS FROM IT THAT THE
                                      # PUBLIC FFX HEADERS DO NOT STATE, each confirmed here: (1)
                                      # `ffx_vk.cpp` does NOT link standalone — `nm` shows an
                                      # undefined `ffxSetFrameGenerationConfigToSwapchainVK`, which
                                      # lives in the FI swapchain TUs; a four-line stub is the whole
                                      # cost of leaving that half out; (2) the FFX scratch buffer
                                      # must be ZERO-INITIALISED (`calloc`) — `ffxGetInterfaceVK`
                                      # reads a refCount at offset 0 BEFORE clearing and rejects
                                      # non-zero as an already-live context; (3) the caller must
                                      # create and pass THREE persistent shared images
                                      # (dilatedDepth R32_SFLOAT, dilatedMotionVectors R16G16_SFLOAT,
                                      # reconstructedPrevNearestDepth R32_UINT, all at render res,
                                      # all resting GENERAL) — FFX's cross-frame temporal state,
                                      # listed in the dispatch struct beside the genuinely optional
                                      # reactive masks. Also: `fpMessage` is not optional in practice
                                      # (without it creation failures collapse to a bare code), and
                                      # FFX rebuilds its own views from the FfxResourceDescription's
                                      # format and ignores the caller's, so those formats must name
                                      # what was actually allocated.
                                      # OUR DIVERGENCES FROM THE REFERENCE, each because this
                                      # renderer's wire differs: R32 reversed-Z clip depth
                                      # (`xess::view_z_to_clip_depth`) with ENABLE_DEPTH_INVERTED
                                      # where quinlight feeds an R16 pseudo-depth and sets neither;
                                      # AUTO_EXPOSURE deliberately OFF (our tonemap is anchored at a
                                      # fixed paper white); a fixed 1000/60 ms clock rather than a
                                      # wall one (the `nrd_gpu::NOMINAL_DT_MS` rule — a
                                      # deterministic gate must not put a real timer into a vendor
                                      # library's internal curves); and IMAGES OWNED IN RUST via
                                      # `ash` rather than in the shim, so the shim calls only `ffx*`
                                      # and image lifetime sits beside every other Vulkan resource.
                                      # THE FIRST *LINKED* VULKAN DEPENDENCY IN THIS TREE, stated
                                      # rather than discovered: `src/vk/` dlopens everything, but
                                      # `ffx_vk.o` leaves nine `vk*` symbols undefined
                                      # (vkCreateBuffer, vkGetPhysicalDeviceMemoryProperties,
                                      # vkGetPhysicalDeviceFeatures2, ... — the list is in build.rs)
                                      # so it needs the loader at link time. Scoped to
                                      # `cfg(ffx_fsr3_src)`, so a bare checkout still links nothing.
                                      # DO NOT delete the link-lib on the evidence of `ldd`: until
                                      # something calls the shim, `--gc-sections` drops the objects
                                      # and `--as-needed` then drops libvulkan from DT_NEEDED.
                                      # THE GCC VECTORIZER WORKAROUND IS TWO FLAGS, both
                                      # `flag_if_supported`: `-fno-tree-slp-vectorize` and
                                      # `-fno-tree-vectorize`. `CreateBackendContextVK` zeroes a run
                                      # of fields in an `alignas(32)` context carved from a 16-byte-
                                      # aligned scratch buffer, and GCC at -O2+ folds them into an
                                      # aligned 128-bit store on a misaligned address (#GP). This
                                      # box is GCC 15.2, i.e. exactly the compiler concerned.
                                      # DEVICE FEATURES GAINED FOR FFX AND FOR NOTHING OF OURS
                                      # (`vk/device.rs`), all enabled-when-present: shaderFloat16 /
                                      # shaderInt16 / shaderInt8 / 16-bit storage / storage-image
                                      # extended+write-without-format, because **FFX selects its
                                      # fp16 shader permutations from what the device SUPPORTS, not
                                      # from what we ENABLED** — so a supported-but-unenabled
                                      # feature is not a missing optimisation but SPIR-V the device
                                      # rejects; and `VK_KHR_get_memory_requirements2`, which must
                                      # be enabled EXPLICITLY even though it is core since 1.1 (we
                                      # require 1.3), because FFX calls the KHR-suffixed alias whose
                                      # proc-addr is null unless the extension name is enabled — a
                                      # jump to zero inside context creation. Our own corpus declares
                                      # none of these, and the bit-identical V6/V7/V9 gates are the
                                      # proof: every recorded number is unmoved (V6 0.045%, V7
                                      # 0.00e0/hot 0, V8 0.0067 / 3.02%, V9 all zero).
                                      # TWO BUGS FOUND ON THE WAY, neither in FFX. (a)
                                      # `mem_type_index` took the FIRST type satisfying the wanted
                                      # flags without excluding `DEVICE_COHERENT_BIT_AMD`, which the
                                      # spec forbids allocating from unless `deviceCoherentMemory`
                                      # is enabled — latent since M2b, and reachable only by an
                                      # allocation whose `memoryTypeBits` excluded the plain types.
                                      # It is a rejection now. (b) `build_ffx_fsr3`'s
                                      # `#[cfg(not(windows))]` describes the HOST, so
                                      # cross-compiling TO Windows FROM Linux ran it and tried to
                                      # build the FFX SDK with clang-cl — a latent B0 defect that
                                      # could only surface once the SDK was actually fetched (the
                                      # absent-sentinel check returned first before that). The guard
                                      # is `CARGO_CFG_TARGET_OS` now, which is the question being
                                      # asked; `tools/win-cross-check.sh` is what caught it.
                                      # ONE KNOWN-ACCEPT, and it is a real cost: `VK_AMD_device_
                                      # coherent_memory` is enabled when present PURELY to make
                                      # FFX's own allocations legal. MEASURED on RADV STRIX_HALO,
                                      # FFX allocates from memory type 7 (DEVICE_LOCAL |
                                      # DEVICE_COHERENT_AMD) while types 0/1/3/4 are plain
                                      # DEVICE_LOCAL and sit before it — the signature of a scan
                                      # keeping the LAST match. That memory is uncached, so FFX's
                                      # internal FSR3 working set lands in a slower class on AMD;
                                      # nothing of ours follows it there (see (a)). Also unresolved
                                      # and FFX's: validation WARNS that its own `rw_luma_history`
                                      # shader declares Rgba8 against an RGBA16F view.
                                      # WHAT V11 ASSERTS, AND WHAT IT DELIBERATELY DOES NOT.
                                      # Asserted: the context creates, the dispatch returns FFX_OK,
                                      # the output is finite and fully written, validation is clean.
                                      # REPORTED ONLY: the quality comparison — and that is a defect
                                      # in the obvious METRIC, not in the upscaler. Mean |d| against
                                      # a CONVERGED reference rewards blur and punishes sharpening,
                                      # and it is strongly scene-dependent (the `--spp` image-A/B
                                      # lesson again): MEASURED, FSR3 beats a bilinear control on
                                      # the procedural scene (0.01400 vs 0.01489) and LOSES on
                                      # san-miguel (0.00569 vs 0.00379) with identical wiring,
                                      # because san-miguel's 1-spp input is ~4x cleaner so FFX's own
                                      # reconstruction bias dominates the noise it removes. The
                                      # reset-every-frame and bogus-motion controls invert with it.
                                      # Asserting any of them would make the gate a statement about
                                      # which scene it was pointed at. OWED: a detail-preserving
                                      # metric (the `--check-oidn` Laplacian shape), and settling
                                      # AUTO_EXPOSURE/preExposure, which this arm leaves off and the
                                      # reference always sets. `FR_VK_FSR3_JITTER=raw|neg` is the
                                      # lever for the one convention the two FFX generations
                                      # disagree about — `fsr::UPSCALE_MV_SIGN` (1,1) matches the
                                      # reference exactly, but `fsr::JITTER_SIGN` is +1 here and the
                                      # reference negates; MEASURED, neg is slightly better on
                                      # san-miguel (0.00524 vs 0.00569) and neither arm changes any
                                      # verdict, so it is recorded rather than flipped.
                                      # SKIPS, both loud: no SDK on disk (the fetch command is
                                      # named), and a SOFTWARE device — llvmpipe fails
                                      # `ffxFsr3UpscalerContextCreate` outright, which is an
                                      # environment fact, while RADV creates and dispatches.
                                      # Touch shim/ffx_fsr3_vk.* / src/vk/fsr3.rs / build.rs's
                                      # build_ffx_fsr3 / vk/device.rs's feature list -> run
                                      # --check-vk on the procedural scene AND san-miguel, on
                                      # llvmpipe, with the SDK moved aside (the degrade is the half
                                      # nothing else covers), plus --check-fsr, --check-spirv,
                                      # --check, cargo test and tools/win-cross-check.sh
                                      # V12 — THE G-BUFFER PACK AND THE PREV-CAMERA MOTION
                                      # VECTORS (B2, 2026-08-11): the first stage here whose
                                      # subject is not the picture but the WIRE the vendor stack
                                      # reads. `--check-gpu`'s M7 / `--check-dxr`'s T4
                                      # transplanted — same 533x400 odd dims, same 0.02*diag
                                      # forward dolly, same `unpack_gbuf_bytes`, same
                                      # `dlss::mv_selftest`, zero new tolerances.
                                      # TWO FLAGS HAD NEVER RUN ON THIS BACKEND, and the second
                                      # is the one worth naming: `FLAG_GBUF`, and
                                      # `FLAG_HAS_PREV` — which means **the cbuffer's four
                                      # prev-camera rows had never been non-zero here**, a region
                                      # of the 4608-byte -fvk-use-dx-layout packing no gate had
                                      # ever read, and one `gbuf_write_hit` dereferences directly
                                      # (`prev_origin`, `prev_forward`, and `prev_right`/
                                      # `prev_up` through `project_prev`). The pack stores are
                                      # the easy half; that block is the new ground.
                                      # A SECOND TRACER, not a flag on V6's, and for attribution
                                      # rather than tidiness: every number V6/V7/V8/V9/V11 record
                                      # stays unmoved BY CONSTRUCTION, and the ODD dims survive
                                      # (533x400 is what catches a stride assumption; the
                                      # V-suite's own 800x600 cannot). It shares the uploaded
                                      # `VkScene`/`VkTextures`, so the cost is one more DXC pass
                                      # plus the per-pixel buffers — measured, the whole suite
                                      # goes 22 -> 26.7 s on the procedural scene.
                                      # BOTH PACK HALVES ARM, and core-only was tried first and
                                      # does not work: `dlss::mv_selftest` ANDs three arms and
                                      # the third (`dlss::sky_dir_check`, dlss.rs:573) reads the
                                      # NORMAL lane, which lives in EXT — `unpack_gbuf_bytes`
                                      # fills the guide fields with zero when handed `ext: None`,
                                      # so every sky pixel would read `got = (0,0,0)`, `err = 1.0`
                                      # against a 5e-3 limit, and the gate would fail for a reason
                                      # unrelated to the pack. Splitting the function to dodge
                                      # that would cost the property the whole transplant rests on
                                      # (the EXACT existing gate, zero new tolerances), so ext
                                      # arms with core — which is also what M7 does, via
                                      # `force_gbuf_ext(true)`, for its own reason: the consumer
                                      # is a CPU readback and not a feed kernel. `fsr_sig` stays
                                      # OFF (a shade-side export this backend does not capture),
                                      # so `core.w` and the sig lanes stay 0 — which
                                      # `unpack_gbuf_bytes` drops anyway.
                                      # `gbuf_full` is a CONSTRUCTION argument on `VkTracer::new`,
                                      # the D3D12 shape: it sizes the two buffers AND is what
                                      # `write_cb` reads to arm `FLAG_GBUF`, so the flag cannot
                                      # arm over a stride-sized pack. That matters because a
                                      # storage buffer bound with WHOLE_SIZE has no bounds check
                                      # and the fallback stand-in (`VkScene::dummy`) is SIXTEEN
                                      # BYTES against a 72 B/px stride — the flag is memory
                                      # safety, not an optimization (gfx/frame.rs's own note).
                                      # The pack is bound in every descriptor variant whether or
                                      # not it is armed, since an unarmed tracer's pair is
                                      # stride-sized rather than absent: that costs nothing and
                                      # removes one way to be unlucky.
                                      # `unpack_gbuf_bytes` LOST ITS `cfg(windows)`, which it only
                                      # ever carried because it spelled `gpu::trace::GBUF_STRIDE`
                                      # — the same constant as `gfx::frame::GBUF_STRIDE` through
                                      # the glob re-export at gpu/trace.rs:39. Swapping the path
                                      # is the whole change; a Vulkan-local copy would have been
                                      # the transcription hazard the derived layout exists to
                                      # remove, on the one function whose entire job is to agree
                                      # with the wire.
                                      # ASSERTS: `dlss::mv_selftest` (all three arms), coverage
                                      # (every px view_z > 0; sky depth BIT-EQUAL far, never a
                                      # tolerance), the ext ripple lane (0.0 or WATER_RIPPLE_AMP
                                      # within 1e-3, sky exactly 0.0), the sky must-fire under
                                      # `structural`, and an ANTI-VACUITY count of non-zero MVs.
                                      # That last is not redundant with mv_selftest: a pack that
                                      # was never written AT ALL reads as zeros and would trip the
                                      # coverage arm first, reporting "the pack is broken" for
                                      # what is really "the pack is absent" (the M3d lesson — an
                                      # operation that never happened compares clean). The two
                                      # arms measurably separate: tooth 2 below fails
                                      # `mv_selftest` while the MV count stays at a full frame.
                                      # MEASURED (RADV, 533x400): view-z<=0 0 | sky-depth-off 0
                                      # (sky px 61561) | ripple bad 0 | mv non-zero 209097 |
                                      # mv/depth/matrix OK, with mv_selftest reading median
                                      # 1.528e-2 against a 1.697e-1 limit, p90 1.421e-1 against
                                      # 1.697e0, the matrix/ray identity OK and the sky arm
                                      # 3.443e-4 against 5e-3. Green on procedural,
                                      # san-miguel-low-poly, rungholt, both FR_VK_RES parities,
                                      # `--sw-rays` and llvmpipe (which reads 210077 non-zero MVs
                                      # — the two-intersector class at silhouettes). **RUNGHOLT
                                      # PROVES THE RIPPLE LANE NON-VACUOUS at water px 1551,
                                      # against D3D12's own recorded 1552** — one pixel apart on
                                      # two backends and two intersectors; the procedural and
                                      # san-miguel poses read 0 water px and gate nothing there.
                                      # TEETH, two fired: `prev_cam: None` on frame B collapses
                                      # the plane (median 2.085e0 = 12x over, AND `mv non-zero 0`
                                      # — both arms, which is what proves FLAG_HAS_PREV is the
                                      # live path); swapping `prev_right`/`prev_up` in
                                      # `with_frame` reads median 8.156e0 = 48x over while the MV
                                      # count stays 213200, i.e. a WRONG pack rather than an
                                      # absent one. The third — arming FLAG_GBUF over the 16-byte
                                      # dummy — was DELIBERATELY NOT FIRED and the reason is
                                      # recorded rather than the result: `robustBufferAccess` is
                                      # not enabled, so that write is genuinely undefined, and on
                                      # this box the Vulkan device is the iGPU driving the
                                      # display. The claim it would test is a spec fact already
                                      # documented on the D3D12 side; a hang is not worth
                                      # re-deriving it. Fire it on a machine with a discrete
                                      # secondary if it is ever wanted.
                                      # NOT COVERED, said rather than implied: the feed kernel.
                                      # V12 stops at the pack, exactly as M7 stops before M8 —
                                      # it never calls a feed and touches no upscaler resource.
                                      # Touch `VkTracer`'s gbuf pair / `write_cb`'s flag row /
                                      # `unpack_gbuf_bytes` -> run --check (goldens byte-identical
                                      # — B2 touches no shading path), cargo test, --check-vk on
                                      # procedural + san-miguel-low-poly + rungholt, both
                                      # FR_VK_RES parities, --sw-rays, llvmpipe, --check-spirv and
                                      # tools/win-cross-check.sh
                                      # V13 — THE GPU-FED FEED (B3, 2026-08-11): the tracer's own
                                      # pack and 1-spp radiance reaching FSR3 through
                                      # `cs_feed_xess` instead of through three host uploads. B1
                                      # put FSR3 on this backend CPU-fed, so the Vulkan TRACER fed
                                      # nothing; B2 gave it the pack those planes come from; this
                                      # closes the loop, and it is the shape every --gpu/--dxr
                                      # session on D3D12 has used since the pack split.
                                      # FOUR THINGS THE CODE ALREADY GAVE, which is why one
                                      # milestone is one pipeline and two methods: `TraceSources.
                                      # feed` was already assembled portably (the M1 payoff);
                                      # `layout.rs` already maps StorageImage -> STORAGE_IMAGE and
                                      # sizes the pool by `desc_type`, so THREE NEW IMAGES ENTER
                                      # THE LAYOUT WITH NO EDIT (M3a paying out a third time); the
                                      # storage-image descriptor write already EXISTED at u14 for
                                      # `cs_resolve`, so this extends a mechanism rather than
                                      # inventing one (the plan had called it the last unwritten
                                      # resource class, and exploration corrected that before any
                                      # code); and `vk::fsr3::Img` already created its planes with
                                      # STORAGE usage in exactly the formats the kernel writes.
                                      # ONE ENTRY POINT of the several `feed.hlsl` declares, which
                                      # is what keeps the footprint to three bindings: DXC drops
                                      # every `feed_*` the kernel does not reference, so the unit
                                      # contributes u16 (RGBA16F colour), u18 (R32F reversed-Z clip
                                      # depth) and u19 (RG16F mvec) and NOTHING else — its other
                                      # inputs (`accum` u0, `gbuf` u15 from trace_common) are the
                                      # SAME declarations the tracer already binds, so they join no
                                      # new slot. u16/u18/u19 are unclaimed by the tracer family
                                      # (ladder u5..u9, hemi u10..u13, pack u15/u32) and the map's
                                      # conflict detector is what would say so at build time.
                                      # OPTIONAL BY CONSTRUCTION (`TracerOpts::feed`), and that is
                                      # not thrift: the map is DERIVED, so compiling the unit
                                      # unconditionally would move V5's slot count 46 -> 49 and
                                      # every tracer's pool sizing — i.e. it would move recorded
                                      # numbers for stages that never dispatch a feed. `TracerOpts`
                                      # also retires the positional pile `VkTracer::new`'s own
                                      # comment warned about at nine arguments; `feed` IMPLIES
                                      # `gbuf_full` and the constructor FORCES it rather than
                                      # asserting, since a feed over a stride-sized pack would read
                                      # one texel's allocation per pixel.
                                      # THE COMPARISON IS BETWEEN TWO ROUTES, NOT TWO RENDERERS,
                                      # and that is what makes it assertable where V11's quality
                                      # claim is not. V11 can only REPORT whether FSR3 beats a
                                      # bilinear control, because that answer is scene-dependent
                                      # (it wins on procedural and loses on san-miguel with
                                      # identical wiring). Here BOTH arms consume the SAME frame
                                      # from the SAME Vulkan tracer — one through `record_feed`,
                                      # one through `Fsr3::frame`'s host upload of the identical
                                      # readback — so any difference is wiring and a tight bar is
                                      # honest. Scoring a CPU-RENDERED arm against a GPU-rendered
                                      # one would not be this test at all: different intersector,
                                      # different RNG stream, and the feed route would be invisible
                                      # inside that difference. TWO FFX CONTEXTS, not one run
                                      # twice, because FSR3 is TEMPORAL and one context would carry
                                      # the first route's history into the second.
                                      # `Fsr3::frame_fed` is `frame`'s sibling: it swaps the three
                                      # buffer->image copies for transition-to-GENERAL, the caller's
                                      # recorded kernel, and transition back — so both arms hand FFX
                                      # images in the same layout BY CONSTRUCTION and only the
                                      # writer differs — and both end in ONE shared `record_ffx`,
                                      # which is what stops the dispatch desc drifting between the
                                      # two arms of a comparison that would then be measuring
                                      # itself.
                                      # `gate_xess_feed` IS THE GATE, cfg-lifted exactly as B2
                                      # lifted `unpack_gbuf_bytes` and for the same reason (plain
                                      # slices, a `dlss::GBufs`, scalars; no D3D12 type in the
                                      # body) — zero new tolerances, the --check-gpu M8 bar
                                      # verbatim, with B2's now-portable unpacker supplying the
                                      # `gb2` oracle out of the pack readback. `mono16` came with
                                      # it, being its ulp metric.
                                      # ANTI-VACUITY IS A SENTINEL, NOT A ZERO CHECK: the three
                                      # images are written once and read once per frame, so a feed
                                      # that never dispatched leaves the PREVIOUS frame's contents,
                                      # which after frame 0 look entirely plausible (the M3d lesson
                                      # — an operation that never happened compares clean; V3's
                                      # 0xEEEEEEEE and V9's re-flooded sentinel exist for this).
                                      # 0xEE is read back off the DEPTH plane because R32F makes a
                                      # surviving word unambiguous where an f16 pair could be a
                                      # real value.
                                      # MEASURED (RADV, 400x300 -> 800x600, 8 frames): depth-ulp>4
                                      # 0 (max 2) | sky-not-0.0 0 (sky px 34498) | mvec-ulp>1 0 |
                                      # color-ulp>1 0, and gpu-fed vs cpu-fed mean |d| / mean
                                      # |gpu-fed| **0.00081** against a 2% bar — 0.00074 on
                                      # san-miguel-low-poly, 0.00072 on rungholt, 0.00081 under
                                      # --sw-rays and at the other FR_VK_RES parity. Every
                                      # pre-existing number unmoved (V5 46 slots / 45 pipelines, V6
                                      # 0.045%, V7 0.00e0 / hot 0, V8 AO 0.0067 / GI 3.02%, V9 all
                                      # zero, V11 0.01400, V12 mv median 1.528e-2). `FR_VK_MAP=1`
                                      # prints `wire_feed <- 15 write(s) over 5 set-0 variant(s)` =
                                      # 3 registers x 5 variants, which is the cheapest proof the
                                      # three bindings really are in the derived map (the V13
                                      # tracer does not print its own slot count; V5 reports V6's,
                                      # and that staying at 46 is the point).
                                      # THREE TEETH, all fired, and they SEPARATE — the B2
                                      # absent-vs-wrong distinction again: skipping `wire_feed`
                                      # trips BOTH arms (120000/120000 sentinel texels survive AND
                                      # every plane gate fails); SWAPPING the depth and mvec views
                                      # trips the plane gate alone (depth-ulp 85502, mvec-ulp
                                      # 180962, colour untouched at 0) with the sentinel SILENT,
                                      # which is the class a derived layout provably cannot catch
                                      # since both are STORAGE_IMAGE to Vulkan and only the VALUES
                                      # differ; and dropping `record_feed`'s `div_ceil` leaves
                                      # exactly 1600 texels = 400 x the 4 rows 300/8 discards,
                                      # caught by both. A FOURTH defect was found by the compiler
                                      # rather than planted: the dead-code warning on `wire_feed`
                                      # is what said the descriptors were never pointed at the
                                      # images, before a single gate ran.
                                      # `Fsr3::read_input`'s first draft named
                                      # SHADER_READ_ONLY_OPTIMAL as a `vkCmdCopyImageToBuffer`
                                      # source, which the spec forbids — the validation layer named
                                      # the VUID exactly rather than the copy returning stale
                                      # bytes, the good failure mode and the --gpu-debug argument
                                      # for the third time in this port.
                                      # COST: a third tracer takes --check-vk 26.7 -> 32.4 s, all
                                      # of it DXC. That buys "V6-V12 unmoved" structurally rather
                                      # than by care, and the odd-dimension coverage V12 needs.
                                      # STILL NOT ASSERTED, and B1's owed half only half retired:
                                      # this proves the two FEED ROUTES agree, not that the upscale
                                      # is GOOD. That still wants a detail-preserving metric (the
                                      # --check-oidn Laplacian shape) and a settled
                                      # AUTO_EXPOSURE/preExposure.
                                      # Touch `TracerOpts` / the feed unit list / `wire_feed` /
                                      # `record_feed` / `Fsr3::frame_fed`/`record_ffx`/`feed_views`
                                      # -> run --check (goldens byte-identical — B3 touches no
                                      # shading path), cargo test, --check-vk on procedural +
                                      # san-miguel-low-poly + rungholt, both FR_VK_RES parities,
                                      # --sw-rays, llvmpipe (V11/V13 SKIP), --check-spirv,
                                      # --check-fsr and tools/win-cross-check.sh
                                      # V14 — THE NRD BRIDGE (B4a, 2026-08-11): `cs_nrd_pack` and
                                      # `cs_nrd_out`, the front and back halves of the ONE denoiser
                                      # seam this renderer has, running on Vulkan with a passthrough
                                      # between them. The first denoiser-adjacent code on this
                                      # backend, and the first consumer of the sig lanes B2
                                      # allocated and deliberately left unarmed.
                                      # THE BRIDGE IS NRD's OWN, which is why this is the NRD path's
                                      # first half rather than a detour: the file is
                                      # nrd_bridge.hlsl, the descriptor set is NRD_FEED_SET, the
                                      # call is wire_nrd_feed, and FRD was the BORROWER
                                      # (gpu/frd_gpu.rs's own header — "FrdGpu carries NrdGpu's
                                      # exact plane contract, which is why arm_denoiser_for wires
                                      # BOTH arms through this one call"). The kernels are
                                      # engine-blind by design, so proving them needs no engine at
                                      # all — which is what makes this cheap enough to precede one.
                                      # IT JOINS THE TRACER FAMILY — no second layout, no new set
                                      # variant — and that is a property of the SOURCE:
                                      # `TraceSources::nrd_bridge()` pastes trace_common.hlsli, so
                                      # it declares the tracer's registers plus u17, u20, u23..u27
                                      # (all free) and u16/u18/u19, which are B3's feed images at
                                      # the SAME descriptor kind. It declares no `t` register at
                                      # all. So B4a is ONE conditional unit, B3's shape exactly,
                                      # and `nrd_bridge()` being a METHOD rather than a keyed
                                      # Option on this side meant the shared core changed by zero
                                      # lines.
                                      # ONE DIFFERENCE FROM D3D12, and it DELETES code: every
                                      # plane rests in GENERAL for its whole life. That layout is
                                      # legal for both SAMPLED_IMAGE and STORAGE_IMAGE, so the pass
                                      # sequence needs only memory barriers and never a layout
                                      # transition — D3D12's NPSR<->UA bracketing has no
                                      # counterpart here and must not be invented. It pays again at
                                      # readback: GENERAL is a legal copy SOURCE, unlike
                                      # SHADER_READ_ONLY_OPTIMAL, which B3 learned from the
                                      # validation layer.
                                      # THE SIG LANES ARE A PER-FRAME CELL, not a construction
                                      # flag, and that is what makes the invariance gateable:
                                      # FLAG_FSR_SIG and its dependents are cbuffer bits over an
                                      # assignment-only capture, so two traces one bit apart need
                                      # no recompile and no second tracer. `remod_exact` rides
                                      # ARMED with them, matching the D3D12 shipping default, so
                                      # the two backends' packs agree about what sig.w's high half
                                      # carries (m_d, on loan from shadow_t). nrd_rejitter stays
                                      # OFF — it is NVIDIA's Jacobian, engine-gated on D3D12 for a
                                      # stated reason, and no engine sits between the halves yet.
                                      # ONE TRACER FOR V13 AND V14, not the fourth B2/B3 each
                                      # added. The coupling is a FEATURE: the only thing V14 adds
                                      # to a frame is a cbuffer bit, so V13's figures holding
                                      # across the bridge's arrival IS the capture-invariance
                                      # claim, stated across two stages instead of asserted once —
                                      # and it saves a whole DXC pass (26.7 -> 30.5 s rather than
                                      # ~37).
                                      # THE CLAIM IS AN IDENTITY, not a tolerance: the recompose is
                                      # col = R + D_out*kd*m_d + S_out*f0 with R = base −
                                      # D_in*kd*m_d − S_in*f0, so with OUT == IN byte for byte the
                                      # correction is algebraically ZERO and col must collapse onto
                                      # base — exactly the colour cs_feed_xess wrote. That is N3
                                      # transplanted, and it is the strongest shape in the suite
                                      # because it compares bytes against a value the SAME device
                                      # produced moments earlier rather than against a model of one.
                                      # MEASURED (RADV, 400x300): colour byte-diff **0** on
                                      # procedural, san-miguel-low-poly, rungholt, both FR_VK_RES
                                      # parities and --sw-rays.
                                      # AND THE IDENTITY ALONE IS NOT ENOUGH — a planted tooth
                                      # PROVED it, which is the finding of this milestone. Skipping
                                      # the plane wiring entirely left the gate GREEN: with the
                                      # planes unbound the recompose reads zeros, D_out − D_in is
                                      # zero, col collapses onto base, and the identity holds FOR
                                      # THE WRONG REASON. A passthrough makes the delta zero BY
                                      # CONSTRUCTION, so the arm that scores the recompose can
                                      # never also score the pack. The missing half is asking
                                      # whether the pack's data ARRIVED — M3d's texture probe in
                                      # another currency — so V14 reads the three IN planes back
                                      # off the tracer's own images (which exist whether or not a
                                      # descriptor points at them, which is exactly what makes an
                                      # unwired bridge show up as an untouched plane) and requires
                                      # them non-trivial: measured in_diff 624463/960000 non-zero
                                      # bytes, in_spec 517082, in_viewz 479311/480000.
                                      # THE FOLD gets gated for free, and it is a half no feed
                                      # route reaches: the dirty pass is `poison_inputs`, which
                                      # floods ALL THREE engine planes rather than D3D12 F3's one,
                                      # and `cs_nrd_pack` writes the engine's depth and mvec guides
                                      # ITSELF (which is why an NRD frame runs no separate feed at
                                      # all) — so V13's own plane gate re-run afterwards scores the
                                      # fold. It comes back with V13's verdict to the digit
                                      # (depth-ulp>4 0 max 2, sky px 34498, mvec-ulp 0, color-ulp
                                      # 0) over bytes that were 0xEE a moment earlier.
                                      # THE INVARIANCE ARM (N6b transplanted): two traces across
                                      # the sig bit — accum BIT-IDENTICAL (measured 0 bytes), every
                                      # hit pixel carrying sig when armed (85280/85280), ZERO
                                      # leaking when disarmed, and m_d in (0, 1].
                                      # THE INERT NOTE FIRES ON EVERY COMMITTED POSE, and that is
                                      # worth stating rather than leaving a reader to infer: m_d is
                                      # sk = 1 − 0.157*sheen blended toward sk*dcav, only the
                                      # `fabric` class sets sheen at all (matclass.rs's
                                      # tela/carpet/individual vocabulary), and dcav needs the
                                      # detail window OPEN, i.e. a magnified surface. The
                                      # procedural scene has neither; san-miguel's NINE fabric
                                      # materials are its tablecloths and chair fabric — in the
                                      # scene, absent from the fitted overview every --check-vk run
                                      # uses. The pose that DOES move it is this tree's own
                                      # documented glassware close-up, `--cam
                                      # 0.71,1.55,0.45,0.71,1.25,-0.35` on san-miguel-low-poly,
                                      # where the arm reads **m_d [0.7700, 1.0000]** and the note
                                      # correctly stays silent. READ THE LOG, NOT THE EXIT CODE
                                      # there: the rest of the suite is known-red at that pose for
                                      # reasons predating this stage (mv_selftest median 3.156
                                      # against a 0.17 limit — the exact figure already recorded
                                      # for it — plus the candidate-loop divergences a
                                      # nearly-all-glass view amplifies), the same caveat
                                      # --dxr-sbt 3 already carries for the identical pose. V14's
                                      # two arms PASS there.
                                      # TEETH, four fired: skipping wire_nrd (identity green,
                                      # arrival 0/960000 on all three — the two arms separating is
                                      # the point); SWAPPING in_diff/in_spec at the descriptor
                                      # write, same kind and same format, so the class a derived
                                      # layout provably cannot catch (byte-diff 503359, and the
                                      # arrival counts SWAP, which NAMES the bug rather than merely
                                      # flagging it); dropping record_nrd_out's div_ceil (12784
                                      # bytes ~= 400 px x the 4 rows 300/8 discards x 8 B); and
                                      # decoding sig.w's LOW half instead of its high (reads
                                      # [0.0017, 5.0938] — that is ao_t, a world-space hit
                                      # distance, provably outside (0,1], which is what proves the
                                      # lane is genuinely decoded and the high half is the right
                                      # half).
                                      # NOT DONE, and it is the whole point of the split: no
                                      # ENGINE. B4b is libNRD.so (a Linux CMake build the installer
                                      # currently skips only because a D3D12 session cannot load
                                      # it — its own words), a libloading twin of nrd.rs's four
                                      # cfg(windows) loader sites (the struct transcription,
                                      # SpirvBindingOffsets, compute_shader_spirv and the
                                      # GetLibraryDesc gate are ALREADY portable and already
                                      # gated), and src/vk/nrd.rs — whose descriptor layout is the
                                      # first in this backend DECLARED by a foreign library at its
                                      # own binding offsets (SREG 0, BREG 2, UREG 3, TREG 20)
                                      # rather than derived from our SPIR-V, so binding_of's
                                      # never-a-literal rule does not apply there.
                                      # Touch `TracerOpts::nrd` / the bridge unit list /
                                      # `wire_nrd` / `record_nrd_*` / `NRD_REGS` / `read_nrd_plane`
                                      # / write_cb's sig arming -> run --check (goldens
                                      # byte-identical — B4a touches no shading path, only which
                                      # lanes the pack writes, and accum is bit-identical across
                                      # that by construction), cargo test, --check-vk on procedural
                                      # + san-miguel-low-poly + rungholt, both FR_VK_RES parities,
                                      # --sw-rays, llvmpipe, the glassware close-up above (for the
                                      # m_d arm, reading the log), --check-spirv, --check-fsr and
                                      # tools/win-cross-check.sh
                                      # V15 — A REAL NRD ENGINE (B4b-ii, 2026-08-12): ReBLUR
                                      # running between the bridge's two halves, i.e. the thing
                                      # the seam exists for. `src/vk/nrd.rs` is `NrdGpu`'s twin
                                      # and V15 is `--check-gpu`'s N4 transplanted with ZERO new
                                      # tolerances — same nine-frame protocol, same six
                                      # assertions, same thresholds — so the two backends'
                                      # numbers are directly comparable rather than merely
                                      # similar. MEASURED (RADV, 400x300 procedural): 14
                                      # pipelines, 8 dispatches on the RESTART frame and 32 at
                                      # peak (N1's 31 + one, and BOTH are reported because the
                                      # gap is the reset latch visible in the dispatch list), 37
                                      # pool sets, `differs 119241/120000`, **Laplacian 0.1089 ->
                                      # 0.0236 (a 78% drop)**, mean 0.3378 -> 0.3356 (0.7%),
                                      # temporal 0.00928 -> 0.00434, restart 0.00893.
                                      # THE SHARED HALF MOVED FIRST: `gfx::denoise` now owns the
                                      # plane vocabulary, `common_settings`, `reblur_settings` and
                                      # `NOMINAL_DT_MS`, and BOTH recorders route through it —
                                      # `nrd_gpu.rs`'s `reg_for` and its plane-format array are
                                      # that map now, not a second opinion. The vocabulary existed
                                      # because it was MISSING: `vk/tracer.rs` cited `P_*`
                                      # constants in `nrd_gpu.rs` that never existed, so B4a had
                                      # to re-invent the names and B4b-ii would have been the
                                      # third naming of one thing. `denoise-vocab` in `--check`
                                      # gates it on every platform (ALL in index order, the
                                      # ResourceType map INJECTIVE and round-tripping, pool types
                                      # resolving to no plane, unique names, the formats, the dt
                                      # clamp band incl. NaN, the denoising-range lockstep with
                                      # cs_nrd_out, and a 4x2 MV-scale test — square dimensions
                                      # cannot tell {1/rw, 1/rh} from its transpose). So a LINUX
                                      # box now gates the matrix convention and the UV motion-
                                      # vector scale the WINDOWS recorder depends on.
                                      # `DispatchDesc` gained Clone/Copy, which makes the snapshot
                                      # both recorders must take ONE call and deletes D3D12's
                                      # 14-line hand copy.
                                      # FOUR DIFFERENCES FROM THE D3D12 TWIN, each API and not
                                      # design: PER-PIPELINE set layouts from
                                      # `PipelineDesc::resource_ranges` (D3D12 sizes one table to
                                      # the per-set maxima and leaves the surplus slots stale —
                                      # legal, and legal here only via
                                      # `descriptorBindingPartiallyBound`; the exact counts make
                                      # nothing stale by construction, at the price of set-1
                                      # compatibility breaking per pipeline, so both sets bind per
                                      # dispatch — one call); NO layout transitions and no state
                                      # tracker (everything rests in GENERAL, so D3D12's
                                      # NPSR<->UA bracketing has no counterpart and must not be
                                      # invented, and one global memory barrier per dispatch
                                      # replaces it — the narrowing A/B measured a WASH on D3D12);
                                      # the CB is a DYNAMIC uniform buffer at a per-dispatch
                                      # offset, persistently mapped (a plain host write is legal
                                      # here where the ladder needed `vkCmdUpdateBuffer` and its
                                      # easy-to-omit write-after-read edge, because NRD's whole
                                      # dispatch list is known BEFORE any of it is recorded);
                                      # and RING_FRAMES collapses to 1, so the descriptor pool is
                                      # simply RESET per record() — A CONTRACT, not an
                                      # observation, licensed by `VkHeadless::run` fencing every
                                      # submit, and a future presenter with real frames in flight
                                      # must ring the pool or keep the fence.
                                      # THE MISSING FEATURE, found the way M3a found ray query —
                                      # by the validation layer, not by reading anything: four of
                                      # ReBLUR's fourteen kernels declare
                                      # `ComputeDerivativeGroupQuadsKHR` +
                                      # `SPV_KHR_compute_shader_derivatives`, and RADV accepted
                                      # every module on a device created without them while the
                                      # layer named it exactly. `VK_KHR_compute_shader_derivatives`
                                      # joins the enabled-when-present list; V15 SKIPs on a device
                                      # that lacks it rather than dispatching modules whose
                                      # declared capability is unavailable. ITS NAME IS THE ONE
                                      # HAND-DECLARED STRING in `vk/device.rs`, and that is a
                                      # version gap rather than a choice: the KHR extension
                                      # promoted the NV one in header 1.3.291 and `ash 0.38.0` —
                                      # the latest release — is generated against 1.3.281. The
                                      # FEATURE STRUCT still comes from ash, the spec making
                                      # `...FeaturesNV` a literal alias (same fields, same sType
                                      # 1000201000), which is why the layer names both as
                                      # acceptable. Drop the literal when ash regenerates.
                                      # TEETH, and the second one is why this stage has a seventh
                                      # assertion: (1) the CMake-order binding trap — pinning
                                      # `{0, 2, 3, 20}` instead of the struct's `{0, 20, 2, 3}` —
                                      # fails with the layer naming the exact variables
                                      # (`gIn_ViewZ` at binding 20, `REBLUR_ClassifyTilesConstants`
                                      # at Set 1 Binding 2). (2) A PLANE SWAP — IN_SPEC routed to
                                      # the IN_DIFF image, same format and same descriptor type,
                                      # so validation is silent — PASSED ALL SIX: NRD still gets a
                                      # plausible radiance signal, so it still denoises, still
                                      # accumulates and still restarts, with the mean moved 6.6%
                                      # and everything inside its band. N4's six are about whether
                                      # the denoiser RAN, not about whether each plane is routed
                                      # to the right image, and that gap is real on both backends.
                                      # THE PLANE-ROUTING ARM closes it: zeroing IN_SPEC between
                                      # the pack and the engine must move OUT_SPEC, because if the
                                      # descriptor points elsewhere then nothing reads that image
                                      # and the clear changes nothing (M3d's texture probe once
                                      # more, GPU-vs-GPU one clear apart). Measured 544096/960000
                                      # B moved clean and **0 B with the swap planted**.
                                      # IT SCORES OUT_SPEC, NOT THE RECOMPOSED COLOUR, and that
                                      # correction is the whole reason it works: the colour was
                                      # the obvious readback and is CONFOUNDED, since
                                      # `cs_nrd_out` reconstructs the residual as
                                      # `R = base - D_in*kd*m_d - S_in*f0` and therefore reads
                                      # IN_SPEC itself — measured, the swap moved 276 kB of colour
                                      # and the gate stayed green. The control is TWO identical
                                      # RESET frames, which must agree: they do bit-for-bit in the
                                      # colour and NOT in OUT_SPEC (90 B of 960000), because
                                      # ACCUM_RESTART resets accumulation while the permanent pool
                                      # survives it and the recompose quantises the residue away
                                      # at f16 — so the bar is that the effect DOMINATES the floor
                                      # (~6000x), not that the floor is zero.
                                      # ONE SCENE-DEPENDENT ARM, gated on `structural` like every
                                      # other must-fire here: TEMPORAL SHRINK asserts accumulation
                                      # is reducing frame-to-frame variance, which needs variance
                                      # to reduce. MEASURED — san-miguel-low-poly reads 0.00195 ->
                                      # 0.00199, FLAT, with `d_early` already BELOW the procedural
                                      # scene's converged `d_late` of 0.00434: the input is at the
                                      # floor by frame 1 and the residual is sampling noise. The
                                      # Laplacian still drops 72% there and RESTART still departs
                                      # by 1.7x, so what carries the accumulation claim on those
                                      # scenes is RESTART — a reset frame departing from a
                                      # converged one by MORE than the converged pair differ is
                                      # only possible if there was a history to discard.
                                      # NOT PORTED, both deliberate: `FR_NRD_DEBUG`'s
                                      # OUT_VALIDATION dump (72 lines under a lever, and the
                                      # readback ring it wants does not exist here — the PLANE is
                                      # still allocated, since NRD names it in `reg_for` whether
                                      # or not it writes it) and `FR_NRD_BARRIER=narrow` (an arm
                                      # for a barrier scheme this file does not have).
                                      # Touch `src/vk/nrd.rs` / `gfx::denoise` / `nrd_plane_handles`
                                      # / `vk_format` / the compute-derivatives enable -> run
                                      # --check (the `denoise-vocab` gate + goldens byte-identical
                                      # — B4b-ii touches no shading path), cargo test, --check-vk
                                      # on procedural + san-miguel-low-poly + rungholt, both
                                      # FR_VK_RES parities, --sw-rays, llvmpipe (V15 SKIPs there
                                      # if the device lacks the derivatives extension),
                                      # --check-nrd, --check-spirv, --check-fsr and
                                      # tools/win-cross-check.sh
                                      # V16 — THE COMPOSED FRAME (B5a, 2026-08-12): trace ->
                                      # nrd_pack -> ReBLUR -> nrd_out -> FSR3, in one frame. Every
                                      # stage before it stopped one step short, and the gap had a
                                      # sharp shape: V13 upscales but has NO denoiser in its frame
                                      # (its FFX history is built entirely from raw 1-spp colour),
                                      # while V14/V15 denoise and stop at `read_input`, a readback
                                      # of the plane. `frame_fed` — the only thing in this tree
                                      # that dispatches FFX — appeared exactly ONCE in main.rs,
                                      # inside V13. **So until now, if `cs_nrd_out` had written
                                      # into a plane FFX never reads, every V13, V14 and V15
                                      # assertion would still have passed.** It is D3D12's own
                                      # `present_trace_fsr3` ordering, including the rule that an
                                      # NRD frame runs NO feed dispatch (the folded `cs_nrd_pack`
                                      # owns the mvec/depth guides, `cs_nrd_out` owns the colour) —
                                      # the fold end to end, which only this stage can score.
                                      # THE LAYOUT DEFECT IT FIXED, and the instrument finding
                                      # that came with it: `Fsr3`'s three INPUT images rest in
                                      # SHADER_READ_ONLY_OPTIMAL (a contract with the shim —
                                      # ffx_fsr3_vk.cpp declares them COMPUTE_READ and ffx_vk
                                      # emits no barrier when its tracked state already matches),
                                      # and `wire_feed` declares those descriptors GENERAL. But
                                      # u16 is ONE binding under two names — the derived map
                                      # prints `feed_color/nrd_color_out` — and it is the
                                      # UPSCALER's image, not one of the seven `nrd_lay_out`
                                      # covers. So V14/V15 were dispatching `cs_nrd_out` against an
                                      # image resting in SRO through a descriptor claiming
                                      # GENERAL. **The validation layer does not report that, and
                                      # not for want of looking**: two plants settled it — an
                                      # illegal layout ON the descriptor fires 10 errors at
                                      # `vkUpdateDescriptorSets`, and a wrong barrier `oldLayout`
                                      # fires 20 including the runtime `vkQueueSubmit(): ... expects
                                      # VkImage ... to be in layout ...`, so the layer both tracks
                                      # actual layouts AND checks descriptor legality; what it does
                                      # not do is compare a storage-image descriptor's DECLARED
                                      # layout against the image's ACTUAL one. On RADV the two
                                      # coincide for these formats, so every write landed and every
                                      # gate stayed green — a spec violation neither the hardware
                                      # nor the armed instrument could surface, i.e. the class that
                                      # needs a structural fix rather than a gate. `Fsr3::bracket`
                                      # is that fix in ONE spelling, with `write_inputs` as
                                      # `frame_fed`'s bracket minus the dispatch for recorders that
                                      # write the planes without upscaling them (V14, V15). It
                                      # moved NO number — a layout transition preserves contents,
                                      # and V13/V14/V15 read digit-for-digit identical after it.
                                      # THE STRONG ARM IS THE DEPENDENCY PROBE, not the picture:
                                      # "how much should a denoise change an upscale" has no
                                      # principled answer and any bar for it would be a new
                                      # tolerance, while "did FFX read the plane our recompose
                                      # wrote" does — zero the colour plane BETWEEN `cs_nrd_out`
                                      # and the dispatch and require the output to move past a
                                      # MEASURED floor. THE FLOOR IS WHY THE PROBE USES ONE
                                      # WARMED CONTEXT: three FRESH contexts read a control drift
                                      # of 994206 of 1440000 channels — two identical runs
                                      # disagreeing on 69% of the image — because FFX declares
                                      # essentially every internal resource
                                      # FFX_RESOURCE_INIT_DATA_TYPE_UNINITIALIZED and `ffx_vk`
                                      # skips the init copy for that type, so a reset frame's
                                      # untouched texels hold PER-ALLOCATION residue by contract
                                      # (the Metal backend records the same thing from the other
                                      # side). One context makes the allocations identical; one
                                      # discarded warm-up makes the residue identical too. After
                                      # that the control reads EXACTLY 0.
                                      # MEASURED (RADV, 400x300 -> 800x600, 8 frames): procedural
                                      # denoised-vs-fed rel 0.02849, differs 998202/1440000, **lap
                                      # 0.0030 vs fed 0.0047** (the denoise SURVIVES the upscale —
                                      # a different claim from V15's "the denoise happened"),
                                      # control drift 0, clearing our recompose moves
                                      # 1440000/1440000. smlp 0.00376 / lap 0.0013 vs 0.0018,
                                      # rungholt 0.00303 / same lap pair, `--sw-rays` 0.03500. Both
                                      # FR_VK_RES parities green; llvmpipe SKIPs through V13's
                                      # software-device arm.
                                      # THREE TEETH FIRED, and they SEPARATE: T2 (drop
                                      # `record_nrd_out`) reads `colour 120000 8-byte runs, depth 0
                                      # words` — the two sentinels distinguishing "the recompose
                                      # never ran" from "the fold never ran", which is why the
                                      # sentinel is checked BEFORE the all-black histogram (an
                                      # unwritten plane presents as a black frame, and that is the
                                      # symptom while the sentinel is the cause); T3 (leave
                                      # `wire_feed` pointed elsewhere) collapses the probe to
                                      # `0/1440000` **while the main arm stays green**, which is
                                      # the class a derived layout provably cannot catch since
                                      # every plane is STORAGE_IMAGE to Vulkan; T4 (move the clear
                                      # BEFORE the recompose, where it gets overwritten) also
                                      # `0/1440000`, proving the probe measures a dependency on our
                                      # FINAL write rather than a coincidence. A per-stage
                                      # validation delta (`validation_errors()` snapshotted at
                                      # entry/exit) makes V16 name itself, since the suite
                                      # otherwise reads that global once at the end and a layout
                                      # regression would surface unattributed.
                                      # KNOWN LIMIT, stated rather than implied: on RADV GENERAL
                                      # and SRO are the same hardware state for these formats, so
                                      # no value-scoring gate on this box can see the layout class
                                      # — the layer is the only instrument, and it is blind to this
                                      # specific comparison. The fix is therefore structural (one
                                      # bracket, one spelling, one call site) and the claim is not
                                      # "a gate would have caught it".
                                      # Touch `Fsr3::{bracket, write_inputs, clear_color_in_place}`
                                      # / `run_check_vk_compose` / `compose_probe` -> run --check
                                      # (goldens byte-identical — B5a touches no shading path),
                                      # cargo test, --check-vk on procedural + san-miguel-low-poly
                                      # + rungholt, both FR_VK_RES parities, --sw-rays, llvmpipe,
                                      # --check-spirv, --check-fsr, --check-nrd and
                                      # tools/win-cross-check.sh
                                      # V17 — STRUCTURE REPLAY WITH THE PACK ARMED (B5c,
                                      # 2026-08-12): the gate that licenses what the capture arm
                                      # now ships. V9 proves a replayed frame is bit-identical to
                                      # a produced one, but its tracer is `TracerOpts::default()`
                                      # — `gbuf_full: false` — so there the pack is not merely
                                      # unchecked, it is not ALLOCATED (`VkTracer::new` collapses
                                      # it to a one-element dummy); and V13/V14/V15/V16 all run on
                                      # a pack-armed tracer that never replays. So "replay with
                                      # the pack armed" was exercised by no gate in EITHER
                                      # direction — while `CineVk::output_frame` now replays every
                                      # sub-frame after the first on a tracer carrying the pack,
                                      # the feed planes and NRD's. The claim is V9's one
                                      # configuration further out: the terminal structure is a
                                      # pure function of (scene, BVH, basis, rw, rh) while
                                      # spp/jitter/frame ride the cbuffer, and the PACK is written
                                      # by the leaf and sky passes, which a replay re-dispatches —
                                      # so a replayed frame must reproduce a produced one BYTE for
                                      # byte in `accum` AND both pack halves, with the ladder
                                      # provably not run. IT SITS BEFORE THE FFX CONTEXTS, and
                                      # that placement is the point rather than tidiness:
                                      # everything after context creation is unreachable on a
                                      # software ICD (V11/V13 SKIP outright), so putting the one
                                      # pure-tracer gate in that function ahead of them is what
                                      # gives it the llvmpipe coverage V6/V7/V8/V9 have — i.e.
                                      # what keeps it gateable in CI without a GPU. Anti-vacuity
                                      # runs FIRST, because every assertion below it is satisfied
                                      # by two buffers that were never written: the produced pack
                                      # must be non-zero and the producing frame must have run a
                                      # ladder at all. MEASURED (RADV, 400x300): byte-diff 0 with
                                      # pack non-zero 1201522 B and split 65 -> 0, on procedural,
                                      # san-miguel-low-poly, `--sw-rays`, both FR_VK_RES parities
                                      # and llvmpipe (1194681 B — a different count because the
                                      # software intersector differs at edges, which the
                                      # self-vs-self comparison is indifferent to). TEETH, both
                                      # fired and DISTINCT: a "replay" that quietly re-traces
                                      # everything reads `replay ran the ladder (split 65)` — the
                                      # anti-vacuity half, since such a replay is bit-identical BY
                                      # CONSTRUCTION — and shading the replay at a different frame
                                      # index reads `accum: 815314 of 1440000 bytes differ`, which
                                      # is the byte comparison proving it bites.
                                      # V18 — THE DISPLAY STAGE, RUNG 1 (B6a, 2026-08-12): the
                                      # first thing this backend ever DREW. Everything before it is
                                      # compute — eighteen stages each ending in a number, and a
                                      # `--cinematic` PNG whose tone curve was applied on the CPU —
                                      # so `src/vk/display.rs` is a graphics pipeline
                                      # (`layout::graphics_pipeline`, DYNAMIC RENDERING, no
                                      # VkRenderPass and no VkFramebuffer), a derived layout
                                      # carrying a sampler and a uniform buffer, a colour
                                      # attachment with a real layout lifecycle, and a fullscreen
                                      # triangle. The gate is `--check-gpu`'s M12 transplanted with
                                      # ZERO new tolerances: the same ramp (TW 64 x TH 32, n 2048,
                                      # RAMP_LO 1e-3 -> RAMP_HI 6e4 with index 0 exactly 0.0 and
                                      # index 1 exactly 1.0 = the HDR knee, plus M12's own
                                      # anti-vacuity assert that the top lands in [4.4e4, 65504]),
                                      # the same three wires at the same limits, the same
                                      # f16-ROUNDED oracle (the source is a real RGBA16F texture, so
                                      # comparing against the exact f32 would charge the port for
                                      # the wire's rounding), the same |got-want|/max(1,|want|)
                                      # metric. MEASURED (RADV, 400x300 unrelated — this stage is
                                      # 64x32): sdr worst 2.37e-3 (limit 3.92e-3), sdr10 9.41e-4
                                      # (1.08e-3), hdr10 9.68e-4 (2.5e-3) — all slightly above
                                      # D3D12's own M12 figures (2.2e-3 / 5.4e-4 / 5.5e-4), which is
                                      # DXC at SM 6.0 against fxc at SM 5.0 for the pow/exp pair and
                                      # is comfortably inside the shared limits.
                                      # THE FORMATS CROSS OVER UNCHANGED, which is what lets the
                                      # oracle transplant at all: Vulkan's
                                      # `A2B10G10R10_UNORM_PACK32` puts R in the low ten bits
                                      # exactly as D3D12's `R10G10B10A2_UNORM` does (the names
                                      # suggest the opposite and the memory layout agrees), and
                                      # `B8G8R8A8_UNORM` is B,G,R,A in memory in both.
                                      # TWO ARMS M12 DOES NOT HAVE, and the first is why this is not
                                      # simply M12 with the formats renamed. **THE BLIT
                                      # EXACT-IDENTITY ARM**: M12's ramp is geometric and therefore
                                      # SMOOTH, and the consequence is measured rather than
                                      # asserted — for a ONE-PIXEL horizontal shift within the
                                      # geometric span the worst per-channel error is 2.02e-3
                                      # against an sdr tolerance of 3.92e-3, i.e. INVISIBLE, with
                                      # ALL 2045 ramp texels individually inside tolerance of their
                                      # own neighbour; the 10-bit wires catch it at 2.02e-3 vs
                                      # 1.08e-3, under 2x and only because their quantum is
                                      # smaller. So the ramp is a CURVE gate that is weak evidence
                                      # about the pixel MAPPING — exactly the class a wrong
                                      # SV_Position convention, a half-texel offset or a transposed
                                      # row length lives in, and exactly the class a backend that
                                      # has never rasterized anything is most likely to get wrong.
                                      # `blit.hlsl` is twelve lines of
                                      # `src.Load(int3(pos.xy,0)).rgb` with alpha forced to 1, so
                                      # over a per-(pixel, channel) byte pattern it is an EXACT byte
                                      # identity with no tolerance to hide in. 8-bit only, and that
                                      # is a precision argument rather than a shortcut: f16 carries
                                      # 11 significant bits so representing k/255 is worst-case
                                      # ~0.12 of a UNORM8 step — an eighth of the rounding boundary
                                      # — while at ten bits the two are the same size and an exact
                                      # bar would be scoring f16 rather than the rasterizer. Per
                                      # CHANNEL rather than per pixel so a swizzle is caught too,
                                      # which the ramp's own per-channel tint cannot do (it is
                                      # monotone in all three). **ALPHA IS A FREE COVERAGE
                                      # WITNESS**: both pixel shaders write 1.0 unconditionally and
                                      # the M12 oracle compares RGB only, so "alpha at maximum
                                      # everywhere" answers a question no colour comparison can —
                                      # did the draw cover this pixel — which is what separates "the
                                      # curve is wrong" from "the triangle missed", a black frame
                                      # reporting both identically. Plus a per-stage validation
                                      # delta (V16's pattern), so the first graphics pipeline in
                                      # this backend names itself instead of surfacing as an
                                      # unattributed count in the suite-wide sweep.
                                      # `cullMode = NONE` IS LOAD-BEARING, not lazy: D3D and Vulkan
                                      # disagree about which way NDC y points, so the IDENTICAL
                                      # `SV_VertexID` triangle has opposite screen-space winding
                                      # under the two APIs. Its COVERAGE is unaffected (it contains
                                      # the whole [-1,1] square either way — the point of the trick)
                                      # but a cull mode correct on D3D12 would discard it outright;
                                      # `gpu/tonemap.rs` sets `D3D12_CULL_MODE_NONE` for these same
                                      # shaders, so this is agreement rather than a workaround. Two
                                      # ash `Default`s are INVALID and are set explicitly —
                                      # `line_width` would be 0.0 (the spec requires exactly 1.0
                                      # without `wideLines`) and `rasterization_samples` 0 (not a
                                      # legal bit) — the class a validation layer catches and a
                                      # driver without one may not.
                                      # THE `Params` BLOCK is the one hand-written thing in a slice
                                      # whose virtue is that everything else is derived: D3D12
                                      # pushes ten f32 ROOT CONSTANTS (`gpu/tonemap.rs`'s
                                      # `Passes::record`), Vulkan has no equivalent for a `b`
                                      # register, so they ride a UBO — and the bytes are identical
                                      # because `-fvk-use-dx-layout` puts the members at DX offsets.
                                      # WHICH WIRE CATCHES A MISORDERING DEPENDS ON WHERE IT LANDS,
                                      # measured: a slide of the WHOLE block zeroes `inv_samples`
                                      # so the frame goes black and all three rows fail (worst
                                      # 1.00e0 on both SDR wires), while an error confined to the
                                      # TAIL is the quiet one — `ToneParams::SDR` has `scale` and
                                      # `mode` BOTH at 1.0, so swapping those two is literally
                                      # undetectable on the SDR wires (they read their exact
                                      # unplanted 2.37e-3 / 9.41e-4) and hdr10 alone fails, at
                                      # 5.83e-1, 233x over. **Do not drop the hdr10 row because the
                                      # SDR rows pass** — on this block it is the only detector for
                                      # half the failure modes.
                                      # ON llvmpipe IT PASSES TOO (sdr 1.96e-3, sdr10 9.61e-4,
                                      # hdr10 8.86e-4, blit EXACT), which is what makes CI coverage
                                      # a fact rather than a hope: the stage renders into an image
                                      # it owns and needs no surface, so `ci.yml`'s `check-vulkan`
                                      # job runs it on every push and V18 joins V6/V7/V8/V9 in the
                                      # forbidden-skip list. The blit identity being EXACT on BOTH
                                      # ICDs is the stronger half — two independent rasterizers
                                      # agreeing bit-for-bit on the pixel mapping.
                                      # TEETH, four fired: swapped VS/PS modules (validation names
                                      # it precisely, and RADV then SEGFAULTS on the invalid
                                      # pipeline — so the layer is the only thing that names it, the
                                      # armed-instrument argument again); the `Params` UBO never
                                      # written (worst 1.00e0 on both SDR wires); the block slid by
                                      # one slot (same); and `scale`/`mode` swapped, the one that
                                      # proves the paragraph above. ALL FOUR leave the blit arm
                                      # EXACT, which is the arms SEPARATING: blit scores the
                                      # mapping, the tonemap rows score the curve and its constants,
                                      # and neither substitutes for the other.
                                      # NOT TRANSPLANTED, deliberately: M12b (`main.rs`, 292 lines)
                                      # — the spike-guard / pre-glare family, which wants a live
                                      # bloom pyramid and carries its own six arms. Its own commit,
                                      # not a rider on the first graphics pipeline this backend has.
                                      # Touch `src/vk/display.rs` / `layout::graphics_pipeline` /
                                      # `gfx::shaders::GFX_VS`/`GFX_PS` / `run_check_vk_display` ->
                                      # run --check (goldens byte-identical — B6a touches no shading
                                      # path), cargo test, --check-vk on procedural +
                                      # san-miguel-low-poly, both FR_VK_RES parities, --sw-rays,
                                      # llvmpipe, --check-spirv and tools/win-cross-check.sh
                                      # V19 — THE PRESENT PATH (B6a rung 2, 2026-08-12): a swapchain
                                      # over `VK_EXT_headless_surface`, so this backend PRESENTS.
                                      # V18 proved it can rasterise; presentation is a different
                                      # resource class (images the presentation engine owns, in a
                                      # layout nothing else here enters) through a different API,
                                      # and a real surface needs a window — which needs a windowing
                                      # crate and an input design (B6b). The headless surface splits
                                      # that: acquire, render, present, engine recycles, no display
                                      # attached. What it CANNOT prove is pacing (no vblank, no
                                      # scanout) — that is B6b's, and this module says so rather
                                      # than implying otherwise. THE CLAIM IS A BYTE IDENTITY AT THE
                                      # NEGOTIATED FORMAT: the same `Passes` into a swapchain image
                                      # must produce the same bytes as into an offscreen image of
                                      # that format — deliberately not V18's fixed sdr wire, since
                                      # that decouples from format negotiation entirely and stays
                                      # EXACT rather than a tolerance (the V14/V17 shape).
                                      # THE IDENTITY ALONE IS VACUOUS, MEASURED NOT ARGUED:
                                      # `record_to` opens with `LOAD_OP_CLEAR`, so deleting the draw
                                      # gives BOTH targets the same zeros and the byte comparison
                                      # PASSES — fired as a tooth, and the identity arm stayed
                                      # silent while three others caught it. Those three: the CPU
                                      # oracle (V18's `tone::map` comparison at the negotiated
                                      # format's own tolerance — worst 2.37e-3, V18's sdr figure to
                                      # the digit); ALPHA at maximum (both display pixel shaders
                                      # write 1.0 unconditionally while the oracle compares RGB
                                      # only, and a clear writes 0); and a `0xEE` SENTINEL flooded
                                      # before the draw. THE SENTINEL IS THREE-WAY and free: the
                                      # pattern = everything ran; the CLEAR colour = the render pass
                                      # ran and the draw did not; the SENTINEL SURVIVING = the draw
                                      # went to a DIFFERENT image than the one flooded and copied.
                                      # Both middle and last outcomes were fired as separate teeth
                                      # and give DIFFERENT diagnoses. Written by a draw-less
                                      # `LOAD_OP_CLEAR` rather than `vkCmdClearColorImage`
                                      # deliberately: the latter needs TRANSFER_DST, which the
                                      # offscreen twin does not carry, so it would put a difference
                                      # between the two images whose equality the gate asserts.
                                      # THE PRESENT IS PROVED BY EXHAUSTION — nothing scans out, so
                                      # `VK_SUCCESS` is a statement about a function call. Run
                                      # `images.len() + 1` cycles: the last acquire can only succeed
                                      # if the engine RELEASED one, and `ACQUIRE_TIMEOUT_NS` is
                                      # FINITE so a present that did nothing reports rather than
                                      # hangs. Acquire order is PRINTED, never asserted (the spec
                                      # constrains none of it) — measured [0, 1, 0, 2, 0] over 4
                                      # images, i.e. recycling visible by cycle 2.
                                      # ONE RENDER-FINISHED SEMAPHORE PER IMAGE, and that is not a
                                      # preference: a single shared one shipped first and the
                                      # validation layer named it exactly ("is being signaled by
                                      # VkQueue ..., but it may still be in use by VkSwapchainKHR").
                                      # A binary semaphore may not be re-signalled while a wait is
                                      # outstanding, and a present's wait has no CPU-visible
                                      # completion — the harness fence covers the SUBMIT. Per-image
                                      # makes reuse PROVABLY safe: acquiring image N means the
                                      # engine released N, which means the present that waited on
                                      # `render_done[N]` completed. The ACQUIRE semaphore needs no
                                      # such treatment for a different reason, not by luck —
                                      # `wait_submit` blocks before the next acquire.
                                      # `Passes::record_to` takes (img, view, w, h) rather than
                                      # `&Image` because a swapchain image owns no `VkDeviceMemory`
                                      # and `Image::mem` is private and non-`Option`; `record` stays
                                      # a thin wrapper so V18's recording is TEXTUALLY unchanged.
                                      # Its two transitions were ALREADY right — `UNDEFINED ->
                                      # COLOR_ATTACHMENT_OPTIMAL` is correct for a freshly acquired
                                      # image and the tail to `TRANSFER_SRC_OPTIMAL` is exactly
                                      # where a copy-before-present wants it — so rung 2 appends ONE
                                      # barrier and edits neither.
                                      # `display::rgb_offsets` IS THE DEFECT THIS CAUGHT BEFORE IT
                                      # SHIPPED: `decode`'s catch-all read BGRA, right for the one
                                      # 8-bit format rung 1 renders and silently wrong for the
                                      # other — and the headless surface's OWN first format, on both
                                      # ICDs here, is `R8G8B8A8_UNORM`, i.e. exactly what a present
                                      # path taking the driver's preference negotiates. Inheriting
                                      # it would have swapped R and B in 255 of every 256 texels of
                                      # a hashed pattern while every other assertion stayed green,
                                      # because a swizzle preserves alpha, coverage, AND a byte
                                      # compare of two images that were BOTH swizzled. One statement
                                      # of byte order now, consulted by `decode` and the identity,
                                      # returning `Option` so an unknown format is REFUSED at
                                      # negotiation rather than guessed (and `decode` returns NaN
                                      # there, which fails every comparison it reaches).
                                      # `_SRGB` is refused too: the pixel shader applies its own
                                      # `pow(1/2.2)` and the hardware would encode it twice — a
                                      # wrong image that still looks plausible.
                                      # V19 SKIPS ON llvmpipe, AND THAT IS A MEASURED lavapipe
                                      # DEFECT rather than a limitation: it advertises
                                      # `VK_KHR_swapchain`, accepts a headless surface, and answers
                                      # EVERY capability query — present support, formats, FIFO,
                                      # `supportedUsageFlags` — and then `vkCreateSwapchainKHR`
                                      # JUMPS TO ADDRESS ZERO inside its own frames (`#0 0x0 / #1..4
                                      # libvulkan_lvp.so`), reproducing identically with validation
                                      # DISABLED while RADV runs the same code clean. A segfault is
                                      # not a failure mode a gate can report — it takes the process
                                      # down mid-suite — and CI runs --check-vk on llvmpipe every
                                      # push, so the skip is PRE-EMPTIVE (before the call) unlike
                                      # V11/V13's skip-on-returned-error. `FR_VK_PRESENT_SOFTWARE=1`
                                      # forces the attempt, verified live to still segfault, so the
                                      # re-test is one variable the day Mesa fixes it. CONSEQUENCE:
                                      # **V19 does NOT join ci.yml's forbidden-skip list** — it
                                      # cannot run there — which is why the display stage's CI
                                      # coverage stops at V18.
                                      # `VkHeadless::run_present`/`wait_submit` are `run`'s siblings:
                                      # a present sits BETWEEN the submit and the wait, so a call
                                      # that blocked before returning could not express it.
                                      # Fence-only WOULD work here (same-queue ordering + a CPU
                                      # wait) and is recorded as the fallback, but B6b needs the
                                      # semaphore path, so rung 2 proves the shape the window
                                      # inherits rather than a gate-only shortcut.
                                      # MEASURED (RADV, 64x32, 4 images, FIFO,
                                      # `R8G8B8A8_UNORM`): byte-identical over 10240 texels across 5
                                      # cycles, oracle worst 2.37e-3 (limit 3.92e-3), alpha full,
                                      # sentinel survivors 0, validation clean — on procedural,
                                      # san-miguel-low-poly, rungholt, both FR_VK_RES parities and
                                      # --sw-rays. FOUR TEETH FIRED, each naming its own defect and
                                      # all four SEPARATING: no draw (clear-colour outcome + oracle
                                      # + alpha, identity SILENT); draw into another image (sentinel
                                      # outcome + identity + oracle + alpha); pipeline format !=
                                      # swapchain format (validation ONLY — every value arm passes,
                                      # since both targets went through the same wrong pipeline and
                                      # RADV honours the view's format, which is the sharpest
                                      # argument in this file for CI running the layer armed);
                                      # missing `PRESENT_SRC_KHR` transition (validation, named).
                                      # Touch `src/vk/swapchain.rs` / `Passes::record_to` /
                                      # `display::rgb_offsets` / `run_present`/`wait_submit` / the
                                      # surface+swapchain extension enables in `device.rs` -> run
                                      # --check (goldens byte-identical — rung 2 touches no shading
                                      # path), cargo test, --check-vk on procedural +
                                      # san-miguel-low-poly + rungholt, both FR_VK_RES parities,
                                      # --sw-rays, llvmpipe (V19 SKIPs), --check-spirv, --check-fsr,
                                      # --check-nrd and tools/win-cross-check.sh — the last is not
                                      # optional here: `mod vk` is cfg(unix), so a new gate function
                                      # needs `#[cfg(unix)]` and nothing else on this box catches
                                      # its absence
                                      # M3k — THE SCALE M3i IS INSURANCE AGAINST, REACHED (2026-08-11), and a
                                      # gate that named the wrong bug. No Vulkan gate had ever loaded a scene
                                      # past ~5.6M tris, so the 95x scratch cut M3i measured was a mechanism
                                      # confirmed at small scale and nothing more. `--check-vk
                                      # san-miguel-low-poly.obj --tile 2` is 22.5M tris, and the whole stack
                                      # carries it: 718 chunks, blas 2803 MB compacted from 3400, **scratch
                                      # 7 MB** (the number the insurance is about — one BLAS at this size is
                                      # the 1891 MB class that removed an Intel device under D3D12), 2317.9 MB
                                      # staged in 52 submits, 313 textures / 405.6 MB with BC7 live, and every
                                      # V6/V8/V9/V10 gate green — radiance 0.006%, hemi AO 0.0023 / GI 1.10%,
                                      # replay tbuf/info/accum 0. THE VERTEX-GATHER ARM RAN FOR THE FIRST TIME
                                      # ON THIS BACKEND (`1 chunk(s) vertex-gathered — id range over the
                                      # 16777216 ceiling`): untiled San Miguel has ~2.8M vertices, so no chunk
                                      # can clear the ceiling and only a tiled or world-scale scene reaches the
                                      # arm at all — the M3h coverage lesson (reach is a property of which
                                      # scene somebody ran) with no lever to fix it, since this one needs the
                                      # geometry.
                                      # AND V7's tmin-overshoot FAILED AT 1 PIXEL, which turned out to be the
                                      # GATE's defect and not the renderer's. The attribution chain, each arm
                                      # a run rather than an argument: deterministic (3/3 byte-identical);
                                      # the CPU tracer's own verify at the SAME scale reads max rel t err
                                      # 0.00e0; the CPU and the GPU reference AGREE at the pixel, so the
                                      # wavefront is the odd one out; `FR_ABL=tzero` (leaf primaries trace
                                      # from 0, making the two calls literally identical arguments) changes
                                      # NOTHING — so it is not the inherited bound; `FR_ABL=nocandtmin` and
                                      # `notrans` change nothing either; `FR_ABL=noalpha` returns it to
                                      # **EXACTLY 0.00e0 with 0 hot channels**; and `--sw-rays` — our own
                                      # fixed-order walk — also reads **0.00e0**. So it is the ALPHA-CUTOUT
                                      # candidate loop, and it is the DRIVER's enumeration rather than ours.
                                      # (`--no-blas-split` does not fix it, it only flips the sign: 1.69e-3
                                      # nearer instead of farther, which is the tell that the partition
                                      # perturbs the same knife-edge rather than causing it.) Note the tzero
                                      # arm is only worth anything because the define was PROVEN to arrive —
                                      # `--check-spirv` assembled bytes move 7680159 -> 7553627 — the
                                      # probe-reach rule, and this file's own instrument for it.
                                      # THE FIX IS A DECOMPOSITION, NOT AN ALLOWANCE, and that distinction is
                                      # the whole point: a leaf primary searches [t_start, inf), so when the
                                      # reference's OWN hit lies inside that interval NO value of t_start
                                      # could have hidden it — the inherited bound is innocent BY
                                      # CONSTRUCTION (here t_start 7.79 against hits at 16.54/16.62), and
                                      # what remains is two intersector RUNS disagreeing. Those pixels are
                                      # counted as `cand-edge` under the same 0.05% allowance the two
                                      # siblings in the same function already carry (the IMAGE half of this
                                      # very gate was passing throughout — `img_hot` and `img_mean` have had
                                      # that allowance since M3c; only the `t` half was absolute). Pixels
                                      # BELOW t_start stay in `tmin-overshoot` at exact zero, so the counter
                                      # now measures what its name says, and the culprit line stops printing
                                      # "the inherited-tmin bug class" over a pixel the bound cannot explain
                                      # — the wrong-diagnosis class this tree names in its own QA commit.
                                      # TEETH, and they are where the argument lives: `cand-edge` is EXACTLY
                                      # ZERO wherever the phenomenon provably cannot occur — an opaque scene
                                      # compiles no `Proceed()` arm (`gfx::shaders::non_opaque`, the same
                                      # derived predicate that drops GEOMETRY_FLAG_OPAQUE) and `--sw-rays`
                                      # walks our own tree, both measured 0.00e0 — which is most of the
                                      # suite; `claim-violation` is untouched at exact zero and is the
                                      # INVARIANT rather than this consequence form; and a genuinely bad
                                      # claim is a property of a whole TILE, so it arrives as thousands of
                                      # pixels against a bound of 240. The split itself is gated by
                                      # inverting the innocence predicate, which routes the same pixel back
                                      # to the hard bucket and FAILS the run — a tooth for the routing, not
                                      # merely for the bound. The bound is deliberately loose against what
                                      # was measured (1 px of 480000 = 2e-6, 240x under) because it must
                                      # hold on hardware whose enumeration differs more than RADV's.
                                      # NOTE THE OPPOSITE ATTRIBUTION one milestone earlier, because the two
                                      # look alike and are not: M3f/M3g's fb-replay divergence SURVIVES
                                      # --sw-rays (same rays, shaded differently — mechanism still open, in
                                      # the shared HLSL), while this one VANISHES under it. Same lever, same
                                      # scene, opposite verdicts — which is exactly why the lever is the
                                      # instrument and a plausible story is not.
                                      # AND SAY WHAT THE PREDICATE IS NOT, because the obvious reading
                                      # over-trusts it: `rt > t_start` is true at essentially EVERY leaf
                                      # pixel, since `claim-violation == 0` asserts exactly
                                      # t_start <= min(cpu_t, ref_t)·(1+1e-4) — so innocence fails only in
                                      # a 1e-4-wide relative band, and on a PASSING run the split barely
                                      # discriminates at all. It discriminates precisely when a claim is
                                      # already violated, i.e. in the case it must not soften, which is the
                                      # sense in which it is a decomposition rather than an allowance. The
                                      # teeth are therefore the ARM, the 0.05% BOUND, `claim-violation`
                                      # itself, and the untouched IMAGE half (a cand-edge pixel is NOT added
                                      # to edge_mask, so its colour is still gated at 1e-2 under its own
                                      # independent 0.05% count) — NOT the selectivity of the predicate.
                                      # The comparison is STRICT (`rt > t_start`, tightened from `>=` when
                                      # the D3D12 twin landed): acceptance is strictly beyond TMin, so a hit
                                      # AT t_start is one the bound WOULD have hidden and belongs in the hard
                                      # bucket. Unreachable in practice — t_start is an AABB frustum bound
                                      # and rt a triangle intersection, so exact equality does not occur —
                                      # but the strict form is the one that states the open interval the
                                      # innocence argument rests on, and the two backends move together.
                                      # THE VULKAN HALF OF THAT TIGHTENING WAS MADE FROM WINDOWS AND IS
                                      # COMPILE-UNVERIFIED — say so rather than let the paragraph below
                                      # imply nothing blind was done. It is one operator inside an existing
                                      # expression (no new identifier, no new borrow, syntactically
                                      # unchanged in shape) and it can only move a case from the soft
                                      # bucket to the HARD one, so the worst outcome is a stricter gate;
                                      # against that, leaving the two backends' predicates DIVERGENT is the
                                      # very drift the shared-closure discipline above exists to prevent.
                                      # `--check-vk san-miguel-low-poly.obj --tile 2` on the next Linux run
                                      # is what closes it.
                                      # OWED, PAID ON THE D3D12 SIDE (2026-08-11): `run_check_gpu` carries
                                      # the same decomposition now — and it had TWO undecomposed copies, not
                                      # one, since the --spp sub-gate re-runs the same comparison over the
                                      # same t_start_of and edge_mask (Vulkan has no spp gate, so there was
                                      # no twin here to decline). Both take ONE closure, so that pair cannot
                                      # drift the way the two BACKENDS just did; the spp arm also gained the
                                      # culprit list it never had, since undecomposed it fails a tiled cutout
                                      # run naming MULTI-SAMPLING — a worse wrong answer than the spp=1
                                      # loop's, nothing about the extra samples being implicated — and it is
                                      # the likelier arm to fire (five probes = five draws at one knife-edge).
                                      # The D3D12 summary line prints the ARM STATE beside the count, because
                                      # `cand-edge 0` armed and unarmed are different facts; owed BACK to
                                      # V7's line next Linux run.
                                      # MEASURED (2026-08-11, 4090 + Arc Pro B70): `cand-edge 0 (arm off)`
                                      # on the procedural default and under `--sw-rays` on san-miguel — the
                                      # two provably-zero teeth — and `cand-edge 0 (arm armed)` on
                                      # san-miguel-low-poly untiled AND at `--tile 2` (22.5M tris) on BOTH
                                      # vendors, primary and every spp probe, at `max rel t err 0.00e0`. So
                                      # the bucket is REACHED and swallows nothing where the hardware is
                                      # bit-exact. And the arm is not cosmetic even on a clean run: the
                                      # SAME scene and pose reads 3966 alpha-cutout rejections on the 4090
                                      # against 2951 on the B70, i.e. the candidate loops demonstrably
                                      # enumerate differently while their reported t still agrees exactly —
                                      # which is the phenomenon existing and simply not manifesting here.
                                      # STILL OWED: THE VENDOR, not the code and not the scale. The discrete
                                      # R9700 is gone from the Windows box and the only AMD adapter left is
                                      # the 7950X3D iGPU, which fails --check-gpu at the spp READBACK
                                      # (pre-existing environment) — and that failure lands BEFORE the spp
                                      # comparison loop, so it cannot reach the second twin even in
                                      # principle. `--check-gpu san-miguel-low-poly.obj --tile 2
                                      # --prefer-amd` stays on the list for the next discrete AMD card.
                                      # NOTE THE DIFFERENT JUDGEMENT from the decline this paragraph used to
                                      # record, because the two look alike: that was an edit to a gate this
                                      # box could not RUN or even typecheck; the D3D12 one compiles and runs
                                      # on Windows on two vendors, and only the vendor that produced the
                                      # phenomenon is missing — a measurement gap, not a blind edit.
                                      # THE FRUSTUM TREE IS THE ONE STREAM THIS FILE UPLOADS: t0 space0 takes
                                      # ftree::FTree::quantized() (the QFNode wire format — the per-processor
                                      # split verdict, not a shortcut: the CPU keeps f32 nodes and the GPU
                                      # trades decode ALU for -56% tree bandwidth, with decoded boxes still
                                      # CONTAINING the true ones so every prune stays conservative) or
                                      # gfx::scene::gpu_bvh_nodes under --no-ftree. GpuBvhNode moved to
                                      # gfx::scene beside GpuMat for GpuMat's reason — one #[repr(C)] mirror
                                      # per HLSL struct, read by both backends.
                                      # THE CLOUD CACHES ARE BUILT AND DISPATCHED HERE, not levered off:
                                      # cs_reference reads the amortized sky lattice (--sky-lod, default 4)
                                      # and the slab-space cloud-shadow cache (--cloud-shadow, default 16)
                                      # at registers the wavefront otherwise uses for tile queues, so a
                                      # tracer that skipped their fills would shade a black sky — and
                                      # forcing both levers off would make the gate cover a configuration
                                      # nobody ships. So cs_sky_lod and cs_cloud_shadow are compiled and
                                      # dispatched exactly as record_sky_lod/record_cloud_shadow run them
                                      # on D3D12, and both caches are covered for free.
                                      # THE M2c SUBGROUP VERDICT (V4) — waveprobe.hlsl, the D3D12 suite's
                                      # OWN kernel unmodified, at every group width the tracer dispatches
                                      # (32 for cs_level*/cs_hemi_*, SKY_GROUP=64, LEAF_GROUP=256), in
                                      # THREE pipeline arms because the difference between them IS the
                                      # decision: default (no flags — what a naive port gets), varying
                                      # (ALLOW_VARYING_SUBGROUP_SIZE — driver picks per shader, the ONLY
                                      # behavior D3D12 offers, so it is the arm that says what the D3D12
                                      # numbers would look like here) and pinned
                                      # (VkPipelineShaderStageRequiredSubgroupSizeCreateInfo — the thing
                                      # D3D12 cannot do at all). MEASURED on RADV STRIX_HALO: g32 -> 32
                                      # lanes x 1 wave, g64 -> 64 x 1, g256 -> 64 x 4, default and
                                      # varying IDENTICAL, every row at 100% lane occupancy.
                                      # THE VERDICT IS NO PIN, and it INVERTS the expectation the plan was
                                      # written on: natural subgroupSize is 64, so a 32-thread group
                                      # looked like a half-empty wave64 — but RADV already narrows to
                                      # wave32 for a 32-thread workgroup, so the lane-waste class (the
                                      # D3D12 bug that cost -38% in cs_leaf) DOES NOT REPRODUCE and M3
                                      # carries no pinning machinery. The reported subgroupSize is a
                                      # DEVICE property and the compiled width is a PER-PIPELINE choice;
                                      # only a probe tells them apart, which is the same lesson the
                                      # D3D12 caps taught. The pin still WORKS and stays gated (g64/g256
                                      # pinned to 32 return exactly 32 at 2/8 waves), so the lever is
                                      # there if a real kernel's register pressure ever makes RADV choose
                                      # differently — waveprobe is trivial BY DESIGN, so it answers "what
                                      # a kernel of this WIDTH gets" and nothing finer, and the real
                                      # kernels want an FR_WIDTH-style in-kernel report in M3.
                                      # llvmpipe: 8 lanes at every width, range [8..8], so the pinned arm
                                      # is legitimately impossible there and says so in a NOTE instead of
                                      # failing — the eligibility predicate lives in wave_probe and
                                      # publishes pin_attempted, so the caller's anti-vacuity check cannot
                                      # drift from what actually ran (a first draft keyed it off the caps
                                      # and failed llvmpipe for asking a legal question with an
                                      # impossible answer).
                                      # VK_EXT_headless_surface is NOT used and the plan that named it was
                                      # wrong: that extension runs the PRESENTATION path without a window,
                                      # and compute needs no surface at all — the harness has one less
                                      # moving part than expected. It becomes relevant when the display
                                      # stage wants gating, not before.
                                      # SCENE-KEYED SINCE V5, and V0-V4 are not: smoke.hlsl and
                                      # waveprobe.hlsl carry no scene-derived defines, so those stages are
                                      # a pure function of the device, while V5/V6 assemble the real
                                      # tracer and therefore inherit every scene-conditional define
                                      # (ALPHA_CUTOUT/HEIGHTFIELD/TRANS_SHADOW) — which is why the slot
                                      # count moves with the scene (46 procedural, 49 on
                                      # san-miguel-low-poly) and why `--check-vk san-miguel-low-poly.obj`
                                      # covers bindings the procedural run cannot reach. SINCE M3d BOTH
                                      # STAGES WANT THE SAME SCENE — a textured one — where before V5
                                      # wanted coverage and V6 skipped exactly those; run the pair on
                                      # san-miguel AND on the procedural default, since the teeth table
                                      # above shows they catch different drops. AND ADD `--tile 2` TO THE
                                      # LIST (M3k): `--check-vk` never loads the world, so `--tile` is the
                                      # only lever that reaches world-class scale here — it is what put
                                      # 22.5M tris, a 2803 MB BLAS and the vertex-GATHER arm through the
                                      # backend for the first time, and every one of those is a code path
                                      # no untiled scene can reach.
                                      # `ash` is the one new dependency — a generated binding, not
                                      # a framework (no allocator, no render graph, no policy), and its
                                      # default `loaded` feature dlopens libvulkan.so.1 and resolves every
                                      # entry point by symbol: the same footprint policy as dxc/oidn/xess/
                                      # nrd, so nothing links Vulkan and every other --check* stays
                                      # unaffected. unix-only today for the same reason src/spirv.rs is —
                                      # nothing here forbids Vulkan on Windows, the Windows build simply
                                      # must not gain a dependency it does not yet use
```
