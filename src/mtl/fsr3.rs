//! FidelityFX FSR3 1.1.4 on Metal — the transpiled shader table and its gates.
//!
//! FidelityFX ships only `ffx_vk` and `ffx_dx12` backends, so Metal reuses the
//! very SPIR-V blobs the Vulkan shaderblob accessor returns: `build.rs::
//! generate_fsr3_metallibs` transpiles them (spirv-cross -> `metal` -> `metallib`)
//! and emits the table this module `include!`s. The key is FNV-1a-64 of the
//! SPIR-V bytes, and `shim/ffx_fsr3_metal.mm` computes the byte-identical hash over
//! `FfxShaderBlob.data` — the same bytes from the same accessor — to find a
//! permutation's metallib at pipeline-create time. That hash is the ONLY thing
//! the two sides agree on, so neither may be changed alone.

/// The permutation count this tree expects. It is a MEASURED total, not a
/// product — the options-per-pass count is wildly non-uniform, and an earlier
/// comment here read "10 passes x {fp32, fp16} x 4 options = 80", which is
/// arithmetically 80 and factually wrong about every term but the first. What
/// the committed corpus actually holds, per precision:
///
/// | pass                  | permutations |
/// |-----------------------|--------------|
/// | accumulate            | 24           |
/// | prepare_inputs        | 8            |
/// | autogen_reactive      | 1            |
/// | debug_view            | 1            |
/// | luma_instability      | 1            |
/// | luma_pyramid          | 1            |
/// | prepare_reactivity    | 1            |
/// | rcas                  | 1            |
/// | shading_change        | 1            |
/// | shading_change_pyramid| 1            |
///
/// `(24 + 8 + 8x1) x {fp32, fp16} = 40 x 2 = 80`. The distinction matters
/// because the false factorization predicts +8 for an eleventh pass when the
/// truth is anywhere from +2 to +48 — so if this number moves, RE-COUNT rather
/// than multiply.
///
/// The wave64 half of the 160 committed blobs is deliberately NOT transpiled:
/// Apple GPUs are SIMD-32, so `GetDeviceCapabilitiesMetal` reports
/// `waveLaneCountMax = 32` and FFX never requests them (they would also
/// mis-execute at width 32). **That skip and that caps hardcode are a PAIR** —
/// change one and FFX asks for a hash that was never emitted.
pub const EXPECTED_PERMUTATIONS: usize = 80;

#[cfg(ffx_fsr3_metal)]
mod table {
    // FFX_FSR3_METALLIBS: &[(u64, &[u8])] sorted by hash, and
    // FFX_FSR3_PERMUTATIONS_FOUND: the count build.rs ENUMERATED (the
    // denominator — see the gate below for why it is emitted at all).
    include!(concat!(env!("OUT_DIR"), "/ffx_fsr3_metallibs.rs"));
}

/// The transpiled permutations, keyed by FNV-1a-64 of their SPIR-V.
#[cfg(ffx_fsr3_metal)]
pub fn metallibs() -> &'static [(u64, &'static [u8])] {
    table::FFX_FSR3_METALLIBS
}

/// How many non-wave64 permutation headers build.rs enumerated. A gap between
/// this and `metallibs().len()` is a partial transpile.
#[cfg(ffx_fsr3_metal)]
pub fn permutations_found() -> usize {
    table::FFX_FSR3_PERMUTATIONS_FOUND
}

/// The metallib table's gates. Pure — no device, no GPU, no FFX.
///
/// These exist because every failure they catch is otherwise a RARE, RUNTIME,
/// SCENE-DEPENDENT pipeline-create failure: a truncated table only bites when
/// some particular FSR3 pass asks for the one hash that is missing. Catching it
/// here makes a `spirv-cross` upgrade that changes MSL emission a gate failure
/// instead of a mystery.
#[cfg(ffx_fsr3_metal)]
pub fn table_self_test() -> Result<(), String> {
    let libs = metallibs();
    let found = permutations_found();

    // TEETH: an empty or truncated table cannot pass. Both bounds matter —
    // `found` catches "build.rs enumerated 80 and emitted 60" (a transpile
    // regression), `EXPECTED_PERMUTATIONS` catches "the committed shader
    // directory itself lost files", which `found` alone would happily accept.
    if libs.len() != found {
        return Err(format!(
            "ffx-metal: {} metallibs but {found} permutations enumerated — a partial \
             transpile (re-read the build's FSR3-Metal warning for the first failure)",
            libs.len()
        ));
    }
    if found != EXPECTED_PERMUTATIONS {
        return Err(format!(
            "ffx-metal: {found} non-wave64 permutations enumerated, expected \
             {EXPECTED_PERMUTATIONS} ((24 + 8 + 8x1) x {{fp32,fp16}}; see the constant's \
             per-pass table) — the committed SPIR-V under SDKs/FidelityFX-SDK-prebuilt/ \
             has changed"
        ));
    }

    let mut prev: Option<u64> = None;
    for &(hash, blob) in libs {
        // Strictly ascending, hence unique: the C side linear-scans for an
        // equal hash, so a duplicate key makes the selection ambiguous rather
        // than wrong-and-loud.
        if let Some(p) = prev {
            if hash <= p {
                return Err(format!(
                    "ffx-metal: table is not strictly ascending at {hash:#018x} (after \
                     {p:#018x}) — build.rs must sort, and duplicates make lookup ambiguous"
                ));
            }
        }
        prev = Some(hash);

        // `[12-byte LE threadgroup header][metallib]`. Metal needs the
        // workgroup size HOST-side at dispatch (Vulkan and DXIL both reflect it
        // out of the bytecode), so build.rs parses `OpExecutionMode LocalSize`
        // and prepends it; the shim strips the same 12 bytes back off.
        if blob.len() < 16 {
            return Err(format!("ffx-metal: {hash:#018x} blob is {} B — truncated", blob.len()));
        }
        let g = |i: usize| u32::from_le_bytes([blob[i], blob[i + 1], blob[i + 2], blob[i + 3]]);
        let (gx, gy, gz) = (g(0), g(4), g(8));
        // A mis-parsed OpExecutionMode reads as a nonsense group. Metal's own
        // ceiling is 1024 threads per threadgroup, so this is the real bound
        // and not a guess.
        if gx == 0 || gy == 0 || gz == 0 || (gx as u64) * (gy as u64) * (gz as u64) > 1024 {
            return Err(format!(
                "ffx-metal: {hash:#018x} threadgroup header is [{gx},{gy},{gz}] — \
                 OpExecutionMode LocalSize was mis-parsed"
            ));
        }
        // `metallib`'s container magic, so a garbage entry fails HERE rather
        // than inside newLibraryWithData at pipeline-create time.
        if &blob[12..16] != b"MTLB" {
            return Err(format!(
                "ffx-metal: {hash:#018x} is not a metallib past the header (got {:02x?})",
                &blob[12..16]
            ));
        }
    }
    Ok(())
}

// ============================================================ the FFX context

#[cfg(ffx_fsr3_metal)]
use objc2::rc::Retained;
#[cfg(ffx_fsr3_metal)]
use objc2::runtime::ProtocolObject;
#[cfg(ffx_fsr3_metal)]
use objc2_metal::{MTLStorageMode, MTLTexture};

/// One transpiled permutation as the shim sees it — the `FrShimFsr3MetalLib` of
/// `shim/ffx_fsr3_metal.h`, and the ONLY struct that crosses this ABI. Every
/// FFX type stays on the C++ side (the `ffx_shim.cpp` rule), so a v1.1.x layout
/// cannot drift out of step with a Rust transcription of it.
#[cfg(ffx_fsr3_metal)]
#[repr(C)]
struct ShimLib {
    hash: u64,
    data: *const u8,
    size: u32,
}

#[cfg(ffx_fsr3_metal)]
unsafe extern "C" {
    fn frshim_fsr3_metal_create(
        device: *mut std::ffi::c_void,
        max_render_w: u32,
        max_render_h: u32,
        upscale_w: u32,
        upscale_h: u32,
        flags: u32,
        libs: *const ShimLib,
        lib_count: u32,
        out_pipelines: *mut u32,
        out_handle: *mut *mut std::ffi::c_void,
    ) -> i32;
    #[allow(clippy::too_many_arguments)]
    fn frshim_fsr3_metal_dispatch(
        handle: *mut std::ffi::c_void,
        cmd_buffer: *mut std::ffi::c_void,
        color: *mut std::ffi::c_void,
        depth: *mut std::ffi::c_void,
        motion: *mut std::ffi::c_void,
        output: *mut std::ffi::c_void,
        render_w: u32,
        render_h: u32,
        upscale_w: u32,
        upscale_h: u32,
        jitter_x: f32,
        jitter_y: f32,
        mv_scale_x: f32,
        mv_scale_y: f32,
        frame_time_delta_ms: f32,
        camera_near: f32,
        camera_far: f32,
        camera_fov_y: f32,
        reset: i32,
    ) -> i32;
    fn frshim_fsr3_metal_destroy(handle: *mut std::ffi::c_void);
}

// Mirrors shim/ffx_fsr3_metal.h's flag enum. Duplicated rather than bound,
// because binding it would mean a build-time bindgen for four bits; the header
// names this file as the other half, so a change to either is one grep away.
#[cfg(ffx_fsr3_metal)]
pub const FLAG_HDR: u32 = 1 << 0;
#[cfg(ffx_fsr3_metal)]
pub const FLAG_DEPTH_INVERTED: u32 = 1 << 1;
#[cfg(ffx_fsr3_metal)]
pub const FLAG_AUTO_EXPOSURE: u32 = 1 << 3;

/// Everything a dispatch needs that is not a texture. A struct rather than
/// nineteen positional arguments, because the four scalars in the middle
/// (`jitter`, `mv_scale`) are the ones a caller can silently transpose.
#[cfg(ffx_fsr3_metal)]
pub struct DispatchParams {
    /// The frame's render extent — at most the `render` given to `new`. FFX
    /// reads only this sub-rect of the max-sized planes.
    pub render: (usize, usize),
    /// The frame's sub-pixel jitter, ALREADY multiplied by `fsr::JITTER_SIGN`.
    pub jitter: (f32, f32),
    pub near: f32,
    pub far: f32,
    /// Vertical field of view, radians.
    pub fov_y: f32,
    pub frame_time_ms: f32,
    /// True on the first frame and after any discontinuity: FFX throws its
    /// temporal history away.
    pub reset: bool,
}

/// A live FSR3 upscaler context on the hand-written `ffx_metal` backend, plus
/// the four planes it reads and writes.
///
/// Creating one is the expensive and the interesting part: it drives
/// `fpCreatePipeline` across every FSR3 pass, so this is where the transpiled
/// metallibs are loaded, specialized against the device's linear-texture
/// alignment, and turned into compute pipeline states. A mis-keyed table, a
/// spirv-cross output Metal will not accept, or a caps/wave64 divergence all
/// surface HERE rather than at the first dispatch that happens to need the one
/// missing permutation.
#[cfg(ffx_fsr3_metal)]
pub struct Fsr3 {
    handle: *mut std::ffi::c_void,
    /// Compute pipeline states actually built during creation — see
    /// `pipelines()`.
    pipelines: u32,
    /// The MAX extents the planes and FFX's temporals were sized to.
    max_render: (usize, usize),
    upscale: (usize, usize),
    // Allocated ONCE at the max extents and dispatched as sub-rects — the
    // discipline `gpu/ffx_up.rs` states for the D3D12 arm, and the reason a
    // dynamic render resolution costs no reallocation. Shared with the MetalFX
    // arm (`super::planes`), which is what makes the two upscalers' inputs
    // identical by construction rather than by two call sites agreeing.
    trio: super::planes::Trio,
    output: Retained<ProtocolObject<dyn MTLTexture>>,
}

#[cfg(ffx_fsr3_metal)]
impl Fsr3 {
    /// `render` is the MAXIMUM render extent — planes and FFX's temporals are
    /// sized to it once and a smaller frame dispatches a sub-rect (the
    /// discipline `gpu/ffx_up.rs` states for the D3D12 arm).
    ///
    /// `DEPTH_INVERTED` is not optional for this renderer: `fsr::stage_depth`
    /// writes `xess::view_z_to_clip_depth`'s reversed-Z, where sky lands on
    /// exactly 0.0. Telling FFX otherwise inverts its whole disocclusion test.
    /// (The quinlight reference sets neither flag because it feeds a flat
    /// pseudo-depth — the plan's "real depth is the untested path".)
    pub fn new(
        mtl: &super::device::Mtl,
        render: (usize, usize),
        upscale: (usize, usize),
        flags: u32,
    ) -> Result<Fsr3, String> {
        let libs: Vec<ShimLib> = metallibs()
            .iter()
            .map(|&(hash, blob)| ShimLib { hash, data: blob.as_ptr(), size: blob.len() as u32 })
            .collect();
        let mut handle: *mut std::ffi::c_void = std::ptr::null_mut();
        let mut pipelines: u32 = 0;
        // SAFETY: `mtl` outlives the call, and the borrowed table is 'static —
        // `metallibs()` returns `include_bytes!`'d data — which is what lets the
        // shim keep the pointers for the handle's whole life.
        let rc = unsafe {
            frshim_fsr3_metal_create(
                mtl.device_ptr(),
                render.0 as u32,
                render.1 as u32,
                upscale.0 as u32,
                upscale.1 as u32,
                flags,
                libs.as_ptr(),
                libs.len() as u32,
                &mut pipelines,
                &mut handle,
            )
        };
        if rc != 0 || handle.is_null() {
            // The shim has already printed FFX's own message through fpMessage
            // and named the failing pass and hash if a metallib was the cause;
            // this line says which PHASE, which those do not.
            return Err(format!(
                "frshim_fsr3_metal_create failed: {} ({})",
                rc,
                match rc {
                    1 => "bad argument from our side",
                    2 => "zero render or upscale extent",
                    3 => "assembling the FfxInterface",
                    4 => "ffxFsr3UpscalerContextCreate — see the [fsr3-metal] lines above",
                    6 => "a Metal object came back nil",
                    _ => "unknown",
                }
            ));
        }
        // THE FORMATS ARE THIS RENDERER'S WIRE, not a choice made here: the
        // shim declares exactly these to FFX, which rebuilds its own views from
        // that description and ignores whatever we hold — so a mismatch between
        // what is allocated and what is declared is read as garbage rather than
        // rejected. They live in `super::planes` so allocation and declaration
        // read one constant.
        //
        // THE PLANES ARE BUILT INTO LOCALS FIRST, and that is about the FAILURE
        // path rather than about style. From the `create` above, `handle` owns a
        // live FFX context, its backend, the three temporals and every pipeline
        // state — and `Fsr3` does not exist yet, so `Drop` cannot run. A `?`
        // inside the struct literal would therefore return with all of that
        // leaked, which is exactly the shape that goes unnoticed: it is an
        // error path, and a gate builds several contexts in a loop.
        let alloc = || -> Result<
            (super::planes::Trio, Retained<ProtocolObject<dyn MTLTexture>>),
            String,
        > {
            let trio = super::planes::Trio::new(mtl, render)?;
            // Shared, unlike the MetalFX arm's output: FFX documents no storage
            // requirement, so the plain `read` path applies.
            let out = mtl
                .texture(upscale.0, upscale.1, super::planes::OUTPUT, MTLStorageMode::Shared)
                .map_err(|e| format!("output plane: {e}"))?;
            Ok((trio, out))
        };
        let (trio, output) = match alloc() {
            Ok(planes) => planes,
            Err(e) => {
                // SAFETY: the context this function just created, never handed
                // out and never submitted against, so the GPU is idle by
                // construction — the precondition `destroy` states.
                unsafe { frshim_fsr3_metal_destroy(handle) };
                return Err(e);
            }
        };
        Ok(Fsr3 { handle, pipelines, max_render: render, upscale, trio, output })
    }

    /// Stage the input trio into the planes' top-left `rw x rh` sub-rect.
    ///
    /// The planes and the recording both live in `super::planes::Trio`, shared
    /// with the MetalFX arm — which is what makes "the two Metal upscalers see
    /// identical inputs" true by construction rather than by inspection, and
    /// therefore what makes reading one gate's verdict against the other's mean
    /// anything. It is a dedup within ONE api, not the `trait Upscaler`
    /// `src/mtl/mod.rs` rules out; that file and `planes.rs`'s own header carry
    /// the argument, including the condition under which it must be un-shared.
    ///
    /// The encodings are `fsr::stage_*` either way — the SAME pure functions
    /// `--check-fsr` gates on every platform. Nothing about them is decided
    /// here or there.
    pub fn upload(
        &self,
        mtl: &super::device::Mtl,
        accum: &[std::sync::atomic::AtomicU32],
        g: &crate::dlss::GBufs,
        render: (usize, usize),
        near: f32,
        far: f32,
    ) {
        self.trio.stage(mtl, accum, g, render, near, far);
    }

    /// Record one upscale, submit it, and block until the GPU is done.
    ///
    /// Blocking is right here and would not be in a presenting session: this is
    /// a headless gate, so there is no next frame to overlap with and a fence
    /// ring would be apparatus with no consumer. It is also what makes `Drop`
    /// safe without an explicit wait.
    pub fn dispatch(
        &self,
        mtl: &super::device::Mtl,
        p: &DispatchParams,
    ) -> Result<(), String> {
        let (rw, rh) = p.render;
        if rw == 0 || rh == 0 || rw > self.max_render.0 || rh > self.max_render.1 {
            return Err(format!(
                "dispatch render extent {rw}x{rh} is not within the {}x{} the context was \
                 created for",
                self.max_render.0, self.max_render.1
            ));
        }
        let ptr = |t: &Retained<ProtocolObject<dyn MTLTexture>>| {
            Retained::as_ptr(t) as *mut std::ffi::c_void
        };
        // Cleared so that "the dispatch wrote nothing" reads as zeros rather
        // than as whatever the allocator handed us. MEASURED: FSR3 does write
        // every output texel (two fresh contexts given one reset frame produce
        // bit-identical output with no clear at all), so this buys no
        // correctness — it buys an unambiguous FAILURE signal, which is the
        // `--check-fsr3` U4 assertion that the mean is nonzero.
        mtl.clear(&self.output, self.upscale.0, self.upscale.1, 8);
        let mut rc = 0i32;
        mtl.run(|cb| {
            // SAFETY: every pointer outlives the call — the textures are ours,
            // the handle is live, and the command buffer is borrowed for the
            // closure. The shim opens and closes its own encoders inside it and
            // leaves none open, which is what lets `run` commit unconditionally.
            rc = unsafe {
                frshim_fsr3_metal_dispatch(
                    self.handle,
                    cb as *const _ as *mut std::ffi::c_void,
                    ptr(&self.trio.color),
                    ptr(&self.trio.depth),
                    ptr(&self.trio.motion),
                    ptr(&self.output),
                    rw as u32,
                    rh as u32,
                    self.upscale.0 as u32,
                    self.upscale.1 as u32,
                    p.jitter.0,
                    p.jitter.1,
                    // THE MV CONVENTION, stated once in src/fsr.rs and read
                    // here. Our plane already holds pixel-space curr->prev
                    // vectors, so the scale is the bare sign — the same value
                    // the D3D12 FSR 3.1 arm passes.
                    crate::fsr::UPSCALE_MV_SIGN.0,
                    crate::fsr::UPSCALE_MV_SIGN.1,
                    p.frame_time_ms,
                    p.near,
                    p.far,
                    p.fov_y,
                    i32::from(p.reset),
                )
            };
        })?;
        if rc != 0 {
            return Err(format!(
                "frshim_fsr3_metal_dispatch failed: {rc} — see the [fsr3-metal] lines above"
            ));
        }
        Ok(())
    }

    /// The upscaled frame as linear f32 RGBA, row-major at the upscale extent.
    ///
    /// Widened from the RGBA16F plane here rather than left as bits, because
    /// every consumer — the gate's scoring, a PNG dump — wants floats, and
    /// doing it once keeps the f16 decode out of each.
    pub fn read_output(&self, mtl: &super::device::Mtl) -> Vec<f32> {
        let (w, h) = self.upscale;
        let bytes = mtl.read(&self.output, w, h, 8);
        (0..w * h * 4)
            .map(|i| {
                f32::from(half::f16::from_le_bytes([bytes[i * 2], bytes[i * 2 + 1]]))
            })
            .collect()
    }

    /// How many compute pipeline states creation actually built.
    ///
    /// THE ANTI-VACUITY NUMBER, and it is worth having as a number rather than
    /// as a comment: `new()` returning `Ok` says the FFX context exists, which
    /// is NOT the same claim as "the transpiled metallibs work" — a backend
    /// that answered every `fpCreatePipeline` with an empty success would be
    /// indistinguishable by return code. It is also the honest measure of
    /// COVERAGE: measured on an M1 this is ELEVEN, one per FSR3 pass at the
    /// option word the context flags select, and a sweep of all eight flag
    /// combinations reaches only 14 distinct blobs of the 80 (most passes
    /// ignore most option bits, and the 40 fp32 variants are never requested
    /// because the caps report fp16). `table_self_test` proves all 80 are
    /// well-formed metallibs; this proves eleven of them become pipelines.
    pub fn pipelines(&self) -> u32 {
        self.pipelines
    }
}

#[cfg(ffx_fsr3_metal)]
impl Drop for Fsr3 {
    fn drop(&mut self) {
        // The shim releases its temporals and then destroys the FFX context,
        // which runs fpDestroyPipeline/fpDestroyResource back through our
        // backend. Every command buffer this module submits is committed with
        // `waitUntilCompleted`, so the GPU is idle by construction here.
        unsafe { frshim_fsr3_metal_destroy(self.handle) };
    }
}
