//! Adapter enumeration, the pick, the limits ask, and the error side
//! channels — everything `--check-wgpu` J0/J1 gate before a kernel exists.
//!
//! The pure half (`type_rank`/`Cand`/`pick`/`needed_bindings`/`ask_limits`/
//! `limit_reject`) is J0's subject and runs with no GPU; it is a textual
//! twin of vk::device's (`mod vk` is cfg(unix) — see mod.rs). Two things
//! the Vulkan twin never faced:
//!
//! - **Cross-backend duplicates.** wgpu enumerates the same physical GPU
//!   once per backend (Vulkan AND D3D12 on Windows), so candidate names
//!   carry the backend — `"Radeon (Vulkan)"` vs `"Radeon (Dx12)"` — and a
//!   bare `FR_WGPU_ADAPTER=radeon` on such a box is CORRECTLY ambiguous.
//! - **The binding-number ceiling.** DXC's register shifts (spirv.rs:
//!   b=0 / t=1000 / u=2000 / s=3000) put UAVs at `@binding(2000+)`, above
//!   WebGPU's DEFAULT `max_bindings_per_bind_group = 1000`. Native wgpu
//!   raises it via `required_limits`; whether BROWSERS will is Stage C2's
//!   go/no-go on a binding-compaction rewrite, and J1's print of this limit
//!   is the day-one probe the stage exists for. An adapter that cannot
//!   grant the ask is REJECTED at pick time (`limit_reject` -> `Cand::
//!   reject`), so `request_device` never fails by construction.

use crate::spirv::{self, Reg};
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

/// Absent-vs-told, the vk::device::VkError shape verbatim: a gate must SKIP
/// (exit 0) when the box simply has no usable adapter, and must FAIL when
/// the session asked for something specific and did not get it. A mistyped
/// `FR_WGPU_ADAPTER` is not an absent GPU, and reporting it as one turns
/// being-told into passing (the `--fsr4` doctrine).
#[derive(Debug)]
pub struct WgpuError {
    pub msg: String,
    pub absent: bool,
}

impl WgpuError {
    fn absent(msg: impl Into<String>) -> Self {
        WgpuError { msg: msg.into(), absent: true }
    }
    fn told(msg: impl Into<String>) -> Self {
        WgpuError { msg: msg.into(), absent: false }
    }
}

impl std::fmt::Display for WgpuError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.msg)
    }
}

/// A wgpu call failing on a device we already chose is a real failure, not
/// an absent GPU — `?` defaults to the reportable side. Only the empty
/// enumeration and the all-rejected arm construct `absent`.
impl From<String> for WgpuError {
    fn from(s: String) -> Self {
        WgpuError::told(s)
    }
}

/// The vk::device ladder verbatim: a software adapter is a legitimate CI
/// target when it is the only one present, and must lose to real hardware
/// when it is not.
pub fn type_rank(t: wgpu::DeviceType) -> u32 {
    match t {
        wgpu::DeviceType::DiscreteGpu => 4,
        wgpu::DeviceType::IntegratedGpu => 3,
        wgpu::DeviceType::VirtualGpu => 2,
        wgpu::DeviceType::Cpu => 1,
        _ => 0,
    }
}

/// One enumerated adapter, reduced to what the choice depends on. `name`
/// carries the backend suffix (see the module note on duplicates).
#[derive(Clone, Debug)]
pub struct Cand {
    pub name: String,
    pub rank: u32,
    /// `Some(reason)` when the adapter cannot run this gate at all. A
    /// rejected adapter is never picked, however high it ranks.
    pub reject: Option<String>,
}

impl Cand {
    pub fn new(name: &str, rank: u32, reject: Option<&str>) -> Self {
        Cand { name: name.to_string(), rank, reject: reject.map(str::to_string) }
    }
}

/// Choose an adapter. `force` is `FR_WGPU_ADAPTER`: an index, or a
/// case-insensitive substring of the candidate name (which includes the
/// backend, so `"radeon (dx12)"` disambiguates a cross-backend duplicate).
///
/// The three vk::device::pick rules, for the same three silent wrong
/// answers: a FORCED adapter that cannot run is an ERROR, not a fallback;
/// an AMBIGUOUS substring is an error rather than first-match; ties break
/// by ENUMERATION INDEX so the pick repeats run to run.
pub fn pick(cands: &[Cand], force: Option<&str>) -> Result<usize, WgpuError> {
    if cands.is_empty() {
        return Err(WgpuError::absent("no wgpu adapters enumerated"));
    }
    if let Some(f) = force {
        // Every failure below is `told`, not `absent`: the box may be full
        // of working GPUs — the lever named none of them.
        let f = f.trim();
        let idx = if let Ok(i) = f.parse::<usize>() {
            if i >= cands.len() {
                return Err(WgpuError::told(format!(
                    "FR_WGPU_ADAPTER={i} is out of range ({} adapter{} enumerated)",
                    cands.len(),
                    if cands.len() == 1 { "" } else { "s" }
                )));
            }
            i
        } else {
            let lower = f.to_lowercase();
            let hits: Vec<usize> = cands
                .iter()
                .enumerate()
                .filter(|(_, c)| c.name.to_lowercase().contains(&lower))
                .map(|(i, _)| i)
                .collect();
            match hits.len() {
                0 => {
                    let all: Vec<&str> = cands.iter().map(|c| c.name.as_str()).collect();
                    return Err(WgpuError::told(format!(
                        "FR_WGPU_ADAPTER={f:?} matched no adapter; enumerated: {}",
                        all.join(", ")
                    )));
                }
                1 => hits[0],
                _ => {
                    let m: Vec<&str> = hits.iter().map(|&i| cands[i].name.as_str()).collect();
                    return Err(WgpuError::told(format!(
                        "FR_WGPU_ADAPTER={f:?} is ambiguous — matched {}",
                        m.join(", ")
                    )));
                }
            }
        };
        if let Some(why) = &cands[idx].reject {
            return Err(WgpuError::told(format!(
                "FR_WGPU_ADAPTER selected {:?}, which {why}",
                cands[idx].name
            )));
        }
        return Ok(idx);
    }

    let best = cands
        .iter()
        .enumerate()
        .filter(|(_, c)| c.reject.is_none())
        .max_by_key(|(i, c)| (c.rank, std::cmp::Reverse(*i)))
        .map(|(i, _)| i);
    // Every adapter rejected IS an absent condition: the box has hardware
    // but none of it can grant what this gate needs — an environment fact a
    // gate skips on rather than a defect it fails on.
    best.ok_or_else(|| {
        let mut s = String::from("no usable wgpu adapter:");
        for c in cands {
            s.push_str(&format!("\n  {} — {}", c.name, c.reject.as_deref().unwrap_or("?")));
        }
        WgpuError::absent(s)
    })
}

/// The highest binding number the corpus reaches, plus one — DERIVED from
/// the one shift rule (`spirv::binding_of`), never a literal, so a moved
/// shift moves the ask with it.
///
/// `s` carries the highest shift (3000) and `s1` is the highest sampler
/// register the corpus declares (`trace_common.hlsli`'s `samp_lin`/
/// `samp_aniso`), so this covers every `b`/`t`/`u` register below it too.
/// The TOOTH is downstream and exact: `tracer.rs` asserts that no unit it
/// compiled declares a binding at or above this number, so a corpus that
/// grows an `s2` fails loudly at construction instead of at some browser's
/// `createBindGroupLayout`.
pub fn needed_bindings() -> u32 {
    spirv::binding_of(Reg::S, 1) + 1
}

/// What a session needs ON TOP of the corpus-wide contract — the two rows
/// that are properties of the SCENE rather than of the shader corpus.
///
/// Everything else in the ask is fixed by `wgsl::BUDGET`, which is what
/// `--check-wgsl` W5 audits every entry point against. These two cannot be:
/// a bucket is a texture the scene has (0 on the default scene, 21 on
/// bistro, 165 on san-miguel), and the biggest buffer is its BVH.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Ask {
    /// `gfx::texweb::plan(scene).buckets.len()` — sampled-texture bindings
    /// ON TOP of `BUDGET.sampled_fixed`.
    pub buckets: u32,
    /// The largest single buffer the session will bind.
    pub buffer_bytes: u64,
}

impl Ask {
    /// The empty ask — no scene, no tracer, so the corpus-wide contract
    /// alone. `self_test`'s fixture, and nothing else: the RUNNER always asks
    /// for the real scene, because a device is opened once and the tracer
    /// runs on the same one the smoke did.
    pub fn none() -> Ask {
        Ask::default()
    }
}

/// The device ask: WebGPU's own defaults with exactly the rows this corpus
/// overruns raised, and no others.
///
/// **The count rows come from [`crate::wgsl::BUDGET`]**, which is the same
/// table `--check-wgsl` W5 audits every entry point against — so the limit a
/// device is asked for and the limit the corpus was PROVEN to fit under are
/// one constant, not two that must be kept in step. That join is the whole
/// reason W5 exists as a gate rather than as a report.
///
/// Deliberately NOT `adapter.limits()` wholesale: a browser grants the
/// defaults plus what the page asks for, and a gate that always requested
/// native maxima would measure nothing about the WebGPU floor. The two
/// SCENE rows are the exception and are argued at [`Ask`] — for those,
/// asking for what the scene needs is exactly what a page does, and J1's
/// print of ask-vs-default is the browser go/no-go datum.
///
/// `self_test` pins the whole delta field by field: a row that drifts away
/// from BUDGET, or a new row raised without a reason, fails there.
pub fn ask_limits(ask: &Ask) -> wgpu::Limits {
    let b = &crate::wgsl::BUDGET;
    let mut l = wgpu::Limits::default();
    l.max_bindings_per_bind_group = needed_bindings();
    l.max_storage_buffers_per_shader_stage = b.storage_bufs;
    l.max_storage_textures_per_shader_stage = b.storage_tex;
    l.max_sampled_textures_per_shader_stage = b.sampled_fixed + ask.buckets;
    l.max_samplers_per_shader_stage = b.samplers;
    l.max_uniform_buffers_per_shader_stage = b.uniform_bufs;
    // `max` rather than `=`: the WebGPU defaults already exceed what a small
    // scene needs, and asking DOWN would be a refusal a browser never makes.
    l.max_buffer_size = l.max_buffer_size.max(ask.buffer_bytes);
    l.max_storage_buffer_binding_size =
        l.max_storage_buffer_binding_size.max(ask.buffer_bytes);
    l
}

/// `Some(reason)` when an adapter cannot grant `ask_limits(ask)` — fed into
/// `Cand::reject` so the pick, not `request_device`, is where refusal
/// surfaces (with the adapter's name attached).
///
/// EVERY field of the ask, not just the raised ones: `request_device` asks
/// for the full WebGPU default set too, and an adapter below a DEFAULT
/// would otherwise slip the pick and surface as a told-FAIL from
/// `request_device` — a driver-defect report for what is really an
/// environment fact. `fatal: false` so the reason names every shortfall,
/// not the first; the "vs asked" phrasing is direction-neutral because the
/// `min_*` alignment limits fail the other way around.
pub fn limit_reject(adapter: &wgpu::Limits, ask: &Ask) -> Option<String> {
    let mut fails = Vec::new();
    ask_limits(ask).check_limits_with_fail_fn(adapter, false, |name, asked, granted| {
        let why = if name == "max_bindings_per_bind_group" { " (DXC register shifts)" } else { "" };
        fails.push(format!("{name} {granted} vs {asked} asked{why}"));
    });
    if fails.is_empty() { None } else { Some(format!("grants {}", fails.join(", "))) }
}

/// A live device + the facts J1 prints. `errors` is the uncaptured-error
/// side channel — the vk `validation_errors()` twin, struct-held rather
/// than global so two devices cannot cross-contaminate.
pub struct Wgpu {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub info: wgpu::AdapterInfo,
    /// What the device was GRANTED.
    pub limits: wgpu::Limits,
    /// What the adapter COULD have granted (J1's risk table).
    pub adapter_limits: wgpu::Limits,
    /// The scene rows this device was opened for.
    pub ask: Ask,
    errors: Arc<AtomicU32>,
}

impl Wgpu {
    /// Instance -> enumerate -> pick -> device. Absence is a normal
    /// condition (`absent`); a mis-set lever or a refused request on a
    /// chosen adapter is not (`told`).
    pub fn new(ask: Ask) -> Result<Wgpu, WgpuError> {
        // `_from_env` honors WGPU_BACKEND — a free lever; announce it when
        // armed (the lever-line rule).
        if let Ok(b) = std::env::var("WGPU_BACKEND") {
            eprintln!("wgpu: WGPU_BACKEND={b} (backend set from the environment)");
        }
        let instance =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());

        let adapters = pollster::block_on(instance.enumerate_adapters(wgpu::Backends::all()));
        let cands: Vec<Cand> = adapters
            .iter()
            .map(|a| {
                let i = a.get_info();
                Cand {
                    // The backend disambiguates cross-backend duplicates —
                    // the same physical GPU enumerates once per backend.
                    name: format!("{} ({:?})", i.name, i.backend),
                    rank: type_rank(i.device_type),
                    reject: limit_reject(&a.limits(), &ask),
                }
            })
            .collect();

        let force = std::env::var("FR_WGPU_ADAPTER").ok();
        if let Some(f) = &force {
            eprintln!("wgpu: FR_WGPU_ADAPTER={f} (adapter pick forced)");
        }
        let idx = pick(&cands, force.as_deref())?;
        let adapter = &adapters[idx];

        // ask <= adapter is guaranteed (the pick rejected any that can't),
        // so a refusal here is a driver defect worth reporting, not a limit
        // negotiation.
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("frustracer check-wgpu"),
            required_features: wgpu::Features::empty(),
            required_limits: ask_limits(&ask),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::default(),
            trace: wgpu::Trace::Off,
        }))
        .map_err(|e| WgpuError::told(format!("request_device on {}: {e}", cands[idx].name)))?;

        // The side channel, installed BEFORE any creation call: it REPLACES
        // wgpu's default native handler, which PANICS — without this, the
        // first validation error is a process abort, not a FAIL line.
        let errors = Arc::new(AtomicU32::new(0));
        let c = errors.clone();
        device.on_uncaptured_error(Arc::new(move |e| {
            c.fetch_add(1, Ordering::Relaxed);
            eprintln!("wgpu uncaptured error: {e}");
        }));

        Ok(Wgpu {
            limits: device.limits(),
            adapter_limits: adapter.limits(),
            info: adapter.get_info(),
            device,
            queue,
            ask,
            errors,
        })
    }

    /// Uncaptured errors so far — the gate's tail sweep reads this.
    pub fn errors(&self) -> u32 {
        self.errors.load(Ordering::Relaxed)
    }

    /// The adapter line, the vk::Vk::line() shape: every fact a later SKIP
    /// or a slow number would need to be read against.
    pub fn line(&self) -> String {
        let i = &self.info;
        format!(
            "wgpu: {} ({:?}, {:?}) — driver {} {}, max_bindings_per_bind_group {} granted \
             (adapter {})",
            i.name,
            i.device_type,
            i.backend,
            i.driver,
            i.driver_info,
            self.limits.max_bindings_per_bind_group,
            self.adapter_limits.max_bindings_per_bind_group,
        )
    }

    /// The J1 risk table: what this session ASKED FOR against the WebGPU
    /// DEFAULT (what a browser grants before any ask) and against what the
    /// adapter could have given. Facts, not verdicts — the runner prints.
    ///
    /// The middle column is the browser question in one number: every row
    /// where `ask > default` is a row a page must request and a browser may
    /// refuse. The right column is why the corpus needs it, and every one of
    /// those reasons is a MEASURED `--check-wgsl` W5 print rather than an
    /// estimate.
    pub fn limits_risks(&self) -> Vec<String> {
        let a = &self.adapter_limits;
        let g = &self.limits;
        let d = wgpu::Limits::default();
        let k = ask_limits(&self.ask);
        let b = &crate::wgsl::BUDGET;
        let bucket_note = format!(
            "BUDGET.sampled_fixed {} + {} WEB_TEX bucket(s) for this scene",
            b.sampled_fixed, self.ask.buckets
        );
        let rows: [(&str, u64, u64, u64, u64, String); 6] = [
            (
                "max_bindings_per_bind_group",
                k.max_bindings_per_bind_group as u64,
                d.max_bindings_per_bind_group as u64,
                g.max_bindings_per_bind_group as u64,
                a.max_bindings_per_bind_group as u64,
                "DXC register shifts reach s:3000+ — if browsers cap at the default, C2 compacts"
                    .into(),
            ),
            (
                "max_storage_buffers_per_shader_stage",
                k.max_storage_buffers_per_shader_stage as u64,
                d.max_storage_buffers_per_shader_stage as u64,
                g.max_storage_buffers_per_shader_stage as u64,
                a.max_storage_buffers_per_shader_stage as u64,
                format!("BUDGET.storage_bufs {} (W5 worst 22)", b.storage_bufs),
            ),
            (
                "max_storage_textures_per_shader_stage",
                k.max_storage_textures_per_shader_stage as u64,
                d.max_storage_textures_per_shader_stage as u64,
                g.max_storage_textures_per_shader_stage as u64,
                a.max_storage_textures_per_shader_stage as u64,
                format!("BUDGET.storage_tex {} (W5 worst 9, frd_temporal)", b.storage_tex),
            ),
            (
                "max_sampled_textures_per_shader_stage",
                k.max_sampled_textures_per_shader_stage as u64,
                d.max_sampled_textures_per_shader_stage as u64,
                g.max_sampled_textures_per_shader_stage as u64,
                a.max_sampled_textures_per_shader_stage as u64,
                bucket_note,
            ),
            (
                "max_storage_buffer_binding_size",
                k.max_storage_buffer_binding_size as u64,
                d.max_storage_buffer_binding_size as u64,
                g.max_storage_buffer_binding_size as u64,
                a.max_storage_buffer_binding_size as u64,
                "this scene's largest single buffer (its BVH)".into(),
            ),
            (
                "max_buffer_size",
                k.max_buffer_size,
                d.max_buffer_size,
                g.max_buffer_size,
                a.max_buffer_size,
                "same, as an allocation rather than a binding".into(),
            ),
        ];
        rows.iter()
            .map(|(name, ask, def, granted, adapter, why)| {
                let flag = if ask > def { " OVER-DEFAULT" } else { "" };
                format!(
                    "{name}: ask {ask} vs WebGPU default {def}{flag} \
                     (granted {granted}, adapter {adapter}) — {why}"
                )
            })
            .collect()
    }

    /// Creation-error capture: wgpu creation calls are infallible by
    /// signature, so without a scope a bad module surfaces later with no
    /// attribution. v30: `push_error_scope` returns a guard whose `pop()`
    /// IS the future — a dropped guard still pops but EATS the error, so
    /// the block_on is load-bearing. Errors inside a scope never reach the
    /// uncaptured counter (no double count).
    pub fn scoped<T>(&self, what: &str, f: impl FnOnce() -> T) -> Result<T, String> {
        let guard = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let v = f();
        match pollster::block_on(guard.pop()) {
            None => Ok(v),
            Some(e) => Err(format!("{what}: {e}")),
        }
    }
}

/// J0 — the pure half, meaningful on a box with no GPU at all. The vk
/// device self_test case-for-case, plus the cases the Vulkan twin never
/// faced (cross-backend duplicates, the limits math).
pub fn self_test() -> Result<(), String> {
    use wgpu::DeviceType as D;

    // Rank order: the ladder must strictly descend.
    let ranks = [D::DiscreteGpu, D::IntegratedGpu, D::VirtualGpu, D::Cpu, D::Other];
    for w in ranks.windows(2) {
        if type_rank(w[0]) <= type_rank(w[1]) {
            return Err(format!("wgpu: type_rank not descending at {:?} vs {:?}", w[0], w[1]));
        }
    }

    let c = |n: &str, r: u32| Cand::new(n, r, None);

    // A software adapter beside real hardware loses, in both enumeration
    // orders; alone, it wins.
    for cands in [
        vec![c("llvmpipe (Vulkan)", 1), c("GeForce (Vulkan)", 4)],
        vec![c("GeForce (Vulkan)", 4), c("llvmpipe (Vulkan)", 1)],
    ] {
        let i = pick(&cands, None).map_err(|e| format!("wgpu: pick refused: {e}"))?;
        if !cands[i].name.starts_with("GeForce") {
            return Err(format!("wgpu: pick took {:?} over real hardware", cands[i].name));
        }
    }
    let soft = [c("llvmpipe (Vulkan)", 1)];
    if pick(&soft, None).map_err(|e| e.msg)? != 0 {
        return Err("wgpu: a lone software adapter was not picked".into());
    }

    // Ties break by enumeration index — the pick is a pure function of the
    // driver's own order.
    let tie = [c("A (Vulkan)", 4), c("B (Vulkan)", 4)];
    if pick(&tie, None).map_err(|e| e.msg)? != 0 {
        return Err("wgpu: a rank tie did not break by enumeration index".into());
    }

    // A rejected adapter is never picked; ALL rejected is absent (an
    // environment fact), naming every reason.
    let rej = [
        Cand::new("A (Vulkan)", 4, Some("grants max_bindings_per_bind_group 640 < 2003 needed")),
        c("B (Vulkan)", 1),
    ];
    if pick(&rej, None).map_err(|e| e.msg)? != 1 {
        return Err("wgpu: a rejected adapter out-ranked a usable one".into());
    }
    let all_rej = [Cand::new("A (Vulkan)", 4, Some("reason-a"))];
    match pick(&all_rej, None) {
        Err(e) if e.absent && e.msg.contains("reason-a") => {}
        Err(e) => return Err(format!("wgpu: all-rejected was not absent-with-reasons: {e}")),
        Ok(_) => return Err("wgpu: an all-rejected list picked something".into()),
    }
    match pick(&[], None) {
        Err(e) if e.absent => {}
        _ => return Err("wgpu: an empty enumeration was not absent".into()),
    }

    // The lever: index, substring (case-insensitive), and the three told
    // failures. `absent == false` IS the told-vs-absent tooth — assert it.
    let two = [c("GeForce RTX 4090 (Vulkan)", 4), c("Intel Arc (Vulkan)", 3)];
    if pick(&two, Some("1")).map_err(|e| e.msg)? != 1 {
        return Err("wgpu: FR_WGPU_ADAPTER by index did not select it".into());
    }
    if pick(&two, Some("arc")).map_err(|e| e.msg)? != 1 {
        return Err("wgpu: FR_WGPU_ADAPTER by substring did not select it".into());
    }
    for (force, what) in [("7", "out-of-range"), ("nope", "no-match")] {
        match pick(&two, Some(force)) {
            Err(e) if !e.absent => {}
            Err(_) => return Err(format!("wgpu: a {what} lever read as ABSENT — told is the point")),
            Ok(_) => return Err(format!("wgpu: a {what} lever picked an adapter")),
        }
    }
    match pick(&rej, Some("0")) {
        Err(e) if !e.absent && e.msg.contains("which grants") => {}
        Err(e) => return Err(format!("wgpu: forcing a rejected adapter: wrong error: {e}")),
        Ok(_) => return Err("wgpu: forcing a rejected adapter succeeded".into()),
    }

    // The case vk never had: the same GPU once per backend. A bare
    // substring is CORRECTLY ambiguous; the backend suffix resolves it.
    let dup = [c("Radeon 780M (Vulkan)", 3), c("Radeon 780M (Dx12)", 3)];
    match pick(&dup, Some("radeon")) {
        Err(e) if !e.absent && e.msg.contains("ambiguous") => {}
        Err(e) => return Err(format!("wgpu: cross-backend duplicate: wrong error: {e}")),
        Ok(_) => return Err("wgpu: a cross-backend duplicate was silently first-matched".into()),
    }
    if pick(&dup, Some("(dx12)")).map_err(|e| e.msg)? != 1 {
        return Err("wgpu: the backend suffix did not disambiguate".into());
    }

    // The limits math: derived from the shift rule, both ways, and the
    // anti-drift equality — a row cannot drift from BUDGET, and a new raised
    // row cannot ride in silently.
    let smoke = Ask::none();
    if needed_bindings() != spirv::binding_of(Reg::S, 1) + 1 {
        return Err("wgpu: needed_bindings is not the derived s1+1".into());
    }
    // It must cover every register class the corpus uses, not just its own —
    // `s` carries the highest shift, so this is the claim that makes one
    // number enough.
    for (r, n) in [(Reg::B, 1u32), (Reg::T, 12), (Reg::U, 32)] {
        if spirv::binding_of(r, n) >= needed_bindings() {
            return Err(format!("wgpu: needed_bindings does not cover {r:?}{n}"));
        }
    }
    let sufficient = ask_limits(&smoke);
    let mut low = sufficient.clone();
    low.max_bindings_per_bind_group = needed_bindings() - 1;
    match limit_reject(&low, &smoke) {
        Some(r) if r.contains(&needed_bindings().to_string()) => {}
        Some(r) => return Err(format!("wgpu: limit_reject names neither number: {r}")),
        None => return Err("wgpu: a too-low binding ceiling was not rejected".into()),
    }
    if limit_reject(&sufficient, &smoke).is_some() {
        return Err("wgpu: an exactly-sufficient adapter was rejected".into());
    }
    // The completeness tooth: the reject covers the WHOLE ask. An adapter
    // below a WebGPU DEFAULT on a field the ask never raises must be
    // rejected too — and with `fatal: false` the reason must name BOTH
    // shortfalls, not stop at the first.
    let mut low2 = sufficient.clone();
    low2.max_compute_workgroup_storage_size -= 1;
    low2.max_bindings_per_bind_group = 1;
    match limit_reject(&low2, &smoke) {
        Some(r)
            if r.contains("max_compute_workgroup_storage_size")
                && r.contains("max_bindings_per_bind_group") => {}
        Some(r) => return Err(format!("wgpu: below-default reject named only: {r}")),
        None => return Err("wgpu: an adapter below a WebGPU default was not rejected".into()),
    }

    // THE JOIN: every count row of the ask IS `wgsl::BUDGET`, the table
    // `--check-wgsl` W5 proves the corpus fits under. If these two ever
    // drift, a corpus audited green would be run under a limit it was never
    // audited against — so the equality is asserted rather than trusted.
    let b = &crate::wgsl::BUDGET;
    let d = wgpu::Limits::default();
    let k = ask_limits(&smoke);
    let rows: [(&str, u32, u32); 5] = [
        ("storage buffers", k.max_storage_buffers_per_shader_stage, b.storage_bufs),
        ("storage textures", k.max_storage_textures_per_shader_stage, b.storage_tex),
        ("sampled textures", k.max_sampled_textures_per_shader_stage, b.sampled_fixed),
        ("samplers", k.max_samplers_per_shader_stage, b.samplers),
        ("uniform buffers", k.max_uniform_buffers_per_shader_stage, b.uniform_bufs),
    ];
    for (what, asked, budget) in rows {
        if asked != budget {
            return Err(format!("wgpu: ask_limits {what} = {asked}, BUDGET says {budget}"));
        }
    }
    // The scene rows move with the Ask and nothing else does — the buckets
    // land on the sampled row, the bytes on the two size rows.
    let scened = ask_limits(&Ask { buckets: 7, buffer_bytes: 3 << 30 });
    if scened.max_sampled_textures_per_shader_stage != b.sampled_fixed + 7 {
        return Err("wgpu: the bucket ask did not reach the sampled-texture row".into());
    }
    if scened.max_buffer_size != 3 << 30 || scened.max_storage_buffer_binding_size != 3 << 30 {
        return Err(format!(
            "wgpu: the byte ask did not reach the size rows ({} / {})",
            scened.max_buffer_size, scened.max_storage_buffer_binding_size
        ));
    }
    // A scene SMALLER than the WebGPU defaults must not ask DOWN — a
    // refusal a browser never makes, and one that would fail a device the
    // defaults already cover.
    let tiny = ask_limits(&Ask { buckets: 0, buffer_bytes: 1024 });
    if tiny.max_buffer_size != d.max_buffer_size
        || tiny.max_storage_buffer_binding_size != d.max_storage_buffer_binding_size
    {
        return Err("wgpu: a tiny scene asked the size rows DOWN from the defaults".into());
    }
    // Nothing outside the named delta moved. `Limits` is `PartialEq`, so
    // reconstructing the expected value field by field and comparing wholesale
    // is what makes a silently-added row a failure.
    let mut want = d.clone();
    want.max_bindings_per_bind_group = needed_bindings();
    want.max_storage_buffers_per_shader_stage = b.storage_bufs;
    want.max_storage_textures_per_shader_stage = b.storage_tex;
    want.max_sampled_textures_per_shader_stage = b.sampled_fixed;
    want.max_samplers_per_shader_stage = b.samplers;
    want.max_uniform_buffers_per_shader_stage = b.uniform_bufs;
    if k != want {
        return Err("wgpu: ask_limits raised a field outside the BUDGET delta".into());
    }

    Ok(())
}
