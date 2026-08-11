//! Vulkan bring-up: loader, instance, physical-device pick, device, queue,
//! memory, buffers. The peer of the D3D12 backend's `D3d`/`HeadlessGpu`
//! construction, and deliberately nothing more — every policy decision above
//! this line (which kernels, which constants, which flags) lives in `gfx::`
//! and is shared with D3D12 rather than mirrored.
//!
//! NOTHING LINKS VULKAN. `ash`'s default `loaded` feature dlopens
//! `libvulkan.so.1` and resolves every entry point by symbol — the same
//! footprint policy `gpu/dxc.rs`, `oidn.rs`, `xess.rs` and `nrd.rs` already
//! follow, so a box with no loader gets a diagnosable error instead of a
//! failed exec, and a build without a GPU still builds.
//!
//! TWO HARD REQUIREMENTS, both traceable to decisions already made:
//!
//! * **Vulkan 1.3.** `vk::spirv` compiles the corpus at
//!   `-fspv-target-env=vulkan1.3` because DXC's SPIR-V default (vulkan1.0)
//!   cannot express subgroup ops or RayQuery at all. A 1.2 device could not
//!   consume the modules we produce, so the version is a floor, not a
//!   preference.
//! * **`scalarBlockLayout`.** `-fvk-use-dx-layout` is what lets ONE Rust
//!   packer serve both backends (`gfx::frame::FrameCb`'s 4608 bytes stay
//!   byte-compatible instead of needing a std140 twin), and the price is that
//!   DX packing rules must be legal — a `uint3` at 12-byte stride is a scalar
//!   layout, not std430's 16. So this is REQUIRED, never preferred: a device
//!   without it would validate every module and then read the wrong bytes.
//!
//! THE PICK IS PURE AND GATED, because this is exactly the place a silent
//! wrong answer hides. A box can expose several ICDs — the dev box exposes a
//! real RDNA3.5 iGPU *and* llvmpipe, a software rasterizer — and a ranking bug
//! that quietly prefers the software device does not fail: it renders,
//! correctly, at a hundredth of the speed, and every measurement taken
//! afterwards is a measurement of llvmpipe. `pick` and `mem_type_index` are
//! therefore ordinary functions over plain data, pinned by `self_test` in
//! `--check-vk` with teeth in both directions.

use ash::vk;
use std::ffi::{c_char, CStr, CString};
use std::sync::atomic::{AtomicU32, Ordering};

/// The API floor. See the module header: this follows the SPIR-V target env,
/// it is not a taste.
pub const REQ_API: u32 = vk::API_VERSION_1_3;

/// Validation messages at ERROR severity, counted by the debug callback.
/// A gate that arms validation must FAIL on a nonzero count — the D3D12
/// `--gpu-debug` discipline, which exists because the debug layer writes to a
/// side channel and a run that ignores it has armed validation and thrown the
/// findings away.
static VALIDATION_ERRORS: AtomicU32 = AtomicU32::new(0);

pub fn validation_errors() -> u32 {
    VALIDATION_ERRORS.load(Ordering::Relaxed)
}

// ---------------------------------------------------------------------------
// The pure half: device ranking and memory-type selection.

/// Preference order among device classes. Discrete beats integrated beats
/// virtual beats CPU; anything unrecognized sorts last. The CPU rung is what
/// keeps a software ICD (llvmpipe, lavapipe, SwiftShader) from winning a box
/// that also has real hardware — and it is a rung rather than an exclusion on
/// purpose, since a software device is a legitimate CI target when it is the
/// only one present.
pub fn type_rank(t: vk::PhysicalDeviceType) -> u32 {
    match t {
        vk::PhysicalDeviceType::DISCRETE_GPU => 4,
        vk::PhysicalDeviceType::INTEGRATED_GPU => 3,
        vk::PhysicalDeviceType::VIRTUAL_GPU => 2,
        vk::PhysicalDeviceType::CPU => 1,
        _ => 0,
    }
}

/// One enumerated device, reduced to what the choice actually depends on.
#[derive(Clone, Debug)]
pub struct Cand {
    pub name: String,
    pub rank: u32,
    /// `Some(reason)` when the device cannot run this renderer at all. A
    /// rejected device is never picked, however high it ranks.
    pub reject: Option<String>,
}

impl Cand {
    pub fn new(name: &str, rank: u32, reject: Option<&str>) -> Self {
        Cand { name: name.to_string(), rank, reject: reject.map(str::to_string) }
    }
}

/// Choose a device. `force` is `FR_VK_DEVICE`: an index, or a
/// case-insensitive substring of the device name.
///
/// Three rules, each of which exists to prevent a specific silent wrong
/// answer. A FORCED device that cannot run is an ERROR, not a fallback — being
/// told is the whole point of the lever (the `--fsr4` doctrine). An AMBIGUOUS
/// substring is an error rather than first-match, because "amd" matching two
/// adapters and quietly taking one is how a measurement ends up describing the
/// other device. And ties break by ENUMERATION INDEX, so the pick is a pure
/// function of the driver's own order and repeats run to run.
pub fn pick(cands: &[Cand], force: Option<&str>) -> Result<usize, VkError> {
    if cands.is_empty() {
        return Err(VkError::absent("no Vulkan devices enumerated"));
    }
    if let Some(f) = force {
        // Every failure below is `told`, not `absent`: the box may be full of
        // working GPUs — the lever named none of them.
        let f = f.trim();
        let idx = if let Ok(i) = f.parse::<usize>() {
            if i >= cands.len() {
                return Err(VkError::told(format!(
                    "FR_VK_DEVICE={i} is out of range ({} device{} enumerated)",
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
                    return Err(VkError::told(format!(
                        "FR_VK_DEVICE={f:?} matched no device; enumerated: {}",
                        all.join(", ")
                    )));
                }
                1 => hits[0],
                _ => {
                    let m: Vec<&str> = hits.iter().map(|&i| cands[i].name.as_str()).collect();
                    return Err(VkError::told(format!(
                        "FR_VK_DEVICE={f:?} is ambiguous — matched {}",
                        m.join(", ")
                    )));
                }
            }
        };
        if let Some(why) = &cands[idx].reject {
            return Err(VkError::told(format!(
                "FR_VK_DEVICE selected {:?}, which {why}",
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
    // Every device rejected IS an absent-Vulkan condition: the box has
    // hardware but none of it can run this renderer, which is an environment
    // fact a gate skips on rather than a defect it fails on.
    best.ok_or_else(|| {
        let mut s = String::from("no usable Vulkan device:");
        for c in cands {
            s.push_str(&format!("\n  {} — {}", c.name, c.reject.as_deref().unwrap_or("?")));
        }
        VkError::absent(s)
    })
}

/// First memory type that is both allowed by the requirement mask and carries
/// every wanted property. FIRST, not best: the Vulkan spec orders memory types
/// so that earlier ones are "more optimal" for the same property set, and
/// picking deterministically is what makes an allocation failure reproducible.
pub fn mem_type_index(
    props: &vk::PhysicalDeviceMemoryProperties,
    type_bits: u32,
    want: vk::MemoryPropertyFlags,
) -> Option<u32> {
    (0..props.memory_type_count).find(|&i| {
        type_bits & (1 << i) != 0 && props.memory_types[i as usize].property_flags.contains(want)
    })
}

// ---------------------------------------------------------------------------
// The impure half.

/// What the pick found, kept for the loud line and for M2c's wave policy.
#[derive(Clone, Debug)]
pub struct DeviceInfo {
    pub name: String,
    pub driver: String,
    pub api: (u32, u32, u32),
    pub kind: vk::PhysicalDeviceType,
    pub vendor_id: u32,
    /// The driver's preferred subgroup width, and the range it will honour
    /// under `subgroup_size_control`. On D3D12 this is a range the driver
    /// picks inside per shader and the caps never predict; Vulkan can PIN it
    /// (`VkPipelineShaderStageRequiredSubgroupSizeCreateInfo`), which is the
    /// M2c decision.
    ///
    /// `subgroup_size` is `VkPhysicalDeviceSubgroupProperties::subgroupSize` —
    /// the width an UNPINNED pipeline gets — and is deliberately a separate
    /// query from the control range, because the two answer different
    /// questions and coincide only by accident (they do on RADV, which is
    /// exactly why filling this from `max_subgroup_size` printed a plausible
    /// number here for a while).
    pub subgroup_size: u32,
    pub subgroup_min: u32,
    pub subgroup_max: u32,
    pub subgroup_size_control: bool,
    /// `requiredSubgroupSizeStages` includes COMPUTE — i.e. a compute pipeline
    /// on this device may pin its width. Probed rather than inferred from the
    /// feature bit: that bit says the device implements size control, this
    /// says it implements it for the one stage this renderer has.
    pub subgroup_pin_compute: bool,
    /// `maxComputeWorkgroupSubgroups` — the pin's own ceiling, since a group of
    /// `g` threads pinned to width `w` needs `ceil(g / w)` subgroups.
    pub max_workgroup_subgroups: u32,
    pub ray_query: bool,
    pub accel_struct: bool,
    pub rt_pipeline: bool,
}

impl DeviceInfo {
    /// PCI vendor id -> the vocabulary `main::vendor_defaults` already speaks.
    /// Kept here rather than derived at the call site because a Vulkan session
    /// will eventually want the same vendor-keyed policy the D3D12 one has,
    /// and that policy must key off a FACT (which device was opened), never
    /// off which one was asked for.
    pub fn vendor_str(&self) -> &'static str {
        match self.vendor_id {
            0x1002 | 0x1022 => "AMD",
            0x10DE => "NVIDIA",
            0x8086 => "Intel",
            0x13B5 => "ARM",
            0x5143 => "Qualcomm",
            0x10005 => "Mesa",
            _ => "unknown vendor",
        }
    }

    pub fn kind_str(&self) -> &'static str {
        match self.kind {
            vk::PhysicalDeviceType::DISCRETE_GPU => "discrete",
            vk::PhysicalDeviceType::INTEGRATED_GPU => "integrated",
            vk::PhysicalDeviceType::VIRTUAL_GPU => "virtual",
            vk::PhysicalDeviceType::CPU => "software",
            _ => "other",
        }
    }
}

pub struct Vk {
    /// Held for its LIFETIME, not read: `ash::Entry` owns the dlopen'd
    /// `libvulkan.so.1`, and dropping it unloads the library out from under
    /// every handle below. Never "clean this up" as an unused field.
    #[allow(dead_code)]
    pub entry: ash::Entry,
    pub instance: ash::Instance,
    /// The physical device we opened. Held because it is IRRECOVERABLE, not
    /// because something reads it today: Vulkan has no device -> physical-device
    /// query, so dropping this would make every later capability read
    /// (acceleration-structure properties, format support, heap budgets)
    /// require re-enumerating and re-running the pick — i.e. it could pick a
    /// different device than the one these handles belong to.
    #[allow(dead_code)]
    pub phys: vk::PhysicalDevice,
    pub device: ash::Device,
    pub queue: vk::Queue,
    pub qfam: u32,
    pub mem: vk::PhysicalDeviceMemoryProperties,
    pub info: DeviceInfo,
    debug: Option<(ash::ext::debug_utils::Instance, vk::DebugUtilsMessengerEXT)>,
}

/// Why bring-up failed, split by what the caller should DO about it.
///
/// The distinction is load-bearing and was found by exercising the lever: a
/// gate must SKIP (exit 0) when the box simply has no usable Vulkan — the
/// bare-checkout rule every SDK-dependent gate here follows — and must FAIL
/// when the session asked for something specific and did not get it. A
/// mistyped `FR_VK_DEVICE` is not an absent GPU, and reporting it as one turns
/// being-told into passing, which is the exact failure the lever exists to
/// prevent (the `--fsr4` doctrine).
#[derive(Debug)]
pub struct VkError {
    pub msg: String,
    pub absent: bool,
}

impl VkError {
    fn absent(msg: impl Into<String>) -> Self {
        VkError { msg: msg.into(), absent: true }
    }
    fn told(msg: impl Into<String>) -> Self {
        VkError { msg: msg.into(), absent: false }
    }
}

impl std::fmt::Display for VkError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.msg)
    }
}

/// A raw Vulkan call failing on a device we already chose is a real failure,
/// not an absent GPU — so the `?` shorthand throughout bring-up defaults to
/// the reportable side. Only the loader and the pick construct `absent`.
impl From<String> for VkError {
    fn from(s: String) -> Self {
        VkError::told(s)
    }
}

pub struct Buffer {
    pub buf: vk::Buffer,
    pub mem: vk::DeviceMemory,
    pub size: u64,
    pub host: bool,
}

unsafe extern "system" fn debug_cb(
    sev: vk::DebugUtilsMessageSeverityFlagsEXT,
    _ty: vk::DebugUtilsMessageTypeFlagsEXT,
    data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    _user: *mut std::ffi::c_void,
) -> vk::Bool32 {
    let msg = unsafe {
        let d = &*data;
        if d.p_message.is_null() {
            String::new()
        } else {
            CStr::from_ptr(d.p_message).to_string_lossy().into_owned()
        }
    };
    if sev.contains(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR) {
        VALIDATION_ERRORS.fetch_add(1, Ordering::Relaxed);
        eprintln!("vk validation ERROR: {msg}");
    } else if sev.contains(vk::DebugUtilsMessageSeverityFlagsEXT::WARNING) {
        eprintln!("vk validation warning: {msg}");
    }
    vk::FALSE
}

fn cstr(bytes: &[c_char]) -> String {
    unsafe { CStr::from_ptr(bytes.as_ptr()) }.to_string_lossy().into_owned()
}

impl Vk {
    /// Bring up loader, instance and device. `validation` arms
    /// `VK_LAYER_KHRONOS_validation` + `VK_EXT_debug_utils`; a request that
    /// cannot be honoured is LOUD and continues unvalidated, since a silently
    /// unarmed validation run is worse than an honestly unarmed one.
    pub fn new(validation: bool) -> Result<Vk, VkError> {
        let entry = unsafe { ash::Entry::load() }.map_err(|e| {
            VkError::absent(format!("no Vulkan loader ({e}) — install the Vulkan ICD loader"))
        })?;

        let app_name = CString::new("frustracer").unwrap();
        let app = vk::ApplicationInfo::default()
            .application_name(&app_name)
            .engine_name(&app_name)
            .api_version(REQ_API);

        let want_layer = CString::new("VK_LAYER_KHRONOS_validation").unwrap();
        let mut layers: Vec<*const c_char> = Vec::new();
        let mut exts: Vec<*const c_char> = Vec::new();
        let mut armed = false;
        if validation {
            let have = unsafe { entry.enumerate_instance_layer_properties() }
                .map_err(|e| format!("enumerate_instance_layer_properties: {e}"))?;
            let found = have.iter().any(|l| cstr(&l.layer_name) == "VK_LAYER_KHRONOS_validation");
            if found {
                layers.push(want_layer.as_ptr());
                exts.push(ash::ext::debug_utils::NAME.as_ptr());
                armed = true;
            } else {
                eprintln!(
                    "vk: FR_VK_VALIDATION requested but VK_LAYER_KHRONOS_validation is not \
                     installed — continuing UNVALIDATED"
                );
            }
        }

        let ici = vk::InstanceCreateInfo::default()
            .application_info(&app)
            .enabled_layer_names(&layers)
            .enabled_extension_names(&exts);
        let instance = unsafe { entry.create_instance(&ici, None) }
            .map_err(|e| format!("vkCreateInstance: {e}"))?;

        let debug = if armed {
            let di = ash::ext::debug_utils::Instance::new(&entry, &instance);
            let info = vk::DebugUtilsMessengerCreateInfoEXT::default()
                .message_severity(
                    vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                        | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING,
                )
                .message_type(
                    vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                        | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                        | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
                )
                .pfn_user_callback(Some(debug_cb));
            match unsafe { di.create_debug_utils_messenger(&info, None) } {
                Ok(m) => Some((di, m)),
                Err(e) => {
                    eprintln!("vk: debug messenger unavailable ({e}) — continuing UNVALIDATED");
                    None
                }
            }
        } else {
            None
        };

        match Self::open_device(entry, instance, debug) {
            Ok(v) => Ok(v),
            Err(e) => Err(e),
        }
    }

    fn open_device(
        entry: ash::Entry,
        instance: ash::Instance,
        debug: Option<(ash::ext::debug_utils::Instance, vk::DebugUtilsMessengerEXT)>,
    ) -> Result<Vk, VkError> {
        let phys_all = unsafe { instance.enumerate_physical_devices() }
            .map_err(|e| format!("vkEnumeratePhysicalDevices: {e}"))?;

        let mut cands = Vec::new();
        let mut probed = Vec::new();
        for &p in &phys_all {
            let pr = Self::probe(&instance, p);
            cands.push(Cand::new(&pr.0.name, type_rank(pr.0.kind), pr.1.as_deref()));
            probed.push(pr);
        }

        let force = std::env::var("FR_VK_DEVICE").ok();
        if force.is_some() {
            eprintln!("vk: FR_VK_DEVICE={} (device pick forced)", force.as_deref().unwrap());
        }
        let idx = pick(&cands, force.as_deref())?;
        let (info, _) = probed.swap_remove(idx);
        let phys = phys_all[idx];

        // Queue family: prefer graphics+compute (one family that can also
        // drive a future display stage), else any compute family. COMPUTE
        // implies TRANSFER by spec, so the staging copies need nothing extra.
        let fams = unsafe { instance.get_physical_device_queue_family_properties(phys) };
        let has = |f: &vk::QueueFamilyProperties, b: vk::QueueFlags| f.queue_flags.contains(b);
        let qfam = fams
            .iter()
            .position(|f| {
                has(f, vk::QueueFlags::COMPUTE) && has(f, vk::QueueFlags::GRAPHICS)
            })
            .or_else(|| fams.iter().position(|f| has(f, vk::QueueFlags::COMPUTE)))
            .ok_or_else(|| format!("{}: no compute queue family", info.name))?
            as u32;

        let prio = [1.0f32];
        let qci = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(qfam)
            .queue_priorities(&prio)];

        // Enable only what is required plus what is free and already probed.
        // scalarBlockLayout is the load-bearing one (module header).
        let mut f12 = vk::PhysicalDeviceVulkan12Features::default().scalar_block_layout(true);
        let mut f13 = vk::PhysicalDeviceVulkan13Features::default();
        if info.subgroup_size_control {
            f13 = f13.subgroup_size_control(true).compute_full_subgroups(true);
        }
        let dci = vk::DeviceCreateInfo::default()
            .queue_create_infos(&qci)
            .push_next(&mut f12)
            .push_next(&mut f13);
        let device = unsafe { instance.create_device(phys, &dci, None) }
            .map_err(|e| format!("vkCreateDevice on {}: {e}", info.name))?;
        let queue = unsafe { device.get_device_queue(qfam, 0) };
        let mem = unsafe { instance.get_physical_device_memory_properties(phys) };

        Ok(Vk { entry, instance, phys, device, queue, qfam, mem, info, debug })
    }

    /// Everything the pick and the loud line need, plus the reason this device
    /// cannot be used if it cannot.
    fn probe(
        instance: &ash::Instance,
        p: vk::PhysicalDevice,
    ) -> (DeviceInfo, Option<String>) {
        let mut driver = vk::PhysicalDeviceDriverProperties::default();
        let mut sg = vk::PhysicalDeviceSubgroupSizeControlProperties::default();
        let mut sgp = vk::PhysicalDeviceSubgroupProperties::default();
        let mut props2 = vk::PhysicalDeviceProperties2::default()
            .push_next(&mut driver)
            .push_next(&mut sg)
            .push_next(&mut sgp);
        unsafe { instance.get_physical_device_properties2(p, &mut props2) };
        let props = props2.properties;

        let mut f12 = vk::PhysicalDeviceVulkan12Features::default();
        let mut f13 = vk::PhysicalDeviceVulkan13Features::default();
        let mut feats2 =
            vk::PhysicalDeviceFeatures2::default().push_next(&mut f12).push_next(&mut f13);
        unsafe { instance.get_physical_device_features2(p, &mut feats2) };

        let exts = unsafe { instance.enumerate_device_extension_properties(p) }.unwrap_or_default();
        let has_ext = |n: &CStr| exts.iter().any(|e| cstr(&e.extension_name) == n.to_string_lossy());

        let api = props.api_version;
        let info = DeviceInfo {
            name: cstr(&props.device_name),
            driver: {
                let d = cstr(&driver.driver_name);
                let v = cstr(&driver.driver_info);
                if v.is_empty() { d } else { format!("{d} {v}") }
            },
            api: (vk::api_version_major(api), vk::api_version_minor(api), vk::api_version_patch(api)),
            kind: props.device_type,
            vendor_id: props.vendor_id,
            subgroup_size: sgp.subgroup_size,
            // A device that reports no control range still HAS a width, so
            // degrade the range onto it rather than printing [0..0] — a probe
            // that says "range zero" reads as broken hardware when what it
            // means is "no size control here".
            subgroup_min: if sg.min_subgroup_size == 0 {
                sgp.subgroup_size
            } else {
                sg.min_subgroup_size
            },
            subgroup_max: if sg.max_subgroup_size == 0 {
                sgp.subgroup_size
            } else {
                sg.max_subgroup_size
            },
            subgroup_size_control: f13.subgroup_size_control == vk::TRUE,
            subgroup_pin_compute: sg
                .required_subgroup_size_stages
                .contains(vk::ShaderStageFlags::COMPUTE),
            max_workgroup_subgroups: sg.max_compute_workgroup_subgroups,
            ray_query: has_ext(ash::khr::ray_query::NAME),
            accel_struct: has_ext(ash::khr::acceleration_structure::NAME),
            rt_pipeline: has_ext(ash::khr::ray_tracing_pipeline::NAME),
        };

        // Hard requirements, each traceable to a decision already made.
        let reject = if api < REQ_API {
            Some(format!(
                "reports Vulkan {}.{} — the SPIR-V corpus targets 1.3",
                info.api.0, info.api.1
            ))
        } else if f12.scalar_block_layout != vk::TRUE {
            Some("lacks scalarBlockLayout, which -fvk-use-dx-layout requires".to_string())
        } else {
            None
        };
        (info, reject)
    }

    /// The `subgroupSize` the driver reports as its natural width — what an
    /// UNPINNED pipeline gets. Kept as a named accessor rather than a bare
    /// field read because the distinction from the control range is the whole
    /// M2c question, and a call site that says `natural_subgroup_size()` cannot
    /// be misread as "the width my kernel got" (which only a probe can answer).
    pub fn natural_subgroup_size(&self) -> u32 {
        self.info.subgroup_size
    }

    pub fn buffer(
        &self,
        size: u64,
        usage: vk::BufferUsageFlags,
        host: bool,
    ) -> Result<Buffer, String> {
        let size = size.max(4);
        let bci = vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let buf = unsafe { self.device.create_buffer(&bci, None) }
            .map_err(|e| format!("vkCreateBuffer({size} B): {e}"))?;
        let req = unsafe { self.device.get_buffer_memory_requirements(buf) };
        let want = if host {
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT
        } else {
            vk::MemoryPropertyFlags::DEVICE_LOCAL
        };
        let idx = mem_type_index(&self.mem, req.memory_type_bits, want).ok_or_else(|| {
            format!("no memory type for {size} B with {want:?} (mask {:#x})", req.memory_type_bits)
        })?;
        let ai = vk::MemoryAllocateInfo::default().allocation_size(req.size).memory_type_index(idx);
        let mem = unsafe { self.device.allocate_memory(&ai, None) }.map_err(|e| {
            unsafe { self.device.destroy_buffer(buf, None) };
            format!("vkAllocateMemory({} B): {e}", req.size)
        })?;
        unsafe { self.device.bind_buffer_memory(buf, mem, 0) }
            .map_err(|e| format!("vkBindBufferMemory: {e}"))?;
        Ok(Buffer { buf, mem, size, host })
    }

    pub fn free_buffer(&self, b: &Buffer) {
        unsafe {
            self.device.destroy_buffer(b.buf, None);
            self.device.free_memory(b.mem, None);
        }
    }

    /// Host-visible write. Coherent by construction (`buffer(.., host: true)`
    /// asks for HOST_COHERENT), so there is no flush to forget.
    pub fn write(&self, b: &Buffer, bytes: &[u8]) -> Result<(), String> {
        if !b.host {
            return Err("write() on a device-local buffer".into());
        }
        if bytes.len() as u64 > b.size {
            return Err(format!("write {} B into a {} B buffer", bytes.len(), b.size));
        }
        unsafe {
            let p = self
                .device
                .map_memory(b.mem, 0, b.size, vk::MemoryMapFlags::empty())
                .map_err(|e| format!("vkMapMemory: {e}"))? as *mut u8;
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), p, bytes.len());
            self.device.unmap_memory(b.mem);
        }
        Ok(())
    }

    pub fn read(&self, b: &Buffer, len: usize) -> Result<Vec<u8>, String> {
        if !b.host {
            return Err("read() on a device-local buffer".into());
        }
        let mut out = vec![0u8; len.min(b.size as usize)];
        unsafe {
            let p = self
                .device
                .map_memory(b.mem, 0, b.size, vk::MemoryMapFlags::empty())
                .map_err(|e| format!("vkMapMemory: {e}"))? as *const u8;
            std::ptr::copy_nonoverlapping(p, out.as_mut_ptr(), out.len());
            self.device.unmap_memory(b.mem);
        }
        Ok(out)
    }

    /// Validation is ACTUALLY armed — not merely requested. The two differ
    /// whenever the layer is not installed (`Vk::new` says so and continues),
    /// and a gate that reports the request instead of the fact would log a
    /// validated run and an unvalidated one identically.
    pub fn validated(&self) -> bool {
        self.debug.is_some()
    }

    pub fn line(&self) -> String {
        let i = &self.info;
        let rt = match (i.ray_query, i.accel_struct, i.rt_pipeline) {
            (true, true, true) => "ray-query + rt-pipeline",
            (true, true, false) => "ray-query",
            (false, true, _) => "accel-struct only",
            _ => "no ray tracing",
        };
        format!(
            "vk: {} ({} {}, {}) — Vulkan {}.{}.{}, subgroup {} [{}..{}]{}, {}",
            i.name,
            i.vendor_str(),
            i.kind_str(),
            i.driver,
            i.api.0,
            i.api.1,
            i.api.2,
            i.subgroup_size,
            i.subgroup_min,
            i.subgroup_max,
            if i.subgroup_size_control { " pinnable" } else { "" },
            rt
        )
    }
}

impl Drop for Vk {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            self.device.destroy_device(None);
            if let Some((di, m)) = &self.debug {
                di.destroy_debug_utils_messenger(*m, None);
            }
            self.instance.destroy_instance(None);
        }
    }
}

// ---------------------------------------------------------------------------

/// The pure half's gate. Runs in `--check-vk` before anything is loaded, so it
/// is meaningful on a box with no Vulkan at all.
pub fn self_test() -> Result<(), String> {
    use vk::PhysicalDeviceType as T;
    let d = type_rank(T::DISCRETE_GPU);
    let ig = type_rank(T::INTEGRATED_GPU);
    let sw = type_rank(T::CPU);
    let other = type_rank(T::OTHER);
    if !(d > ig && ig > sw && sw > other) {
        return Err(format!("type_rank order: discrete {d}, integrated {ig}, cpu {sw}, other {other}"));
    }

    // THE DEV BOX'S OWN HAZARD: a real iGPU beside llvmpipe. Getting this
    // backwards does not fail — it renders correctly and a hundred times
    // slower, and every number taken afterwards describes the software device.
    let two = [Cand::new("Radeon 8060S Graphics (RADV STRIX_HALO)", ig, None),
               Cand::new("llvmpipe (LLVM 20.1.0, 256 bits)", sw, None)];
    if pick(&two, None).map_err(|e| e.msg)? != 0 {
        return Err("pick chose the software device over real hardware".into());
    }
    // ...and the same list in the other order, so the answer is a property of
    // the ranking and not of enumeration order.
    let flipped = [two[1].clone(), two[0].clone()];
    if pick(&flipped, None).map_err(|e| e.msg)? != 1 {
        return Err("pick depends on enumeration order".into());
    }

    let three = [Cand::new("iGPU", ig, None), Cand::new("dGPU", d, None), Cand::new("sw", sw, None)];
    if pick(&three, None).map_err(|e| e.msg)? != 1 {
        return Err("pick did not prefer the discrete device".into());
    }
    // A rejected device is never chosen, however it ranks — the teeth against
    // "highest rank wins" quietly outranking a hard requirement.
    let three_bad = [
        Cand::new("iGPU", ig, None),
        Cand::new("dGPU", d, Some("lacks scalarBlockLayout")),
        Cand::new("sw", sw, None),
    ];
    if pick(&three_bad, None).map_err(|e| e.msg)? != 0 {
        return Err("pick chose a device it had rejected".into());
    }
    // Ties break by index, deterministically.
    let tie = [Cand::new("a", ig, None), Cand::new("b", ig, None)];
    if pick(&tie, None).map_err(|e| e.msg)? != 0 {
        return Err("tie did not break by enumeration index".into());
    }
    // All rejected: an error naming every reason, never a fallback pick.
    let none = [Cand::new("a", d, Some("too old")), Cand::new("b", ig, Some("no scalar layout"))];
    match pick(&none, None) {
        Ok(i) => return Err(format!("pick returned {i} with every device rejected")),
        Err(e) => {
            if !e.msg.contains("too old") || !e.msg.contains("no scalar layout") {
                return Err(format!("all-rejected error dropped a reason: {}", e.msg));
            }
            // ...and it is an ABSENT condition: hardware present, none usable.
            if !e.absent {
                return Err("all-rejected did not report as an absent-Vulkan condition".into());
            }
        }
    }
    if pick(&[], None).is_ok() {
        return Err("pick accepted an empty device list".into());
    }

    // The force lever. It must be able to pick a LOWER-ranked device (that is
    // what it is for), by index or by substring...
    if pick(&three, Some("sw")).map_err(|e| e.msg)? != 2 {
        return Err("FR_VK_DEVICE substring did not override rank".into());
    }
    if pick(&three, Some("2")).map_err(|e| e.msg)? != 2 {
        return Err("FR_VK_DEVICE index did not override rank".into());
    }
    if pick(&three, Some("IGPU")).map_err(|e| e.msg)? != 0 {
        return Err("FR_VK_DEVICE substring is case-sensitive".into());
    }
    // ...and must REFUSE rather than silently fall back, in all three ways it
    // can fail to name exactly one usable device.
    if pick(&three, Some("9")).is_ok() {
        return Err("FR_VK_DEVICE accepted an out-of-range index".into());
    }
    if pick(&three, Some("nope")).is_ok() {
        return Err("FR_VK_DEVICE accepted a name matching nothing".into());
    }
    let amb = [Cand::new("AMD one", ig, None), Cand::new("AMD two", ig, None)];
    if pick(&amb, Some("amd")).is_ok() {
        return Err("FR_VK_DEVICE accepted an ambiguous substring".into());
    }
    if pick(&three_bad, Some("dGPU")).is_ok() {
        return Err("FR_VK_DEVICE forced a rejected device instead of erroring".into());
    }

    // Memory-type selection.
    let mut mp = vk::PhysicalDeviceMemoryProperties::default();
    let dl = vk::MemoryPropertyFlags::DEVICE_LOCAL;
    let hv = vk::MemoryPropertyFlags::HOST_VISIBLE;
    let hc = vk::MemoryPropertyFlags::HOST_COHERENT;
    mp.memory_type_count = 4;
    mp.memory_types[0].property_flags = vk::MemoryPropertyFlags::empty();
    mp.memory_types[1].property_flags = dl;
    mp.memory_types[2].property_flags = hv | hc;
    mp.memory_types[3].property_flags = dl | hv | hc;
    if mem_type_index(&mp, !0, dl) != Some(1) {
        return Err("mem_type_index did not take the first DEVICE_LOCAL type".into());
    }
    if mem_type_index(&mp, !0, hv | hc) != Some(2) {
        return Err("mem_type_index did not take the first host-coherent type".into());
    }
    // A superset satisfies the request; a type outside the mask does NOT —
    // the requirement mask is a hardware constraint, not a preference, and
    // ignoring it is a bind failure at best and a wrong heap at worst.
    if mem_type_index(&mp, 1 << 3, hv | hc) != Some(3) {
        return Err("mem_type_index rejected a superset of the wanted flags".into());
    }
    if mem_type_index(&mp, (1 << 0) | (1 << 1), hv) != None {
        return Err("mem_type_index ignored the requirement mask".into());
    }
    if mem_type_index(&mp, 0, dl) != None {
        return Err("mem_type_index matched under an empty mask".into());
    }
    Ok(())
}
