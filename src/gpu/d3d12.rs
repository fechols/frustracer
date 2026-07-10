//! Core D3D12 machinery: device, queue, swapchain, per-frame ring, fence
//! sync, and the small barrier/copy helpers everything else records with.
//!
//! Frame pacing: FRAMES_IN_FLIGHT per-frame slots {command allocator, fence
//! value, upload ranges}. The fence is waited *before touching a slot's
//! upload memory*, not after present — v1 is effectively synchronous (the CPU
//! trace dwarfs GPU work), but CPU/GPU overlap later is a loop reorder, not a
//! rewrite.

use std::mem::ManuallyDrop;
use windows::core::Interface;
use windows::Win32::Foundation::{CloseHandle, HANDLE, HWND};
use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_12_0;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::Win32::Graphics::Dxgi::*;
use windows::Win32::System::Threading::{CreateEventW, WaitForSingleObject, INFINITE};

pub const FRAMES_IN_FLIGHT: usize = 2;
pub const BACKBUFFERS: u32 = 3;
pub const SWAPCHAIN_FORMAT: DXGI_FORMAT = DXGI_FORMAT_B8G8R8A8_UNORM;

pub type Result<T> = std::result::Result<T, String>;

fn err<T>(ctx: &str, e: windows::core::Error) -> Result<T> {
    Err(format!("{ctx}: {e}"))
}

pub struct FrameSlot {
    pub allocator: ID3D12CommandAllocator,
    pub fence_value: u64,
}

pub struct D3d {
    pub device: ID3D12Device,
    pub queue: ID3D12CommandQueue,
    /// Declared before `swapchain`: the buffer refs pin the native swapchain,
    /// so they must release first (fields drop in declaration order) or SL
    /// warns about destroying the proxy while the native is still referenced.
    pub backbuffers: Vec<ID3D12Resource>,
    pub swapchain: IDXGISwapChain3,
    pub rtv_heap: ID3D12DescriptorHeap,
    pub rtv_size: u32,
    pub list: ID3D12GraphicsCommandList,
    pub slots: Vec<FrameSlot>,
    pub fence: ID3D12Fence,
    pub fence_event: HANDLE,
    pub next_fence: u64,
    pub frame_index: usize,
    pub width: u32,
    pub height: u32,
}

/// Create the native D3D12 device on the picked adapter (debug layer first —
/// it must be enabled before device creation).
pub fn create_device(adapter: &IDXGIAdapter4, debug: bool) -> Result<ID3D12Device> {
    if debug {
        let mut dbg: Option<ID3D12Debug> = None;
        if unsafe { D3D12GetDebugInterface(&mut dbg) }.is_ok() {
            if let Some(d) = dbg {
                unsafe { d.EnableDebugLayer() };
                eprintln!("gpu: D3D12 debug layer enabled");
            }
        }
    }
    let mut device: Option<ID3D12Device> = None;
    if let Err(e) = unsafe { D3D12CreateDevice(adapter, D3D_FEATURE_LEVEL_12_0, &mut device) } {
        return err("D3D12CreateDevice", e);
    }
    Ok(device.unwrap())
}

/// Create a direct command queue. In the Streamline path the `device` passed
/// here is the SL *proxy* device — that hook is what lets DLSS Frame
/// Generation own presentation pacing later.
pub fn create_queue(device: &ID3D12Device) -> Result<ID3D12CommandQueue> {
    unsafe {
        device.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
            Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
            ..Default::default()
        })
    }
    .map_err(|e| format!("CreateCommandQueue: {e}"))
}

impl D3d {
    /// Build the swapchain + frame machinery around an existing device/queue
    /// (M3 passes the Streamline proxy queue here so present is hooked).
    pub fn with_queue(
        factory: &IDXGIFactory6,
        device: ID3D12Device,
        queue: ID3D12CommandQueue,
        hwnd: HWND,
        width: u32,
        height: u32,
    ) -> Result<Self> {
        let sc_desc = DXGI_SWAP_CHAIN_DESC1 {
            Width: width,
            Height: height,
            Format: SWAPCHAIN_FORMAT,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            BufferUsage: DXGI_USAGE_RENDER_TARGET_OUTPUT,
            BufferCount: BACKBUFFERS,
            SwapEffect: DXGI_SWAP_EFFECT_FLIP_DISCARD,
            ..Default::default()
        };
        let sc1: IDXGISwapChain1 = unsafe {
            factory.CreateSwapChainForHwnd(&queue, hwnd, &sc_desc, None, None)
        }
        .map_err(|e| format!("CreateSwapChainForHwnd: {e}"))?;
        let swapchain: IDXGISwapChain3 =
            sc1.cast().map_err(|e| format!("IDXGISwapChain3 cast: {e}"))?;

        let rtv_heap: ID3D12DescriptorHeap = unsafe {
            device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
                NumDescriptors: BACKBUFFERS,
                ..Default::default()
            })
        }
        .map_err(|e| format!("CreateDescriptorHeap(RTV): {e}"))?;
        let rtv_size =
            unsafe { device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV) };
        let rtv0 = unsafe { rtv_heap.GetCPUDescriptorHandleForHeapStart() };
        let mut backbuffers = Vec::with_capacity(BACKBUFFERS as usize);
        for i in 0..BACKBUFFERS {
            let buf: ID3D12Resource = unsafe { swapchain.GetBuffer(i) }
                .map_err(|e| format!("swapchain GetBuffer({i}): {e}"))?;
            let handle = D3D12_CPU_DESCRIPTOR_HANDLE {
                ptr: rtv0.ptr + (i * rtv_size) as usize,
            };
            unsafe { device.CreateRenderTargetView(&buf, None, handle) };
            backbuffers.push(buf);
        }

        let mut slots = Vec::with_capacity(FRAMES_IN_FLIGHT);
        for _ in 0..FRAMES_IN_FLIGHT {
            let allocator: ID3D12CommandAllocator = unsafe {
                device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
            }
            .map_err(|e| format!("CreateCommandAllocator: {e}"))?;
            slots.push(FrameSlot { allocator, fence_value: 0 });
        }
        let list: ID3D12GraphicsCommandList = unsafe {
            device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &slots[0].allocator, None)
        }
        .map_err(|e| format!("CreateCommandList: {e}"))?;
        unsafe { list.Close() }.map_err(|e| format!("initial list Close: {e}"))?;

        let fence: ID3D12Fence = unsafe { device.CreateFence(0, D3D12_FENCE_FLAG_NONE) }
            .map_err(|e| format!("CreateFence: {e}"))?;
        let fence_event = unsafe { CreateEventW(None, false, false, None) }
            .map_err(|e| format!("CreateEventW: {e}"))?;

        Ok(Self {
            device,
            queue,
            swapchain,
            backbuffers,
            rtv_heap,
            rtv_size,
            list,
            slots,
            fence,
            fence_event,
            next_fence: 1,
            frame_index: 0,
            width,
            height,
        })
    }

    pub fn rtv_handle(&self, backbuffer: u32) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        let start = unsafe { self.rtv_heap.GetCPUDescriptorHandleForHeapStart() };
        D3D12_CPU_DESCRIPTOR_HANDLE { ptr: start.ptr + (backbuffer * self.rtv_size) as usize }
    }

    fn wait_for_fence(&self, value: u64) -> Result<()> {
        if value == 0 || unsafe { self.fence.GetCompletedValue() } >= value {
            return Ok(());
        }
        unsafe { self.fence.SetEventOnCompletion(value, self.fence_event) }
            .map_err(|e| format!("SetEventOnCompletion: {e}"))?;
        unsafe { WaitForSingleObject(self.fence_event, INFINITE) };
        Ok(())
    }

    /// Wait until this frame's slot is free, reset its allocator + the list.
    /// Returns the slot index; upload memory for that slot is safe to touch
    /// after this returns.
    pub fn begin_frame(&mut self) -> Result<usize> {
        let slot = self.frame_index % FRAMES_IN_FLIGHT;
        self.wait_for_fence(self.slots[slot].fence_value)?;
        unsafe { self.slots[slot].allocator.Reset() }
            .map_err(|e| format!("allocator Reset: {e}"))?;
        unsafe { self.list.Reset(&self.slots[slot].allocator, None) }
            .map_err(|e| format!("list Reset: {e}"))?;
        Ok(slot)
    }

    /// Close, execute, present, signal. Call after recording the frame.
    pub fn end_frame(&mut self, slot: usize) -> Result<()> {
        unsafe { self.list.Close() }.map_err(|e| format!("list Close: {e}"))?;
        let lists = [Some(self.list.cast::<ID3D12CommandList>().unwrap())];
        unsafe { self.queue.ExecuteCommandLists(&lists) };
        unsafe { self.swapchain.Present(1, DXGI_PRESENT(0)) }
            .ok()
            .map_err(|e| format!("Present: {e}"))?;
        let v = self.next_fence;
        self.next_fence += 1;
        unsafe { self.queue.Signal(&self.fence, v) }.map_err(|e| format!("queue Signal: {e}"))?;
        self.slots[slot].fence_value = v;
        self.frame_index += 1;
        Ok(())
    }

    /// Close, execute, signal, and BLOCK until the GPU finishes — a frame
    /// submission without a Present, for work whose output the CPU reads
    /// back immediately (the post-upscale denoise path). The slot's fence
    /// bookkeeping matches `end_frame`, so the frame ring stays consistent
    /// and the caller's Map of a readback buffer is safe on return.
    pub fn submit_and_wait(&mut self, slot: usize) -> Result<()> {
        unsafe { self.list.Close() }.map_err(|e| format!("list Close: {e}"))?;
        let lists = [Some(self.list.cast::<ID3D12CommandList>().unwrap())];
        unsafe { self.queue.ExecuteCommandLists(&lists) };
        let v = self.next_fence;
        self.next_fence += 1;
        unsafe { self.queue.Signal(&self.fence, v) }.map_err(|e| format!("queue Signal: {e}"))?;
        self.slots[slot].fence_value = v;
        self.frame_index += 1;
        self.wait_for_fence(v)
    }

    pub fn wait_idle(&mut self) -> Result<()> {
        let v = self.next_fence;
        self.next_fence += 1;
        unsafe { self.queue.Signal(&self.fence, v) }.map_err(|e| format!("queue Signal: {e}"))?;
        self.wait_for_fence(v)
    }

    /// Abandon a partially recorded frame: close the list without executing
    /// it (recorded barriers/copies never reach the GPU, so tracked resource
    /// states stay truthful) and leave the slot's fence untouched so the next
    /// begin_frame reuses it. Recovery path for a mid-frame error.
    pub fn abort_frame(&mut self) {
        let _ = unsafe { self.list.Close() };
    }

    /// Record and synchronously execute a one-off command list outside the
    /// frame loop (drains in-flight work before borrowing slot 0's allocator
    /// and drains again after). For rare operations like readbacks.
    pub fn run_once<F: FnOnce(&ID3D12GraphicsCommandList)>(&mut self, f: F) -> Result<()> {
        self.wait_idle()?;
        unsafe { self.slots[0].allocator.Reset() }
            .map_err(|e| format!("run_once allocator Reset: {e}"))?;
        unsafe { self.list.Reset(&self.slots[0].allocator, None) }
            .map_err(|e| format!("run_once list Reset: {e}"))?;
        f(&self.list);
        unsafe { self.list.Close() }.map_err(|e| format!("run_once list Close: {e}"))?;
        let lists = [Some(self.list.cast::<ID3D12CommandList>().unwrap())];
        unsafe { self.queue.ExecuteCommandLists(&lists) };
        self.wait_idle()
    }
}

impl Drop for D3d {
    fn drop(&mut self) {
        let _ = self.wait_idle();
        let _ = unsafe { CloseHandle(self.fence_event) };
    }
}

/// Transition barrier. The `transmute_copy` pattern is the standard
/// windows-rs idiom: the union field is `ManuallyDrop<Option<ID3D12Resource>>`
/// and we hand it a borrowed refcount that ManuallyDrop keeps from releasing.
pub fn transition(
    res: &ID3D12Resource,
    from: D3D12_RESOURCE_STATES,
    to: D3D12_RESOURCE_STATES,
) -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Transition: ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                pResource: unsafe { std::mem::transmute_copy(res) },
                StateBefore: from,
                StateAfter: to,
                Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
            }),
        },
    }
}

/// Copy location for a texture subresource — either side of a
/// CopyTextureRegion (upload dst, readback src).
pub fn loc_subresource(res: &ID3D12Resource) -> D3D12_TEXTURE_COPY_LOCATION {
    D3D12_TEXTURE_COPY_LOCATION {
        pResource: unsafe { std::mem::transmute_copy(res) },
        Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 },
    }
}

/// Copy location for a buffer with a placed footprint — either side of a
/// CopyTextureRegion (upload src, readback dst).
pub fn loc_footprint(
    res: &ID3D12Resource,
    footprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
) -> D3D12_TEXTURE_COPY_LOCATION {
    D3D12_TEXTURE_COPY_LOCATION {
        pResource: unsafe { std::mem::transmute_copy(res) },
        Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: footprint },
    }
}

pub fn default_heap() -> D3D12_HEAP_PROPERTIES {
    D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() }
}

pub fn upload_heap() -> D3D12_HEAP_PROPERTIES {
    D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_UPLOAD, ..Default::default() }
}

pub fn readback_heap() -> D3D12_HEAP_PROPERTIES {
    D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() }
}

pub fn tex2d_desc(w: u32, h: u32, format: DXGI_FORMAT, flags: D3D12_RESOURCE_FLAGS) -> D3D12_RESOURCE_DESC {
    D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        Alignment: 0,
        Width: w as u64,
        Height: h,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: format,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
        Flags: flags,
    }
}

pub fn buffer_desc(size: u64) -> D3D12_RESOURCE_DESC {
    D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Alignment: 0,
        Width: size,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: DXGI_FORMAT_UNKNOWN,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: D3D12_RESOURCE_FLAG_NONE,
    }
}

pub fn committed_tex(
    device: &ID3D12Device,
    w: u32,
    h: u32,
    format: DXGI_FORMAT,
    flags: D3D12_RESOURCE_FLAGS,
    initial: D3D12_RESOURCE_STATES,
) -> Result<ID3D12Resource> {
    let mut res: Option<ID3D12Resource> = None;
    unsafe {
        device.CreateCommittedResource(
            &default_heap(),
            D3D12_HEAP_FLAG_NONE,
            &tex2d_desc(w, h, format, flags),
            initial,
            None,
            &mut res,
        )
    }
    .map_err(|e| format!("CreateCommittedResource(tex {w}x{h} {format:?}): {e}"))?;
    Ok(res.unwrap())
}

/// Persistently-mapped upload buffer.
pub struct UploadBuffer {
    pub resource: ID3D12Resource,
    pub ptr: *mut u8,
    pub size: usize,
}

// The mapped pointer is only written from the render thread between
// begin_frame (fence wait) and end_frame; D3D12 upload heaps are always
// CPU-visible write-combined memory.
unsafe impl Send for UploadBuffer {}

impl UploadBuffer {
    pub fn new(device: &ID3D12Device, size: usize) -> Result<Self> {
        let mut res: Option<ID3D12Resource> = None;
        unsafe {
            device.CreateCommittedResource(
                &upload_heap(),
                D3D12_HEAP_FLAG_NONE,
                &buffer_desc(size as u64),
                D3D12_RESOURCE_STATE_GENERIC_READ,
                None,
                &mut res,
            )
        }
        .map_err(|e| format!("CreateCommittedResource(upload {size}): {e}"))?;
        let resource = res.unwrap();
        let mut ptr = std::ptr::null_mut();
        unsafe { resource.Map(0, None, Some(&mut ptr)) }.map_err(|e| format!("Map: {e}"))?;
        Ok(Self { resource, ptr: ptr as *mut u8, size })
    }
}

/// GPU→CPU staging buffer for rare readbacks (screenshots). Created on
/// demand; mapped only for the duration of the read.
pub struct ReadbackBuffer {
    pub resource: ID3D12Resource,
}

impl ReadbackBuffer {
    pub fn new(device: &ID3D12Device, size: usize) -> Result<Self> {
        let mut res: Option<ID3D12Resource> = None;
        unsafe {
            device.CreateCommittedResource(
                &readback_heap(),
                D3D12_HEAP_FLAG_NONE,
                &buffer_desc(size as u64),
                D3D12_RESOURCE_STATE_COPY_DEST,
                None,
                &mut res,
            )
        }
        .map_err(|e| format!("CreateCommittedResource(readback {size}): {e}"))?;
        Ok(Self { resource: res.unwrap() })
    }
}

pub const ROW_ALIGN: usize = D3D12_TEXTURE_DATA_PITCH_ALIGNMENT as usize; // 256

pub fn aligned_pitch(row_bytes: usize) -> usize {
    (row_bytes + ROW_ALIGN - 1) & !(ROW_ALIGN - 1)
}

pub fn footprint(
    format: DXGI_FORMAT,
    w: u32,
    h: u32,
    bytes_per_px: usize,
    offset: u64,
) -> D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
        Offset: offset,
        Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
            Format: format,
            Width: w,
            Height: h,
            Depth: 1,
            RowPitch: aligned_pitch(w as usize * bytes_per_px) as u32,
        },
    }
}
