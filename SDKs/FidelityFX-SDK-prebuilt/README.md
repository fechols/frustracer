# Pre-built FidelityFX FSR3-Upscaler SPIR-V shader headers (vendored)

**200 generated `.h` files, ~17 MiB, plain git.** These are the only files under
`SDKs/` that are committed besides the v2.3.0 ffx-api header subset and the NRD
submodule, and they are committed for one reason:

> The FidelityFX SDK compiles its HLSL to SPIR-V with `FidelityFX_SC.exe`, a
> **Windows-only** driver bundled in the SDK's own `sdk/tools/binary_store/`, and
> upstream ships **no pre-compiled SPIR-V**. A Linux or macOS checkout therefore
> cannot produce them at all.

Vendoring the output is what makes FSR3 buildable off Windows. They are MIT and
redistributable — unlike every other SDK this tree consumes, which is why this
one may be committed while the rest are fetched.

## Why FidelityFX 1.1.4 and not the ffx-api the Windows build uses

Two FidelityFX generations coexist here on purpose. `install-prerequisites.sh`'s
`FFX_SRC_TAG` carries the full argument; the short version is that ffx-api
v2.3.0 ships as **signed prebuilt DX12 provider DLLs**, has no Vulkan backend
(its own readme lists *"Vulkan is currently not supported in SDK"* under known
issues), and **removed the `FfxInterface` custom-backend seam** that a Metal
implementation would have to be written against. v1.1.4 is the last generation
that is MIT source with a stock first-party `ffx_vk` backend and a seam. Windows
keeps v2.3.0 (it has FSR4, Ray Regeneration and frame generation); Vulkan and
Metal use these.

## Both backends read this one directory

Vulkan consumes the SPIR-V directly. **Metal consumes the same bytes** —
`build.rs` transpiles each permutation to a `.metallib` at build time
(`spirv-cross --msl` → `xcrun metal -c` → `metallib`), content-addressed by an
FNV-1a hash of the SPIR-V so the shim can look up the pass FFX asks for. So
there is no Metal shader artifact to vendor and none is committed.

Only the **non-wave64** permutations are ever used on Metal (Apple GPUs are
SIMD-32, and the shim reports `waveLaneCount = 32` so FFX never requests the
wave64 set). They are kept here anyway because the Vulkan backend may want them
on hardware that has wave64.

```
SDKs/FidelityFX-SDK-prebuilt/
├── README.md          # this file
└── shaders/
    └── vk/            # 200 generated headers: ffx_fsr3upscaler_*
```

The `*.h.d` depfiles the generator drops alongside are build litter and are
gitignored.

## Regenerating them (needs a Windows host)

Only necessary when bumping `FFX_SRC_TAG`. Nothing in the normal build runs this.

**Prerequisites:** Windows 10/11 with Visual Studio 2022 (v1.1.4's
`toolchain.cmake` hard-codes MSVC), the LunarG Vulkan SDK ≥ 1.3.250 (the SDK's
`CheckVulkanSDKVersion()` rejects older), and CMake ≥ 3.23.

```powershell
git clone https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK.git
cd FidelityFX-SDK
git checkout v1.1.4          # commit c6efa6bf

mkdir build_vk && cd build_vk
cmake ..\sdk -A x64 ^
    -DFFX_API_BACKEND=VK_X64 ^
    -DFFX_FSR3UPSCALER=ON ^
    -DFFX_FI=OFF ^
    -DFFX_OF=OFF ^
    -DFFX_FSR3=OFF ^
    -DFFX_FSR2=OFF ^
    -DFFX_FSR1=OFF ^
    -DFFX_AUTO_COMPILE_SHADERS=ON
cmake --build . --config Release --target ffx_backend_vk_x64
```

Turning `FFX_FI` / `FFX_OF` / `FFX_FSR3` **off** is not just thrift: those are
the frame-generation and optical-flow components, whose sources are MSVC-only
and have known portability problems. This tree builds the **upscaler only**.

The headers land in `build_vk\shaders\vk\`. Copy only the `.h` files back:

```powershell
robocopy build_vk\shaders\vk <frustracer>\SDKs\FidelityFX-SDK-prebuilt\shaders\vk *.h /MIR
```

Then bump `FFX_SRC_TAG` in `install-prerequisites.sh` and the version probe in
`build.rs` **together** — they are a pin pair, like `NRD_TAG` and `src/nrd.rs`.

## What is NOT here

The SDK **source** (`SDKs/FidelityFX-SDK/`) is gitignored and fetched:

```
./install-prerequisites.sh fsr3src
```

That pulls the v1.1.4 tarball and extracts four paths (`sdk/include`, `sdk/src`,
`sdk/libs`, `sdk/CMakeLists.txt`) — ~19 MB of the 189 MB archive. The 154 MB
`sdk/tools/` tree is deliberately skipped: it exists to run the Windows-only
shader compiler above, which is exactly the step this directory removes.

`build.rs` needs **both halves** and degrades loudly with a `cargo:warning` when
either is missing — FSR3 is simply absent from that build, never a hard error.
