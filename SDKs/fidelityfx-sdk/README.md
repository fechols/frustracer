# FidelityFX SDK header subset (vendored)

MIT-licensed header subset of the AMD FidelityFX SDK, vendored from
https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK at tag **v2.3.0**:

- `api/include/` — `Kits/FidelityFX/api/include` (ffx-api core + DX12 backend descs)
- `denoisers/include/` — `Kits/FidelityFX/denoisers/include` (FSR Ray Regeneration 1.2.0)
- `upscalers/include/` — `Kits/FidelityFX/upscalers/include` (FSR4 upscaler)

The directory layout preserves the headers' own relative includes
(`ffx_denoiser.h` includes `../../api/include/ffx_api.h`). Every vendored file
carries the MIT license block. Nothing links these at build time beyond the
`shim/ffx_shim.cpp` translation unit; the runtime is reached by loading
`amd_fidelityfx_loader_dx12.dll` dynamically.

The signed runtime DLLs (`amd_fidelityfx_loader_dx12.dll`,
`amd_fidelityfx_denoiser_dx12.dll`, `amd_fidelityfx_upscaler_dx12.dll`) are NOT
committed — they ship in the `FidelityFX-Samples-v2.3.0-prebuilt.zip` release
asset, extracted at `SDKs/FidelityFX-Samples-prebuilt/`; the default
`--ffx-path` points at its `Samples/Denoisers/.../dx12/x64/Release` directory
(override with `--ffx-path` or `FRUSTRACER_FFX_PATH`).
