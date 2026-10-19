// OrtApi — runtime resolution of ONNX Runtime (DirectML EP) for the AI pass.
// onnxruntime.dll / DirectML.dll are LoadLibrary'd from the proxy's directory so the
// proxy's import table stays KERNEL32-only. Missing DLLs never crash — Load() fails.
#pragma once

#if RETRO_HAVE_ORT

#include <d3d12.h>

#pragma warning(push, 0)
#include <onnxruntime_c_api.h>
#include <dml_provider_factory.h>
#pragma warning(pop)

namespace ai
{

// Resolves OrtApi + OrtDmlApi once (latched). dir = folder holding the DLLs (UTF-8/ANSI).
bool Load(const char* dir) noexcept;

const OrtApi*    Api() noexcept;
const OrtDmlApi* Dml() noexcept;

// DMLCreateDevice from the DirectML.dll pinned by Load() (same module ORT uses).
using FnDMLCreateDevice = HRESULT(WINAPI*)(ID3D12Device*, DML_CREATE_DEVICE_FLAGS, REFIID, void**);
FnDMLCreateDevice DmlCreateDevice() noexcept;

// D3D12CreateDevice from d3d12.dll.
PFN_D3D12_CREATE_DEVICE D3D12Create() noexcept;

// Logs + releases a non-null status. Returns true when st == nullptr (success).
bool Ok(OrtStatus* st, const char* what) noexcept;

} // namespace ai

#endif // RETRO_HAVE_ORT
