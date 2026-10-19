# CRender — D3D11 proxy DLL + test harness (/MT, C++20). Builds x86 and x64.
cmake_minimum_required(VERSION 3.21)
project(CRender LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Static CRT: the proxy DLL must not depend on vcruntime*.dll
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

if(CMAKE_SIZEOF_VOID_P EQUAL 8)
  message(STATUS "CRender: x64 build (works in 64-bit D3D11 games)")
else()
  message(STATUS "CRender: x86 build (works in 32-bit D3D11 games)")
endif()

add_subdirectory(third_party/minhook)

# --- proxy DLL (d3d11.dll) ------------------------------------------
# Target is retro_proxy so the import lib (retro_proxy.lib) never shadows the SDK's
# d3d11.lib; the produced binary is still d3d11.dll.
add_library(retro_proxy SHARED
  src/dllmain.cpp
  src/core/Log.cpp
  src/proxy/ProxyDll.cpp
  src/proxy/ProxyExports.cpp
  src/proxy/exports.def
  src/d3d11/HookManager.cpp
  src/d3d11/RetroPass.cpp
  src/d3d11/ConsoleWire.cpp
  src/d3d11/Targets.cpp
  src/term/Tracker.cpp
  src/term/Hud.cpp
  src/ai/AiPass.cpp
  src/ai/Detect.cpp
  src/ai/OrtApi.cpp
)
target_include_directories(retro_proxy PRIVATE src third_party/minhook/include)

# x64: MSVC has no __asm, so the 49 naked jmp stubs are generated as MASM instead.
# Each stub tail-jumps through its resolved slot; if unresolved it calls the resolver
# (args are in rcx/rdx/r8/r9/xmm0-3 on x64, so they are saved/restored around the call).
if (CMAKE_SIZEOF_VOID_P EQUAL 8)
  enable_language(ASM_MASM)
  file(READ "${CMAKE_SOURCE_DIR}/src/proxy/FwdList.inc" _fwdlist)
  string(REGEX MATCHALL "FWD_ENTRY\\([A-Za-z0-9_]+\\)[\r\n]" _fwd_entries "${_fwdlist}")
  set(_asm ".code\nEXTERN EnsureRealD3D11Impl:proc\n")
  foreach(_e ${_fwd_entries})
    string(REGEX REPLACE "FWD_ENTRY\\(([A-Za-z0-9_]+)\\).*" "\\1" _n "${_e}")
    string(STRIP "${_n}" _n)
    string(APPEND _asm
"EXTERN g_fwd_${_n}:qword
PUBLIC ${_n}
${_n} proc
    cmp qword ptr [g_fwd_${_n}], 0
    jne short @f
    push rcx
    push rdx
    push r8
    push r9
    sub rsp, 68h
    movdqu [rsp+20h], xmm0
    movdqu [rsp+30h], xmm1
    movdqu [rsp+40h], xmm2
    movdqu [rsp+50h], xmm3
    call EnsureRealD3D11Impl
    movdqu xmm0, [rsp+20h]
    movdqu xmm1, [rsp+30h]
    movdqu xmm2, [rsp+40h]
    movdqu xmm3, [rsp+50h]
    add rsp, 68h
    pop r9
    pop r8
    pop rdx
    pop rcx
@@:
    jmp qword ptr [g_fwd_${_n}]
${_n} endp
")
  endforeach()
  string(APPEND _asm "end\n")
  set(_asm_file "${CMAKE_BINARY_DIR}/fwd_stubs_x64.asm")
  file(WRITE "${_asm_file}" "${_asm}")
  target_sources(retro_proxy PRIVATE "${_asm_file}")
  set_source_files_properties("${_asm_file}" PROPERTIES LANGUAGE ASM_MASM)
endif()

# Terminator ONNX base: ONNX Runtime (DirectML EP) headers + arch DLLs fetched by
# tools/ai/fetch_ort.ps1 (x86 -> bin/, x64 -> bin64/). DLLs are LoadLibrary'd at
# runtime — nothing is linked, the import table stays KERNEL32-only. Headers are
# arch-neutral; the define is gated on the matching-arch runtime actually existing.
set(ORT_DIR "${CMAKE_SOURCE_DIR}/third_party/onnxruntime")
if (CMAKE_SIZEOF_VOID_P EQUAL 4)
  set(ORT_BIN "${ORT_DIR}/bin")
else()
  set(ORT_BIN "${ORT_DIR}/bin64")
endif()
if (EXISTS "${ORT_DIR}/include/onnxruntime_c_api.h" AND EXISTS "${ORT_DIR}/include/DirectML.h"
    AND EXISTS "${ORT_BIN}/onnxruntime.dll")
  target_include_directories(retro_proxy PRIVATE "${ORT_DIR}/include")
  target_compile_definitions(retro_proxy PRIVATE RETRO_HAVE_ORT=1)
  message(STATUS "CRender: Terminator ONNX path enabled (ONNX Runtime at ${ORT_BIN})")
else()
  target_compile_definitions(retro_proxy PRIVATE RETRO_HAVE_ORT=0)
  message(STATUS "CRender: Terminator ONNX path disabled (HLSL fallback) — run tools/ai/fetch_ort.ps1")
endif()
target_link_libraries(retro_proxy PRIVATE minhook)
target_compile_options(retro_proxy PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:/W4 /WX /permissive- /utf-8>)
target_compile_definitions(retro_proxy PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX UNICODE _UNICODE)
set_target_properties(retro_proxy PROPERTIES
  PREFIX "" OUTPUT_NAME "d3d11" SUFFIX ".dll"
  ARCHIVE_OUTPUT_NAME "retro_proxy")
# Emit a linker .map so crash RVAs (e.g. from the VEH forensics log) resolve to
# source-level symbols without needing the PDB tooling on the game machine.
if (MSVC)
  target_link_options(retro_proxy PRIVATE "/MAP")
endif()

# Auto-deploy: copy the proxy next to Heaven.exe on every x86 build (Heaven is 32-bit).
set(HEAVEN_BIN "F:/Heaven Benchmark 4.0/bin" CACHE PATH "Heaven benchmark bin dir")
if (CMAKE_SIZEOF_VOID_P EQUAL 4 AND EXISTS "${HEAVEN_BIN}")
  add_custom_command(TARGET retro_proxy POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:retro_proxy> "${HEAVEN_BIN}/d3d11.dll"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_SOURCE_DIR}/config/retro.json" "${HEAVEN_BIN}/retro.json"
    COMMENT "Deploying d3d11.dll + retro.json to ${HEAVEN_BIN}")
  foreach(_ai_file "${ORT_BIN}/onnxruntime.dll" "${ORT_BIN}/DirectML.dll"
                   "${CMAKE_SOURCE_DIR}/config/retro_ai.onnx"
                   "${CMAKE_SOURCE_DIR}/config/retro_ai_temporal.onnx"
                   "${CMAKE_SOURCE_DIR}/config/terminator_vision.onnx"
                   "${CMAKE_SOURCE_DIR}/config/yolov8n.onnx"
                   "${CMAKE_SOURCE_DIR}/config/det_weapons.onnx")
    if (EXISTS "${_ai_file}")
      get_filename_component(_ai_name "${_ai_file}" NAME)
      add_custom_command(TARGET retro_proxy POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${_ai_file}" "${HEAVEN_BIN}/${_ai_name}"
        COMMENT "Deploying ${_ai_name} to ${HEAVEN_BIN}")
    endif()
  endforeach()
endif()

# --- test harness (rotating cube, heavy/light pixel shaders) --------
add_executable(test_cube WIN32 tools/test_cube/main.cpp)
target_compile_options(test_cube PRIVATE /W4 /WX /permissive- /utf-8)
target_compile_definitions(test_cube PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX UNICODE _UNICODE)
target_link_libraries(test_cube PRIVATE d3d11.lib dxgi.lib d3dcompiler.lib)
