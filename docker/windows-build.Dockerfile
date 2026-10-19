# escape=`
# Windows build image — compiles retro_proxy (d3d11.dll) + test_cube.exe for
# BOTH Win32/x86 and x64 using VS Build Tools + CMake + Ninja inside a Windows
# container. The resulting DLL still has to be run on a real Windows desktop —
# a container has no display, GPU games, or D3D11 swap chains.
#
# Requirements on the host:
#   - Docker Desktop in "Windows containers" mode (Windows host, WSL2/Hyper-V)
#   - Container OS must match the host kernel family. ltsc2022 works on
#     Windows 10 21H2+/11 via --isolation=process; add that flag to docker
#     build/run if the default hyper-v isolation complains.
#
# Build (from repo root — image is ~10-15 GB, first pull/install is slow):
#   docker build -f docker/windows-build.Dockerfile -t crender-build-win .
#
# Extract artifacts:
#   docker create --name cbuild crender-build-win
#   docker cp cbuild:C:/src/build/win32/Release/d3d11.dll ./d3d11-x86.dll
#   docker cp cbuild:C:/src/build/x64/Release/d3d11.dll   ./d3d11-x64.dll
#   docker rm cbuild
FROM mcr.microsoft.com/windows/servercore:ltsc2022

# VS channel: project is built with VS 2026 ("Visual Studio 18 2026" preset).
# Stable aka.ms channels: 17/release (VS2022) or 18/release once available.
# VS2022 (v14.4x) compiles this C++20 code fine, but /WX is strict — if newer
# warnings break the build, set VS_CHANNEL=18/release or drop /WX locally.
ARG VS_CHANNEL=17/release

SHELL ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command"]

RUN Invoke-WebRequest -UseBasicParsing `
      -Uri "https://aka.ms/vs/$env:VS_CHANNEL/vs_BuildTools.exe" `
      -OutFile C:\vs_BuildTools.exe ; `
    $p = Start-Process -Wait -PassThru -FilePath C:\vs_BuildTools.exe -ArgumentList `
      '--quiet','--wait','--norestart','--nocache', `
      '--installPath','C:\BuildTools', `
      '--add','Microsoft.VisualStudio.Workload.VCTools', `
      '--add','Microsoft.VisualStudio.Component.Windows11SDK.22621', `
      '--add','Microsoft.VisualStudio.Component.VC.Tools.x86.x64', `
      '--add','Microsoft.VisualStudio.Component.VC.CMake.Project' ; `
    Remove-Item C:\vs_BuildTools.exe -Force ; `
    if ($p.ExitCode -notin 0,3010) { throw "vs_BuildTools exit $($p.ExitCode)" }

WORKDIR C:/src
COPY . .

# ORT-DML headers/DLLs for the Terminator path (comment out to build RETRO_HAVE_ORT=0).
RUN & 'C:\src\tools\ai\fetch_ort.ps1'

SHELL ["cmd", "/S", "/C"]

# vcvars sets cl/ml64/etc; prepend the bundled CMake+Ninja so presets aren't
# needed (the "Visual Studio 18 2026" generator doesn't exist under VS2022 —
# Ninja drives the same compiler). Builds both architectures.
ENV CMAKE_DIR=C:\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake

RUN call "C:\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x86 && `
    set "PATH=%CMAKE_DIR%\CMake\bin;%CMAKE_DIR%\Ninja;%PATH%" && `
    cmake -S . -B build\win32 -G Ninja -DCMAKE_BUILD_TYPE=Release && `
    cmake --build build\win32

RUN call "C:\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" amd64 && `
    set "PATH=%CMAKE_DIR%\CMake\bin;%CMAKE_DIR%\Ninja;%PATH%" && `
    cmake -S . -B build\x64 -G Ninja -DCMAKE_BUILD_TYPE=Release && `
    cmake --build build\x64

# Keep the image alive long enough for `docker create`/`docker cp` extraction.
ENTRYPOINT ["cmd", "/c", "dir C:\\src\\build /s /b | findstr /i \".dll .exe\" && echo. && echo Build tree ready — docker cp the artifacts out."]
