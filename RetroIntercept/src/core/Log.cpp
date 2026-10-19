// Thread-safe rotating file logger; writes RetroIntercept.log next to the proxy DLL.
#include "Log.h"

#include <Windows.h>
#include <cstdarg>
#include <cstdio>
#include <mutex>

namespace
{

std::mutex g_mutex;
FILE* g_file = nullptr;
char g_dir[MAX_PATH] = {};
constexpr unsigned kMaxLogBytes = 4u * 1024u * 1024u;

const char* SeverityName(Log::Severity sev) noexcept
{
    switch (sev)
    {
    case Log::Severity::Debug: return "DBG";
    case Log::Severity::Info:  return "INF";
    case Log::Severity::Warn:  return "WRN";
    case Log::Severity::Error: return "ERR";
    }
    return "???";
}

// Resolves the directory containing this module (the proxy DLL lives next to the game exe).
void ResolveModuleDir() noexcept
{
    HMODULE self = nullptr;
    if (GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                           GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           reinterpret_cast<LPCWSTR>(&ResolveModuleDir), &self) == FALSE)
        return;

    wchar_t path[MAX_PATH] = {};
    const DWORD len = GetModuleFileNameW(self, path, MAX_PATH);
    if (len == 0 || len >= MAX_PATH)
        return;

    wchar_t* slash = path;
    for (wchar_t* p = path; *p; ++p)
        if (*p == L'\\' || *p == L'/')
            slash = p;
    *slash = L'\0';
    WideCharToMultiByte(CP_UTF8, 0, path, -1, g_dir, MAX_PATH, nullptr, nullptr);
}

void RotateIfLarge(const char* fullPath) noexcept
{
    WIN32_FILE_ATTRIBUTE_DATA attr = {};
    wchar_t wpath[MAX_PATH] = {};
    MultiByteToWideChar(CP_UTF8, 0, fullPath, -1, wpath, MAX_PATH);
    if (GetFileAttributesExW(wpath, GetFileExInfoStandard, &attr) == FALSE)
        return;
    if (attr.nFileSizeHigh != 0 || attr.nFileSizeLow < kMaxLogBytes)
        return;

    char prev[MAX_PATH] = {};
    _snprintf_s(prev, _TRUNCATE, "%s\\RetroIntercept.1.log", g_dir);
    wchar_t wprev[MAX_PATH] = {};
    MultiByteToWideChar(CP_UTF8, 0, prev, -1, wprev, MAX_PATH);
    MoveFileExW(wpath, wprev, MOVEFILE_REPLACE_EXISTING);
}

} // namespace

void Log::Init() noexcept
{
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_file != nullptr)
        return;

    ResolveModuleDir();

    char path[MAX_PATH] = {};
    _snprintf_s(path, _TRUNCATE, "%s\\RetroIntercept.log", g_dir[0] ? g_dir : ".");
    RotateIfLarge(path);

    g_file = _fsopen(path, "ab", _SH_DENYNO); // shared: tail-able while running
}

const char* Log::ModuleDir() noexcept
{
    return g_dir;
}

void Log::Shutdown() noexcept
{
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_file != nullptr)
    {
        fclose(g_file);
        g_file = nullptr;
    }
}

void FlushIfDue() noexcept
{
    static DWORD last = 0;
    const DWORD now = GetTickCount();
    if (now - last >= 250)
    {
        last = now;
        if (g_file != nullptr)
            fflush(g_file);
    }
}

void Log::VWrite(Severity sev, const char* tag, const char* fmt, va_list args) noexcept
{
    char msg[2048] = {};
    _vsnprintf_s(msg, _TRUNCATE, fmt, args);

    SYSTEMTIME st = {};
    GetLocalTime(&st);
    const DWORD tid = GetCurrentThreadId();

    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_file != nullptr)
    {
        fprintf_s(g_file, "[%02u:%02u:%02u.%03u] [t%05u] [%s] [%s] %s\n",
                  st.wHour, st.wMinute, st.wSecond, st.wMilliseconds,
                  tid, SeverityName(sev), tag ? tag : "-", msg);
        fflush(g_file);
    }
    // Mirror to debugger output when no file sink is available.
    if (g_file == nullptr)
    {
        char line[2300] = {};
        _snprintf_s(line, _TRUNCATE, "[RI] [%s] [%s] %s\n",
                    SeverityName(sev), tag ? tag : "-", msg);
        OutputDebugStringA(line);
    }
}

void Log::Debug(const char* tag, const char* fmt, ...) noexcept
{
    va_list args;
    va_start(args, fmt);
    VWrite(Severity::Debug, tag, fmt, args);
    va_end(args);
}

void Log::Info(const char* tag, const char* fmt, ...) noexcept
{
    va_list args;
    va_start(args, fmt);
    VWrite(Severity::Info, tag, fmt, args);
    va_end(args);
}

void Log::Warn(const char* tag, const char* fmt, ...) noexcept
{
    va_list args;
    va_start(args, fmt);
    VWrite(Severity::Warn, tag, fmt, args);
    va_end(args);
}

void Log::Error(const char* tag, const char* fmt, ...) noexcept
{
    va_list args;
    va_start(args, fmt);
    VWrite(Severity::Error, tag, fmt, args);
    va_end(args);
}
