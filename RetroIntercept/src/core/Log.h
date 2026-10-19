// Thread-safe rotating file logger; writes RetroIntercept.log next to the proxy DLL.
#pragma once

#include <cstdarg>

class Log
{
public:
    enum class Severity { Debug, Info, Warn, Error };

    // Opens the log in append mode in the directory containing this module.
    // Rotates to RetroIntercept.1.log when the file exceeds kMaxBytes.
    static void Init() noexcept;
    static void Shutdown() noexcept;

    // Directory containing the proxy DLL (valid after Init). Never null.
    static const char* ModuleDir() noexcept;

    static void Debug(const char* tag, const char* fmt, ...) noexcept;
    static void Info (const char* tag, const char* fmt, ...) noexcept;
    static void Warn (const char* tag, const char* fmt, ...) noexcept;
    static void Error(const char* tag, const char* fmt, ...) noexcept;

private:
    static void VWrite(Severity sev, const char* tag, const char* fmt, va_list args) noexcept;
};
