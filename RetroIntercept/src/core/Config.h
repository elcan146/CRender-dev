// Config — tiny helpers on top of the proxy's strstr-style retro.json scanning.
//
// The legacy parser finds keys anywhere in the file (nested "Output"/"ShaderStrip" keys rely
// on that). Preset blocks such as "TerminatorLens" reuse key names ("Crt", "Vignette"), so
// ExtractObject copies such a block out and blanks it in the source buffer: the legacy scan
// never sees it, and the block is parsed on its own.
#pragma once

#include <cstdlib>
#include <cstring>

namespace cfg
{

// Finds "key": { ... } in buf, copies the braces-inclusive object into out (NUL-terminated,
// truncated to outSize), and overwrites the whole `"key": {...}` span in buf with spaces.
// String-aware brace matching. Returns false if the key or a well-formed object is absent.
inline bool ExtractObject(char* buf, const char* key, char* out, size_t outSize) noexcept
{
    if (buf == nullptr || key == nullptr || out == nullptr || outSize == 0)
        return false;
    out[0] = 0;
    char quoted[64];
    const size_t kl = strlen(key);
    if (kl + 3 > sizeof(quoted))
        return false;
    quoted[0] = '"';
    memcpy(quoted + 1, key, kl);
    quoted[kl + 1] = '"';
    quoted[kl + 2] = 0;
    char* k = strstr(buf, quoted);
    if (k == nullptr)
        return false;
    char* p = k + kl + 2;
    while (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n') ++p;
    if (*p != ':')
        return false;
    ++p;
    while (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n') ++p;
    if (*p != '{')
        return false;
    char* begin = p;
    int depth = 0;
    bool inStr = false;
    for (; *p != 0; ++p)
    {
        if (inStr)
        {
            if (*p == '\\' && p[1] != 0) { ++p; continue; }
            if (*p == '"') inStr = false;
            continue;
        }
        if (*p == '"') inStr = true;
        else if (*p == '{') ++depth;
        else if (*p == '}' && --depth == 0) break;
    }
    if (*p != '}')
        return false;
    const size_t n = static_cast<size_t>(p - begin + 1);
    const size_t c = n < outSize - 1 ? n : outSize - 1;
    memcpy(out, begin, c);
    out[c] = 0;
    for (char* q = k; q <= p; ++q)
        if (*q != '\r' && *q != '\n')
            *q = ' ';
    return true;
}

// Points just past the ':' of "key" in text, or nullptr.
inline const char* ValueOf(const char* text, const char* key) noexcept
{
    if (text == nullptr || key == nullptr)
        return nullptr;
    char quoted[64];
    const size_t kl = strlen(key);
    if (kl + 3 > sizeof(quoted))
        return nullptr;
    quoted[0] = '"';
    memcpy(quoted + 1, key, kl);
    quoted[kl + 1] = '"';
    quoted[kl + 2] = 0;
    const char* k = strstr(text, quoted);
    if (k == nullptr)
        return nullptr;
    const char* c = strchr(k + kl + 2, ':');
    return c != nullptr ? c + 1 : nullptr;
}

inline bool ReadFloat(const char* text, const char* key, float* v) noexcept
{
    const char* c = ValueOf(text, key);
    if (c == nullptr)
        return false;
    *v = static_cast<float>(atof(c));
    return true;
}

inline bool ReadInt(const char* text, const char* key, int* v) noexcept
{
    const char* c = ValueOf(text, key);
    if (c == nullptr)
        return false;
    *v = atoi(c);
    return true;
}

// Accepts true/false or a number (non-zero = true).
inline bool ReadBool(const char* text, const char* key, bool* v) noexcept
{
    const char* c = ValueOf(text, key);
    if (c == nullptr)
        return false;
    while (*c == ' ' || *c == '\t') ++c;
    if (strncmp(c, "true", 4) == 0)       *v = true;
    else if (strncmp(c, "false", 5) == 0) *v = false;
    else                                  *v = atoi(c) != 0;
    return true;
}

// "key": "value" — copies the quoted string into out (NUL-terminated, truncated).
inline bool ReadString(const char* text, const char* key, char* out, size_t outSize) noexcept
{
    if (out == nullptr || outSize == 0)
        return false;
    out[0] = 0;
    const char* c = ValueOf(text, key);
    if (c == nullptr)
        return false;
    while (*c == ' ' || *c == '\t') ++c;
    if (*c != '"')
        return false;
    const char* e = strchr(c + 1, '"');
    if (e == nullptr)
        return false;
    const size_t n = static_cast<size_t>(e - c - 1);
    const size_t m = n < outSize - 1 ? n : outSize - 1;
    memcpy(out, c + 1, m);
    out[m] = 0;
    return true;
}

// "key": [a, b, c]
inline bool ReadFloat3(const char* text, const char* key, float v[3]) noexcept
{
    const char* c = ValueOf(text, key);
    if (c == nullptr)
        return false;
    c = strchr(c, '[');
    if (c == nullptr)
        return false;
    char* end = nullptr;
    float t[3];
    ++c;
    for (int i = 0; i < 3; ++i)
    {
        t[i] = strtof(c, &end);
        if (end == c)
            return false;
        c = end;
        while (*c == ' ' || *c == ',' || *c == '\t' || *c == '\r' || *c == '\n') ++c;
    }
    memcpy(v, t, sizeof(t));
    return true;
}

} // namespace cfg
