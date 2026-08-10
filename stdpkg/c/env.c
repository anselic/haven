// Process environment, exposed to the language via std/env. Windows and POSIX
// disagree on how you set a variable (`setenv` vs `_putenv_s`) and spell the
// working-directory call (`getcwd` vs `_getcwd`), and neither hands `main` its
// argv the way haven's zero-arg `main` wants; those differences are normalized
// here so std/env stays a thin, portable wrapper (the same role rt_fs_* plays
// for std/fs).
//
// The error convention mirrors std/fs: the side-effecting `set` returns 0 on
// success or the captured C `errno`, and the value-returning `get`/`cwd` hand
// back a malloc'd, NUL-terminated buffer (freeable by rt_free, so std/env can
// adopt it directly into a `Vec<u8>`/`String`) or NULL.

#define _CRT_SECURE_NO_WARNINGS

#include <errno.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
  #include <direct.h>            // _getcwd
  #define RT_GETCWD _getcwd
#else
  #include <unistd.h>            // getcwd
  #define RT_GETCWD getcwd
#endif

// --- process arguments ------------------------------------------------------
// haven's `main` takes no argc/argv, so the runtime captures them out of band.
// On glibc/musl and macOS, `.init_array` constructors are invoked with
// (argc, argv, envp); on Windows the CRT exposes them as the __argc/__argv
// globals. Either way the argv strings live for the whole program, so std/env
// hands each one back as a borrowed `str`.
#ifdef _WIN32
static int    rt_g_argc(void) { return __argc; }
static char** rt_g_argv(void) { return __argv; }
#else
static int    g_argc = 0;
static char** g_argv = NULL;
__attribute__((constructor))
static void rt_capture_args(int argc, char** argv, char** envp) {
    (void)envp;
    g_argc = argc;
    g_argv = argv;
}
static int    rt_g_argc(void) { return g_argc; }
static char** rt_g_argv(void) { return g_argv; }
#endif

uint64_t rt_env_argc(void) {
    int n = rt_g_argc();
    return n > 0 ? (uint64_t)n : 0;
}

// The i-th argument (argv[i]), or NULL if out of range.
const char* rt_env_arg(uint64_t i) {
    char** v = rt_g_argv();
    if (!v || i >= rt_env_argc()) return NULL;
    return v[i];
}

// --- variables --------------------------------------------------------------
// A malloc'd, NUL-terminated copy of the value of $name, or NULL if it is unset.
// *out_len gets the length (excluding the NUL). getenv's own buffer is not
// guaranteed stable across calls, so we copy it for the caller to own.
uint8_t* rt_env_get(const char* name, uint64_t* out_len) {
    *out_len = 0;
    const char* v = getenv(name);
    if (!v) return NULL;
    size_t n = strlen(v);
    uint8_t* buf = (uint8_t*)malloc(n + 1);
    if (!buf) return NULL;
    memcpy(buf, v, n + 1);   // include the trailing NUL
    *out_len = (uint64_t)n;
    return buf;
}

int32_t rt_env_set(const char* name, const char* value) {
    errno = 0;
#ifdef _WIN32
    return _putenv_s(name, value) == 0 ? 0 : (errno ? errno : 1);
#else
    return setenv(name, value, 1) == 0 ? 0 : (errno ? errno : 1);
#endif
}

// --- working directory ------------------------------------------------------
// The current working directory as a malloc'd, NUL-terminated buffer, or NULL
// with *out_err set. Grows the buffer until the path fits (ERANGE = too small).
uint8_t* rt_env_cwd(uint64_t* out_len, int32_t* out_err) {
    *out_len = 0;
    size_t cap = 512;
    for (;;) {
        char* buf = (char*)malloc(cap);
        if (!buf) { *out_err = ENOMEM; return NULL; }
        errno = 0;
        if (RT_GETCWD(buf, cap) != NULL) {
            *out_len = (uint64_t)strlen(buf);
            return (uint8_t*)buf;
        }
        free(buf);
        if (errno != ERANGE) { *out_err = errno ? errno : 1; return NULL; }
        cap *= 2;
    }
}

// --- exit -------------------------------------------------------------------
[[noreturn]] void rt_env_exit(int32_t code) { exit((int)code); }

const char* rt_env_strerror(int32_t code) { return strerror(code); }
