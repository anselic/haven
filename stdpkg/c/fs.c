// Filesystem primitives, exposed to the language via std/fs. Windows and POSIX
// disagree on `mkdir`'s arity, on whether `remove` unlinks a directory, and on
// `stat`'s spelling; those differences are normalized here so std/fs stays a
// thin, portable wrapper (the same role rt_alloc plays for std/alloc).
//
// The error convention mirrors std/alloc's "failure is a value": every
// side-effecting call returns 0 on success or the captured C `errno` (a positive
// int) on failure, which std/fs wraps in an `FsError`. The value-returning read
// hands back a malloc'd, NUL-terminated buffer (freeable by rt_free, so std/fs
// can adopt it directly into a `Vec<u8>`), or NULL with the errno in *out_err.

// std/fs uses fopen/strerror; MSVC deprecates them in favor of _s variants that
// don't exist elsewhere. We use them portably and correctly, so opt out of the nag.
#define _CRT_SECURE_NO_WARNINGS

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>


#ifdef _WIN32
  #include <direct.h>            // _mkdir, _rmdir
  #define RT_MKDIR(p) _mkdir(p)
  #define RT_RMDIR(p) _rmdir(p)
  #ifndef S_ISDIR
    #define S_ISDIR(m) (((m) & S_IFMT) == S_IFDIR)
  #endif
#else
  #include <sys/types.h>
  #include <unistd.h>            // rmdir
  #define RT_MKDIR(p) mkdir((p), 0777)
  #define RT_RMDIR(p) rmdir(p)
#endif

int32_t rt_fs_mkdir(const char* path) {
    errno = 0;
    return RT_MKDIR(path) == 0 ? 0 : (errno ? errno : 1);
}

int32_t rt_fs_rename(const char* src, const char* dst) {
    errno = 0;
    return rename(src, dst) == 0 ? 0 : (errno ? errno : 1);
}

// Remove a file or (empty) directory. `remove` unlinks files everywhere but on
// some platforms (notably Windows) won't touch a directory, so fall back to
// rmdir when the path turns out to be one.
int32_t rt_fs_remove(const char* path) {
    errno = 0;
    if (remove(path) == 0) return 0;
    struct stat s;
    if (stat(path, &s) == 0 && S_ISDIR(s.st_mode)) {
        errno = 0;
        if (RT_RMDIR(path) == 0) return 0;
    }
    return errno ? errno : 1;
}

uint8_t* rt_fs_read(const char* path, uint64_t* out_len, int32_t* out_err) {
    errno = 0;
    FILE* f = fopen(path, "rb");
    if (!f) { *out_err = errno ? errno : 1; return NULL; }
    if (fseek(f, 0, SEEK_END) != 0) { *out_err = errno ? errno : 1; fclose(f); return NULL; }
    long n = ftell(f);
    if (n < 0) { *out_err = errno ? errno : 1; fclose(f); return NULL; }
    rewind(f);
    // one extra byte for the NUL std/string relies on for a borrowable `str`.
    uint8_t* buf = (uint8_t*)malloc((size_t)n + 1);
    if (!buf) { *out_err = ENOMEM; fclose(f); return NULL; }
    size_t rd = fread(buf, 1, (size_t)n, f);
    if (ferror(f)) { *out_err = errno ? errno : 1; free(buf); fclose(f); return NULL; }
    fclose(f);
    buf[rd] = 0;
    *out_len = (uint64_t)rd;
    return buf;
}

int32_t rt_fs_write(const char* path, const uint8_t* data, uint64_t len, uint64_t* out_written) {
    errno = 0;
    FILE* f = fopen(path, "wb");
    if (!f) { *out_written = 0; return errno ? errno : 1; }
    size_t wr = fwrite(data, 1, (size_t)len, f);
    int err = (wr != (size_t)len) ? (errno ? errno : 1) : 0;
    if (fclose(f) != 0 && !err) err = errno ? errno : 1;
    *out_written = (uint64_t)wr;
    return err;
}

int32_t rt_fs_copy(const char* src, const char* dst, uint64_t* out_bytes) {
    errno = 0;
    *out_bytes = 0;
    FILE* in = fopen(src, "rb");
    if (!in) return errno ? errno : 1;
    FILE* out = fopen(dst, "wb");
    if (!out) { int e = errno ? errno : 1; fclose(in); return e; }
    char b[65536];
    size_t n;
    int err = 0;
    uint64_t total = 0;
    while ((n = fread(b, 1, sizeof b, in)) > 0) {
        if (fwrite(b, 1, n, out) != n) { err = errno ? errno : 1; break; }
        total += (uint64_t)n;
    }
    if (!err && ferror(in)) err = errno ? errno : 1;
    if (fclose(out) != 0 && !err) err = errno ? errno : 1;
    fclose(in);
    *out_bytes = total;
    return err;
}

int32_t rt_fs_exists(const char* path) {
    struct stat s;
    return stat(path, &s) == 0 ? 1 : 0;
}

int32_t rt_fs_is_dir(const char* path) {
    struct stat s;
    if (stat(path, &s) != 0) return 0;
    return S_ISDIR(s.st_mode) ? 1 : 0;
}

const char* rt_fs_strerror(int32_t code) { return strerror(code); }