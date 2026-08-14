// Process spawning, exposed to the language via std/process. Windows and POSIX
// disagree on how you launch a child, wait for it, and read its exit status;
// those differences are normalized here so std/process stays a thin, portable
// wrapper (the same role rt_fs_* plays for std/fs).
//
// Two shapes are offered. `rt_proc_run` spawns a child that inherits this
// process's stdout/stderr (its output flows straight to the terminal) and hands
// back only the exit code. `rt_proc_capture` redirects the child's stdout and
// stderr into temp files, then slurps each into a malloc'd, NUL-terminated
// buffer (freeable by rt_free, so std/process can adopt it directly into a
// `Vec<u8>`/`String`).
//
// The error convention mirrors std/fs: every call returns 0 when the child was
// launched and waited on, or a positive C `errno` when the *spawn itself* failed
// (program not found, etc.), which std/process wraps in a `ProcError`. A child
// that runs and exits nonzero is NOT an error here - that shows up as the exit
// code in *out_code, for the caller to interpret.

#define _CRT_SECURE_NO_WARNINGS

#include <errno.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#ifdef _WIN32
  #include <io.h>            // _dup, _dup2, _close, _read, _lseeki64, _open
  #include <fcntl.h>         // _O_* flags
  #include <process.h>       // _spawnvp, _P_WAIT
  #include <windows.h>       // GetTempPathA, GetTempFileNameA
  #define RT_DUP    _dup
  #define RT_DUP2   _dup2
  #define RT_CLOSE  _close
  #define RT_READ   _read
  #define RT_LSEEK  _lseeki64
#else
  #include <unistd.h>        // dup, dup2, close, read, lseek, unlink
  #include <spawn.h>         // posix_spawnp
  #include <sys/wait.h>      // waitpid, WIF*
  extern char** environ;
  #define RT_DUP    dup
  #define RT_DUP2   dup2
  #define RT_CLOSE  close
  #define RT_READ   read
  #define RT_LSEEK  lseek
#endif

// Build a NUL-terminated argv: [program, args[0..nargs), NULL]. `args` may be
// NULL when nargs == 0 (an empty Vec has a null buffer). Caller frees the array
// (not the strings, which are borrowed). Returns NULL on allocation failure.
static char** rt_build_argv(const char* program, const char* const* args, uint64_t nargs) {
    char** argv = (char**)malloc((size_t)(nargs + 2) * sizeof(char*));
    if (!argv) return NULL;
    argv[0] = (char*)program;
    for (uint64_t i = 0; i < nargs; i++) {
        argv[i + 1] = (char*)args[i];
    }
    argv[nargs + 1] = NULL;
    return argv;
}

// Spawn `program` (searched on PATH) with `argv`, wait for it, and store its
// exit status in *out_code. Returns 0 if the child was launched and reaped, or a
// positive errno if the spawn failed. A child killed by a signal reports as
// 128 + signum (the shell convention), matching what a POSIX shell would show.
static int32_t rt_spawn_wait(const char* program, char* const argv[], int32_t* out_code) {
#ifdef _WIN32
    errno = 0;
    intptr_t status = _spawnvp(_P_WAIT, program, (const char* const*)argv);
    if (status == -1) return errno ? errno : 1;
    *out_code = (int32_t)status;
    return 0;
#else
    pid_t pid;
    int rc = posix_spawnp(&pid, program, NULL, NULL, argv, environ);
    if (rc != 0) return rc;  // posix_spawnp returns an errno directly, not via errno
    int status = 0;
    while (waitpid(pid, &status, 0) < 0) {
        if (errno != EINTR) return errno ? errno : 1;
    }
    if (WIFEXITED(status))        *out_code = (int32_t)WEXITSTATUS(status);
    else if (WIFSIGNALED(status)) *out_code = 128 + (int32_t)WTERMSIG(status);
    else                          *out_code = -1;
    return 0;
#endif
}

// Open a fresh temp file for read+write, deleted automatically on close. Returns
// an fd, or -1 on failure (with errno set).
static int rt_temp_fd(void) {
#ifdef _WIN32
    char dir[MAX_PATH];
    char path[MAX_PATH];
    if (GetTempPathA(MAX_PATH, dir) == 0) { errno = EACCES; return -1; }
    if (GetTempFileNameA(dir, "hvp", 0, path) == 0) { errno = EACCES; return -1; }
    // GetTempFileNameA created the file; reopen it, delete-on-last-close.
    return _open(path, _O_RDWR | _O_BINARY | _O_TEMPORARY);
#else
    const char* tmp = getenv("TMPDIR");
    char path[4096];
    snprintf(path, sizeof path, "%s/hvpXXXXXX", (tmp && *tmp) ? tmp : "/tmp");
    int fd = mkstemp(path);
    if (fd >= 0) unlink(path);  // unlink now; the fd keeps it alive until close
    return fd;
#endif
}

// Read all of `fd` from offset 0 into a malloc'd, NUL-terminated buffer. Sets
// *out_len to the byte count (excluding the NUL). Returns the buffer, or NULL on
// failure (leaving *out_len at 0).
static uint8_t* rt_slurp_fd(int fd, uint64_t* out_len) {
    *out_len = 0;
    if (RT_LSEEK(fd, 0, SEEK_SET) < 0) return NULL;
    size_t cap = 4096;
    size_t len = 0;
    uint8_t* buf = (uint8_t*)malloc(cap);
    if (!buf) return NULL;
    for (;;) {
        // keep room for a full chunk plus the trailing NUL.
        if (len + 4096 + 1 > cap) {
            size_t ncap = cap * 2;
            uint8_t* nb = (uint8_t*)realloc(buf, ncap);
            if (!nb) { free(buf); return NULL; }
            buf = nb;
            cap = ncap;
        }
        int n = RT_READ(fd, buf + len, 4096);
        if (n < 0) { free(buf); return NULL; }
        if (n == 0) break;
        len += (size_t)n;
    }
    buf[len] = 0;
    *out_len = (uint64_t)len;
    return buf;
}

int32_t rt_proc_run(const char* program, const char* const* args, uint64_t nargs, int32_t* out_code) {
    *out_code = -1;
    char** argv = rt_build_argv(program, args, nargs);
    if (!argv) return ENOMEM;
    int32_t rc = rt_spawn_wait(program, argv, out_code);
    free(argv);
    return rc;
}

int32_t rt_proc_capture(const char* program, const char* const* args, uint64_t nargs,
                        int32_t* out_code,
                        uint8_t** out_stdout, uint64_t* out_stdout_len,
                        uint8_t** out_stderr, uint64_t* out_stderr_len) {
    *out_code = -1;
    *out_stdout = NULL; *out_stdout_len = 0;
    *out_stderr = NULL; *out_stderr_len = 0;

    char** argv = rt_build_argv(program, args, nargs);
    if (!argv) return ENOMEM;

    int ofd = rt_temp_fd();
    int efd = rt_temp_fd();
    if (ofd < 0 || efd < 0) {
        int e = errno ? errno : 1;
        if (ofd >= 0) RT_CLOSE(ofd);
        if (efd >= 0) RT_CLOSE(efd);
        free(argv);
        return e;
    }

    // Redirect this process's stdout/stderr onto the temp files across the spawn
    // (the child inherits fds 0/1/2), then restore. Flush first so any buffered
    // parent output lands on the real terminal, not in the child's capture file.
    fflush(stdout);
    fflush(stderr);
    int save_out = RT_DUP(1);
    int save_err = RT_DUP(2);
    RT_DUP2(ofd, 1);
    RT_DUP2(efd, 2);

    int32_t rc = rt_spawn_wait(program, argv, out_code);

    RT_DUP2(save_out, 1); RT_CLOSE(save_out);
    RT_DUP2(save_err, 2); RT_CLOSE(save_err);

    if (rc == 0) {
        *out_stdout = rt_slurp_fd(ofd, out_stdout_len);
        *out_stderr = rt_slurp_fd(efd, out_stderr_len);
    }

    RT_CLOSE(ofd);
    RT_CLOSE(efd);
    free(argv);
    return rc;
}

const char* rt_proc_strerror(int32_t code) { return strerror(code); }
