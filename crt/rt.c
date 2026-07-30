#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdarg.h>

struct Slice {
    void* data;
    int length;
};

uint64_t rt_slice_len(struct Slice* slice) { return (uint64_t)slice->length; }

[[noreturn]] void rt_abort(const char* msg) {
    fprintf(stderr, "abort: %s\n", msg);
    exit(101);
}

void rt_printf(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    vprintf(fmt, args);
    va_end(args);
}

// Float printing takes a concrete `double` parameter (a fixed prototype), not
// the erased/variadic path `rt_printf` uses. The language emits a plain
// fixed-arity call, so the value travels in an XMM register the callee reads
// directly - no caller-set %al, which the language no longer emits. The inner
// `printf` is a normal C-to-C variadic call the C compiler handles correctly.
void rt_printf_f64(const char* fmt, double d) { printf(fmt, d); }

char* rt_f32_to_str(float f, int32_t precision) {
    static char buf[32];
    if (precision < 0)
        snprintf(buf, sizeof(buf), "%g", f);
    else
        snprintf(buf, sizeof(buf), "%.*g", precision, f);
    return buf;
}

char* rt_f64_to_str(double d, int32_t precision) {
    static char buf[32];
    if (precision < 0)
        snprintf(buf, sizeof(buf), "%g", d);
    else
        snprintf(buf, sizeof(buf), "%.*g", precision, d);
    return buf;
}

// Heap allocation, exposed to the language via `std/alloc` (see crt/std/alloc.hv).
// On failure these return NULL rather than aborting: `std/alloc` wraps the result
// in `Option<*T>`, so out-of-memory surfaces as `none` for the caller to handle.
void* rt_alloc(uint64_t size) {
    return malloc((size_t)size);
}
void* rt_realloc(void* ptr, uint64_t size) {
    return realloc(ptr, (size_t)size);
}
void rt_free(void* ptr) { free(ptr); }

// Bridges between haven's `str` (a NUL-terminated C string) and a `*u8` byte
// buffer, used by std/string. In haven's type system `str` is its own type: it
// is neither indexable nor `ptr_cast`-able, even though at the machine level it
// is just a bare pointer - the same representation as `*u8`. These are pure
// reinterpretations (no copy), the same way `strlen` already reads a `str`'s
// bytes. `rt_str_as_bytes` views a `str`'s bytes for reading; `rt_bytes_as_str`
// hands a byte buffer back as a `str` (the caller guarantees a trailing NUL).
uint8_t* rt_str_as_bytes(const char* s) { return (uint8_t*)s; }
const char* rt_bytes_as_str(uint8_t* p) { return (const char*)p; }