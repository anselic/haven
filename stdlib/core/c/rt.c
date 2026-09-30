#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

[[noreturn]] void rt_abort(const char* msg) {
    fprintf(stderr, "abort: %s\n", msg);
    exit(101);
}

struct Slice {
    void* data;
    int length;
};

uint64_t rt_slice_len(struct Slice* slice) { return (uint64_t)slice->length; }

void rt_slice_slice(struct Slice* slice, uint64_t start, uint64_t end, uint64_t elem_sz) {
    if (start > end || end > (uint64_t)slice->length)
        rt_abort("slice: invalid range");
    slice->data = (char*)slice->data + start * elem_sz;
    slice->length = (int)(end - start);
}
