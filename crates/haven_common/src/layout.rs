//! Memory layout of types, following the C rules (as used by the x86-64 System V
//! ABI on our target). This is the prerequisite for by-value struct FFI: to pass
//! a struct in registers the way C does, we first have to know its size, its
//! alignment, and the byte offset of every field so we can carve it into
//! eightbytes and classify them.
//!
//! Everything here is target-specific (LP64: 8-byte pointers). It is deliberately
//! kept separate from `emit_type` in `llvm.rs`, which describes the *IR* shape of
//! a type, not its concrete byte layout.

use std::collections::HashMap;

use crate::ast::Type;

/// Struct definitions, keyed by name: each maps to its ordered `(field, type)`
/// list. This is the same shape `LowerCtx.structs` already carries (see
/// `mil.rs`), so callers can hand theirs straight in.
pub type StructTable<'a> = HashMap<&'a str, Vec<(&'a str, Type<'a>)>>;

/// Pointer size/alignment for our LP64 target. A function value is just an
/// opaque pointer, so it shares these.
const POINTER_SIZE: usize = 8;
const POINTER_ALIGN: usize = 8;

/// Round `n` up to the next multiple of `align` (which must be a power of two).
fn round_up(n: usize, align: usize) -> usize {
    debug_assert!(align.is_power_of_two(), "alignment {align} is not a power of two");
    (n + align - 1) & !(align - 1)
}

/// Size in bytes of a value of type `ty`, including any tail padding needed to
/// keep it a multiple of its own alignment (so `size_of` doubles as the stride
/// of an array element).
pub fn size_of<'a>(ty: &Type<'a>, structs: &StructTable<'a>) -> usize {
    use Type::*;
    match ty {
        Void => 0,
        // C `_Bool` occupies one byte in memory even though it is `i1` in IR.
        Bool => 1,
        Int8 | Uint8 => 1,
        Int32 | Uint32 | Float32 => 4,
        Int64 | Uint64 | Float64 => 8,
        Pointer(_) | Function { .. } => POINTER_SIZE,

        // An array is `n` elements laid end to end at the element stride.
        Array(elem, n) => size_of(elem, structs) * n.expect_lit(),

        // A SIMD vector is `n` packed elements with no interior padding.
        Simd(elem, n) => size_of(elem, structs) * n.expect_lit(),

        // A slice is a `{ ptr, len }` fat pointer.
        Slice(_) => aggregate_layout(fat_pointer_fields().iter(), structs).0,
        // `str` is a raw `*const u8` - a single machine pointer.
        Str => POINTER_SIZE,
        // a field-less enum is stored as its integer discriminant repr; a
        // data-carrying enum is the aggregate `{ tag, payload }` registered as a
        // synthetic struct of the same name (see typecheck's forward-decl pass).
        Enum { has_payload: false, repr, .. } => size_of(repr, structs),
        Enum { has_payload: true, name, .. } =>
            aggregate_layout(struct_fields(name, structs).iter().map(|(_, t)| t), structs).0,

        Struct { name, .. } => aggregate_layout(struct_fields(name, structs).iter().map(|(_, t)| t), structs).0,

        Path { path, .. } => Type::unresolved(path),
        Param(name) => panic!("type parameter `{name}` survived to layout"),
    }
}

/// Alignment in bytes required by a value of type `ty`.
pub fn align_of<'a>(ty: &Type<'a>, structs: &StructTable<'a>) -> usize {
    use Type::*;
    match ty {
        // Zero-sized things still need a non-zero alignment for `round_up`.
        Void | Bool => 1,
        Int8 | Uint8 => 1,
        Int32 | Uint32 | Float32 => 4,
        Int64 | Uint64 | Float64 => 8,
        Pointer(_) | Function { .. } => POINTER_ALIGN,

        // An array is as aligned as its element.
        Array(elem, _) => align_of(elem, structs),

        // SysV aligns a vector to its size, rounded up to a power of two (an
        // 8-byte vector => 8, a 16-byte vector => 16), but never below the
        // element's own alignment.
        Simd(elem, _) => size_of(ty, structs)
            .next_power_of_two()
            .max(align_of(elem, structs)),

        Slice(_) => aggregate_layout(fat_pointer_fields().iter(), structs).1,
        Str => POINTER_ALIGN,
        Enum { has_payload: false, repr, .. } => align_of(repr, structs),
        Enum { has_payload: true, name, .. } =>
            aggregate_layout(struct_fields(name, structs).iter().map(|(_, t)| t), structs).1,

        Struct { name, .. } => aggregate_layout(struct_fields(name, structs).iter().map(|(_, t)| t), structs).1,

        Path { path, .. } => Type::unresolved(path),
        Param(name) => panic!("type parameter `{name}` survived to layout"),
    }
}

/// Byte offset of field `index` within `struct_name`, following C field
/// placement (each field bumped up to its own alignment).
pub fn field_offset<'a>(struct_name: &str, index: usize, structs: &StructTable<'a>) -> usize {
    let fields = struct_fields(struct_name, structs);
    assert!(
        index < fields.len(),
        "field index {index} out of range for struct `{struct_name}` with {} field(s)",
        fields.len()
    );

    let mut offset = 0;
    for (_, fty) in &fields[..index] {
        offset = round_up(offset, align_of(fty, structs));
        offset += size_of(fty, structs);
    }
    // `offset` now sits at the end of the previous field; bump it up to where
    // the requested field actually starts.
    round_up(offset, align_of(&fields[index].1, structs))
}

/// Look up a struct's field list or panic with a clear message. A missing entry
/// is always a compiler bug (typecheck should have rejected unknown types).
fn struct_fields<'t, 'a>(name: &str, structs: &'t StructTable<'a>) -> &'t [(&'a str, Type<'a>)] {
    structs
        .get(name)
        .unwrap_or_else(|| panic!("layout of unknown struct `{name}`"))
}

/// Shared layout core: walk `fields` in order applying C placement rules and
/// return `(size, align)` of the resulting aggregate. `size` includes tail
/// padding to a multiple of `align`.
fn aggregate_layout<'a, 'b>(
    fields: impl Iterator<Item = &'b Type<'a>>,
    structs: &StructTable<'a>,
) -> (usize, usize)
where
    'a: 'b,
{
    let mut offset = 0;
    let mut align = 1;
    for fty in fields {
        let a = align_of(fty, structs);
        offset = round_up(offset, a);
        offset += size_of(fty, structs);
        align = align.max(a);
    }
    (round_up(offset, align), align)
}

/// The `{ ptr, i32 }` field list backing a slice fat pointer. The pointer's
/// pointee is irrelevant to layout, so `*void` stands in. (`str` no longer uses
/// this - it is a bare `*const u8`.)
fn fat_pointer_fields<'a>() -> [Type<'a>; 2] {
    [Type::Pointer(Box::new(Type::Void)), Type::Int32]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn table<'a>(defs: &[(&'a str, Vec<(&'a str, Type<'a>)>)]) -> StructTable<'a> {
        defs.iter().cloned().collect()
    }

    #[test]
    fn scalars() {
        let s = table(&[]);
        assert_eq!((size_of(&Type::Bool, &s), align_of(&Type::Bool, &s)), (1, 1));
        assert_eq!((size_of(&Type::Int32, &s), align_of(&Type::Int32, &s)), (4, 4));
        assert_eq!((size_of(&Type::Float32, &s), align_of(&Type::Float32, &s)), (4, 4));
        assert_eq!((size_of(&Type::Int64, &s), align_of(&Type::Int64, &s)), (8, 8));
        let p = Type::Pointer(Box::new(Type::Float32));
        assert_eq!((size_of(&p, &s), align_of(&p, &s)), (8, 8));
    }

    #[test]
    fn color_is_four_bytes_align_one() {
        // struct Color { r,g,b,a: u8 } - four packed bytes, no padding.
        let s = table(&[(
            "Color",
            vec![("r", Type::Uint8), ("g", Type::Uint8), ("b", Type::Uint8), ("a", Type::Uint8)],
        )]);
        let c = Type::plain_struct("Color");
        assert_eq!(size_of(&c, &s), 4);
        assert_eq!(align_of(&c, &s), 1);
        assert_eq!(field_offset("Color", 0, &s), 0);
        assert_eq!(field_offset("Color", 3, &s), 3);
    }

    #[test]
    fn vector2_and_rectangle() {
        let s = table(&[
            ("Vector2", vec![("x", Type::Float32), ("y", Type::Float32)]),
            (
                "Rectangle",
                vec![
                    ("x", Type::Float32), ("y", Type::Float32),
                    ("w", Type::Float32), ("h", Type::Float32),
                ],
            ),
        ]);
        let v2 = Type::plain_struct("Vector2");
        assert_eq!((size_of(&v2, &s), align_of(&v2, &s)), (8, 4));
        assert_eq!(field_offset("Vector2", 1, &s), 4);

        let rect = Type::plain_struct("Rectangle");
        assert_eq!((size_of(&rect, &s), align_of(&rect, &s)), (16, 4));
        assert_eq!(field_offset("Rectangle", 3, &s), 12);
    }

    #[test]
    fn mixed_fields_get_padded() {
        // struct { a: bool, b: i64 } - b must land at offset 8, size 16 align 8.
        let s = table(&[("Mixed", vec![("a", Type::Bool), ("b", Type::Int64)])]);
        let m = Type::plain_struct("Mixed");
        assert_eq!(field_offset("Mixed", 0, &s), 0);
        assert_eq!(field_offset("Mixed", 1, &s), 8);
        assert_eq!((size_of(&m, &s), align_of(&m, &s)), (16, 8));
    }

    #[test]
    fn nested_struct_and_array() {
        // struct Inner { x: i32, y: i32 } (8/4); Outer { a: bool, inner: Inner }
        // -> inner at offset 4, total 12/4.
        let s = table(&[
            ("Inner", vec![("x", Type::Int32), ("y", Type::Int32)]),
            ("Outer", vec![("a", Type::Bool), ("inner", Type::plain_struct("Inner"))]),
        ]);
        let outer = Type::plain_struct("Outer");
        assert_eq!(field_offset("Outer", 1, &s), 4);
        assert_eq!((size_of(&outer, &s), align_of(&outer, &s)), (12, 4));

        // [Inner; 3] is 24 bytes, align 4.
        let arr = Type::Array(Box::new(Type::plain_struct("Inner")), crate::ast::ConstVal::Lit(3));
        assert_eq!((size_of(&arr, &s), align_of(&arr, &s)), (24, 4));
    }

    #[test]
    fn data_enum_aggregate() {
        // enum Msg { Note(u8, f32) }: the payload struct {u8@0, f32@4} is 8/4
        // (mixed alignment), and the aggregate { tag: i32, payload: [8 x i8] } is
        // 12/4 with the payload blob right after the tag at offset 4.
        let s = table(&[
            ("Msg$Note", vec![("0", Type::Uint8), ("1", Type::Float32)]),
            ("Msg", vec![
                ("$tag", Type::Int32),
                ("$payload", Type::Array(Box::new(Type::Int8), crate::ast::ConstVal::Lit(8))),
            ]),
        ]);
        let note = Type::plain_struct("Msg$Note");
        assert_eq!((size_of(&note, &s), align_of(&note, &s)), (8, 4));
        assert_eq!(field_offset("Msg$Note", 1, &s), 4); // f32 padded past the u8

        // a data-carrying enum sizes via its synthetic aggregate struct.
        let msg = Type::Enum { name: "Msg", repr: Box::new(Type::Int32), has_payload: true, args: Vec::new() };
        assert_eq!((size_of(&msg, &s), align_of(&msg, &s)), (12, 4));
        assert_eq!(field_offset("Msg", 1, &s), 4); // payload blob after the tag
    }

    #[test]
    fn simd_and_fat_pointers() {
        let s = table(&[]);
        // <2 x f32> -> 8 bytes, align 8; <4 x f32> -> 16 bytes, align 16.
        let v2 = Type::Simd(Box::new(Type::Float32), crate::ast::ConstVal::Lit(2));
        assert_eq!((size_of(&v2, &s), align_of(&v2, &s)), (8, 8));
        let v4 = Type::Simd(Box::new(Type::Float32), crate::ast::ConstVal::Lit(4));
        assert_eq!((size_of(&v4, &s), align_of(&v4, &s)), (16, 16));

        // slice fat pointer: { ptr@0, i32@8 } -> 16 bytes, align 8.
        let sl = Type::Slice(Box::new(Type::Float32));
        assert_eq!((size_of(&sl, &s), align_of(&sl, &s)), (16, 8));
        // `str` is a raw `*const u8` -> one pointer, 8 bytes, align 8.
        assert_eq!((size_of(&Type::Str, &s), align_of(&Type::Str, &s)), (8, 8));
    }
}
