//! x86-64 System V ABI classification: decides how a value crosses the C FFI
//! boundary - passed/returned in registers as coerced "eightbyte" pieces, or via
//! memory (a `byval` pointer argument / `sret` return slot).
//!
//! This is step 2 of by-value struct FFI. It only *classifies*; emitting the
//! pack/unpack glue at call/return boundaries is a separate step. It builds on
//! the byte geometry from [`crate::layout`].
//!
//! Reference: System V AMD64 ABI, §3.2.3 "Parameter Passing". We implement the
//! subset needed for real code: aggregates up to two eightbytes (16 bytes) of
//! integers, floats, pointers and small SIMD. Anything larger is MEMORY. We do
//! not model `__m256`/`__m512` (SSEUP) or unaligned/packed structs, since the
//! language produces neither.
//!
//! On Windows we instead follow the Microsoft x64 convention, which diverges
//! only for aggregates: a struct/array of size 1/2/4/8 bytes rides in a single
//! *integer* register (even an all-float one like `Vector2`), and every other
//! aggregate is passed by reference (`byval`/`sret`). Scalars are unchanged. See
//! [`classify_win64`].

use std::collections::HashMap;
use haven_common::ast::Type;
use haven_common::defs::DefId;
use crate::layout::{self, TypeTable, TypeInfo, EnumRepr};

/// Data-enum aggregate name -> its variant payload-struct names. Lets
/// `classify_into` treat an enum's `$payload` byte blob as a real union of the
/// variant payload structs (see its `Type::Struct` arm) instead of raw bytes.
pub type UnionTable = HashMap<DefId, Vec<DefId>>;

/// Size of one "eightbyte" register slot.
const EIGHTBYTE: usize = 8;
/// Aggregates larger than two eightbytes are always passed in memory.
const MAX_REGISTER_BYTES: usize = 2 * EIGHTBYTE;

/// The coerced LLVM type for one eightbyte carried in a register.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reg {
    /// A general-purpose register holding an integer, `n` bytes wide. `n` is
    /// always one of 1/2/4/8, rendering as `i8`/`i16`/`i32`/`i64`.
    Int(usize),
    /// A general-purpose register holding a pointer. Same INTEGER class and same
    /// machine register as `Int(8)` - a distinct spelling only so pointer fields
    /// keep their `ptr` type (matching Clang, and sparing later stages an
    /// integer<->pointer round-trip).
    Ptr,
    /// A single 32-bit float occupying (the low half of) an SSE register.
    Float,
    /// Two packed 32-bit floats in one SSE register: `<2 x float>`.
    Float2,
    /// A single 64-bit double in an SSE register.
    Double,
}

impl Reg {
    /// The LLVM IR type used to pass this eightbyte.
    pub fn to_llvm(self) -> &'static str {
        match self {
            Reg::Int(1) => "i8",
            Reg::Int(2) => "i16",
            Reg::Int(4) => "i32",
            Reg::Int(8) => "i64",
            Reg::Int(n) => unreachable!("integer eightbyte of {n} bytes"),
            Reg::Ptr => "ptr",
            Reg::Float => "float",
            Reg::Float2 => "<2 x float>",
            Reg::Double => "double",
        }
    }
}

/// How a value is passed or returned across the C ABI.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Abi {
    /// Passed in registers as these coerced pieces, in order (0, 1, or 2 for our
    /// ≤16-byte subset). Empty means a zero-sized value (nothing to pass).
    Direct(Vec<Reg>),
    /// Passed/returned via memory: a `byval(%T)` pointer argument, or an
    /// `sret(%T)` return slot.
    Memory,
}

impl Abi {
    pub fn is_memory(&self) -> bool {
        matches!(self, Abi::Memory)
    }
}

/// Provisional class of an eightbyte (or a leaf field) during classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Class {
    NoClass,
    Integer,
    Sse,
    Memory,
}

/// SysV merge rule for combining the class already in an eightbyte with the
/// class of a field that lands in it (§3.2.3, step 4).
fn merge(a: Class, b: Class) -> Class {
    use Class::*;
    match (a, b) {
        _ if a == b => a,
        (NoClass, other) | (other, NoClass) => other,
        (Memory, _) | (_, Memory) => Memory,
        (Integer, _) | (_, Integer) => Integer,
        _ => Sse,
    }
}

/// Classify how `ty` is passed/returned across the C ABI, picking the flavor for
/// the target we compile to. haven currently emits code for the host, so the
/// choice is made by `cfg!` - a Windows-hosted compiler uses the Microsoft x64
/// convention, everything else x86-64 System V. When a real `--target` flag
/// lands, thread the selector through here instead.
pub fn classify<'a>(ty: &Type<'a>, types: &TypeTable<'a>, unions: &UnionTable) -> Abi {
    if cfg!(target_os = "windows") {
        classify_win64(ty, types, unions)
    } else {
        classify_sysv(ty, types, unions)
    }
}

/// Classify how `ty` is passed/returned on x86-64 System V (Linux, macOS, BSDs).
pub fn classify_sysv<'a>(ty: &Type<'a>, types: &TypeTable<'a>, unions: &UnionTable) -> Abi {
    let size = layout::size_of(ty, types);

    // Zero-sized: nothing to pass.
    if size == 0 {
        return Abi::Direct(vec![]);
    }
    // Too large for two registers => memory.
    if size > MAX_REGISTER_BYTES {
        return Abi::Memory;
    }

    let n = size.div_ceil(EIGHTBYTE);
    let mut eb = vec![Eightbyte::default(); n];

    classify_into(ty, 0, types, unions, &mut eb);

    // Post-merge: any MEMORY eightbyte poisons the whole aggregate.
    if eb.iter().any(|e| e.class == Class::Memory) {
        return Abi::Memory;
    }

    let regs = eb
        .iter()
        .map(|e| match e.class {
            Class::Sse if e.has_double => Reg::Double,
            Class::Sse if e.used > 4 => Reg::Float2,
            Class::Sse => Reg::Float,
            // An INTEGER eightbyte filled by a single pointer keeps its `ptr`
            // type; anything else (mixed content, narrower ints) is a plain int.
            Class::Integer if e.is_sole_pointer() => Reg::Ptr,
            // Integer, or a padding-only eightbyte (defensive: shouldn't occur).
            Class::Integer | Class::NoClass => Reg::Int(int_width(e.used)),
            Class::Memory => unreachable!("memory eightbyte handled above"),
        })
        .collect();
    Abi::Direct(regs)
}

/// Classify how `ty` is passed/returned under the Microsoft x64 convention
/// (Windows).
///
/// The rules only diverge from SysV for aggregates:
///
///   * A struct/array whose size is exactly 1/2/4/8 bytes rides in a single
///     *integer* register of that width - even an all-float one like `Vector2`,
///     which SysV would hand to an SSE register as `<2 x float>`. This is why
///     by-value struct FFI to raylib silently misdrew on Windows before.
///   * Any other aggregate (sizes 3/5/6/7, or larger than 8 bytes) is passed by
///     reference: a `byval` pointer argument / `sret` return slot.
///
/// Scalars - including pointers and SIMD vector types - keep their natural
/// register, exactly as under SysV, so we delegate anything that isn't a struct
/// or array to [`classify_sysv`]. The downstream pack/unpack glue is bytewise, so
/// coercing an all-float eightbyte to `i64` needs no other changes.
pub fn classify_win64<'a>(ty: &Type<'a>, types: &TypeTable<'a>, unions: &UnionTable) -> Abi {
    match ty {
        _ if is_aggregate(ty, types) =>
            match layout::size_of(ty, types) {
                0 => Abi::Direct(vec![]),
                size @ (1 | 2 | 4 | 8) => Abi::Direct(vec![Reg::Int(size)]),
                _ => Abi::Memory,
            },
        // Scalars/pointers/SIMD/field-less enums are placed identically to SysV.
        _ => classify_sysv(ty, types, unions),
    }
}

/// Accumulated state for one eightbyte as leaves are folded into it.
#[derive(Debug, Clone, Copy)]
struct Eightbyte {
    class: Class,
    /// Bytes actually occupied (drives the coerced width so a 1-byte tail
    /// becomes `i8`, not `i64`, matching Clang).
    used: usize,
    /// A 64-bit float lands here, so it must coerce to `double`, not `<2 x float>`.
    has_double: bool,
    /// A pointer leaf touched this eightbyte.
    has_ptr: bool,
    /// A non-pointer leaf touched this eightbyte.
    has_nonptr: bool,
}

impl Default for Eightbyte {
    fn default() -> Self {
        Eightbyte { class: Class::NoClass, used: 0, has_double: false, has_ptr: false, has_nonptr: false }
    }
}

impl Eightbyte {
    /// True when this eightbyte is filled by exactly one pointer and nothing
    /// else - the case where `ptr` is the right coercion rather than `i64`.
    fn is_sole_pointer(&self) -> bool {
        self.has_ptr && !self.has_nonptr && self.used == EIGHTBYTE
    }
}

/// Walk `ty`'s leaf fields, folding each leaf's class into the eightbyte(s) it
/// occupies at absolute byte `offset`.
fn classify_into<'a>(ty: &Type<'a>, offset: usize, types: &TypeTable<'a>, unions: &UnionTable, eb: &mut [Eightbyte]) {
    match ty {
        // Aggregates: recurse into members at their real offsets. A struct and
        // a data-carrying enum are the same shape here - the enum's `{tag,
        // payload}` aggregate is a field list like any other - so one arm covers
        // both, where they used to need two.
        Type::Named { def, .. } if is_aggregate(ty, types) => {
            let fields = &layout::fields_of(*def, types).to_vec();
            if let Some(variants) = unions.get(def) {
                // a data-enum aggregate: field 0 is the tag, field 1 the
                // `$payload` blob - a union of the variant payload structs, not
                // literal bytes. classify the tag, then every variant at the
                // payload's offset, and let `merge` combine them (the SysV union
                // rule). this lets an all-float variant land in an SSE register
                // instead of being forced INTEGER by `$payload`'s placeholder.
                let (_, tag_ty) = &fields[0];
                classify_into(tag_ty, offset, types, unions, eb);
                let payload_off = offset + layout::field_offset(*def, 1, types);
                for variant in variants {
                    classify_into(&Type::named(*variant), payload_off, types, unions, eb);
                }
                return;
            }
            for (i, (_, fty)) in fields.iter().enumerate() {
                let foff = offset + layout::field_offset(*def, i, types);
                classify_into(fty, foff, types, unions, eb);
            }
        }
        Type::Array(elem, count) => {
            let stride = layout::size_of(elem, types);
            for i in 0..count.expect_lit() {
                classify_into(elem, offset + i * stride, types, unions, eb);
            }
        }
        // Leaf: a scalar / pointer / SIMD vector / field-less enum.
        _ => {
            let size = layout::size_of(ty, types);
            let class = leaf_class(ty);
            let is_double = leaf_is_double(ty);
            let is_ptr = leaf_is_pointer(ty);
            let first = offset / EIGHTBYTE;
            let last = (offset + size - 1) / EIGHTBYTE;
            for i in first..=last {
                eb[i].class = merge(eb[i].class, class);
                eb[i].has_double |= is_double;
                eb[i].has_ptr |= is_ptr;
                eb[i].has_nonptr |= !is_ptr;
                // Bytes of this leaf reaching into eightbyte `i`, capped at 8.
                let local_end = ((offset + size) - i * EIGHTBYTE).min(EIGHTBYTE);
                eb[i].used = eb[i].used.max(local_end);
            }
        }
    }
}

/// Whether `ty` is laid out as an aggregate: an array, a struct, or a
/// data-carrying enum. A field-less enum is a bare scalar and is not.
///
/// This used to be a pattern match, back when `Type` said which kind of named
/// type it was. Now it is a table lookup: the answer belongs to the definition,
/// not to every type value that mentions it.
fn is_aggregate<'a>(ty: &Type<'a>, types: &TypeTable<'a>) -> bool {
    match ty {
        Type::Array(..) => true,
        Type::Named { def, .. } => !matches!(
            types.get(def),
            Some(TypeInfo { enum_: Some(EnumRepr { has_payload: false, .. }), .. }) | None,
        ),
        _ => false,
    }
}

/// The SysV class of a leaf (non-aggregate) type.
fn leaf_class(ty: &Type) -> Class {
    match ty {
        Type::Path { path, .. } => Type::unresolved(path),
        Type::Float32 | Type::Float64 => Class::Sse,
        Type::Simd(inner, _) if inner.is_numeric() && !inner.is_integer() => Class::Sse,
        Type::Simd(_, _) => Class::Integer, // integer-lane vectors go in GP-class regs
        Type::Bool
        | Type::Int8 | Type::Int16 | Type::Int32 | Type::Int64
        | Type::Uint8 | Type::Uint16 | Type::Uint32 | Type::Uint64
        | Type::Pointer(_) | Type::Function { .. } => Class::Integer,
        // A slice is a two-INTEGER-word fat pointer; FFI bans it, but classify
        // sanely. `str` is a single `*const u8`, handled as a pointer leaf below.
        Type::Slice(_) | Type::Str => Class::Integer,
        // a field-less enum is its integer discriminant repr - an INTEGER-class
        // leaf. Anything named that *is* an aggregate is handled by
        // `classify_into` and never reaches here.
        Type::Named { .. } => Class::Integer,
        Type::Void => Class::NoClass,
        Type::Array(..) => {
            unreachable!("aggregates are handled by classify_into, not leaf_class")
        }
        Type::Param(n) => panic!("type parameter `{n}` survived to ABI classification"),
        Type::Never => panic!("the bottom type `!` reached ABI classification - it has no values"),
    }
}

/// Whether a leaf occupies its eightbyte(s) with 64-bit float lanes, which must
/// be coerced to `double` rather than packed as `<2 x float>`.
fn leaf_is_double(ty: &Type) -> bool {
    matches!(ty, Type::Float64)
        || matches!(ty, Type::Simd(inner, _) if matches!(**inner, Type::Float64))
}

/// Whether a leaf is a bare pointer (data or function). These fill exactly one
/// eightbyte and, when alone in it, coerce to `ptr`. `str` is a raw `*const u8`,
/// so it counts too. A slice fat pointer is deliberately excluded - it is two
/// words, not a single `ptr`.
fn leaf_is_pointer(ty: &Type) -> bool {
    matches!(ty, Type::Pointer(_) | Type::Function { .. } | Type::Str)
}

/// Smallest power-of-two integer width (in bytes) covering `used_bytes`.
fn int_width(used_bytes: usize) -> usize {
    match used_bytes {
        0..=1 => 1,
        2 => 2,
        3..=4 => 4,
        _ => 8,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use haven_common::ast::ConstVal;

    /// A stable identity per test name, so a table can still be written out with
    /// readable names while the code under test keys on identities.
    fn d(name: &str) -> DefId {
        let mut h: u32 = 0x811c9dc5;
        for b in name.as_bytes() {
            h ^= *b as u32;
            h = h.wrapping_mul(0x01000193);
        }
        DefId(h)
    }

    /// The named type `name` refers to.
    fn nt<'a>(name: &str) -> Type<'a> {
        Type::named(d(name))
    }

    fn table<'a>(defs: &[(&'a str, Vec<(&'a str, Type<'a>)>)]) -> TypeTable<'a> {
        defs.iter().map(|(n, f)| (d(n), TypeInfo::struct_(f.clone()))).collect()
    }

    /// Like `table`, but marks `enum_name` as a data-carrying enum - which is
    /// what makes `is_aggregate` treat its `{ $tag, $payload }` entry as one.
    fn table_with_enum<'a>(
        defs: &[(&'a str, Vec<(&'a str, Type<'a>)>)],
        enum_name: &str,
        repr: Type<'a>,
    ) -> TypeTable<'a> {
        let mut t = table(defs);
        t.get_mut(&d(enum_name)).unwrap().enum_ = Some(EnumRepr { repr, has_payload: true });
        t
    }

    fn unions(defs: &[(&str, Vec<&str>)]) -> UnionTable {
        defs.iter()
            .map(|(n, vs)| (d(n), vs.iter().map(|v| d(v)).collect()))
            .collect()
    }

    fn render(abi: Abi) -> Vec<String> {
        match abi {
            Abi::Memory => vec!["memory".into()],
            Abi::Direct(regs) => regs.iter().map(|r| r.to_llvm().to_string()).collect(),
        }
    }

    /// Classify on SysV and render as the list of LLVM coercion types (or "memory").
    fn llvm<'a>(ty: &Type<'a>, s: &TypeTable<'a>) -> Vec<String> {
        render(classify_sysv(ty, s, &HashMap::new()))
    }

    /// Same, but with an enum-union table in play.
    fn llvm_u<'a>(ty: &Type<'a>, s: &TypeTable<'a>, u: &UnionTable) -> Vec<String> {
        render(classify_sysv(ty, s, u))
    }

    /// Same, but classify under the Microsoft x64 convention.
    fn llvm_win64<'a>(ty: &Type<'a>, s: &TypeTable<'a>) -> Vec<String> {
        render(classify_win64(ty, s, &HashMap::new()))
    }

    #[test]
    fn scalars_pass_in_their_natural_register() {
        let s = table(&[]);
        assert_eq!(llvm(&Type::Int8, &s), ["i8"]);
        assert_eq!(llvm(&Type::Int32, &s), ["i32"]);
        assert_eq!(llvm(&Type::Int64, &s), ["i64"]);
        assert_eq!(llvm(&Type::Float32, &s), ["float"]);
        assert_eq!(llvm(&Type::Float64, &s), ["double"]);
    }

    #[test]
    fn color_packs_to_one_i32() {
        // struct Color { r,g,b,a: u8 } -> 4 bytes, one INTEGER eightbyte.
        let s = table(&[(
            "Color",
            vec![("r", Type::Uint8), ("g", Type::Uint8), ("b", Type::Uint8), ("a", Type::Uint8)],
        )]);
        assert_eq!(llvm(&nt("Color"), &s), ["i32"]);
    }

    #[test]
    fn vector2_is_two_packed_floats() {
        // struct Vector2 { x,y: f32 } -> one SSE eightbyte, two floats.
        let s = table(&[("Vector2", vec![("x", Type::Float32), ("y", Type::Float32)])]);
        assert_eq!(llvm(&nt("Vector2"), &s), ["<2 x float>"]);
    }

    #[test]
    fn vector3_is_packed_then_scalar_float() {
        // struct Vector3 { x,y,z: f32 } -> 12 bytes: <2 x float>, then a tail float.
        let s = table(&[(
            "Vector3",
            vec![("x", Type::Float32), ("y", Type::Float32), ("z", Type::Float32)],
        )]);
        assert_eq!(llvm(&nt("Vector3"), &s), ["<2 x float>", "float"]);
    }

    #[test]
    fn rectangle_is_two_float_pairs() {
        // struct Rectangle { x,y,w,h: f32 } -> 16 bytes, two SSE eightbytes.
        let s = table(&[(
            "Rectangle",
            vec![
                ("x", Type::Float32), ("y", Type::Float32),
                ("w", Type::Float32), ("h", Type::Float32),
            ],
        )]);
        assert_eq!(llvm(&nt("Rectangle"), &s), ["<2 x float>", "<2 x float>"]);
    }

    #[test]
    fn mixed_int_and_float_in_one_eightbyte_is_integer() {
        // struct { i: i32, f: f32 } -> both share eightbyte 0; INTEGER wins -> i64.
        let s = table(&[("Mix", vec![("i", Type::Int32), ("f", Type::Float32)])]);
        assert_eq!(llvm(&nt("Mix"), &s), ["i64"]);
    }

    #[test]
    fn two_ints_coalesce_to_i64() {
        let s = table(&[("Pair", vec![("a", Type::Int32), ("b", Type::Int32)])]);
        assert_eq!(llvm(&nt("Pair"), &s), ["i64"]);
    }

    #[test]
    fn tail_byte_uses_narrow_integer_not_i64() {
        // struct { a: i64, b: i8 } -> 16 bytes; eightbyte 1 holds only 1 byte.
        // Coercion must be { i64, i8 }, matching Clang - not { i64, i64 }.
        let s = table(&[("Tail", vec![("a", Type::Int64), ("b", Type::Int8)])]);
        assert_eq!(llvm(&nt("Tail"), &s), ["i64", "i8"]);
    }

    #[test]
    fn float_then_double_stays_split() {
        // { f: f32, d: f64 } -> f32 in eb0 (float), f64 in eb1 (double).
        let s = table(&[("FD", vec![("f", Type::Float32), ("d", Type::Float64)])]);
        assert_eq!(llvm(&nt("FD"), &s), ["float", "double"]);
    }

    #[test]
    fn lone_pointer_field_stays_ptr() {
        let s = table(&[("Ref", vec![("p", Type::Pointer(Box::new(Type::Float32)))])]);
        assert_eq!(llvm(&nt("Ref"), &s), ["ptr"]);
    }

    #[test]
    fn pointer_alongside_scalars_keeps_ptr_per_eightbyte() {
        // { i: i32, p: ptr } -> eb0 = i32 (used 4), eb1 = a whole pointer -> ptr.
        let s = table(&[(
            "IntPtr",
            vec![("i", Type::Int32), ("p", Type::Pointer(Box::new(Type::Int8)))],
        )]);
        assert_eq!(llvm(&nt("IntPtr"), &s), ["i32", "ptr"]);

        // { p: ptr, q: ptr } -> two pointer eightbytes.
        let s2 = table(&[(
            "TwoPtr",
            vec![
                ("p", Type::Pointer(Box::new(Type::Int8))),
                ("q", Type::Pointer(Box::new(Type::Int8))),
            ],
        )]);
        assert_eq!(llvm(&nt("TwoPtr"), &s2), ["ptr", "ptr"]);
    }

    #[test]
    fn pointer_sharing_an_eightbyte_falls_back_to_integer() {
        // A pointer can only sit at an 8-aligned offset, so it never truly shares
        // an eightbyte - but guard the merge anyway: a bool then a pointer would
        // require the pointer at offset 8, leaving eb0 as the bool (i8).
        let s = table(&[(
            "BoolPtr",
            vec![("b", Type::Bool), ("p", Type::Pointer(Box::new(Type::Int8)))],
        )]);
        assert_eq!(llvm(&nt("BoolPtr"), &s), ["i8", "ptr"]);
    }

    #[test]
    fn enum_payload_classifies_as_union_not_raw_bytes() {
        // enum Msg { Note(f64) }: tag i32 (4/4), payload padded to offset 8 (f64
        // needs align 8), aggregate is 16 bytes / align 8.
        let s = table_with_enum(&[
            ("Msg$Note", vec![("0", Type::Float64)]),
            ("Msg", vec![
                ("$tag", Type::Int32),
                ("$payload", Type::Array(Box::new(Type::Int64), ConstVal::Lit(1))),
            ]),
        ], "Msg", Type::Int32);
        let u = unions(&[("Msg", vec!["Msg$Note"])]);

        // Without the union table, `$payload`'s placeholder `i64` element
        // classifies as INTEGER - wrong for a variant that actually holds a double.
        assert_eq!(llvm(&nt("Msg"), &s), ["i32", "i64"]);

        // With it, the payload eightbyte is classified from the real variant
        // field and lands in an SSE register, matching what C would do for the
        // equivalent `struct { int tag; union { double v; } payload; }`.
        assert_eq!(llvm_u(&nt("Msg"), &s, &u), ["i32", "double"]);
    }

    #[test]
    fn over_sixteen_bytes_goes_to_memory() {
        // struct { a,b,c,d: f64 } -> 32 bytes -> MEMORY.
        let s = table(&[(
            "Big",
            vec![
                ("a", Type::Float64), ("b", Type::Float64),
                ("c", Type::Float64), ("d", Type::Float64),
            ],
        )]);
        assert!(classify(&nt("Big"), &s, &HashMap::new()).is_memory());
    }

    #[test]
    fn nested_struct_flattens_across_eightbytes() {
        // Inner { x,y: f32 } (8 bytes, SSE). Outer { a: Inner, b: i32 } -> 12 bytes:
        // eb0 = the two floats -> <2 x float>; eb1 = the i32 -> i32.
        let s = table(&[
            ("Inner", vec![("x", Type::Float32), ("y", Type::Float32)]),
            ("Outer", vec![("a", nt("Inner")), ("b", Type::Int32)]),
        ]);
        assert_eq!(llvm(&nt("Outer"), &s), ["<2 x float>", "i32"]);
    }

    #[test]
    fn array_field_classifies_elementwise() {
        // struct { xs: [f32; 3] } -> same shape as Vector3.
        let s = table(&[(
            "Arr",
            vec![("xs", Type::Array(Box::new(Type::Float32), ConstVal::Lit(3)))],
        )]);
        assert_eq!(llvm(&nt("Arr"), &s), ["<2 x float>", "float"]);
    }

    // --- Microsoft x64 (Windows) ------------------------------------------

    #[test]
    fn win64_scalars_match_sysv() {
        // Bare scalars keep their natural register on both ABIs: floats in SSE,
        // ints/pointers in GP.
        let s = table(&[]);
        assert_eq!(llvm_win64(&Type::Int8, &s), ["i8"]);
        assert_eq!(llvm_win64(&Type::Int64, &s), ["i64"]);
        assert_eq!(llvm_win64(&Type::Float32, &s), ["float"]);
        assert_eq!(llvm_win64(&Type::Float64, &s), ["double"]);
        assert_eq!(llvm_win64(&Type::Pointer(Box::new(Type::Int8)), &s), ["ptr"]);
    }

    #[test]
    fn win64_color_still_packs_to_one_i32() {
        // 4-byte aggregate -> one integer register on both ABIs (this is why
        // colors always rendered on Windows).
        let s = table(&[(
            "Color",
            vec![("r", Type::Uint8), ("g", Type::Uint8), ("b", Type::Uint8), ("a", Type::Uint8)],
        )]);
        assert_eq!(llvm_win64(&nt("Color"), &s), ["i32"]);
    }

    #[test]
    fn win64_vector2_is_one_integer_register() {
        // The regression: an 8-byte all-float struct goes in a *GP* register on
        // Win64 (i64), not an SSE `<2 x float>` as under SysV.
        let s = table(&[("Vector2", vec![("x", Type::Float32), ("y", Type::Float32)])]);
        assert_eq!(llvm_win64(&nt("Vector2"), &s), ["i64"]);
        assert_eq!(llvm(&nt("Vector2"), &s), ["<2 x float>"]); // SysV, for contrast
    }

    #[test]
    fn win64_pointer_wrapper_is_one_integer_register() {
        // An 8-byte struct wrapping a pointer also passes as a single i64.
        let s = table(&[("Ref", vec![("p", Type::Pointer(Box::new(Type::Float32)))])]);
        assert_eq!(llvm_win64(&nt("Ref"), &s), ["i64"]);
    }

    #[test]
    fn win64_odd_sized_aggregate_goes_to_memory() {
        // 3 bytes is not 1/2/4/8, so it is passed by reference - unlike SysV,
        // which would coalesce it into one small integer eightbyte.
        let s = table(&[(
            "Rgb",
            vec![("r", Type::Uint8), ("g", Type::Uint8), ("b", Type::Uint8)],
        )]);
        assert!(classify_win64(&nt("Rgb"), &s, &HashMap::new()).is_memory());
    }

    #[test]
    fn win64_over_eight_bytes_goes_to_memory() {
        // Vector3 (12 bytes) and Rectangle (16 bytes) both pass by reference on
        // Win64, where SysV would split them across two registers.
        let s = table(&[
            ("Vector3", vec![("x", Type::Float32), ("y", Type::Float32), ("z", Type::Float32)]),
            ("Rectangle", vec![
                ("x", Type::Float32), ("y", Type::Float32),
                ("w", Type::Float32), ("h", Type::Float32),
            ]),
        ]);
        assert!(classify_win64(&nt("Vector3"), &s, &HashMap::new()).is_memory());
        assert!(classify_win64(&nt("Rectangle"), &s, &HashMap::new()).is_memory());
        // SysV keeps them in registers.
        assert_eq!(llvm(&nt("Vector3"), &s), ["<2 x float>", "float"]);
    }
}
