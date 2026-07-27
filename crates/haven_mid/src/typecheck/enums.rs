use std::collections::HashMap;
use haven_common::ast::*;
use super::context::{Context, EnumDef};

/// Whether every aggregate (struct, or data-enum) that `ename`'s payload fields
/// reference is already registered in `structs` - i.e. whether it's safe to call
/// `layout::size_of` on `ename`'s payload structs yet. Used to sequence data-enum
/// "pass 2" (aggregate registration) into a fixpoint: a data enum's payload can
/// itself be another (unregistered) data enum (`Option<Option<i32>>`).
pub(crate) fn enum_agg_deps_ready<'a>(
    ename: &str,
    enums: &HashMap<&'a str, EnumDef<'a>>,
    structs: &HashMap<&'a str, Vec<(&'a str, Type<'a>)>>,
) -> bool {
    fn ty_ready<'a>(ty: &Type<'a>, structs: &HashMap<&'a str, Vec<(&'a str, Type<'a>)>>) -> bool {
        match ty {
            Type::Struct { name, .. } => structs.contains_key(name),
            Type::Enum { name, has_payload: true, .. } => structs.contains_key(name),
            // a pointer is a fixed-size opaque handle: doesn't need its pointee's
            // own layout, so it never blocks readiness.
            Type::Pointer(_) => true,
            Type::Array(inner, _) | Type::Slice(inner) | Type::Simd(inner, _) => ty_ready(inner, structs),
            _ => true,
        }
    }
    enums[ename].payloads.values().all(|fields| fields.iter().all(|(_, ty)| ty_ready(ty, structs)))
}

/// The `$payload` byte-blob type for a data enum's aggregate: an array sized to
/// hold the largest variant, chunked so the array's own (LLVM/C "natural
/// layout") alignment matches `needed_align` - the true max alignment any
/// variant's payload requires. A plain `[N x i8]` is always align 1, which
/// under-reports the requirement whenever a variant holds something like a
/// pointer or `f64` (align 8): the tag/payload split would then place the
/// payload right after a 4-byte `i32` tag at offset 4, misaligned relative to
/// what an equivalent C `struct { int tag; union { ... }; }` would produce.
/// Picking `i32`/`i64` chunks instead makes LLVM's own struct layout (and our
/// `layout::size_of`/`align_of`, which mirrors it) naturally pad and align the
/// field correctly - with no change needed at the field-access sites, since a
/// `FieldPtr` into the payload re-bases through the variant's own payload
/// struct type and never observes `$payload`'s element type. Alignments above
/// 8 (e.g. a SIMD payload) aren't modeled and stay under-aligned; nothing in
/// the language currently exercises that.
pub(crate) fn payload_blob_type<'a>(bytes: usize, needed_align: usize) -> Type<'a> {
    let (chunk, chunk_size) = if needed_align >= 8 {
        (Type::Int64, 8)
    } else if needed_align >= 4 {
        (Type::Int32, 4)
    } else {
        (Type::Int8, 1)
    };
    let count = bytes.div_ceil(chunk_size);
    Type::Array(Box::new(chunk), ConstVal::Lit(count))
}

/// The discriminant repr type for an enum, from its `@repr(<int>)` attribute,
/// and whether that attribute was actually written (vs. defaulted). Defaults to
/// `i32` (C `int`) when absent; `@repr(C)` is an explicit spelling of that same
/// default. haven has no 16-bit integer, so `u16`/`i16` are not accepted yet.
/// The explicitness is used to gate a data-carrying enum's `@export`/`extern`
/// crossing (see `check_export_type`): like Rust's `#[repr(C)]`, the layout is
/// only a committed FFI contract once the author has written `@repr` themselves.
pub(crate) fn enum_repr<'a>(attributes: &[Attribute<'a>]) -> Result<(Type<'a>, bool), String> {
    for a in attributes {
        if a.value.name == "repr" {
            return match a.value.value.as_deref() {
                None | Some("C") | Some("i32") => Ok((Type::Int32, true)),
                Some("i8")  => Ok((Type::Int8, true)),
                Some("i64") => Ok((Type::Int64, true)),
                Some("u8")  => Ok((Type::Uint8, true)),
                Some("u32") => Ok((Type::Uint32, true)),
                Some("u64") => Ok((Type::Uint64, true)),
                Some(other) => Err(format!(
                    "unknown @repr('{}') on enum; expected an integer type \
                     (i8/i32/i64/u8/u32/u64) or C", other)),
            };
        }
    }
    Ok((Type::Int32, false))
}

/// If `name` is an `Enum::Variant` reference to a declared enum, return the
/// enum's name, the variant's discriminant value, and the discriminant repr.
/// The enum name comes from the table key so it carries the `'a` lifetime.
pub(crate) fn enum_variant<'a>(cx: &Context<'a>, name: &str) -> Option<(&'a str, i64, Type<'a>)> {
    let (ename, variant) = name.split_once("::")?;
    let (&ekey, def) = cx.enums.get_key_value(ename)?;
    let val = *def.variants.get(variant)?;
    Some((ekey, val, def.repr.clone()))
}

/// Validate that `path` (a joined `E::V`) names a variant of enum `en`; returns
/// the variant name. Shared by the field-less `Path` and the destructuring
/// `Variant` match-arm patterns.
pub(crate) fn check_variant_pattern<'a>(cx: &Context<'a>, en: &'a str, path: &'a str, span: &Span)
-> Result<&'a str, Error> {
    let (pat_enum, variant) = path.split_once("::")
        .ok_or_else(|| Error { msg: format!("invalid enum pattern `{}`", path), span: span.clone() })?;
    // `en` may be a monomorphized instance (`std.option$Option$i32`) while the
    // pattern names the generic template (`std.option$Option::Some`): mono rewrites
    // construction call/struct-literal sites to the instance name, but never
    // touches match-pattern text (it has no type info to know which instantiation
    // a bare pattern refers to; see mono.rs).
    //
    // So the pattern's enum matches if it *is* `en`, or if `en` is one of its
    // instances - i.e. `en` is `pat_enum` followed by a `$`-separated argument
    // list. Requiring the `$` boundary is what keeps enum `Foo` from matching an
    // instance of a different enum `FooBar`.
    let is_instance = en.len() > pat_enum.len()
        && en.starts_with(pat_enum)
        && en.as_bytes()[pat_enum.len()] == b'$';
    if pat_enum != en && !is_instance {
        return Err(Error { msg: format!("pattern `{}` is not a variant of enum '{}'", path, en), span: span.clone() });
    }
    if !cx.enums[en].variants.contains_key(variant) {
        return Err(Error { msg: format!("enum '{}' has no variant '{}'", en, variant), span: span.clone() });
    }
    Ok(variant)
}

/// If `name` is `E::V` naming a variant of a declared enum, returns the enum's
/// name and the variant's payload field types (empty for a unit variant). Used
/// to recognize a constructor call `E::V(...)` in `infer`/`lower_expr`.
pub(crate) fn enum_variant_ctor<'a>(cx: &Context<'a>, name: &str) -> Option<(&'a str, Vec<Type<'a>>)> {
    let (ename, variant) = name.split_once("::")?;
    let (&ekey, def) = cx.enums.get_key_value(ename)?;
    if !def.variants.contains_key(variant) { return None; }
    let tys = def.payloads.get(variant)
        .map(|fs| fs.iter().map(|(_, t)| t.clone()).collect())
        .unwrap_or_default();
    Some((ekey, tys))
}

/// If `name` is `E::V` naming a variant of a declared enum, returns the (enum,
/// variant) names carrying the `'a` lifetime from the table keys. Used to route
/// a struct-literal `E::V { ... }` and a `StructVariant` pattern to the variant.
pub(crate) fn split_enum_variant<'a>(cx: &Context<'a>, name: &str) -> Option<(&'a str, &'a str)> {
    let (ename, variant) = name.split_once("::")?;
    let (&ekey, def) = cx.enums.get_key_value(ename)?;
    let (&vkey, _) = def.variants.get_key_value(variant)?;
    Some((ekey, vkey))
}
