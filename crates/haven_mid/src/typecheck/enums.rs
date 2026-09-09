use std::collections::HashMap;
use haven_common::ast::*;
use haven_common::defs::DefId;
use haven_common::layout::{TypeTable, TypeInfo, EnumRepr};
use super::context::{Context, EnumDef};

/// Whether every aggregate (struct, or data-enum) that `ename`'s payload fields
/// reference is already registered in `structs` - i.e. whether it's safe to call
/// `layout::size_of` on `ename`'s payload structs yet. Used to sequence data-enum
/// "pass 2" (aggregate registration) into a fixpoint: a data enum's payload can
/// itself be another (unregistered) data enum (`Option<Option<i32>>`).
pub(crate) fn enum_agg_deps_ready<'a>(
    ename: DefId,
    enums: &HashMap<DefId, EnumDef<'a>>,
    types: &TypeTable<'a>,
) -> bool {
    fn ty_ready<'a>(ty: &Type<'a>, types: &TypeTable<'a>) -> bool {
        match ty {
            // a named type blocks readiness until it can actually be measured.
            //
            // Every enum is registered with its discriminant repr as soon as it
            // is declared, so mere presence in the table is not enough: a data
            // enum's `{ $tag, $payload }` field list is filled in by the very
            // pass this gates, and until then it is empty and would measure as
            // zero bytes. A field-less enum is a bare scalar and is ready at
            // once; so is any struct, whose fields are known at declaration.
            Type::Named { def, .. } => match types.get(def) {
                None => false,
                Some(TypeInfo { enum_: Some(EnumRepr { has_payload: true, .. }), fields }) =>
                    !fields.is_empty(),
                Some(_) => true,
            },
            // a pointer is a fixed-size opaque handle: doesn't need its pointee's
            // own layout, so it never blocks readiness.
            Type::Pointer(_) => true,
            Type::Array(inner, _) | Type::Slice(inner) | Type::Simd(inner, _) => ty_ready(inner, types),
            _ => true,
        }
    }
    enums[&ename].payloads.values().all(|fields| fields.iter().all(|(_, ty)| ty_ready(ty, types)))
}

/// The `$payload` byte-blob type for a data enum's aggregate: an array sized to
/// hold the largest variant, chunked so the array's natural alignment matches
/// `needed_align` - the max any variant's payload requires. A plain `[N x i8]`
/// is always align 1, which under-reports whenever a variant holds a pointer or
/// `f64` (align 8): the payload would then sit right after a 4-byte `i32` tag at
/// offset 4, misaligned versus a C `struct { int tag; union { ... }; }`. `i32`/
/// `i64` chunks instead let LLVM's layout (and our `layout` module, which
/// mirrors it) pad and align the field correctly - no change at access sites,
/// since a `FieldPtr` re-bases through the variant's own payload struct and
/// never sees `$payload`'s element type. Alignments above 8 (e.g. SIMD) are not
/// modeled and stay under-aligned; nothing exercises that yet.
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
                Some("i16") => Ok((Type::Int16, true)),
                Some("i64") => Ok((Type::Int64, true)),
                Some("u8")  => Ok((Type::Uint8, true)),
                Some("u16") => Ok((Type::Uint16, true)),
                Some("u32") => Ok((Type::Uint32, true)),
                Some("u64") => Ok((Type::Uint64, true)),
                Some(other) => Err(format!("unknown @repr('{}') on enum", other)),
            };
        }
    }
    Ok((Type::Int32, false))
}

/// If `r` refers to a variant of a declared enum, return the enum, the
/// variant's discriminant value, and the discriminant repr.
pub(crate) fn enum_variant<'a>(cx: &Context<'a>, r: &NameRef<'a>) -> Option<(DefId, i64, Type<'a>)> {
    let def = cx.enums.get(&r.def)?;
    let val = *def.variants.get(r.variant())?;
    Some((r.def, val, def.repr.clone()))
}

/// Validate that `r` names a variant of enum `en`; returns the variant name.
/// Shared by the field-less `Path` and the destructuring `Variant` match-arm
/// patterns.
pub(crate) fn check_variant_pattern<'a>(cx: &Context<'a>, en: DefId, r: &NameRef<'a>, span: &Span)
-> Result<&'a str, Error> {
    let variant = r.variant();
    // `en` may be a monomorphized instance while the pattern names the generic
    // template: mono rewrites construction call/struct-literal sites to the
    // instance, but never touches match patterns (it has no type info to know
    // which instantiation a bare pattern refers to; see mono.rs).
    //
    // So the pattern's enum matches if it *is* `en`, or if `en` is an instance
    // that mono recorded as specializing it.
    let is_instance = cx.instances.get(&en).is_some_and(|i| i.template == r.def);
    if r.def != en && !is_instance {
        return Err(Error::new(*span, format!(
            "pattern `{}` is not a variant of enum '{}'", r, cx.name_of(en))));
    }
    if !cx.enums[&en].variants.contains_key(variant) {
        return Err(Error::new(*span, format!(
            "enum '{}' has no variant '{}'", cx.name_of(en), variant)));
    }
    Ok(variant)
}

/// If `r` names a variant of a declared enum, returns the enum and the
/// variant's payload field types (empty for a unit variant). Used to recognize
/// a constructor call `E::V(...)` in `infer`/`lower_expr`.
pub(crate) fn enum_variant_ctor<'a>(cx: &Context<'a>, r: &NameRef<'a>) -> Option<(DefId, Vec<Type<'a>>)> {
    let def = cx.enums.get(&r.def)?;
    let variant = r.variant();
    if !def.variants.contains_key(variant) { return None; }
    let tys = def.payloads.get(variant)
        .map(|fs| fs.iter().map(|(_, t)| t.clone()).collect())
        .unwrap_or_default();
    Some((r.def, tys))
}

/// If `r` names a variant of a declared enum, returns the enum and the variant
/// name carrying the `'a` lifetime from the table key. Used to route a
/// struct-literal `E::V { ... }` and a `StructVariant` pattern to the variant.
pub(crate) fn split_enum_variant<'a>(cx: &Context<'a>, r: &NameRef<'a>) -> Option<(DefId, &'a str)> {
    let def = cx.enums.get(&r.def)?;
    let (&vkey, _) = def.variants.get_key_value(r.variant())?;
    Some((r.def, vkey))
}
