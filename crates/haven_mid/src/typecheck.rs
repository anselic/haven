use std::collections::HashMap;
use haven_common::ast::*;
use haven_common::defs::{DefId, Defs, TyHead};
use haven_common::layout::{self, TypeInfo, TypeTable, EnumRepr};

mod context;
mod generics;
mod enums;
mod infer;

// Public surface re-exported for the rest of the crate (mil.rs) and the driver.
pub use context::{Context, EnumDef, GenericFnSig, MethodCall, RecvAdjust, TraitDef, TraitMethodSig, ENUM_TAG_FIELD, ENUM_PAYLOAD_FIELD};

use generics::{check_const_scope, check_type_resolves, subst_self_assoc};
use enums::{enum_variant, enum_repr, enum_agg_deps_ready, payload_blob_type};
use infer::{check_stmt, check_expr, always_returns};

/// Check whether a type is allowed in an @export function signature.
/// Some types are not allowed because they have unknown layout or calling convention,
/// or I just don't know how to handle it.
fn check_export_type<'a>(
    ty: &Type<'a>,
    types: &TypeTable<'a>,
    enums: &HashMap<DefId, EnumDef<'a>>,
    names: &HashMap<DefId, String>,
) -> Result<(), String> {
    match ty {
        // TODO check if this is correct
        Type::Array(inner, _) =>
            Err(format!("fixed-size array type '[{}; N]' is not allowed in @export functions, use a raw pointer '*{}' and an explicit length parameter instead", inner, inner)),
        Type::Slice(inner) =>
            Err(format!("slice type '{}' is not allowed in @export functions, use a raw pointer '*{}' and an explicit length parameter instead", ty, inner)),
        Type::Path { path, .. } => Type::unresolved(path),
        // `str` is a raw `*const u8` (a C string) - a single machine pointer,
        // so it is ABI-stable and maps directly to C's `const char*`.
        Type::Str => Ok(()),
        // A pointer is a single machine word regardless of what it points to, so
        // it is ABI-stable as an opaque handle even when the pointee's layout is
        // opaque to C (e.g. `*State`, `*u8`, `**u8`). We still reject pointers to
        // the genuinely fat / target-specific pointees (slice/simd/array), whose
        // *value* representation isn't a plain pointer. `*str` is fine - it is a
        // pointer to a pointer (`char**`).
        Type::Pointer(inner) => match &**inner {
            Type::Named { .. }
            | Type::Void | Type::Bool
            | Type::Int8 | Type::Int16 | Type::Int32 | Type::Int64
            | Type::Uint8 | Type::Uint16 | Type::Uint32 | Type::Uint64
            | Type::Float32 | Type::Float64
            | Type::Str
            | Type::Pointer(_) => Ok(()),
            _ => check_export_type(inner, types, enums, names), // *[]f32, *simd<...> stay banned
        },
        Type::Simd(_, _) =>
            Err(format!("SIMD type '{}' is not allowed in @export functions because its calling convention is target-specific and not guaranteed to match the expected caller, or that's what I'm told", ty)),
        Type::Function { .. } =>
            Err("function pointer types are not supported in @export functions".into()),
        Type::Param(_) =>
            Err("generic type parameters are not allowed in @export functions".into()),
        Type::Projection { .. } =>
            Err("associated type projections are not allowed in @export functions".into()),
        // `!` is not source-spellable and only ever an expression's inferred type,
        // so it cannot appear in a written signature - but reject it explicitly
        // rather than fall through.
        Type::Never =>
            Err("the bottom type '!' is not allowed in @export functions".into()),
        // A named type. A by-value struct is ABI-lowered (SysV eightbyte
        // classification), so it may cross the FFI boundary as long as every
        // field is itself export-safe - recurse, so a struct hiding a
        // slice/str/array is rejected.
        //
        // A field-less enum is its integer repr across FFI: a plain C enum,
        // nothing to check. A data-carrying one is laid out as a `{ tag: repr,
        // payload: union }` aggregate (see the data-enum pass below), the same
        // shape a C `struct { <repr> tag; union { ... }; }` tagged union takes -
        // `haven_back::abi` classifies the payload as a real union of the variant
        // payload structs, not raw bytes - so it may cross FFI as long as every
        // variant's fields are export-safe. That layout is identical whether or
        // not `@repr` was written, but - mirroring Rust's requirement that
        // `#[repr(C)]` be explicit before an enum is FFI-safe - we still require
        // the author to have written it, so crossing the boundary is a
        // deliberate, visible commitment rather than an accident of the default.
        Type::Named { def, .. } => {
            let name = names.get(def).cloned().unwrap_or_else(|| format!("#{}", def.0));
            if let Some(edef) = enums.get(def) {
                if !edef.has_payload { return Ok(()); }
                if !edef.has_explicit_repr {
                    return Err(format!(
                        "data-carrying enum '{}' needs an explicit `@repr` (e.g. `@repr(C)`) \
                         to cross an @export/extern boundary", name));
                }
                for (vname, fields) in &edef.payloads {
                    for (fname, fty) in fields {
                        check_export_type(fty, types, enums, names)
                            .map_err(|e| format!("field '{}' of variant '{}::{}': {}", fname, name, vname, e))?;
                    }
                }
                return Ok(());
            }
            let info = types.get(def)
                .ok_or_else(|| format!("unknown struct '{}'", name))?;
            for (fname, fty) in &info.fields {
                check_export_type(fty, types, enums, names)
                    .map_err(|e| format!("field '{}' of struct '{}': {}", fname, name, e))?;
            }
            Ok(())
        }
        // these are all fine across FFI
        Type::Void | Type::Bool
        | Type::Int8 | Type::Int16 | Type::Int32 | Type::Int64
        | Type::Uint8 | Type::Uint16 | Type::Uint32 | Type::Uint64
        | Type::Float32 | Type::Float64 => Ok(()),
    }
}

/// Verify that a global's initializer is a compile-time constant we can emit as
/// an LLVM `constant` aggregate: literals (optionally negated), struct literals of
/// constants, and bare function names (a function's address is a link-time
/// constant). Anything that would need to run code - a call, a load of another
/// global's value, indexing - is rejected.
fn check_const_initializer<'a>(cx: &Context<'a>, expr: &Expr<'a>) -> Result<(), Error> {
    match &expr.value {
        ExprNode::Bool(_)
        | ExprNode::Int8(_) | ExprNode::Int16(_) | ExprNode::Int32(_) | ExprNode::Int64(_)
        | ExprNode::Uint8(_) | ExprNode::Uint16(_) | ExprNode::Uint32(_) | ExprNode::Uint64(_)
        | ExprNode::Float32(_) | ExprNode::Float64(_)
        | ExprNode::IntLit(_) | ExprNode::FloatLit(_) => Ok(()),
        // a string literal is the address of a read-only global blob (`@.str.N`),
        // a link-time constant - exactly like a function's address below.
        ExprNode::Str(_) => Ok(()),
        // a negated numeric literal, e.g. `-1.0`, is still a constant
        ExprNode::Unary { op: UnaryOp::Neg, operand }
            if matches!(operand.value,
                ExprNode::Int8(_) | ExprNode::Int16(_) | ExprNode::Int32(_) | ExprNode::Int64(_)
                | ExprNode::Uint8(_) | ExprNode::Uint16(_) | ExprNode::Uint32(_) | ExprNode::Uint64(_)
                | ExprNode::Float32(_) | ExprNode::Float64(_)
                | ExprNode::IntLit(_) | ExprNode::FloatLit(_)) => Ok(()),
        // an `Enum::Variant` is a compile-time integer constant.
        ExprNode::Path(path) if enum_variant(cx, path).is_some() => Ok(()),
        // a bare top-level function name: its address is a link-time constant.
        // a *global* of function type is excluded - reading its value isn't const.
        ExprNode::Var(name)
            if matches!(cx.lookup(name), Some((_, Type::Function { .. })))
                && !cx.global_consts.contains(name) => Ok(()),
        // a generic function taken by value (`foo::<T>`): its monomorphized
        // instance's address is a link-time constant, exactly like a bare fn name.
        // The turbofish/arity is validated by the type pass; here we only certify
        // constant-ness.
        ExprNode::FnRef { .. } => Ok(()),
        // a struct literal is constant iff every field initializer is constant
        // (nested structs recurse). field names/types are checked by check_expr.
        ExprNode::Struct { fields, .. } => {
            for (_, fexpr) in fields {
                check_const_initializer(cx, fexpr)?;
            }
            Ok(())
        }
        // an array literal is constant iff every element is.
        ExprNode::Slice(elements) => {
            for elem in elements {
                check_const_initializer(cx, elem)?;
            }
            Ok(())
        }
        _ => Err(Error::new(expr.span, "global initializer is not a constant")
            .with_note("a global may be initialized with a literal, a string, a \
                        struct/array literal of constants, or a function name")),
    }
}

fn check_toplevel<'a>(
    cx: &mut Context<'a>,
    node: &TopLevel<'a>,
) -> Result<(), Error> {
    match &node.value {
        TopLevelNode::Function { name, attributes, generics, params, return_type, body, .. } => {
            if attributes.iter().any(|a| a.value.name == "export") {
                // monomorphization isn't implemented yet, so a generic function
                // has no single concrete ABI to export
                if !generics.is_empty() {
                    return Err(Error::new(node.span, format!(
                        "@export function '{}' cannot be generic", name)));
                }
                for (param_name, ty) in params {
                    // resolve enum names first (@export can't be generic) so an
                    // enum param is checked as its integer repr, not an unknown struct.
                    let ty = ty.clone();
                    if let Err(msg) = check_export_type(&ty, &cx.types, &cx.enums, &cx.names) {
                        return Err(Error::new(node.span, format!(
                            "parameter '{}' in @export function '{}': {}", param_name, name, msg))
                            .with_label(node.span, format!("parameter '{}'", param_name)));
                    }
                }
                let return_type = return_type.clone();
                if let Err(msg) = check_export_type(&return_type, &cx.types, &cx.enums, &cx.names) {
                    return Err(Error::new(node.span, format!(
                        "return type in @export function '{}': {}", name, msg))
                        .with_label(node.span, "return type"));
                }
            }

            // bind type params for the duration of this function so that bare
            // idents in the signature/body resolve to `Type::Param` rather than
            // an (undeclared) struct
            cx.generics = generics.iter().filter_map(|g| match g {
                GenericParam::Type { name: n, .. } => Some(*n),
                GenericParam::Const(_, _) => None,
            }).collect();
            cx.const_generics = generics.iter().filter_map(|g| match g {
                GenericParam::Const(n, _) => Some(*n),
                GenericParam::Type { .. } => None,
            }).collect();
            // trait bounds on this fn's type params, so a method call on a
            // `T`-typed receiver in the body resolves through the bound trait.
            cx.generic_bounds = generics.iter().filter_map(|g| match g {
                GenericParam::Type { name, bounds } if !bounds.is_empty() =>
                    Some((*name, bounds.iter().map(|b| b.def).collect())),
                _ => None,
            }).collect();
            // every bound must name a declared trait.
            for g in generics {
                if let GenericParam::Type { name: pn, bounds } = g {
                    for b in bounds {
                        if !cx.traits.contains_key(&b.def) {
                            return Err(Error::new(node.span, format!(
                                "unknown trait '{}' in bound '{}: {}' of '{}'", b, pn, b, name)));
                        }
                    }
                }
            }

            // every `ConstVal::Param` in the signature must name a declared const
            // param. runs for non-generic fns too (empty scope), so a stray
            // `[f32; N]` outside a generic is rejected rather than silently
            // producing an unresolved param.
            for (pname, ty) in params {
                if let Err(msg) = check_const_scope(&cx.const_generics, ty) {
                    return Err(Error::new(node.span, format!(
                        "parameter '{}' of '{}': {}", pname, name, msg))
                        .with_label(node.span, format!("parameter '{}'", pname)));
                }
            }
            if let Err(msg) = check_const_scope(&cx.const_generics, return_type) {
                return Err(Error::new(node.span, format!("return type of '{}': {}", name, msg))
                    .with_label(node.span, "return type"));
            }

            let return_ty = return_type.clone();
            if let Err(msg) = check_type_resolves(cx, &return_ty) {
                return Err(Error::new(node.span, format!("return type of '{}': {}", name, msg))
                    .with_label(node.span, "return type"));
            }

            cx.push_scope();

            // const generics are ordinary compile-time values, visible in the body
            for g in generics {
                if let GenericParam::Const(cname, cty) = g {
                    // const generics are substituted with literals by mono, so
                    // they never surface as `Var` uses in the lowered program.
                    cx.insert(cname, None, cty.clone());
                }
            }

            // push params into scope, resolving type-param references
            for (pname, ty) in params {
                let resolved = ty.clone();
                if let Err(msg) = check_type_resolves(cx, &resolved) {
                    return Err(Error::new(node.span, format!(
                        "parameter '{}' of '{}': {}", pname, name, msg))
                        .with_label(node.span, format!("parameter '{}'", pname)));
                }
                cx.insert(pname, Some(Binding::Param(pname)), resolved);
            }

            for stmt in body {
                check_stmt(cx, &return_ty, stmt)?;
            }

            // a non-void function must return on every path, or control can fall
            // off the end with no value. structural over the body, so it also
            // covers generic templates (return_type may still hold `Param`s).
            if return_ty != Type::Void && !body.iter().any(|s| always_returns(s, &cx.node_types)) {
                cx.pop_scope();
                cx.generics = Vec::new();
                cx.const_generics = Vec::new();
                cx.generic_bounds = HashMap::new();
                // a `!` function promises never to return; the failure is that
                // control can fall off its end, not that a value is missing.
                return Err(if return_ty == Type::Never {
                    Error::new(node.span,
                        format!("function '{}' can fall off its end", name))
                        .with_label(node.span,
                            "declared '!', so it must never return")
                        .with_note("end it with `abort(...)` or a call that never returns")
                } else {
                    Error::new(node.span,
                        format!("function '{}': not all paths return a value", name))
                        .with_label(node.span,
                            format!("declared return type is '{}'", return_type))
                });
            }

            cx.pop_scope();
            cx.generics = Vec::new();
            cx.const_generics = Vec::new();
            cx.generic_bounds = HashMap::new();
        }

        // extern declarations have no body to check, but their signature types
        // must still resolve (so an unknown or not-yet-supported generic struct
        // type is caught at typecheck, not by a codegen panic).
        TopLevelNode::Extern { name, params, return_type, .. } => {
            for (pname, ty) in params {
                let resolved = ty.clone();
                if let Err(msg) = check_type_resolves(cx, &resolved) {
                    return Err(Error::new(node.span, format!(
                        "parameter '{}' of extern '{}': {}", pname, name, msg))
                        .with_label(node.span, format!("parameter '{}'", pname)));
                }
            }
            let resolved_ret = return_type.clone();
            if let Err(msg) = check_type_resolves(cx, &resolved_ret) {
                return Err(Error::new(node.span, format!("return type of extern '{}': {}", name, msg))
                    .with_label(node.span, "return type"));
            }
        }

        TopLevelNode::Struct { name, generics, fields, .. } => {
            // the struct's own type and const params are in scope inside its fields:
            // `T` resolves to `Param`, and a `[T; N]` size names the const param `N`.
            let const_params: Vec<&'a str> = generics.iter().filter_map(|g| match g {
                GenericParam::Const(n, _) => Some(*n),
                GenericParam::Type { .. } => None,
            }).collect();
            // ensure no duplicate field names and that referenced struct types exist
            let mut seen = std::collections::HashSet::new();
            for (field_name, field_ty) in fields {
                if !seen.insert(*field_name) {
                    return Err(Error::new(node.span, format!(
                        "Duplicate field '{}' in struct '{}'", field_name, name)));
                }
                // resolve the struct's own type params to `Param` first, so they
                // aren't reported as unknown struct names.
                let resolved = field_ty.clone();
                if let Err(msg) = check_type_resolves(cx, &resolved) {
                    return Err(Error::new(node.span, format!(
                        "in field '{}' of struct '{}': {}", field_name, name, msg))
                        .with_label(node.span, format!("field '{}'", field_name)));
                }
                // every `ConstVal::Param` in the field (an `[T; N]` size) must name
                // one of the struct's declared const params.
                if let Err(msg) = check_const_scope(&const_params, &resolved) {
                    return Err(Error::new(node.span, format!(
                        "in field '{}' of struct '{}': {}", field_name, name, msg))
                        .with_label(node.span, format!("field '{}'", field_name)));
                }
            }
        }

        TopLevelNode::Global { name, ty, value, .. } => {
            if let Err(msg) = check_type_resolves(cx, ty) {
                return Err(Error::new(node.span, format!("in global '{}': {}", name, msg))
                    .with_label(node.span, format!("global '{}'", name)));
            }
            check_const_initializer(cx, value)?;
            check_expr(cx, ty, value)?;
        }
        // field-less enums are fully validated in the forward-declaration pass
        // (duplicate variants, `@repr` value); nothing more to check here.
        TopLevelNode::Enum { .. } => {}
        // traits are registered + conformance-checked in the forward pass; the
        // signatures carry no bodies to check here.
        TopLevelNode::Trait { .. } => {}
        // methods were desugared to functions in the module resolver.
        TopLevelNode::Alias { .. } => unreachable!("aliases expanded before typecheck"),
        TopLevelNode::Extend { .. } => unreachable!("extend desugared before typecheck"),
    }

    Ok(())
}

pub fn typecheck_program<'a>(
    cx: &mut Context<'a>,
    program: &[TopLevel<'a>],
    impls: &[ImplDecl<'a>],
    defs: &Defs<'a>,
) -> Vec<Error> {
    let mut errors = Vec::new();

    // methods are resolved through the table the module resolver built, not by
    // reconstructing names. cheap to clone: one entry per declared method, and
    // both typecheck passes need it.
    cx.members = defs.members().clone();
    // every monomorphized instance: for matching a template-named pattern
    // against an instance-typed scrutinee, and for putting an instance-typed
    // receiver back into the template form every type-keyed table uses. Empty
    // until mono has run.
    cx.instances = defs.instances().clone();
    // the synthetic payload struct of each data variant, minted at resolution
    // (and by `mono` for each instance), so nothing here has to build one.
    cx.payloads = defs.payloads().clone();
    // how each definition reads in a diagnostic.
    cx.load_names(defs);
    // the definitions the compiler itself knows about, resolved once by the
    // module loader from the prelude. Read rather than rediscovered by scanning
    // trait declarations, so the post-mono pass - which has no trait nodes left,
    // mono having dropped them - is as well informed as the first pass.
    cx.delete_trait = defs.lang().delete;

    // --- forward declaration pass

    // enums come first: a struct field or function parameter may name an enum,
    // and the aggregate pass below needs every enum's repr already recorded.
    //
    // A duplicate declaration is caught at resolution now (two `struct Point` in
    // one module collide in that module's symbol table), so there is nothing to
    // check here: each declaration has its own identity by construction.
    for node in program {
        if let TopLevelNode::Enum { def, name, attributes, generics, variants, .. } = &node.value {
            let def = *def;
            let (repr, has_explicit_repr) = match enum_repr(attributes) {
                Ok(r) => r,
                Err(msg) => {
                    errors.push(Error::new(node.span, msg)
                        .with_note("expected an integer type \
                                    (i8/i16/i32/i64/u8/u16/u32/u64) or C"));
                    continue;
                }
            };
            if !generics.is_empty() {
                cx.generic_enums.insert(def, generics.clone());
            }
            // C-style discriminants: an unspecified variant is the previous one
            // plus one, starting at 0. Payload field types are collected here but
            // resolved against the enum table in a later pass (below), once every
            // enum is known.
            let mut vmap = HashMap::new();
            let mut payloads: HashMap<&str, Vec<(&str, Type)>> = HashMap::new();
            let mut next: i64 = 0;
            let mut dup = false;
            let mut has_payload = false;
            for (vname, explicit, payload) in variants {
                let val = explicit.unwrap_or(next);
                if vmap.insert(*vname, val).is_some() {
                    errors.push(Error::new(node.span, format!(
                        "Duplicate variant '{}' in enum '{}'", vname, name)));
                    dup = true;
                    break;
                }
                if !payload.is_empty() {
                    has_payload = true;
                    // keep the field names (real names for struct variants, "0"/"1"
                    // for tuple variants); the payload struct is built from them.
                    payloads.insert(*vname, payload.clone());
                }
                next = val + 1;
            }
            if dup { continue; }
            // register the discriminant repr for layout/codegen. A field-less
            // enum is a bare scalar and stops here; a data enum's `{ $tag,
            // $payload }` field list is filled in by the aggregate pass below,
            // once every payload struct can be measured.
            cx.types.insert(def, TypeInfo {
                fields: Vec::new(),
                enum_: Some(EnumRepr { repr: repr.clone(), has_payload }),
            });
            cx.enums.insert(def, EnumDef { repr, variants: vmap, payloads, has_payload, has_explicit_repr });
        }
    }

    // forward declare structs so that they can be referenced in function signatures
    for node in program {
        if let TopLevelNode::Struct { def, generics, fields, .. } = &node.value {
            // field types arrive already resolved: a field naming the struct's
            // own type param is a `Type::Param`, anything else an identity.
            cx.types.insert(*def, TypeInfo::struct_(fields.clone()));
            if !generics.is_empty() {
                cx.generic_structs.insert(*def, generics.clone());
            }
        }
    }

    // data enums: now that every enum and struct is declared, resolve payload
    // field types and register the synthetic structs that back the aggregate.
    // For `enum Msg { Note(u8, f32) }` this registers a payload struct
    // `Msg$Note = { "0": u8, "1": f32 }` and the aggregate `Msg = { $tag: repr,
    // $payload: [P x i8] }` where P is the largest variant payload. These reuse
    // the whole struct machinery (layout, FieldPtr, AllocaStruct, copy_struct,
    // sret) so the backend needs almost no new aggregate code.
    let data_enums: Vec<DefId> = cx.enums.iter()
        .filter(|(_, d)| d.has_payload).map(|(n, _)| *n).collect();
    // pass 1: register every payload struct and stash the resolved payloads. The
    // payload struct's field names are exactly the variant's field names, so a
    // struct-style variant's `{ id, val }` is a struct `Msg$Cc = { id, val }`.
    // A generic enum's own type params (e.g. `Option<T>`) are in scope while
    // resolving its payloads, so a field naming one becomes `Type::Param`, exactly
    // like a generic struct's own fields - construction/match-arm checking then
    // substitutes it via `subst_param_type`, and monomorphization flattens it.
    for &ename in &data_enums {
        let raw: Vec<(&'a str, Vec<(&'a str, Type<'a>)>)> = cx.enums[&ename].payloads.iter()
            .map(|(v, fs)| (*v, fs.clone())).collect();
        for (vname, fields) in raw {
            let pdef = cx.payloads[&(ename, vname)];
            cx.types.insert(pdef, TypeInfo::struct_(fields));
        }
    }
    // pass 2: register each aggregate, sizing its byte blob from the (now present)
    // payload structs. Skips a GENERIC enum template: its payload structs still
    // hold `Type::Param` fields (no concrete args bound yet), and `layout::size_of`
    // panics on those - only a concrete instance (monomorphized, or never generic
    // to begin with) gets laid out and lowered to codegen.
    //
    // Iterates to a FIXPOINT rather than a single pass: one data enum's payload
    // can itself be another data enum (`Option<Option<i32>>`), whose own aggregate
    // must be registered first so `layout::size_of` can measure it - but
    // `data_enums`'s order (from a HashMap) is unspecified, so a naive single pass
    // can reach the outer enum before its dependency is ready. Round-robin: process
    // whatever's ready, retry the rest, stop when a full round makes no progress
    // (which - for anything that survived monomorphization's own type-depth limit -
    // only happens once every enum is done).
    let mut pending: Vec<DefId> = data_enums.iter()
        .filter(|e| !cx.generic_enums.contains_key(e)).cloned().collect();
    loop {
        let mut still_pending = Vec::new();
        let mut progressed = false;
        for ename in pending {
            if !enum_agg_deps_ready(ename, &cx.enums, &cx.types) {
                still_pending.push(ename);
                continue;
            }
            let repr = cx.enums[&ename].repr.clone();
            let repr2 = repr.clone();
            let variants: Vec<&'a str> = cx.enums[&ename].payloads.keys().copied().collect();
            let payload_bytes = variants.iter()
                .map(|v| layout::size_of(&Type::named(cx.payloads[&(ename, *v)]), &cx.types))
                .max().unwrap_or(0);
            let payload_align = variants.iter()
                .map(|v| layout::align_of(&Type::named(cx.payloads[&(ename, *v)]), &cx.types))
                .max().unwrap_or(1);
            let agg_fields = vec![
                (ENUM_TAG_FIELD, repr),
                (ENUM_PAYLOAD_FIELD, payload_blob_type(payload_bytes, payload_align)),
            ];
            cx.types.insert(ename, TypeInfo {
                fields: agg_fields,
                enum_: Some(EnumRepr { repr: repr2, has_payload: true }),
            });
            progressed = true;
        }
        if still_pending.is_empty() || !progressed { break; }
        pending = still_pending;
    }

    // forward declare functions so that they can be called before their definition
    for node in program {
        match &node.value {
            // generic functions go into a separate table; not callable via the
            // ordinary function-type path, only through turbofish.
            TopLevelNode::Function { name, generics, params, return_type, .. }
                if !generics.is_empty() => {
                // only type params get reclassified `Struct`->`Param`; const params
                // already arrive as `ConstVal::Param` from the parser.
                let resolved_params = params.iter()
                    .map(|(_, ty)| ty.clone())
                    .collect();
                let resolved_return = return_type.clone();
                cx.generic_fns.insert(name, GenericFnSig {
                    generics: generics.clone(),
                    params: resolved_params,
                    return_type: resolved_return,
                });
            }
            // generic externs (`extern printf<T>(...)`) are the same story as
            // generic functions: they live in `generic_fns` and are only reachable
            // through turbofish, never via the ordinary function-type path. codegen
            // still emits the single underlying C symbol (mono keeps the name, drops
            // the turbofish), so there's nothing to monomorphize.
            TopLevelNode::Extern { name, generics, params, return_type, .. }
                if !generics.is_empty() => {
                let resolved_params = params.iter()
                    .map(|(_, ty)| ty.clone())
                    .collect();
                let resolved_return = return_type.clone();
                cx.generic_fns.insert(name, GenericFnSig {
                    generics: generics.clone(),
                    params: resolved_params,
                    return_type: resolved_return,
                });
            }
            TopLevelNode::Function { name, params, return_type, .. }
            | TopLevelNode::Extern { name, params, return_type, .. } => {
                // resolve enum-typed params/return (no generics here) so the stored
                // signature uses `Type::Enum`, matching how a call's args infer;
                // otherwise an enum param stays `Struct` and mismatches the arg.
                cx.insert(name, None, Type::Function {
                    params: params.iter().map(|(_, ty)| ty.clone()).collect(),
                    return_type: Box::new(return_type.clone()),
                });
            }
            // globals share the value namespace with functions; register the name
            // so references resolve as ordinary variables.
            TopLevelNode::Global { name, ty, .. } => {
                cx.insert(name, None, ty.clone());
                cx.global_consts.insert(name);
            }
            // enums were collected in their own pass above; nothing to register
            // in the value namespace (variant refs resolve directly). traits are
            // registered in their own pass below.
            TopLevelNode::Struct { .. } | TopLevelNode::Enum { .. } | TopLevelNode::Trait { .. } => {}
            TopLevelNode::Alias { .. } => unreachable!("aliases expanded before typecheck"),
            TopLevelNode::Extend { .. } => unreachable!("extend desugared before typecheck"),
        }
    }

    // register each trait's required method signatures. runs after structs and
    // enums are declared so a method's param/return types resolve (`String` ->
    // the struct, an enum name -> `Type::Enum`); `Self` is left symbolic and
    // substituted per-impl / per-bound later.
    for node in program {
        if let TopLevelNode::Trait { def, name, assoc_types, methods, .. } = &node.value {
            let mut ms: HashMap<&'a str, TraitMethodSig<'a>> = HashMap::new();
            let mut dup = false;
            for m in methods {
                if ms.contains_key(m.name) {
                    errors.push(Error::new(node.span, format!(
                        "Duplicate method '{}' in trait '{}'", m.name, name)));
                    dup = true;
                    break;
                }
                let params = m.params.iter()
                    .map(|(_, t)| t.clone()).collect();
                let return_type = m.return_type.clone();
                ms.insert(m.name, TraitMethodSig { receiver: m.receiver, params, return_type });
            }
            if dup { continue; }
            cx.traits.insert(*def, TraitDef { methods: ms, assoc_types: assoc_types.clone() });
        }
    }

    // Trait names were resolved by the front end, but the type namespace also
    // contains structs and enums. Once every trait has been registered, reject
    // associated-type bounds that resolved to some other kind of definition.
    for node in program {
        let TopLevelNode::Trait { assoc_types, .. } = &node.value else { continue };
        for decl in assoc_types {
            for bound in &decl.bounds {
                if !cx.traits.contains_key(&bound.def) {
                    errors.push(Error::new(node.span, format!(
                        "'{}' is not a trait", cx.name_of(bound.def))));
                }
            }
        }
    }

    // check every `extend T: Trait` conformance and record the satisfied impls,
    // so a `T: Trait` bound at a generic call site can be verified. Runs after
    // function forward-declaration, since a method's desugared function (`T$m`)
    // must be visible to compare its signature.
    // Make the complete impl set available before validating any one impl. An
    // associated-type bound may be satisfied by an impl declared later in the
    // source, and declaration order must not change conformance.
    cx.impls.extend(impls.iter().cloned());
    for imp in impls {
        check_impl_conformance(cx, imp, &mut errors);
    }

    // --- typecheck pass
    for node in program {
        // the module a receiver call inside this item belongs to, so a private
        // method reached across a module boundary can be rejected. Items with no
        // definition (none, currently) leave the placeholder in place.
        if let Some(def) = toplevel_def(&node.value) {
            cx.current_module = defs.get(def).module;
        }
        if let Err(err) = check_toplevel(cx, node) {
            errors.push(err);
        }
    }

    errors
}

/// The definition identity of a top-level item, if it has one. Every item kind
/// carries a `def`; `extend` is desugared before typecheck, so it never appears.
fn toplevel_def(node: &TopLevelNode) -> Option<DefId> {
    match node {
        TopLevelNode::Function { def, .. }
        | TopLevelNode::Extern { def, .. }
        | TopLevelNode::Struct { def, .. }
        | TopLevelNode::Global { def, .. }
        | TopLevelNode::Enum { def, .. }
        | TopLevelNode::Trait { def, .. } => Some(*def),
        TopLevelNode::Alias { .. } => None,
        TopLevelNode::Extend { .. } => None,
    }
}

/// Verify that `imp.target` implements every method of `imp.trait_` with a
/// matching signature, and record the `(target, trait)` impl regardless (a
/// conformance error already stops compilation before mono, so recording it
/// avoids a duplicate "does not implement" at the bound site). A method's
/// receiver is expanded to the concrete `self` type and any `Self` in the trait
/// signature is substituted with the implementing type before comparison.
fn check_impl_conformance<'a>(cx: &mut Context<'a>, imp: &ImplDecl<'a>, errors: &mut Vec<Error>) {
    // the implementing type, which is no longer necessarily nameable as a
    // definition - `[T]` and `i32` have only their written form.
    let self_ty = imp.self_ty.clone();
    let (target, trait_) = (cx.show(&self_ty), cx.name_of(imp.trait_));
    let trait_def = match cx.traits.get(&imp.trait_) {
        Some(d) => d.clone(),
        None => {
            errors.push(Error::new(imp.span, format!(
                "unknown trait '{}' in `extend {}: {}`", trait_, target, trait_)));
            return;
        }
    };

    // the associated-type bindings this impl supplies, keyed by name. Every
    // associated type the trait declares must be bound exactly once; a binding
    // naming a type the trait never declared is a mistake, not silently ignored.
    let mut assoc: HashMap<&'a str, Type<'a>> = HashMap::new();
    for (name, ty) in &imp.assoc_bindings {
        if trait_def.assoc_type(name).is_none() {
            errors.push(Error::new(imp.span, format!(
                "trait '{}' has no associated type '{}'", trait_, name)));
            continue;
        }
        if assoc.insert(*name, ty.clone()).is_some() {
            errors.push(Error::new(imp.span, format!(
                "associated type '{}' is bound more than once", name)));
        }
    }
    for decl in &trait_def.assoc_types {
        if !assoc.contains_key(decl.name) {
            errors.push(Error::new(imp.span, format!(
                "type '{}' does not implement trait '{}': missing associated type '{}'",
                target, trait_, decl.name)));
        }
    }

    let impl_scope: ParamBounds<'a> = imp.generics.iter().filter_map(|generic| match generic {
        GenericParam::Type { name, bounds } =>
            Some((*name, bounds.iter().map(|bound| bound.def).collect())),
        GenericParam::Const(..) => None,
    }).collect();
    for decl in &trait_def.assoc_types {
        let Some(binding) = assoc.get(decl.name) else { continue };
        for bound in &decl.bounds {
            if !cx.implements_in_scope(binding, bound.def, &impl_scope) {
                errors.push(Error::new(imp.span, format!(
                    "associated type '{}' for '{}' does not satisfy bound '{}'",
                    decl.name, target, cx.name_of(bound.def))));
            }
        }
    }

    for (mname, sig) in &trait_def.methods {
        // the member table knows what the impl actually declared; this used to
        // rebuild `Target$method`, which missed entirely when the type's emitted
        // name isn't slug-prefixed but its methods' are (enums, `@export`ed
        // structs) - reporting "does not implement" for a method that was right
        // there.
        //
        // A method of a *generic* impl is a generic function, so it lives in
        // `generic_fns` rather than the ordinary value scope. Its signature is
        // compared symbolically - in terms of the impl's own parameters, which
        // is also how `self_ty` is spelled - so `extend Vec<T>: Delete` is
        // checked once for all `T` rather than per instance.
        let found = match cx.members.get(&(imp.head, *mname)) {
            Some(m) if m.generics.is_empty() => match cx.lookup(m.name) {
                Some((_, Type::Function { params, return_type })) =>
                    Some((params.clone(), (**return_type).clone())),
                _ => None,
            },
            Some(m) => cx.generic_fns.get(m.name)
                .map(|sig| (sig.params.clone(), sig.return_type.clone())),
            None => None,
        };
        let Some((params, return_type)) = found else {
            errors.push(Error::new(imp.span,
                format!("type '{}' does not implement trait '{}'", target, trait_))
                .with_label(imp.span, format!("missing method '{}'", mname)));
            continue;
        };
        // expected parameter list: the receiver's `self` type (if any) followed
        // by the declared params, with `Self` substituted to the concrete type.
        let mut expected: Vec<Type<'a>> = Vec::new();
        match sig.receiver {
            Receiver::Associated => {}
            Receiver::Value => expected.push(self_ty.clone()),
            Receiver::Pointer => expected.push(Type::Pointer(Box::new(self_ty.clone()))),
        }
        for p in &sig.params { expected.push(subst_self_assoc(p, &self_ty, &assoc)); }
        let expected_ret = subst_self_assoc(&sig.return_type, &self_ty, &assoc);

        if params.len() != expected.len()
            || params.iter().zip(&expected).any(|(a, b)| a != b)
            || return_type != expected_ret
        {
            errors.push(Error::new(imp.span,
                format!("signature mismatch on '{}::{}'", trait_, mname))
                .with_label(imp.span,
                    format!("'{}' implements it differently from the trait", target))
                .with_note(format!("expected `({}) {}`",
                    expected.iter().map(|t| cx.show(t)).collect::<Vec<_>>().join(", "),
                    cx.show(&expected_ret))));
        }
    }

    // `Delete` is the one trait whose subject must be a *named* type. Its
    // destructor is called from code the ownership pass synthesizes after
    // monomorphization, so the specialized `delete` has to have been minted
    // already - and mono only mints per generic-type instance, an event a
    // structural type has no equivalent of. Allowing it would compile to a
    // silent leak rather than an error, so it is rejected here.
    if Some(imp.trait_) == cx.delete_trait && !matches!(imp.head, TyHead::Def(_)) {
        errors.push(Error::new(imp.span,
            format!("`{}` cannot implement `Delete`", target))
            .with_label(imp.span, "only a struct or enum may own a resource")
            .with_note("a primitive or structural type ([T], *T, [T; N]) is a value or a \
                        borrowed view, and destroying one would destroy something it does \
                        not own"));
    }
}
