//! Externs bound straight to an LLVM intrinsic, for the standard library.
//!
//! ```hv
//! @llvm_intrinsic_exact("llvm.x86.sse.rcp.ps")
//! extern rcp_ps(x: simd<f32, 4>) simd<f32, 4> with [];
//!
//! @llvm_intrinsic_infer("llvm.sqrt")
//! extern llvm_sqrt<T>(x: T) T with [];
//!
//! @llvm_intrinsic_infer("llvm.vector.reduce.fadd", v)
//! extern llvm_reduce_fadd<T, const N: u64>(acc: T, v: simd<T, N>) T with [];
//! ```
//!
//! An `exact` extern is called by the symbol as written. An `infer` extern is
//! *overloaded*: LLVM spells the concrete types into the name
//! (`llvm.sqrt.f32`, `llvm.sqrt.v4f32`), so the compiler builds one symbol per
//! instance from the types of the listed *sources*, in the order listed. A
//! source is `return` or a parameter name; with none listed, the return type is
//! the only one. The order matters because it is LLVM's, and it differs between
//! intrinsics (`llvm.fptosi.sat.i32.f32` is the return type, then the argument).
//!
//! Nothing checks a signature against LLVM's own definition of the intrinsic: a
//! wrong one is rejected by LLVM, not by Haven. This is a tool for writing the
//! standard library, which wraps each binding in an ordinary typed proc.

use crate::ast::{Attribute, Type};

/// `@llvm_intrinsic_exact("...")`
pub const EXACT_ATTR: &str = "llvm_intrinsic_exact";
/// `@llvm_intrinsic_infer("...", sources...)`
pub const INFER_ATTR: &str = "llvm_intrinsic_infer";
/// The source keyword that names the return type.
pub const RETURN_SOURCE: &str = "return";

/// Where one overload suffix comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Source {
    Return,
    /// The parameter at this position.
    Param(usize),
}

/// How an extern names the intrinsic it calls.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LlvmIntrinsic {
    Exact(String),
    Infer { base: String, sources: Vec<Source> },
}

impl LlvmIntrinsic {
    /// Read the binding off an extern's attributes, or `None` for an ordinary
    /// extern. Assumes the attributes passed `check_extern`; an unresolvable
    /// source is skipped rather than reported again.
    pub fn from_extern<'a>(attrs: &[Attribute<'a>], params: &[(&'a str, Type<'a>)]) -> Option<Self> {
        let attr = attrs.iter()
            .find(|a| a.value.name == EXACT_ATTR || a.value.name == INFER_ATTR)?;
        let values = attr.value.values();
        let name = unquote(values.first()?)?.to_string();
        if attr.value.name == EXACT_ATTR {
            return Some(Self::Exact(name));
        }
        let mut sources: Vec<Source> = values[1..].iter()
            .filter_map(|v| resolve_source(v, params))
            .collect();
        if sources.is_empty() {
            sources.push(Source::Return);
        }
        Some(Self::Infer { base: name, sources })
    }

    /// The symbol a call with these concrete argument and return types reaches.
    /// `Err` is the first source type LLVM has no overload suffix for.
    pub fn symbol<'t>(&self, args: &[Type<'t>], ret: &Type<'t>) -> Result<String, Type<'t>> {
        match self {
            Self::Exact(name) => Ok(name.clone()),
            Self::Infer { base, sources } => {
                let mut symbol = base.clone();
                for source in sources {
                    let ty = match source {
                        Source::Return => ret,
                        Source::Param(i) => &args[*i],
                    };
                    symbol.push('.');
                    symbol.push_str(&overload_suffix(ty).ok_or_else(|| ty.clone())?);
                }
                Ok(symbol)
            }
        }
    }
}

/// LLVM's spelling of an overloaded type inside an intrinsic name: `f32`,
/// `i64`, `v4f32`, `p0`. Signedness is not part of an LLVM type, so `u8` and
/// `i8` are both `i8`. `None` for a type with no such spelling (an aggregate).
pub fn overload_suffix(ty: &Type<'_>) -> Option<String> {
    Some(match ty {
        Type::Bool => "i1".to_string(),
        Type::Int8 | Type::Uint8 => "i8".to_string(),
        Type::Int16 | Type::Uint16 => "i16".to_string(),
        Type::Int32 | Type::Uint32 => "i32".to_string(),
        Type::Int64 | Type::Uint64 => "i64".to_string(),
        Type::Float32 => "f32".to_string(),
        Type::Float64 => "f64".to_string(),
        // every pointer is an opaque `ptr` in address space 0
        Type::Pointer(_) | Type::Str | Type::Function { .. } => "p0".to_string(),
        Type::Simd(elem, n) => format!("v{}{}", n.expect_lit(), overload_suffix(elem)?),
        _ => return None,
    })
}

/// Whether `ty` is fixed enough to spell as a suffix. Inside a generic body a
/// type can still be a parameter (`T`, `simd<f32, N>`); the check is left to
/// the concrete instance.
pub fn is_concrete(ty: &Type<'_>) -> bool {
    match ty {
        Type::Param(_) | Type::Projection { .. } => false,
        Type::Simd(elem, n) => matches!(n, crate::ast::ConstVal::Lit(_)) && is_concrete(elem),
        _ => true,
    }
}

/// The text inside a quoted attribute value, or `None` for a bare word.
pub fn unquote(value: &str) -> Option<&str> {
    value.strip_prefix('"')?.strip_suffix('"')
}

/// Whether `name` can be an intrinsic symbol: `llvm.` and then dot-separated
/// parts of letters, digits and `_`.
pub fn valid_name(name: &str) -> bool {
    name.strip_prefix("llvm.").is_some_and(|rest| {
        !rest.is_empty()
            && rest.split('.').all(|part| !part.is_empty()
                && part.chars().all(|c| c.is_ascii_alphanumeric() || c == '_'))
    })
}

fn resolve_source(word: &str, params: &[(&str, Type<'_>)]) -> Option<Source> {
    if word == RETURN_SOURCE {
        return Some(Source::Return);
    }
    params.iter().position(|(name, _)| *name == word).map(Source::Param)
}

/// Check the intrinsic attributes on one extern against its own signature.
/// Each `Err` is a (headline, note) pair, like `ast::check_attribute`; the
/// syntax of each attribute on its own was checked there already.
pub fn check_extern<'a>(
    attrs: &[Attribute<'a>],
    generic: bool,
    params: &[(&'a str, Type<'a>)],
    return_type: &Type<'a>,
) -> Result<(), (String, String)> {
    let mut found = attrs.iter()
        .filter(|a| a.value.name == EXACT_ATTR || a.value.name == INFER_ATTR);
    let Some(attr) = found.next() else { return Ok(()) };
    if found.next().is_some() {
        return Err((
            format!("`@{EXACT_ATTR}` and `@{INFER_ATTR}` cannot both be written on one extern"),
            "use `exact` for one fixed symbol, `infer` to add type suffixes per instance".to_string()));
    }
    if attr.value.name == EXACT_ATTR {
        if generic {
            return Err((
                format!("`@{EXACT_ATTR}` cannot be written on a generic extern"),
                format!("every instance would call the same symbol with different types; \
                         use `@{INFER_ATTR}` to add the type suffixes")));
        }
        return Ok(());
    }

    let values = attr.value.values();
    let returns_void = matches!(return_type, Type::Void);
    if values.len() == 1 && returns_void {
        return Err((
            "an extern returning `void` must name the overloaded parameters".to_string(),
            format!("the suffix comes from the return type by default; list parameter names, \
                     e.g. `@{INFER_ATTR}({}, {})`", values[0],
                params.first().map_or("x", |(name, _)| *name))));
    }
    for word in &values[1..] {
        match resolve_source(word, params) {
            Some(Source::Return) if returns_void => return Err((
                "`return` names no type: this extern returns `void`".to_string(),
                "list the parameters whose types select the overload".to_string())),
            Some(_) => {}
            None => return Err((
                format!("`{word}` is not a parameter of this extern"),
                format!("an overload source is `{RETURN_SOURCE}` or a parameter name"))),
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::ConstVal;

    fn simd(elem: Type<'static>, n: usize) -> Type<'static> {
        Type::Simd(Box::new(elem), ConstVal::Lit(n))
    }

    #[test]
    fn suffixes() {
        assert_eq!(overload_suffix(&Type::Float32).unwrap(), "f32");
        assert_eq!(overload_suffix(&Type::Float64).unwrap(), "f64");
        assert_eq!(overload_suffix(&Type::Uint8).unwrap(), "i8");
        assert_eq!(overload_suffix(&Type::Int64).unwrap(), "i64");
        assert_eq!(overload_suffix(&Type::Bool).unwrap(), "i1");
        assert_eq!(overload_suffix(&Type::Pointer(Box::new(Type::Float32))).unwrap(), "p0");
        assert_eq!(overload_suffix(&simd(Type::Float32, 4)).unwrap(), "v4f32");
        assert_eq!(overload_suffix(&simd(Type::Uint16, 8)).unwrap(), "v8i16");
        assert!(overload_suffix(&Type::Slice(Box::new(Type::Float32))).is_none());
        assert!(overload_suffix(&Type::Array(Box::new(Type::Float32), ConstVal::Lit(4))).is_none());
    }

    #[test]
    fn symbols_follow_the_listed_sources_in_order() {
        let sat = LlvmIntrinsic::Infer {
            base: "llvm.fptosi.sat".to_string(),
            sources: vec![Source::Return, Source::Param(0)],
        };
        assert_eq!(sat.symbol(&[Type::Float32], &Type::Int32).unwrap(), "llvm.fptosi.sat.i32.f32");

        let reduce = LlvmIntrinsic::Infer {
            base: "llvm.vector.reduce.fadd".to_string(),
            sources: vec![Source::Param(1)],
        };
        assert_eq!(reduce.symbol(&[Type::Float64, simd(Type::Float64, 2)], &Type::Float64).unwrap(),
            "llvm.vector.reduce.fadd.v2f64");

        let exact = LlvmIntrinsic::Exact("llvm.x86.sse.rcp.ps".to_string());
        assert_eq!(exact.symbol(&[], &Type::Void).unwrap(), "llvm.x86.sse.rcp.ps");
    }

    #[test]
    fn names() {
        assert!(valid_name("llvm.sqrt"));
        assert!(valid_name("llvm.x86.sse.rcp.ps"));
        assert!(valid_name("llvm.vector.reduce.fadd"));
        assert!(!valid_name("llvm."));
        assert!(!valid_name("llvm"));
        assert!(!valid_name("sqrt"));
        assert!(!valid_name("llvm..sqrt"));
        assert!(!valid_name("llvm.sqrt."));
        assert!(!valid_name("llvm.sq rt"));
        assert!(!valid_name("llvm.sqrt\""));
    }
}
