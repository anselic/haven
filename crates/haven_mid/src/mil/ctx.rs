use std::collections::HashMap;
use haven_common::ast::*;
use crate::typecheck::EnumDef;
use super::ir::*;

/// A constant integer of the given integer `repr` type (an enum discriminant).
pub(crate) fn int_const(repr: &Type, val: i64) -> Const {
    match repr {
        Type::Int8   => Const::Int8(val as i8),
        Type::Int32  => Const::Int32(val as i32),
        Type::Int64  => Const::Int64(val),
        Type::Uint8  => Const::Uint8(val as u8),
        Type::Uint32 => Const::Uint32(val as u32),
        Type::Uint64 => Const::Uint64(val as u64),
        _ => unreachable!("non-integer enum repr {:?}", repr),
    }
}

/// If `path` is an `Enum::Variant` reference, the discriminant as a typed const.
pub(crate) fn enum_const<'a>(enums: &HashMap<&'a str, EnumDef<'a>>, path: &Path<'a>) -> Option<Const> {
    let (ename, variant) = path.as_variant()?;
    let def = enums.get(ename)?;
    Some(int_const(&def.repr, *def.variants.get(variant)?))
}

/// The discriminant const for a match PATTERN's variant, looked up against
/// `ename` - the scrutinee's own (possibly monomorphized) enum name - rather
/// than the pattern's own textual qualifier. A pattern is always written against
/// the ORIGINAL generic enum name (`Option::Some`), since mono has no type info
/// to rewrite match patterns to a mangled instance (`Option$i32::Some`) the way
/// it rewrites construction call/struct-literal sites; only the variant part
/// (after `::`) of `path` is trustworthy here. See `check_variant_pattern`'s
/// matching base-name relaxation on the typecheck side of this same problem.
pub(crate) fn pattern_variant_const<'a>(enums: &HashMap<&'a str, EnumDef<'a>>, ename: &'a str, path: &Path<'a>) -> Const {
    let variant = path.as_variant().expect("enum pattern validated in typecheck").1;
    let def = &enums[ename];
    int_const(&def.repr, def.variants[variant])
}

/// Rewrites a declared type's `Struct(name)` into `Type::Enum` for any declared
/// enum, recursing through compound types. Struct field types arrive already
/// resolved (via typecheck's `cx.structs`), but declared param/return/local
/// types are the raw parser types, so lowering resolves them here.
pub(crate) fn resolve_enum_ty<'a>(enums: &HashMap<&'a str, EnumDef<'a>>, ty: &Type<'a>) -> Type<'a> {
    match ty {
        // by the time MIL runs, mono has already flattened every generic enum
        // instance to a concrete (args-empty) name, so `args` is always empty here.
        Type::Struct { name, args } if args.is_empty() && enums.contains_key(name) =>
            Type::Enum { name, repr: Box::new(enums[name].repr.clone()), has_payload: enums[name].has_payload, args: Vec::new() },
        Type::Pointer(i) => Type::Pointer(Box::new(resolve_enum_ty(enums, i))),
        Type::Array(i, n) => Type::Array(Box::new(resolve_enum_ty(enums, i)), n.clone()),
        Type::Slice(i)    => Type::Slice(Box::new(resolve_enum_ty(enums, i))),
        Type::Simd(i, n)  => Type::Simd(Box::new(resolve_enum_ty(enums, i)), n.clone()),
        Type::Function { params, return_type } => Type::Function {
            params: params.iter().map(|p| resolve_enum_ty(enums, p)).collect(),
            return_type: Box::new(resolve_enum_ty(enums, return_type)),
        },
        other => other.clone(),
    }
}

/// The synthetic struct backing an aggregate value passed/stored by pointer: a
/// real struct, or a data-carrying enum (whose `{ tag, payload }` aggregate is
/// registered as a struct of the same name). Every "this is an aggregate, route
/// it by pointer" site funnels through here so a data enum is never mistaken for
/// a scalar. A field-less enum returns `None` - it is a bare scalar.
pub fn aggregate_struct_name<'a>(ty: &Type<'a>) -> Option<&'a str> {
    match ty {
        Type::Struct { name, .. } => Some(name),
        Type::Enum { name, has_payload: true, .. } => Some(name),
        _ => None,
    }
}

#[derive(Clone, Debug)]
pub struct LoopTargets {
    pub continue_block: BlockId, // where to jump for `continue`
    pub break_block: BlockId,    // where to jump for `break`
}

#[derive(Clone, Debug)]
pub struct LowerCtx<'a> {
    pub reg_counter: usize,
    pub block_counter: usize,
    pub current_block: BlockId,
    pub blocks: Vec<BasicBlock<'a>>,
    pub loop_stack: Vec<LoopTargets>,

    /// Storage slot per param/local, keyed by resolved binding identity (not by
    /// name) so shadowed same-named locals get distinct slots. Cleared per fn.
    pub env: HashMap<Binding<'a>, (Register, Type<'a>)>,
    /// Module-level globals in scope: final emitted name -> type. Referenced as
    /// bare vars, distinct from `env` (locals/params), which shadow these.
    pub globals: HashMap<&'a str, Type<'a>>,
    pub structs: HashMap<&'a str, Vec<(&'a str, Type<'a>)>>, // from typecheck
    /// Declared enums (from typecheck): resolves an `Enum::Variant` reference to
    /// its discriminant constant during lowering.
    pub enums: HashMap<&'a str, EnumDef<'a>>,
    pub node_types: HashMap<usize, Type<'a>>, // from typecheck
    /// Name resolution from typecheck: `Var` node id -> its param/local binding.
    /// Absent for globals/functions, which resolve via `globals` / direct calls.
    pub resolved: HashMap<usize, Binding<'a>>, // from typecheck
    /// Receiver method calls from typecheck (`recv.method(...)`), keyed by the
    /// `Call` node id. Lowered to a direct call with the adjusted receiver
    /// prepended to the args.
    pub method_calls: HashMap<usize, crate::typecheck::MethodCall<'a>>, // from typecheck
    pub current_return_type: Type<'a>,        // return type of the function being lowered
    pub sret_param: Option<Register>,         // out-pointer slot, if the current fn returns a struct
    /// When set, the next struct/array literal lowered fills this pre-allocated
    /// slot instead of allocating a fresh one. Used to hoist a loop-local's slot
    /// to the entry block (see `collect_locals`) so it isn't re-alloca'd every
    /// iteration. The literal lowering `take()`s it, so it applies to exactly the
    /// outermost literal and never leaks into nested field/element literals.
    pub store_target: Option<Register>,
    pub strings: Vec<Vec<u8>>,                // interned string-literal blobs (-> @.str.N)
}

impl<'a> LowerCtx<'a> {
    pub fn fresh_reg(&mut self) -> Register {
        let r = Register(self.reg_counter);
        self.reg_counter += 1;
        r
    }

    pub fn fresh_block(&mut self) -> BlockId {
        let b = BlockId(self.block_counter);
        self.block_counter += 1;
        self.blocks.push(BasicBlock {
            id: b,
            instructions: vec![],
            terminator: None,
        });
        b
    }

    /// Intern a string literal's raw source text, resolving escape sequences and
    /// appending a NUL terminator, and return its index into `strings`. `str` is
    /// a raw `*const u8` C string, so the stored blob carries the trailing `\0`;
    /// callers get a bare pointer to it. Identical blobs are deduplicated so
    /// repeated literals share one global.
    pub fn intern_string(&mut self, raw: &str) -> usize {
        let mut bytes = resolve_escapes(raw);
        bytes.push(0); // NUL terminator, so the raw pointer is a valid C string
        if let Some(i) = self.strings.iter().position(|b| *b == bytes) {
            return i;
        }
        let idx = self.strings.len();
        self.strings.push(bytes);
        idx
    }

    pub fn emit(&mut self, inst: Inst<'a>) {
        let block = self.blocks.iter_mut().find(|b| b.id == self.current_block).unwrap();
        block.instructions.push(inst);
    }

    pub fn terminate(&mut self, term: Terminator<'a>) {
        let block = self.blocks.iter_mut().find(|b| b.id == self.current_block).unwrap();
        block.terminator = Some(term);
    }
}

/// Resolve the escape sequences in a string literal's raw source text into the
/// actual bytes. Recognizes `\n \t \r \0 \\ \"`; an unknown escape `\x` keeps
/// the char `x` verbatim. The lexer guarantees a backslash is always followed
/// by at least one char, so a trailing lone backslash cannot occur.
fn resolve_escapes(raw: &str) -> Vec<u8> {
    let mut out = Vec::with_capacity(raw.len());
    let mut chars = raw.chars();
    while let Some(c) = chars.next() {
        if c != '\\' {
            let mut buf = [0u8; 4];
            out.extend_from_slice(c.encode_utf8(&mut buf).as_bytes());
            continue;
        }
        match chars.next() {
            Some('n') => out.push(b'\n'),
            Some('t') => out.push(b'\t'),
            Some('r') => out.push(b'\r'),
            Some('0') => out.push(0),
            Some('\\') => out.push(b'\\'),
            Some('"') => out.push(b'"'),
            Some(other) => {
                let mut buf = [0u8; 4];
                out.extend_from_slice(other.encode_utf8(&mut buf).as_bytes());
            }
            None => {} // should be unreachable per lexer invariant
        }
    }
    out
}

/// Materialize an array->slice coercion: build a { ptr, i32 N } fat pointer from
/// an array value. Returns the original value unchanged if no coercion applies.
pub(crate) fn coerce<'a>(cx: &mut LowerCtx<'a>, val: Value, from_ty: &Type<'a>, to_ty: &Type<'a>) -> Value {
    match (from_ty, to_ty) {
        (Type::Array(inner, len), Type::Slice(_)) => {
            let fat0 = cx.fresh_reg();
            cx.emit(Inst::InsertValue {
                dst: fat0,
                elem: Value::Const(Const::Undef),
                ty: Type::Pointer(Box::new(*inner.clone())),
                val,
                index: 0,
            });
            let fat1 = cx.fresh_reg();
            cx.emit(Inst::InsertValue {
                dst: fat1,
                elem: Value::Reg(fat0),
                ty: Type::Int32,
                val: Value::Const(Const::Int32(len.expect_lit() as i32)),
                index: 1,
            });
            Value::Reg(fat1)
        }
        _ => val,
    }
}

/// Pulls a concrete type from a turbofish type argument. Type params are skipped
/// in lowering (generic functions aren't monomorphized), so these are concrete.
pub(crate) fn ta_type<'a>(type_args: &[GenericArg<'a>], i: usize) -> Type<'a> {
    match &type_args[i] {
        GenericArg::Type(t) => t.clone(),
        _ => unreachable!("expected a type argument at position {i}"),
    }
}

/// Pulls a const from a turbofish const argument.
pub(crate) fn ta_const(type_args: &[GenericArg<'_>], i: usize) -> usize {
    match &type_args[i] {
        GenericArg::Const(cv) => cv.expect_lit(),
        _ => unreachable!("expected a const argument at position {i}"),
    }
}
