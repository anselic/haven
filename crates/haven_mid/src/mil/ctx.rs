use std::collections::HashMap;
use haven_common::ast::*;
use haven_common::defs::DefId;
use haven_common::layout::TypeTable;
use crate::typecheck::EnumDef;
use super::ir::*;

/// A constant integer of the given integer `repr` type (an enum discriminant).
pub(crate) fn int_const(repr: &Type, val: i64) -> Const {
    match repr {
        Type::Int8   => Const::Int8(val as i8),
        Type::Int16  => Const::Int16(val as i16),
        Type::Int32  => Const::Int32(val as i32),
        Type::Int64  => Const::Int64(val),
        Type::Uint8  => Const::Uint8(val as u8),
        Type::Uint16 => Const::Uint16(val as u16),
        Type::Uint32 => Const::Uint32(val as u32),
        Type::Uint64 => Const::Uint64(val as u64),
        _ => unreachable!("non-integer enum repr {:?}", repr),
    }
}

/// The constant a width-less literal (`ExprNode::IntLit`/`FloatLit`) lowers to,
/// at the type the checker settled on for it - which is why `ty` comes from
/// `node_types` rather than from the node, the way it does for a suffixed
/// literal. The narrowing casts here cannot lose anything: `check_expr` already
/// verified the value fits the target's range.
pub(crate) fn lit_const(ty: &Type<'_>, node: &ExprNode<'_>) -> Const {
    match node {
        ExprNode::IntLit(v) => match ty {
            Type::Int8    => Const::Int8(*v as i8),
            Type::Int16   => Const::Int16(*v as i16),
            Type::Int32   => Const::Int32(*v as i32),
            Type::Int64   => Const::Int64(*v as i64),
            Type::Uint8   => Const::Uint8(*v as u8),
            Type::Uint16  => Const::Uint16(*v as u16),
            Type::Uint32  => Const::Uint32(*v as u32),
            Type::Uint64  => Const::Uint64(*v as u64),
            // `let x: f64 = 1;` - an integer literal in a float's place.
            Type::Float32 => Const::Float32(*v as f32),
            Type::Float64 => Const::Float64(*v as f64),
            other => unreachable!("integer literal typed as {:?}", other),
        },
        ExprNode::FloatLit(f) => match ty {
            Type::Float32 => Const::Float32(*f as f32),
            Type::Float64 => Const::Float64(*f),
            other => unreachable!("float literal typed as {:?}", other),
        },
        other => unreachable!("lit_const on a non-literal node: {}", other),
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct ConstEvalError {
    pub span: Span,
    pub message: &'static str,
}

fn const_eval_error(span: Span, message: &'static str) -> ConstEvalError {
    ConstEvalError { span, message }
}

/// Fold scalar literal arithmetic to the exact constant emitted for a global.
/// Width-less leaves use their inferred type. Integer operations are checked so
/// a constant cannot silently produce a value that its literal spelling would
/// have been forbidden to represent.
pub(crate) fn eval_const_scalar<'a>(
    expr: &Expr<'a>,
    node_types: &HashMap<usize, Type<'a>>,
) -> Result<Const, ConstEvalError> {
    let literal = match &expr.value {
        ExprNode::Bool(b)    => Some(Const::Bool(*b)),
        ExprNode::Int8(n)    => Some(Const::Int8(*n)),
        ExprNode::Int16(n)   => Some(Const::Int16(*n)),
        ExprNode::Int32(n)   => Some(Const::Int32(*n)),
        ExprNode::Int64(n)   => Some(Const::Int64(*n)),
        ExprNode::Uint8(n)   => Some(Const::Uint8(*n)),
        ExprNode::Uint16(n)  => Some(Const::Uint16(*n)),
        ExprNode::Uint32(n)  => Some(Const::Uint32(*n)),
        ExprNode::Uint64(n)  => Some(Const::Uint64(*n)),
        ExprNode::Float32(f) => Some(Const::Float32(*f)),
        ExprNode::Float64(f) => Some(Const::Float64(*f)),
        ExprNode::IntLit(_) | ExprNode::FloatLit(_) =>
            Some(lit_const(&node_types[&expr.id], &expr.value)),
        _ => None,
    };
    if let Some(value) = literal { return Ok(value); }

    match &expr.value {
        ExprNode::Unary { op: UnaryOp::Neg, operand } => {
            // An unsuffixed negative literal is one context-typed unit. Fold the
            // sign before narrowing so `-128: i8` remains valid even though its
            // positive magnitude is not an `i8` value by itself.
            match &operand.value {
                ExprNode::IntLit(v) => return Ok(lit_const(
                    &node_types[&operand.id], &ExprNode::IntLit(-*v))),
                ExprNode::FloatLit(v) => return Ok(lit_const(
                    &node_types[&operand.id], &ExprNode::FloatLit(-*v))),
                _ => {}
            }

            Ok(match eval_const_scalar(operand, node_types)? {
                Const::Int8(n)    => Const::Int8(n.checked_neg().ok_or_else(||
                    const_eval_error(expr.span, "overflow in constant expression during negation"))?),
                Const::Int16(n)   => Const::Int16(n.checked_neg().ok_or_else(||
                    const_eval_error(expr.span, "overflow in constant expression during negation"))?),
                Const::Int32(n)   => Const::Int32(n.checked_neg().ok_or_else(||
                    const_eval_error(expr.span, "overflow in constant expression during negation"))?),
                Const::Int64(n)   => Const::Int64(n.checked_neg().ok_or_else(||
                    const_eval_error(expr.span, "overflow in constant expression during negation"))?),
                Const::Uint8(n)   => Const::Uint8(n.checked_neg().ok_or_else(||
                    const_eval_error(expr.span, "overflow in constant expression during negation"))?),
                Const::Uint16(n)  => Const::Uint16(n.checked_neg().ok_or_else(||
                    const_eval_error(expr.span, "overflow in constant expression during negation"))?),
                Const::Uint32(n)  => Const::Uint32(n.checked_neg().ok_or_else(||
                    const_eval_error(expr.span, "overflow in constant expression during negation"))?),
                Const::Uint64(n)  => Const::Uint64(n.checked_neg().ok_or_else(||
                    const_eval_error(expr.span, "overflow in constant expression during negation"))?),
                Const::Float32(f) => Const::Float32(-f),
                Const::Float64(f) => Const::Float64(-f),
                _ => return Err(const_eval_error(
                    expr.span, "constant negation needs a numeric operand")),
            })
        }
        ExprNode::Binary { op, left, right } => {
            let lhs = eval_const_scalar(left, node_types)?;
            let rhs = eval_const_scalar(right, node_types)?;
            eval_const_binary(expr.span, *op, lhs, rhs)
        }
        _ => Err(const_eval_error(expr.span, "expression is not a scalar constant")),
    }
}

fn eval_const_binary(
    span: Span,
    op: BinaryOp,
    lhs: Const,
    rhs: Const,
) -> Result<Const, ConstEvalError> {
    use BinaryOp::*;

    macro_rules! integer {
        ($variant:ident, $a:expr, $b:expr) => {{
            let value = match op {
                Add => $a.checked_add($b),
                Sub => $a.checked_sub($b),
                Mul => $a.checked_mul($b),
                Div if $b == 0 => return Err(const_eval_error(
                    span, "division by zero in constant expression")),
                Div => $a.checked_div($b),
                Mod if $b == 0 => return Err(const_eval_error(
                    span, "remainder by zero in constant expression")),
                Mod => $a.checked_rem($b),
                _ => return Err(const_eval_error(
                    span, "operator is not supported in a constant arithmetic expression")),
            }.ok_or_else(|| const_eval_error(span, match op {
                Add => "overflow in constant expression during addition",
                Sub => "overflow in constant expression during subtraction",
                Mul => "overflow in constant expression during multiplication",
                Div => "overflow in constant expression during division",
                Mod => "overflow in constant expression during remainder",
                _ => unreachable!(),
            }))?;
            Const::$variant(value)
        }};
    }
    macro_rules! float {
        ($variant:ident, $a:expr, $b:expr) => {{
            let value = match op {
                Add => $a + $b,
                Sub => $a - $b,
                Mul => $a * $b,
                Div => $a / $b,
                Mod => $a % $b,
                _ => return Err(const_eval_error(
                    span, "operator is not supported in a constant arithmetic expression")),
            };
            Const::$variant(value)
        }};
    }

    Ok(match (lhs, rhs) {
        (Const::Int8(a), Const::Int8(b))       => integer!(Int8, a, b),
        (Const::Int16(a), Const::Int16(b))     => integer!(Int16, a, b),
        (Const::Int32(a), Const::Int32(b))     => integer!(Int32, a, b),
        (Const::Int64(a), Const::Int64(b))     => integer!(Int64, a, b),
        (Const::Uint8(a), Const::Uint8(b))     => integer!(Uint8, a, b),
        (Const::Uint16(a), Const::Uint16(b))   => integer!(Uint16, a, b),
        (Const::Uint32(a), Const::Uint32(b))   => integer!(Uint32, a, b),
        (Const::Uint64(a), Const::Uint64(b))   => integer!(Uint64, a, b),
        (Const::Float32(a), Const::Float32(b)) => float!(Float32, a, b),
        (Const::Float64(a), Const::Float64(b)) => float!(Float64, a, b),
        _ => return Err(const_eval_error(
            span, "constant arithmetic operands have different types")),
    })
}

/// If `r` is an `Enum::Variant` reference, the discriminant as a typed const.
pub(crate) fn enum_const<'a>(enums: &HashMap<DefId, EnumDef<'a>>, r: &NameRef<'a>) -> Option<Const> {
    let def = enums.get(&r.def)?;
    Some(int_const(&def.repr, *def.variants.get(r.variant())?))
}

/// The discriminant const for a match PATTERN's variant, looked up against
/// `ename` - the scrutinee's own (possibly monomorphized) enum name - rather
/// than the pattern's own textual qualifier. A pattern is always written against
/// the ORIGINAL generic enum name (`Option::Some`), since mono has no type info
/// to rewrite match patterns to a mangled instance (`Option$i32::Some`) the way
/// it rewrites construction call/struct-literal sites; only the variant part
/// (after `::`) of `path` is trustworthy here. See `check_variant_pattern`'s
/// matching base-name relaxation on the typecheck side of this same problem.
pub(crate) fn pattern_variant_const<'a>(enums: &HashMap<DefId, EnumDef<'a>>, ename: DefId, r: &NameRef<'a>) -> Const {
    let def = &enums[&ename];
    int_const(&def.repr, def.variants[r.variant()])
}

/// The aggregate backing a value passed/stored by pointer: a real struct, or a
/// data-carrying enum (whose `{ $tag, $payload }` aggregate is registered as a
/// struct under the same identity). Every "this is an aggregate, route it by
/// pointer" site funnels through here so a data enum is never mistaken for a
/// scalar. A field-less enum returns `None` - it is a bare scalar.
///
/// This used to be answerable from the type alone, when `Type::Enum` carried a
/// `has_payload` copy. Now it is a table lookup: the fact belongs to the
/// definition, not to every mention of it.
pub fn aggregate_def<'a>(ty: &Type<'a>, enums: &HashMap<DefId, EnumDef<'a>>) -> Option<DefId> {
    match ty {
        Type::Named { def, .. } => match enums.get(def) {
            Some(e) if !e.has_payload => None,
            _ => Some(*def),
        },
        _ => None,
    }
}

/// Whether a value of this type lives in memory and is handed around by
/// pointer: a struct, a data-carrying enum, or a fixed-size array.
///
/// [`aggregate_def`] answers the same question but can only speak for types that
/// *have* a definition, so it says `None` for `[T; N]` - which is structural,
/// but inline storage all the same. Ask this wherever the question is "is this
/// an aggregate", and `aggregate_def` only where a `DefId` is actually needed:
/// to walk a struct's fields, or to name its LLVM type.
pub fn is_aggregate_ty<'a>(ty: &Type<'a>, enums: &HashMap<DefId, EnumDef<'a>>) -> bool {
    matches!(ty, Type::Array(..) | Type::Tuple(..)) || aggregate_def(ty, enums).is_some()
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
    pub types: TypeTable<'a>, // from typecheck
    /// Emitted symbol per definition, for the aggregates this module declares.
    pub symbols: HashMap<DefId, &'a str>,
    /// `(enum, variant) -> payload struct`, from typecheck.
    pub payloads: HashMap<(DefId, &'a str), DefId>,
    /// Declared enums (from typecheck): resolves an `Enum::Variant` reference to
    /// its discriminant constant during lowering.
    pub enums: HashMap<DefId, EnumDef<'a>>,
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

    /// Whether the current block already has a terminator. A diverging expression
    /// (`abort(...)`) terminates its block with `unreachable` mid-statement; the
    /// statement lowerer checks this afterwards to skip the now-dead tail (e.g.
    /// the `return`'s own terminator, or an aggregate copy of a value that was
    /// never produced).
    pub fn is_terminated(&self) -> bool {
        self.blocks.iter().find(|b| b.id == self.current_block).unwrap().terminator.is_some()
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
