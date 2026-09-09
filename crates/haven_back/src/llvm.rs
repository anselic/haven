
use haven_common::ast::*;
use haven_mid::mil::*;
use crate::abi::{self, Abi, Reg, UnionTable};
use std::borrow::Cow;
use std::collections::HashMap;

use haven_common::defs::DefId;
use crate::layout::{self, TypeTable, TypeInfo, EnumRepr};

/// Quote a symbol when it is not a valid bare LLVM identifier.
/// Package names may begin with a digit, which LLVM would otherwise parse as an
/// unnamed value number.
fn ir_symbol(name: &str) -> Cow<'_, str> {
    fn bare(c: char) -> bool {
        c.is_ascii_alphanumeric() || matches!(c, '-' | '$' | '.' | '_')
    }
    let mut chars = name.chars();
    let ok = match chars.next() {
        Some(c) => !c.is_ascii_digit() && bare(c) && chars.all(bare),
        // an empty symbol is not something we expect to emit, but `%` alone is a
        // parse error where `%""` is merely odd.
        None => false,
    };
    if ok {
        return Cow::Borrowed(name);
    }
    // Inside quotes only `"` and `\` must be escaped, as `\xx` hex pairs. Other
    // bytes - including multi-byte UTF-8 from a filename - pass through as-is,
    // which is why this iterates `chars` rather than `bytes`.
    let mut out = String::with_capacity(name.len() + 2);
    out.push('"');
    for ch in name.chars() {
        match ch {
            '"' => out.push_str("\\22"),
            '\\' => out.push_str("\\5C"),
            _ => out.push(ch),
        }
    }
    out.push('"');
    Cow::Owned(out)
}

fn emit_value(val: Value) -> String {
    match val {
        Value::Const(Const::Float32(f)) => format!("0x{:016X}", (f as f64).to_bits()),
        Value::Const(Const::Float64(f)) => format!("0x{:016X}", f.to_bits()),
        Value::Const(Const::Bool(b)) => if b { "1".to_string() } else { "0".to_string() },
        _ => format!("{}", val),
    }
}

fn emit_type<'a>(ty: &Type<'a>, types: &TypeTable<'a>, symbols: &HashMap<DefId, &'a str>) -> String {
    use Type::*;

    match ty {
        Path { path, .. } => Type::unresolved(path),
        Void => "void".to_string(),
        Bool => "i1".to_string(),
        Int8 => "i8".to_string(),
        Int16 => "i16".to_string(),
        Int32 => "i32".to_string(),
        Int64 => "i64".to_string(),
        // LLVM just use one width for unsigned integers and rely on the
        // instructions (or us) to interpret them correctly
        Uint8 => "i8".to_string(),
        Uint16 => "i16".to_string(),
        Uint32 => "i32".to_string(),
        Uint64 => "i64".to_string(),
        Float32 => "float".to_string(),
        Float64 => "double".to_string(),
        Pointer(_) => "ptr".to_string(),

        // an array is contiguous inline storage - `[N x %Name]`, what
        // `size_of`/`alloc` lay out and `Index` strides by - never `[N x ptr]`
        // handles, wherever it appears. so the element goes through
        // `emit_field_type`, not a recursion here that would render a struct as
        // the `ptr` a struct value is.
        //
        // `[N x ptr]` is not a cosmetic mismatch: a whole-array `load`/`store`
        // then copies `N*8` bytes between inline objects, overrunning both when
        // the element is under 8 bytes and dropping the tail when it is over.
        Array(t, n) => format!("[{} x {}]", n.expect_lit(), emit_field_type(t, types, symbols)),
        Slice(_) => "{ ptr, i32 }".to_string(), // struct { ptr, len }
        // `str` is a raw NUL-terminated `*const u8` (a C string), so a bare `ptr`
        Str => "ptr".to_string(),
        // a field-less enum lowers to its integer discriminant repr; a data
        // enum, like a struct, is an aggregate referenced by opaque pointer.
        // Struct values are always referenced via pointer in our codegen (see
        // `lower_function` and `lower_expr` for `ExprNode::Struct`); the named
        // LLVM type `%Name` is only used by `AllocaStruct` and `FieldPtr`, which
        // emit it directly rather than going through here.
        Named { def, .. } => match types.get(def) {
            Some(TypeInfo { enum_: Some(EnumRepr { repr, has_payload: false }), .. }) =>
                emit_type(repr, types, symbols),
            _ => "ptr".to_string(),
        },
        // <size x element_type>
        Simd(ty, size) => format!("<{} x {}>", size.expect_lit(), emit_field_type(ty, types, symbols)),
        // a function pointer is a plain opaque pointer
        Function { .. } => "ptr".to_string(),
        // generic functions are skipped during MIL lowering, so a type param
        // should never reach codegen.
        Param(name) => unreachable!("generic type parameter `{name}` survived to LLVM codegen"),
        // `abort()` is the only source of `!`, and it lowers to a call + an
        // `unreachable` terminator, never to a value or a slot - so no live value
        // ever has this type at codegen.
        Never => unreachable!("the bottom type `!` reached LLVM codegen - it has no values"),
    }
}

/// Layout type for a field *inside* a struct definition. Differs from
/// `emit_type` only for structs because a struct value is normally a `ptr`
/// handle, but as a field it is inlined as the named type `%Name`.
fn emit_field_type<'a>(ty: &Type<'a>, types: &TypeTable<'a>, symbols: &HashMap<DefId, &'a str>) -> String {
    match ty {
        // an aggregate inlined as a field is its named type `%Name`; a
        // field-less enum is just its scalar repr (falls through to emit_type).
        Type::Named { def, .. } if !matches!(
            types.get(def),
            Some(TypeInfo { enum_: Some(EnumRepr { has_payload: false, .. }), .. }),
        ) => format!("%{}", ir_symbol(symbols[def])),
        // sequences agree with `emit_type` (which defers to this for its
        // element), so they could equally fall through; kept explicit because
        // this is the definition the other one refers to.
        Type::Array(t, n) => format!("[{} x {}]", n.expect_lit(), emit_field_type(t, types, symbols)),
        Type::Simd(t, n) => format!("<{} x {}>", n.expect_lit(), emit_field_type(t, types, symbols)),
        _ => emit_type(ty, types, symbols),
    }
}

#[derive(Debug, Clone, Copy)]
enum FastMathFlags {
    None,
    Fast,
    Reassoc,
    NNaN,
    NInf,
    NSZ,
    Arcp,
    Contract,
}

impl FastMathFlags {
    fn from_str(s: &str) -> Option<Self> {
        match s {
            "fast"     => Some(FastMathFlags::Fast),
            "reassoc"  => Some(FastMathFlags::Reassoc),
            "nnan"     => Some(FastMathFlags::NNaN),
            "ninf"     => Some(FastMathFlags::NInf),
            "nsz"      => Some(FastMathFlags::NSZ),
            "arcp"     => Some(FastMathFlags::Arcp),
            "contract" => Some(FastMathFlags::Contract),
            _ => None,
        }
    }

    fn to_str(&self) -> &'static str {
        match self {
            FastMathFlags::None     => "",
            FastMathFlags::Fast     => "fast",
            FastMathFlags::Reassoc  => "reassoc",
            FastMathFlags::NNaN     => "nnan",
            FastMathFlags::NInf     => "ninf",
            FastMathFlags::NSZ      => "nsz",
            FastMathFlags::Arcp     => "arcp",
            FastMathFlags::Contract => "contract",
        }
    }
}

struct EmitCtx<'a> {
    buf: String,
    current_fast_math_flags: FastMathFlags,
    /// Named-type layouts, for on-the-fly SysV ABI classification of by-value
    /// aggregates.
    types: TypeTable<'a>,
    /// Emitted symbol per definition. This is the only thing codegen needs from
    /// the definition table: everything else about a type is in `types`.
    symbols: HashMap<DefId, &'a str>,
    /// Data-enum aggregate -> its variant payload structs, so a by-value enum
    /// crossing an FFI boundary classifies its payload as a real union (see
    /// `abi::classify_into`) instead of raw bytes.
    unions: UnionTable,
    /// Fresh-name counter for ABI coercion temporaries (`%abi.N`), kept separate
    /// from MIL's `%tN` registers so the two never collide.
    abi_ctr: usize,
    /// When the function currently being emitted returns a struct in registers
    /// (SysV Direct class), this holds the local slot the body writes the result
    /// into, plus its coerced register pieces - so `ret void` becomes a
    /// load-and-return of the coerced value. `None` for void/scalar/sret returns.
    sret_direct: Option<(Register, Vec<Reg>)>,
}

impl<'a> EmitCtx<'a> {
    fn emit(&mut self, str: String) {
        self.buf.push_str(&str);
    }

    /// A fresh `%abi.N` temporary name for coercion glue.
    fn abi_tmp(&mut self) -> String {
        let name = format!("%abi.{}", self.abi_ctr);
        self.abi_ctr += 1;
        name
    }

    /// ABI classification of a by-value aggregate: a struct, a data enum's
    /// tag+payload aggregate, or a fixed-size array.
    fn abi_of(&self, ty: &Type<'a>) -> Abi {
        abi::classify(ty, &self.types, &self.unions)
    }

    /// Byte alignment of an aggregate, for `byval`/`sret` attributes.
    fn abi_align(&self, ty: &Type<'a>) -> usize {
        layout::align_of(ty, &self.types)
    }

    /// Whether a value of `ty` crosses a call boundary as an aggregate - held in
    /// memory and classified into registers or `byval`/`sret` - rather than as a
    /// plain scalar.
    ///
    /// This is a question about the *type*, not about a definition: `[T; N]` is
    /// inline storage exactly as a struct is, and has no `DefId` to ask about.
    /// Keying it on one is what left arrays out of the ABI path entirely, so a
    /// by-value array parameter reached LLVM as a raw `[N x T]` value while the
    /// caller passed a pointer to it, and the verifier rejected the mismatch.
    fn is_aggregate_ty(&self, ty: &Type<'a>) -> bool {
        match ty {
            Type::Array(..) => true,
            Type::Named { def, .. } => !matches!(
                self.types.get(def),
                Some(TypeInfo { enum_: Some(EnumRepr { has_payload: false, .. }), .. }),
            ),
            _ => false,
        }
    }

    /// The LLVM type named inside a `byval(..)`/`sret(..)` attribute: the
    /// aggregate's storage layout, `%Name` for a struct and `[N x %Elem]` for an
    /// array. LLVM accepts any type there, not only a named one.
    fn abi_storage_ty(&self, ty: &Type<'a>) -> String {
        emit_field_type(ty, &self.types, &self.symbols)
    }

    /// The symbol an aggregate is declared under: its LLVM `%Name`, quoted if the
    /// mangled name is not a bare LLVM identifier. Every `%` reference to a named
    /// type goes through here, so it stays consistent with the `= type` line.
    fn sym(&self, def: DefId) -> Cow<'a, str> {
        ir_symbol(self.symbols[&def])
    }
}

/// The LLVM type a Direct-class struct is passed/returned as: a single register
/// type for one eightbyte, or an anonymous `{ .. }` aggregate for two.
fn coerced_aggregate_ty(regs: &[Reg]) -> String {
    if regs.len() == 1 {
        regs[0].to_llvm().to_string()
    } else {
        let body = regs.iter().map(|r| r.to_llvm()).collect::<Vec<_>>().join(", ");
        format!("{{ {body} }}")
    }
}

macro_rules! emitln {
    ($cx:expr, $($arg:tt)*) => {{
        $cx.emit(format!($($arg)*));
        $cx.emit("\n".to_string())
    }};
}

fn emit_inst<'a>(cx: &mut EmitCtx<'a>, inst: Inst<'a>) {
    use Inst::*;

    match inst {
        Comment(s) => emitln!(cx, "    ; {}", s),
        // Negation. Floats (scalar or SIMD lanes) use `fneg`; integers, signed or
        // unsigned, use `0 - x` (a vector zero is `zeroinitializer`). Signedness
        // is irrelevant: two's-complement negation is the same bit pattern.
        Unary { dst, op: UnaryOp::Neg, val, ty } => {
            let inner = match &ty {
                Type::Simd(inner, _) => inner.as_ref(),
                _ => &ty,
            };
            let ty_str = emit_type(&ty, &cx.types, &cx.symbols);
            if matches!(inner, Type::Float32 | Type::Float64) {
                emitln!(cx, "    {dst} = fneg {ty_str} {}", emit_value(val));
            } else {
                let zero = if matches!(ty, Type::Simd(_, _)) { "zeroinitializer" } else { "0" };
                emitln!(cx, "    {dst} = sub {ty_str} {zero}, {}", emit_value(val));
            }
        }
        Unary { dst, op, val, ty } =>
            emitln!(cx, "    {dst} = {}, {}", match op {
                UnaryOp::Neg => unreachable!("handled above"),
                UnaryOp::Not if ty == Type::Int32 => "xor i32 -1",
                UnaryOp::Not => "xor i1 1", // logical not
                // this should've be mapped to different instructions (mil.rs)
                UnaryOp::Deref | UnaryOp::AddrOf => unreachable!(),
            }, emit_value(val)),
        Binary { dst, op, lhs, rhs, ty } => {
            use BinaryOp::*;

            // extract SIMD inner type for instruction selection
            let inner = match &ty {
                Type::Simd(inner, _) => inner.as_ref(),
                _ => &ty,
            };

            let is_float = matches!(inner, Type::Float32 | Type::Float64);
            let is_signed_int = matches!(inner, Type::Int8 | Type::Int16 | Type::Int32 | Type::Int64);
            let is_unsigned_int = matches!(inner, Type::Uint8 | Type::Uint16 | Type::Uint32 | Type::Uint64);

            let flags_str = if *inner == Type::Float32 || *inner == Type::Float64 {
                format!(" {}", cx.current_fast_math_flags.to_str())
            } else {
                String::new()
            };

            emitln!(cx, "    {dst} = {}{flags_str} {} {}, {}", match op {
                Add if is_signed_int || is_unsigned_int => "add",
                Add if is_float => "fadd",
                Add => unreachable!("{}", ty),

                Sub if is_signed_int || is_unsigned_int => "sub",
                Sub if is_float => "fsub",
                Sub => unreachable!("{}", ty),

                Mul if is_signed_int || is_unsigned_int => "mul",
                Mul if is_float => "fmul",
                Mul => unreachable!("{}", ty),

                Div if is_signed_int => "sdiv",
                Div if is_unsigned_int => "udiv",
                Div if is_float => "fdiv",
                Div => unreachable!("{}", ty),

                Mod if is_signed_int => "srem",
                Mod if is_unsigned_int => "urem",
                Mod if is_float => "frem",
                Mod => unreachable!("{}", ty),

                Eq if is_float => "fcmp oeq",
                Ne if is_float => "fcmp one",
                Lt if is_float => "fcmp olt",
                Le if is_float => "fcmp ole",
                Gt if is_float => "fcmp ogt",
                Ge if is_float => "fcmp oge",
                Eq => "icmp eq",
                Ne => "icmp ne",

                Lt if is_signed_int => "icmp slt",
                Le if is_signed_int => "icmp sle",
                Gt if is_signed_int => "icmp sgt",
                Ge if is_signed_int => "icmp sge",

                Lt if is_unsigned_int => "icmp ult",
                Le if is_unsigned_int => "icmp ule",
                Gt if is_unsigned_int => "icmp ugt",
                Ge if is_unsigned_int => "icmp uge",

                Lt | Le | Gt | Ge => unreachable!("{}", ty),

                // logical and/or reach here only via the non-short-circuit
                // fallback path; on i1 the bitwise ops are the logical ops.
                And => "and",
                Or => "or",

                BitAnd => "and",
                BitOr => "or",
                BitXor => "xor",
                Shl => "shl",
                // arithmetic shift for signed, logical for unsigned
                Shr if is_signed_int => "ashr",
                Shr if is_unsigned_int => "lshr",
                Shr => unreachable!("{}", ty),
            }, emit_type(&ty, &cx.types, &cx.symbols), emit_value(lhs), emit_value(rhs));
        }
        Call { dst, callee, args, return_type, sret } => {
            // %result = call <return_type> <callee>(<arg_type> <arg_val>, ...)
            // where <callee> is either a function symbol @name or a fn-pointer %reg
            let callee_str = match &callee {
                Callee::Direct(name) => format!("@{}", ir_symbol(name)),
                Callee::Indirect(val) => emit_value(val.clone()),
            };
            // Expand arguments, coercing Direct-class by-value structs into their
            // eightbyte registers (the loads land before the call, above).
            let mut arg_frags: Vec<String> = Vec::new();
            for (val, ty) in args {
                match &ty {
                    _ if cx.is_aggregate_ty(&ty) => match cx.abi_of(&ty) {
                        Abi::Direct(regs) => {
                            let ptr = emit_value(val);
                            for (t, v) in emit_struct_to_regs(cx, &ptr, &regs) {
                                arg_frags.push(format!("{t} {v}"));
                            }
                        }
                        // Memory aggregate: pass a `byval` pointer. The backend
                        // copies our storage into the callee's frame, so the
                        // callee owns its copy (matches the C ABI).
                        Abi::Memory => arg_frags.push(format!(
                            "ptr byval({}) align {} {}",
                            cx.abi_storage_ty(&ty), cx.abi_align(&ty), emit_value(val))),
                    },
                    _ => arg_frags.push(format!("{} {}", emit_type(&ty, &cx.types, &cx.symbols), emit_value(val))),
                }
            }
            let args_str = arg_frags.join(", ");
            match sret {
                Some((slot, ret_ty)) => match cx.abi_of(&ret_ty) {
                    // Direct struct return: the call yields the coerced aggregate;
                    // unpack it into the caller's result slot.
                    Abi::Direct(regs) => {
                        let agg_ty = coerced_aggregate_ty(&regs);
                        let r = cx.abi_tmp();
                        emitln!(cx, "    {r} = call {agg_ty} {callee_str}({args_str})");
                        let vals: Vec<String> = if regs.len() == 1 {
                            vec![r]
                        } else {
                            (0..regs.len()).map(|i| {
                                let v = cx.abi_tmp();
                                emitln!(cx, "    {v} = extractvalue {agg_ty} {r}, {i}");
                                v
                            }).collect()
                        };
                        emit_regs_to_struct(cx, &slot.to_string(), &regs, &vals);
                    }
                    // Memory struct return: void call with a leading sret slot arg.
                    Abi::Memory => {
                        let sret_arg = format!("ptr sret({}) align {} {slot}",
                            cx.abi_storage_ty(&ret_ty), cx.abi_align(&ret_ty));
                        let all_args = if args_str.is_empty() {
                            sret_arg
                        } else {
                            format!("{sret_arg}, {args_str}")
                        };
                        emitln!(cx, "    call void {callee_str}({all_args})");
                    }
                },
                None => {
                    let dst_prefix = dst.map(|dst| format!("{dst} = ")).unwrap_or_default();
                    let call_ty = emit_type(&return_type, &cx.types, &cx.symbols);
                    emitln!(cx, "    {dst_prefix}call {call_ty} {callee_str}({args_str})");
                }
            }
        }
        // %result = getelementptr <PointeeTy>, ptr <BasePtr> {, <IdxTy> <Idx> }*
        // The index type must match the operand's real width (e.g. a u64
        // index emits an i64 operand), not a hardcoded i32.
        //
        // The stride is the element's *storage* type: a sequence of structs is
        // contiguous inline `%Name` records (what `alloc`/`size_of` produce),
        // not an array of `ptr` handles, so an aggregate element strides by its
        // named type. For a scalar element this is identical to `emit_type`.
        Index { dst, slice, index, index_ty, element_ty } => {
            let elem = emit_field_type(&element_ty, &cx.types, &cx.symbols);
            // LLVM widens a narrow gep index to pointer width by *sign*
            // extension, whatever the source language meant by it. An unsigned
            // index therefore addresses backwards once it passes half its range
            // - `buf[i]` with a `u32` `i >= 2^31` reads 2^31 elements the wrong
            // side of `buf` - so widen it here, explicitly and unsigned, rather
            // than let the implicit rule pick. Cheaper as well as correct: the
            // vectorizer has to guard a sign-extended induction variable against
            // overflow before it can use it, and a `zext` needs no such guard.
            match unsigned_narrow_index(&index_ty) {
                Some(bits) => {
                    emitln!(cx, "    {dst}.zx = zext i{bits} {index} to i64");
                    emitln!(cx, "    {dst} = getelementptr {elem}, ptr {slice}, i64 {dst}.zx");
                }
                // Signed, or already pointer-width: the implicit widening is the
                // one that was wanted.
                None => emitln!(cx, "    {dst} = getelementptr {elem}, ptr {slice}, {} {index}", emit_type(&index_ty, &cx.types, &cx.symbols)),
            }
        }
        InsertValue { dst, elem, ty, val, index } =>
            emitln!(cx, "    {dst} = insertvalue {{ ptr, i32 }} {elem}, {} {}, {index}", emit_type(&ty, &cx.types, &cx.symbols), emit_value(val)),
        ExtractValue { dst, val, index } => emitln!(cx, "    {dst} = extractvalue {{ ptr, i32 }} {}, {index}", emit_value(val)),

        Alloca { dst, ty, align } =>  emitln!(cx, "    {dst} = alloca {}{}", emit_type(&ty, &cx.types, &cx.symbols), align_suffix(align)),
        Store { ptr, val, ty, align } => emitln!(cx, "    store {} {}, ptr {ptr}{}", emit_type(&ty, &cx.types, &cx.symbols), emit_value(val), align_suffix(align)),
        Load { dst, ptr, ty, align } => emitln!(cx, "    {dst} = load {}, ptr {ptr}{}", emit_type(&ty, &cx.types, &cx.symbols), align_suffix(align)),

        // an array's elements are inline storage, so an array of structs is
        // `[N x %Name]` (matching a struct's array *field*, and `size_of`), not
        // `[N x ptr]`. `emit_field_type` is identical to `emit_type` for scalars.
        AllocaArray { dst, ty, length } => emitln!(cx, "    {dst} = alloca [{} x {}]", length, emit_field_type(&ty, &cx.types, &cx.symbols)),
        IndexArray { dst, ty, length, array, index } =>
            emitln!(cx, "    {dst} = getelementptr [{length} x {}], ptr {array}, i32 0, i32 {index}", emit_field_type(&ty, &cx.types, &cx.symbols)),

        AllocaStruct { dst, def, align } =>
            emitln!(cx, "    {dst} = alloca %{}{}", cx.sym(def), align_suffix(align)),
        FieldPtr { dst, struct_def, base, field_index } =>
            emitln!(cx, "    {dst} = getelementptr %{}, ptr {base}, i32 0, i32 {field_index}", cx.sym(struct_def)),
        // a zero-offset gep off the global symbol yields its address as a `ptr`
        GlobalPtr { dst, name } =>
            emitln!(cx, "    {dst} = getelementptr i8, ptr @{}, i64 0", ir_symbol(name)),

        Sizeof { dst, ty } => {
            // classic LLVM sizeof: index one element past a null base, then
            // reinterpret the resulting address as an integer. The target data
            // layout decides the stride, so struct padding/alignment is exact.
            let elem = emit_field_type(&ty, &cx.types, &cx.symbols);
            emitln!(cx, "    {dst}.szp = getelementptr {elem}, ptr null, i32 1");
            emitln!(cx, "    {dst} = ptrtoint ptr {dst}.szp to i64");
        }
        Extend { dst, val, from_ty, to_ty } => {
            use Type::*;
            let value = emit_value(val);
            // die typkonvertierungstabelle
            match (from_ty, to_ty) {
                // same-type or identity casts (no-op)
                (Int32, Int32) | (Int32, Uint32) | (Uint32, Int32) | (Uint32, Uint32) => emitln!(cx, "    {dst} = bitcast i32 {} to i32", value),
                (Int64, Int64) | (Int64, Uint64) | (Uint64, Int64) | (Uint64, Uint64) => emitln!(cx, "    {dst} = bitcast i64 {} to i64", value),
                (Float32, Float32) => emitln!(cx, "    {dst} = bitcast float {} to float", value),
                (Float64, Float64) => emitln!(cx, "    {dst} = bitcast double {} to double", value),

                // downcasting
                (Int64, Int32) | (Int64, Uint32)   => emitln!(cx, "    {dst} = trunc i64 {} to i32", value),
                (Uint64, Int32) | (Uint64, Uint32) => emitln!(cx, "    {dst} = trunc i64 {} to i32", value),

                // upcasting/extension
                // signed source -> sign extension
                (Int32, Int64) | (Int32, Uint64) => emitln!(cx, "    {dst} = sext i32 {} to i64", value),
                // unsigned source -> zero extension
                (Uint32, Int64) | (Uint32, Uint64) => emitln!(cx, "    {dst} = zext i32 {} to i64", value),

                // float to float
                (Float32, Float64) => emitln!(cx, "    {dst} = fpext float {} to double", value),
                (Float64, Float32) => emitln!(cx, "    {dst} = fptrunc double {} to float", value),

                // integer to float
                (Int32, Float32) => emitln!(cx, "    {dst} = sitofp i32 {} to float", value),
                (Int32, Float64) => emitln!(cx, "    {dst} = sitofp i32 {} to double", value),
                (Int64, Float32) => emitln!(cx, "    {dst} = sitofp i64 {} to float", value),
                (Int64, Float64) => emitln!(cx, "    {dst} = sitofp i64 {} to double", value),

                (Uint32, Float32) => emitln!(cx, "    {dst} = uitofp i32 {} to float", value),
                (Uint32, Float64) => emitln!(cx, "    {dst} = uitofp i32 {} to double", value),
                (Uint64, Float32) => emitln!(cx, "    {dst} = uitofp i64 {} to float", value),
                (Uint64, Float64) => emitln!(cx, "    {dst} = uitofp i64 {} to double", value),

                // float to integer (saturating)
                // this is like Rust where it defaults to saturating casts (`@llvm.fptosi.sat` / `@llvm.fptoui.sat`)
                // to prevent UB if a float overflows the target integer type (i hope)
                (Float32, Int32)  => emitln!(cx, "    {dst} = call i32 @llvm.fptosi.sat.i32.f32(float {value})"),
                (Float32, Uint32) => emitln!(cx, "    {dst} = call i32 @llvm.fptoui.sat.i32.f32(float {value})"),
                (Float32, Int64)  => emitln!(cx, "    {dst} = call i64 @llvm.fptosi.sat.i64.f32(float {value})"),
                (Float32, Uint64) => emitln!(cx, "    {dst} = call i64 @llvm.fptoui.sat.i64.f32(float {value})"),

                (Float64, Int32)  => emitln!(cx, "    {dst} = call i32 @llvm.fptosi.sat.i32.f64(double {value})"),
                (Float64, Uint32) => emitln!(cx, "    {dst} = call i32 @llvm.fptoui.sat.i32.f64(double {value})"),
                (Float64, Int64)  => emitln!(cx, "    {dst} = call i64 @llvm.fptosi.sat.i64.f64(double {value})"),
                (Float64, Uint64) => emitln!(cx, "    {dst} = call i64 @llvm.fptoui.sat.i64.f64(double {value})"),

                // --- 8-bit integer (i8 / u8) conversions ---
                // same 8-bit width: identity bitcast
                (Int8, Int8) | (Int8, Uint8) | (Uint8, Int8) | (Uint8, Uint8) =>
                    emitln!(cx, "    {dst} = bitcast i8 {} to i8", value),

                // widening from 8-bit: signed source sign-extends, unsigned zero-extends
                (Int8, Int32) | (Int8, Uint32)  => emitln!(cx, "    {dst} = sext i8 {} to i32", value),
                (Int8, Int64) | (Int8, Uint64)  => emitln!(cx, "    {dst} = sext i8 {} to i64", value),
                (Uint8, Int32) | (Uint8, Uint32) => emitln!(cx, "    {dst} = zext i8 {} to i32", value),
                (Uint8, Int64) | (Uint8, Uint64) => emitln!(cx, "    {dst} = zext i8 {} to i64", value),

                // narrowing to 8-bit: truncate (signedness of source is irrelevant)
                (Int32, Int8) | (Int32, Uint8) | (Uint32, Int8) | (Uint32, Uint8) =>
                    emitln!(cx, "    {dst} = trunc i32 {} to i8", value),
                (Int64, Int8) | (Int64, Uint8) | (Uint64, Int8) | (Uint64, Uint8) =>
                    emitln!(cx, "    {dst} = trunc i64 {} to i8", value),

                // 8-bit integer to float
                (Int8, Float32)  => emitln!(cx, "    {dst} = sitofp i8 {} to float", value),
                (Int8, Float64)  => emitln!(cx, "    {dst} = sitofp i8 {} to double", value),
                (Uint8, Float32) => emitln!(cx, "    {dst} = uitofp i8 {} to float", value),
                (Uint8, Float64) => emitln!(cx, "    {dst} = uitofp i8 {} to double", value),

                // float to 8-bit integer (saturating, matching the wider widths above)
                (Float32, Int8)  => emitln!(cx, "    {dst} = call i8 @llvm.fptosi.sat.i8.f32(float {value})"),
                (Float32, Uint8) => emitln!(cx, "    {dst} = call i8 @llvm.fptoui.sat.i8.f32(float {value})"),
                (Float64, Int8)  => emitln!(cx, "    {dst} = call i8 @llvm.fptosi.sat.i8.f64(double {value})"),
                (Float64, Uint8) => emitln!(cx, "    {dst} = call i8 @llvm.fptoui.sat.i8.f64(double {value})"),

                // --- 16-bit integer (i16 / u16) conversions ---
                // same 16-bit width: identity bitcast
                (Int16, Int16) | (Int16, Uint16) | (Uint16, Int16) | (Uint16, Uint16) =>
                    emitln!(cx, "    {dst} = bitcast i16 {} to i16", value),

                // widening from 16-bit: signed source sign-extends, unsigned zero-extends
                (Int16, Int32) | (Int16, Uint32)  => emitln!(cx, "    {dst} = sext i16 {} to i32", value),
                (Int16, Int64) | (Int16, Uint64)  => emitln!(cx, "    {dst} = sext i16 {} to i64", value),
                (Uint16, Int32) | (Uint16, Uint32) => emitln!(cx, "    {dst} = zext i16 {} to i32", value),
                (Uint16, Int64) | (Uint16, Uint64) => emitln!(cx, "    {dst} = zext i16 {} to i64", value),

                // widening from 8-bit to 16-bit
                (Int8, Int16) | (Int8, Uint16)  => emitln!(cx, "    {dst} = sext i8 {} to i16", value),
                (Uint8, Int16) | (Uint8, Uint16) => emitln!(cx, "    {dst} = zext i8 {} to i16", value),

                // narrowing to 16-bit: truncate (signedness of source is irrelevant)
                (Int32, Int16) | (Int32, Uint16) | (Uint32, Int16) | (Uint32, Uint16) =>
                    emitln!(cx, "    {dst} = trunc i32 {} to i16", value),
                (Int64, Int16) | (Int64, Uint16) | (Uint64, Int16) | (Uint64, Uint16) =>
                    emitln!(cx, "    {dst} = trunc i64 {} to i16", value),

                // narrowing 16-bit to 8-bit: truncate
                (Int16, Int8) | (Int16, Uint8) | (Uint16, Int8) | (Uint16, Uint8) =>
                    emitln!(cx, "    {dst} = trunc i16 {} to i8", value),

                // 16-bit integer to float
                (Int16, Float32)  => emitln!(cx, "    {dst} = sitofp i16 {} to float", value),
                (Int16, Float64)  => emitln!(cx, "    {dst} = sitofp i16 {} to double", value),
                (Uint16, Float32) => emitln!(cx, "    {dst} = uitofp i16 {} to float", value),
                (Uint16, Float64) => emitln!(cx, "    {dst} = uitofp i16 {} to double", value),

                // float to 16-bit integer (saturating, matching the other widths above)
                (Float32, Int16)  => emitln!(cx, "    {dst} = call i16 @llvm.fptosi.sat.i16.f32(float {value})"),
                (Float32, Uint16) => emitln!(cx, "    {dst} = call i16 @llvm.fptoui.sat.i16.f32(float {value})"),
                (Float64, Int16)  => emitln!(cx, "    {dst} = call i16 @llvm.fptosi.sat.i16.f64(double {value})"),
                (Float64, Uint16) => emitln!(cx, "    {dst} = call i16 @llvm.fptoui.sat.i16.f64(double {value})"),

                (f, t) => unreachable!("unsupported type extension from {f:?} to {t:?}"),
            };
            // emitln!(cx, "    {dst} = {inst} {from_ty_str} {} to {to_ty_str}", emit_value(val));
        }
        Splat { dst, val, ty, size } => {
            let ty_str = emit_type(&ty, &cx.types, &cx.symbols);
            let simd_ty = format!("<{size} x {}>", ty_str);
            emitln!(cx, "    {dst} = insertelement {simd_ty} undef, {ty_str} {}, i32 0", emit_value(val));
        }
        Shuffle { dst, value_size, v0, v1, ty, size, mask } => {
            let ty_str = emit_type(&ty, &cx.types, &cx.symbols);
            let simd_ty = format!("<{} x {}>", value_size, ty_str);
            let mask = mask.into_iter().map(|i| format!("i32 {}", i)).collect::<Vec<_>>().join(", ");
            emitln!(cx, "    {dst} = shufflevector {simd_ty} {v0}, {simd_ty} {v1}, <{size} x i32> <{mask}>");
        }
    };
}

fn emit_terminator<'a>(cx: &mut EmitCtx<'a>, term: Terminator<'a>) {
    use Terminator::*;

    match term {
        Return(None) => match cx.sret_direct.clone() {
            // Direct struct return: the body wrote the result into the slot; pull
            // the coerced eightbytes back out and return them by value.
            Some((slot, regs)) => {
                let pairs = emit_struct_to_regs(cx, &slot.to_string(), &regs);
                if pairs.len() == 1 {
                    let (ty, v) = &pairs[0];
                    emitln!(cx, "    ret {ty} {v}");
                } else {
                    let agg_ty = coerced_aggregate_ty(&regs);
                    let mut cur = "undef".to_string();
                    for (i, (ty, v)) in pairs.iter().enumerate() {
                        let next = cx.abi_tmp();
                        emitln!(cx, "    {next} = insertvalue {agg_ty} {cur}, {ty} {v}, {i}");
                        cur = next;
                    }
                    emitln!(cx, "    ret {agg_ty} {cur}");
                }
            }
            None => emitln!(cx, "    ret void"),
        },
        Return(Some((value, ty))) => emitln!(cx, "    ret {} {}", emit_type(&ty, &cx.types, &cx.symbols), emit_value(value)),
        Jump(label) => emitln!(cx, "    br label %{label}"),
        Branch { cond, then_block, else_block } =>
            emitln!(cx, "    br i1 {}, label %{then_block}, label %{else_block}", emit_value(cond)),
        Switch { value, value_ty, default, cases } => {
            let ty = emit_type(&value_ty, &cx.types, &cx.symbols);
            let arms = cases.iter()
                .map(|(c, b)| format!("{ty} {}, label %{b}", emit_value(Value::Const(c.clone()))))
                .collect::<Vec<_>>()
                .join(" ");
            emitln!(cx, "    switch {ty} {}, label %{default} [ {arms} ]", emit_value(value));
        }
        Unreachable => emitln!(cx, "    unreachable"),
    }
}

fn emit_block<'a>(cx: &mut EmitCtx<'a>, block: BasicBlock<'a>) {
    emitln!(cx, "  {}:", block.id);
    block.instructions.into_iter().for_each(|inst| emit_inst(cx, inst));
    if let Some(terminator) = block.terminator {
        emit_terminator(cx, terminator);
    } else {
        panic!("block {} has no terminator", block.id);
    }
}

/// The bit width of an index type that is unsigned *and* narrower than a
/// pointer - the only case where LLVM's implicit sign-extension of a gep index
/// is the wrong widening. `None` for a signed index (sign-extension is right)
/// and for a 64-bit one (already pointer width, nothing to widen).
fn unsigned_narrow_index(ty: &Type<'_>) -> Option<u32> {
    match ty {
        Type::Uint8 => Some(8),
        Type::Uint16 => Some(16),
        Type::Uint32 => Some(32),
        _ => None,
    }
}

/// The `, align N` suffix for a `load`/`store`/`alloca`, or nothing at all.
///
/// When omitted, LLVM derives alignment from the target data layout. Explicit
/// alignment wins. The ABI pack/unpack helpers keep `align 1` because they may
/// access eightbytes through a less-aligned struct.
fn align_suffix(align: Option<usize>) -> String {
    align.map(|n| format!(", align {n}")).unwrap_or_default()
}

/// Emit loads pulling each eightbyte of a Direct-class struct out of the storage
/// at `struct_ptr`, returning the `(llvm_type, value)` pairs to pass or return.
fn emit_struct_to_regs<'a>(cx: &mut EmitCtx<'a>, struct_ptr: &str, regs: &[Reg]) -> Vec<(String, String)> {
    regs.iter().enumerate().map(|(i, r)| {
        let ty = r.to_llvm();
        let ptr = gep_eightbyte(cx, struct_ptr, i);
        let val = cx.abi_tmp();
        emitln!(cx, "    {val} = load {ty}, ptr {ptr}, align 1");
        (ty.to_string(), val)
    }).collect()
}

/// Emit stores writing each coerced `value` into the struct storage at
/// `struct_ptr`, at its eightbyte offset - the inverse of `emit_struct_to_regs`.
fn emit_regs_to_struct<'a>(cx: &mut EmitCtx<'a>, struct_ptr: &str, regs: &[Reg], values: &[String]) {
    for (i, (r, v)) in regs.iter().zip(values).enumerate() {
        let ty = r.to_llvm();
        let ptr = gep_eightbyte(cx, struct_ptr, i);
        emitln!(cx, "    store {ty} {v}, ptr {ptr}, align 1");
    }
}

/// A pointer to eightbyte `i` (byte offset `i*8`) within `struct_ptr`. Offset 0
/// reuses the base pointer directly.
fn gep_eightbyte<'a>(cx: &mut EmitCtx<'a>, struct_ptr: &str, i: usize) -> String {
    if i == 0 {
        return struct_ptr.to_string();
    }
    let ptr = cx.abi_tmp();
    emitln!(cx, "    {ptr} = getelementptr i8, ptr {struct_ptr}, i64 {}", i * 8);
    ptr
}

fn emit_function<'a>(cx: &mut EmitCtx<'a>, func: Function<'a>) {
    let mut export = false;
    // `@fastmath` is per function: without this reset the previous function's
    // flags would ride along into every one emitted after it
    cx.current_fast_math_flags = FastMathFlags::None;

    let attrs = func.attributes.iter()
        .filter_map(|a| match (a.value.name, a.value.value.as_deref()) {
            // the value sets below are the ones `ast::KNOWN_ATTRIBUTES` admits;
            // anything else was rejected as a diagnostic long before codegen
            ("inline", Some("always")) => Some(String::from("alwaysinline")),
            ("inline", Some("never"))  => Some(String::from("noinline")),
            ("inline", v) => unreachable!("`@inline({v:?})` passed attribute validation"),

            ("export", None) => { export = true; None },
            ("export", v) => unreachable!("`@export({v:?})` passed attribute validation"),

            ("fastmath", Some(flag)) => {
                cx.current_fast_math_flags = FastMathFlags::from_str(flag)
                    .unwrap_or_else(|| unreachable!("`@fastmath({flag})` passed attribute validation"));
                None
            }
            ("fastmath", None) => unreachable!("bare `@fastmath` passed attribute validation"),

            _ => None,
        }).collect::<Vec<_>>()
        .join(" ");
    let linkage = if export { "dso_local dllexport" } else { "internal" };
    let attrs_str = if attrs.is_empty() { String::new() } else { format!(" {attrs}") };

    // Build the parameter list, coercing by-value Direct structs into their
    // eightbyte registers. Each such struct also needs entry glue that rebuilds
    // it into the `ptr` register the MIL body expects - recorded here, emitted
    // once inside the entry block below.
    let mut sig_params: Vec<String> = Vec::new();
    let mut param_rebuilds: Vec<(Register, Type<'a>, Vec<Reg>, Vec<String>)> = Vec::new();
    for (reg, ty) in &func.params {
        match ty {
            _ if cx.is_aggregate_ty(ty) => match cx.abi_of(ty) {
                Abi::Direct(regs) => {
                    let names: Vec<String> = regs.iter().map(|_| cx.abi_tmp()).collect();
                    for (r, n) in regs.iter().zip(&names) {
                        sig_params.push(format!("{} {n}", r.to_llvm()));
                    }
                    param_rebuilds.push((*reg, ty.clone(), regs, names));
                }
                // Memory aggregate: received as a `byval` pointer - the caller's
                // copy, which this frame owns. (The MIL body still copies it into
                // a local on entry; harmless, just a second copy.)
                Abi::Memory => sig_params.push(format!(
                    "ptr byval({}) align {} {reg}", cx.abi_storage_ty(ty), cx.abi_align(ty))),
            },
            _ => sig_params.push(format!("{} {reg}", emit_type(ty, &cx.types, &cx.symbols))),
        }
    }

    // Return type: a Direct struct returns its coerced aggregate (no sret param);
    // a Memory struct keeps the sret out-pointer; anything else is itself.
    let ret_ty_str = match &func.sret {
        Some((reg, ret_ty)) => match cx.abi_of(ret_ty) {
            Abi::Direct(regs) => {
                // The slot the body writes into is now a local, not a parameter;
                // record it so each `ret` loads and returns the coerced value.
                cx.sret_direct = Some((*reg, regs.clone()));
                coerced_aggregate_ty(&regs)
            }
            Abi::Memory => {
                cx.sret_direct = None;
                sig_params.insert(0, format!("ptr sret({}) align {} {reg}",
                    cx.abi_storage_ty(ret_ty), cx.abi_align(ret_ty)));
                "void".to_string()
            }
        },
        None => {
            cx.sret_direct = None;
            // a `!`-returning proc never returns; LLVM has no bottom type, so its
            // signature return type is `void` (the body ends in `unreachable`).
            match func.return_type {
                Type::Never => "void".to_string(),
                ref ty => emit_type(ty, &cx.types, &cx.symbols),
            }
        }
    };

    let params_str = sig_params.join(", ");
    emitln!(cx, "define {linkage} {ret_ty_str} @{}({params_str}){attrs_str} {{", ir_symbol(func.name));

    // Entry preamble: rebuild Direct struct params into their `ptr` registers,
    // and allocate the local slot for a Direct struct return. These allocas live
    // in an unnamed entry block that falls through to the MIL entry.
    let has_preamble = !param_rebuilds.is_empty() || cx.sret_direct.is_some();
    let first_block = func.blocks.first().map(|b| b.id);
    for (reg, ty, regs, names) in param_rebuilds {
        emitln!(cx, "    {reg} = alloca {}", cx.abi_storage_ty(&ty));
        emit_regs_to_struct(cx, &reg.to_string(), &regs, &names);
    }
    if let Some((reg, _)) = &cx.sret_direct {
        // the slot's type for the alloca comes from func.sret
        let (_, ret_ty) = func.sret.as_ref().unwrap();
        emitln!(cx, "    {reg} = alloca {}", cx.abi_storage_ty(ret_ty));
    }
    if has_preamble {
        if let Some(id) = first_block {
            emitln!(cx, "    br label %{id}");
        }
    }

    func.blocks.into_iter().for_each(|block| emit_block(cx, block));
    emitln!(cx, "}}");
    cx.sret_direct = None;
}

fn emit_extern<'a>(cx: &mut EmitCtx<'a>, ext: ExternDecl<'a>) {
    let attrs = ext.attributes.iter()
        .filter_map(|a| match (a.value.name, a.value.value.as_deref()) {
            ("inline", Some("always")) => Some("alwaysinline"),
            ("inline", Some("never"))  => Some("noinline"),
            _ => None,
        }).collect::<Vec<_>>()
        .join(" ");
    let attrs_str = if attrs.is_empty() { String::new() } else { format!(" {attrs}") };

    let mut params: Vec<String> = ext.params.iter()
        .flat_map(|ty| abi_param_types(cx, ty))
        .collect();
    // A Memory-class struct return uses a hidden leading sret pointer and returns
    // void - the same shape MIL lowers the matching call to.
    let ret_str = match &ext.return_type {
        _ if cx.is_aggregate_ty(&ext.return_type) => match cx.abi_of(&ext.return_type) {
            Abi::Direct(regs) => coerced_aggregate_ty(&regs),
            Abi::Memory => {
                params.insert(0, format!("ptr sret({}) align {}",
                    cx.abi_storage_ty(&ext.return_type), cx.abi_align(&ext.return_type)));
                "void".to_string()
            }
        },
        Type::Never => "void".to_string(),
        _ => emit_type(&ext.return_type, &cx.types, &cx.symbols),
    };
    emitln!(cx, "declare {ret_str} @{}({}){attrs_str}", ir_symbol(ext.name), params.join(", "));
}

/// The LLVM parameter type(s) for a single source-level parameter of type `ty`.
/// A by-value Direct struct expands into its coerced eightbyte registers; a
/// Memory struct is one `byval` pointer; anything else is one type as usual.
fn abi_param_types<'a>(cx: &EmitCtx<'a>, ty: &Type<'a>) -> Vec<String> {
    match ty {
        _ if cx.is_aggregate_ty(ty) => match cx.abi_of(ty) {
            Abi::Direct(regs) => regs.iter().map(|r| r.to_llvm().to_string()).collect(),
            Abi::Memory => vec![format!("ptr byval({}) align {}",
                cx.abi_storage_ty(ty), cx.abi_align(ty))],
        },
        _ => vec![emit_type(ty, &cx.types, &cx.symbols)],
    }
}

/// Emit a byte blob as the body of an LLVM `c"..."` string constant.
/// Printable ASCII passes through and everything else is emitted as a `\XX` hex escape.
fn emit_string_blob(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len());
    for &b in bytes {
        if b == b'"' || b == b'\\' || b < 0x20 || b > 0x7e {
            out.push_str(&format!("\\{:02X}", b));
        } else {
            out.push(b as char);
        }
    }
    out
}

/// Render a global's constant initializer as an LLVM constant expression (the
/// text after the type in `@g = constant <ty> <init>`).
fn emit_const_init<'a>(init: &ConstInit<'a>, types: &TypeTable<'a>, symbols: &HashMap<DefId, &'a str>) -> String {
    match init {
        ConstInit::Scalar(c) => emit_value(Value::Const(c.clone())),
        // `{ <fty> <finit>, ... }` - the surrounding type (`%Name`) is emitted by
        // the caller, and each field carries its own inline type.
        ConstInit::Struct(fields) => {
            let body = fields.iter()
                .map(|(ty, init)| format!("{} {}", emit_field_type(ty, types, symbols), emit_const_init(init, types, symbols)))
                .collect::<Vec<_>>()
                .join(", ");
            format!("{{ {body} }}")
        }
        // `[ <ety> e0, <ety> e1, ... ]` - the surrounding `[N x <ety>]` type is
        // emitted by the caller (top-level global or enclosing struct field).
        ConstInit::Array(elem_ty, elems) => {
            let body = elems.iter()
                .map(|init| format!("{} {}", emit_field_type(elem_ty, types, symbols), emit_const_init(init, types, symbols)))
                .collect::<Vec<_>>()
                .join(", ");
            format!("[{body}]")
        }
        // a function's address is written as the bare symbol
        ConstInit::FnAddr(name) => format!("@{}", ir_symbol(name)),
    }
}

fn emit_module<'a>(cx: &mut EmitCtx<'a>, module: Module<'a>) {
    // read-only global blobs backing string literals (@.str.N)
    for (i, bytes) in module.strings.iter().enumerate() {
        emitln!(cx, "@.str.{i} = private unnamed_addr constant [{} x i8] c\"{}\"",
            bytes.len(), emit_string_blob(bytes));
    }
    if !module.strings.is_empty() {
        emitln!(cx, "");
    }

    // typecheck rejected unknown field types so there can be no forward reference
    // to a not yet declared struct
    for def in &module.structs {
        let body = module.types[def].fields.iter()
            .map(|(_, ty)| emit_field_type(ty, &module.types, &module.symbols))
            .collect::<Vec<_>>()
            .join(", ");
        emitln!(cx, "%{} = type {{ {body} }}", ir_symbol(module.symbols[def]));
    }
    if !module.structs.is_empty() {
        emitln!(cx, "");
    }

    // module-level constants. `@export` gets external (dllexport) linkage so a
    // host can resolve the symbol; everything else stays `internal`.
    for g in &module.globals {
        let linkage = if g.export { "dso_local dllexport constant" } else { "internal constant" };
        emitln!(cx, "@{} = {linkage} {} {}",
            ir_symbol(g.name), emit_field_type(&g.ty, &module.types, &module.symbols),
            emit_const_init(&g.init, &module.types, &module.symbols));
    }
    if !module.globals.is_empty() {
        emitln!(cx, "");
    }

    module.externs.into_iter().for_each(|ext| emit_extern(cx, ext));
    module.functions.into_iter().for_each(|func| emit_function(cx, func));
}

// I hope LLVM will optimize these away if it's not used
const PREPEND: &str = r#"
; casts (for numeric_cast)
declare i32 @llvm.fptosi.sat.i32.f32(float)
declare i32 @llvm.fptoui.sat.i32.f32(float)
declare i64 @llvm.fptosi.sat.i64.f32(float)
declare i64 @llvm.fptoui.sat.i64.f32(float)
declare i32 @llvm.fptosi.sat.i32.f64(double)
declare i32 @llvm.fptoui.sat.i32.f64(double)
declare i64 @llvm.fptosi.sat.i64.f64(double)
declare i64 @llvm.fptoui.sat.i64.f64(double)
declare i8 @llvm.fptosi.sat.i8.f32(float)
declare i8 @llvm.fptoui.sat.i8.f32(float)
declare i8 @llvm.fptosi.sat.i8.f64(double)
declare i8 @llvm.fptoui.sat.i8.f64(double)

"#;

pub fn emit<'a>(module: Module<'a>) -> String {
    let mut header = format!("; ModuleID = 'compiled_module'\n");
    header.push_str(PREPEND.trim_start());

    // Layouts + symbols, for ABI classification and `%Name` references.
    let types = module.types.clone();
    let symbols = module.symbols.clone();
    let unions: UnionTable = module.enum_unions.clone();

    let mut cx = EmitCtx {
        buf: header,
        current_fast_math_flags: FastMathFlags::None,
        types,
        symbols,
        unions,
        abi_ctr: 0,
        sret_direct: None,
    };
    emit_module(&mut cx, module);
    cx.buf
}
