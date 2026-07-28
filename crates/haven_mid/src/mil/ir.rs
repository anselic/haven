use std::collections::HashMap;
use std::fmt::{Display, Formatter};
use haven_common::ast::*;
use haven_common::defs::DefId;
use haven_common::layout::TypeTable;

// Unique SSA temps that will never be reassigned
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Register(pub usize);

impl Display for Register {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "%t{}", self.0)
    }
}

#[derive(Clone, Debug)]
pub enum Value {
    Reg(Register),
    Const(Const),
}

impl Display for Value {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Value::Reg(r) => write!(f, "{}", r),
            Value::Const(c) => write!(f, "{}", c),
        }
    }
}

#[derive(Clone, Debug)]
pub enum Const {
    Null,
    Undef, // for uninitialized values (e.g. insertvalue with undef)
    Bool(bool),
    Int8(i8),
    Int32(i32),
    Int64(i64),
    Uint8(u8),
    Uint32(u32),
    Uint64(u64),
    Float32(f32),
    Float64(f64),
    /// Address of a module-level, NUL-terminated string blob, emitted as
    /// `@.str.{0}` - the raw `*const u8` value of a `str`. The index refers into
    /// `Module::strings`.
    GlobalStr(usize),
}

impl Display for Const {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Const::Null => write!(f, "null"),
            Const::Undef => write!(f, "undef"),
            Const::Bool(b) => write!(f, "{}", b),
            Const::Int8(n) => write!(f, "{}", n),
            Const::Int32(n) => write!(f, "{}", n),
            Const::Int64(n) => write!(f, "{}", n),
            Const::Uint8(n) => write!(f, "{}", n),
            Const::Uint32(n) => write!(f, "{}", n),
            Const::Uint64(n) => write!(f, "{}", n),
            Const::Float32(n)=> write!(f, "{:?}", n),
            Const::Float64(n)=> write!(f, "{:?}", n),
            Const::GlobalStr(n) => write!(f, "@.str.{}", n),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BlockId(pub usize);

impl Display for BlockId {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "b{}", self.0)
    }
}

/// The target of a call: either a statically-known function symbol (`@name`) or a
/// function-pointer value computed at runtime (`%reg`).
#[derive(Clone, Debug)]
pub enum Callee<'a> {
    Direct(&'a str),
    Indirect(Value),
}

#[derive(Clone, Debug)]
pub enum Inst<'a> {
    Comment(String),

    // %dst = <unary op> %val
    Unary { dst: Register, op: UnaryOp, val: Value, ty: Type<'a> },
    // %dst = <binary op> %lhs, %rhs
    Binary { dst: Register, op: BinaryOp, ty: Type<'a>, lhs: Value, rhs: Value },

    // %dst = call callee(args...)
    Call {
        dst: Option<Register>, // None if return type is void
        callee: Callee<'a>,
        args: Vec<(Value, Type<'a>)>,
        return_type: Type<'a>,
        // when the callee returns a struct, the result is passed back through
        // this caller-allocated slot (sret) instead of a return value, and the
        // call itself returns void. Some((slot, struct_name))
        sret: Option<(Register, DefId)>,
    },

    // %dst = getelemptr %slice, %index (for slice indexing)
    // using Register here because slice will be { ptr, len } fat pointer struct
    Index { dst: Register, slice: Register, index: Value, index_ty: Type<'a>, element_ty: Type<'a> },
    // %dst = insertvalue {{ ptr, i32 }} %elem, ty %val, %index
    InsertValue { dst: Register, elem: Value, ty: Type<'a>, val: Value, index: usize },
    // %dst = extractvalue %val, index (for extracting from fat pointer struct, e.g. data pointer or length)
    ExtractValue { dst: Register, val: Value, index: usize },
    // %dst = getelementptr %Name, ptr %base, i32 0, i32 <field_index>
    FieldPtr { dst: Register, struct_def: DefId, base: Register, field_index: usize },
    // %dst = ptr to a module-level global @name (a zero-offset gep, so %dst == @name)
    GlobalPtr { dst: Register, name: &'a str },

    // %dst = alloca ty (for mutable locals, if needed)
    Alloca { dst: Register, ty: Type<'a>, align: Option<usize> },
    // store ty %val, ptr %ptr
    Store { ptr: Register, val: Value, ty: Type<'a>, align: Option<usize> },
    // %dst = load ty %ptr
    Load { dst: Register, ptr: Register, ty: Type<'a>, align: Option<usize> },

    // since Type doesn't encode length for slices and I'm too lazy to change it
    // %dst = alloca [length x ty]
    AllocaArray { dst: Register, ty: Type<'a>, length: usize },
    // %dst = getelemptr [length X ty] %array, %index
    IndexArray { dst: Register, ty: Type<'a>, length: usize, array: Register, index: usize },
    // %dst = alloca %struct_ty, align N
    AllocaStruct { dst: Register, def: DefId, align: Option<usize> },

    // Intrinsic/SIMD related instructions
    // %dst = ptrtoint of `getelementptr ty, ptr null, i32 1` -> size of `ty` in bytes (u64)
    Sizeof { dst: Register, ty: Type<'a> },
    Extend { dst: Register, val: Value, from_ty: Type<'a>, to_ty: Type<'a> },
    // %v0 = insertelement ty(simd) undef, %value, 0
    Splat { dst: Register, val: Value, ty: Type<'a>, size: usize },
    // %dst = shufflevector ty(simd) %v0, ty(simd) %v1, <size x i32> <mask>
    Shuffle { dst: Register, value_size: usize, v0: Value, v1: Value, ty: Type<'a>, size: usize, mask: Vec<usize> },
}

#[derive(Clone, Debug)]
pub enum Terminator<'a> {
    // return %val / return void
    Return(Option<(Value, Type<'a>)>),

    // br label %block
    Jump(BlockId),

    // br i1 %cond, label %then, label %else
    // the condition is a Value, not a Register: a constant condition (`if (true)`,
    // `while (true)`, `true && f()`) branches on the immediate directly.
    Branch {
        cond: Value,
        then_block: BlockId,
        else_block: BlockId,
    },

    // switch <ty> %value, label %default [ <ty> <const>, label %block ... ]
    // used by `match`; the scrutinee is an integer/enum-discriminant value.
    Switch {
        value: Value,
        value_ty: Type<'a>,
        default: BlockId,
        cases: Vec<(Const, BlockId)>,
    },

    // marks a provably unreachable point (e.g. the merge block after an if/else
    // whose branches both return). only emitted where the typechecker has already
    // proven control can't actually reach it.
    Unreachable,
}

impl <'a> Display for Terminator<'a> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Terminator::Return(Some((val, ty))) => write!(f, "return {} : {}", val, ty),
            Terminator::Return(None) => write!(f, "return"),
            Terminator::Jump(block) => write!(f, "br label {}", block),
            Terminator::Branch { cond, then_block, else_block } => {
                write!(f, "br i1 {}, label {}, label {}", cond, then_block, else_block)
            }
            Terminator::Switch { value, value_ty, default, cases } => {
                let arms = cases.iter().map(|(c, b)| format!("{} {}, label {}", value_ty, c, b))
                    .collect::<Vec<_>>().join(" ");
                write!(f, "switch {} {}, label {} [ {} ]", value_ty, value, default, arms)
            }
            Terminator::Unreachable => write!(f, "unreachable"),
        }
    }
}

#[derive(Clone, Debug)]
pub struct BasicBlock<'a> {
    pub id: BlockId,
    pub instructions: Vec<Inst<'a>>,
    pub terminator: Option<Terminator<'a>>,
}

#[derive(Clone, Debug)]
pub struct ExternDecl<'a> {
    pub name: &'a str,
    pub attributes: Vec<Attribute<'a>>,
    pub params: Vec<Type<'a>>,
    pub return_type: Type<'a>,
}

#[derive(Clone, Debug)]
pub struct Function<'a> {
    pub name: &'a str,
    pub attributes: Vec<Attribute<'a>>,
    pub params: Vec<(Register, Type<'a>)>,
    pub return_type: Type<'a>,
    pub blocks: Vec<BasicBlock<'a>>, // blocks[0] is always the entry
    // for struct-returning functions: the hidden sret out-pointer parameter
    // (its register and the struct name). The function returns void in LLVM
    pub sret: Option<(Register, DefId)>,
}

/// The constant value initializing a module-level global. A restricted subset of
/// expressions that map onto an LLVM constant aggregate - no code runs to produce
/// it.
#[derive(Clone, Debug)]
pub enum ConstInit<'a> {
    Scalar(Const),
    /// A constant struct aggregate: one `(field type, field constant)` per field,
    /// in declaration order. Nested structs recurse.
    Struct(Vec<(Type<'a>, ConstInit<'a>)>),
    /// A constant array aggregate: the element type plus one constant per element.
    Array(Type<'a>, Vec<ConstInit<'a>>),
    /// The address of a top-level function, emitted as `@name` - a link-time
    /// constant `ptr`. Used for function-pointer fields (e.g. CLAP's `clap_entry`).
    FnAddr(&'a str),
}

#[derive(Clone, Debug)]
pub struct Global<'a> {
    pub name: &'a str,
    /// `@export`: give the symbol external (dllexport) linkage so a host can look
    /// it up, instead of the default `internal`.
    pub export: bool,
    pub ty: Type<'a>,
    pub init: ConstInit<'a>,
}

#[derive(Clone, Debug)]
pub struct Module<'a> {
    pub functions: Vec<Function<'a>>,
    pub externs: Vec<ExternDecl<'a>>,
    /// Struct definitions, in declaration order: name -> ordered (field name, field type)
    /// Aggregates to declare, in emission order. Their field lists live in
    /// `types`; this is just the order, which a `HashMap` cannot carry and which
    /// matters for readable IR (a struct must precede its users).
    pub structs: Vec<DefId>,
    /// Layout of every named type, for `layout`/`abi` during emission.
    pub types: TypeTable<'a>,
    /// Emitted symbol per definition. This is the seam: everything upstream
    /// works in identities, everything downstream in names, and this map is the
    /// one place the two meet.
    pub symbols: HashMap<DefId, &'a str>,
    /// Module-level constants, emitted as LLVM `constant` globals.
    pub globals: Vec<Global<'a>>,
    /// Interned, escape-resolved string-literal blobs. Index `i` is emitted as
    /// the global `@.str.{i}` and referenced by `Const::GlobalStr(i)`.
    pub strings: Vec<Vec<u8>>,
    /// Data-enum aggregate name -> its variant payload-struct names (only
    /// entries with a payload; a unit variant has none). Lets FFI classification
    /// (`haven_back::abi`) treat an enum's `$payload` byte blob as a real union of
    /// the variant payload structs - each variant classified independently and
    /// merged (SysV union rule) - instead of raw bytes, which would wrongly force
    /// every payload into the INTEGER class regardless of what it actually holds.
    pub enum_unions: HashMap<DefId, Vec<DefId>>,
}
