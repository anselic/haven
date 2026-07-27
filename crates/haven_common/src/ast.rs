use std::fmt::{Display, Formatter};

/// Index of a source file in the [`crate::diag::Files`] table. Every token and
/// every AST node carries one inside its `Span`, so it is deliberately a `Copy`
/// integer: the filename itself is stored once, in `Files`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct FileId(pub u32);

impl FileId {
    /// Placeholder for a span that doesn't point anywhere yet (a cursor field
    /// initialized before the first real span is seen). `Files` renders it as
    /// `<unknown>` rather than panicking, so a stray one degrades a diagnostic
    /// instead of aborting the compiler.
    pub const UNKNOWN: FileId = FileId(u32::MAX);
}

#[derive(Clone, Copy, Debug)]
pub struct Span {
    pub file: FileId,
    pub start: usize,
    pub end: usize,
}

impl Span {
    pub fn new(file: FileId, start: usize, end: usize) -> Self {
        Self { file, start, end }
    }

    /// A span pointing nowhere, for placeholder/cursor fields.
    pub fn unknown() -> Self {
        Self { file: FileId::UNKNOWN, start: 0, end: 0 }
    }
}

impl chumsky::span::Span for Span {
    type Context = FileId;
    type Offset = usize;

    fn context(&self) -> Self::Context { self.file }
    fn new(context: Self::Context, range: std::ops::Range<Self::Offset>) -> Self {
        Self {
            file: context,
            start: range.start,
            end: range.end,
        }
    }
    fn start(&self) -> usize { self.start }
    fn end(&self) -> usize { self.end }
}

#[derive(Clone, Debug)]
pub struct Error {
    pub msg: String,
    pub span: Span,
}

impl Error {
    pub fn new(span: Span, msg: String) -> Self {
        Self { span, msg }
    }
}

#[derive(Clone, Debug)]
pub struct Metadata<T> {
    pub span: Span,
    pub id: usize,
    pub value: T,
}

static GLOBAL_ID_COUNTER: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

impl<T> Metadata<T> {
    pub fn new(value: T, span: Span) -> Self {
        let id = GLOBAL_ID_COUNTER.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        Self { value, id, span }
    }
}

/// A resolved value binding: which specific declaration a `Var` use refers to.
/// Produced by name resolution (in the typechecker) and consumed by MIL lowering
/// to key each variable's storage slot. Locals are identified by their `Declare`
/// statement's node id - globally unique, so two same-named locals in different
/// scopes (shadowing) never collide. Params are identified by name, which is
/// unique within a single function's parameter list.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Binding<'a> {
    Local(usize),
    Param(&'a str),
}

/// Extension trait to add a convenient method for creating metadata from a value and span.
// pub trait MetadataExt<T> {
//     fn make_metadata(self, span: Span) -> Metadata<T>;
// }

// impl<T> MetadataExt<T> for T {
//     fn make_metadata(self, span: Span) -> Metadata<T> {
//         Metadata::new(self, span)
//     }
// }

#[derive(Clone, Debug, PartialEq)]
pub enum Token<'a> {
    Bool(bool),
    Int8(i8), Int32(i32), Int64(i64),
    Uint8(u8), Uint32(u32), Uint64(u64),
    Float32(f32), Float64(f64),
    Str(&'a str),
    Var(&'a str),
    BinaryOp(BinaryOp),
    UnaryOp(UnaryOp),

    LParen, RParen,
    LBrace, RBrace,
    LBracket, RBracket,

    Dot, Comma, Semicolon,
    Colon, ColonColon, Assign, At,
    Arrow,

    Let, If, Else, Return,
    While, Break, Continue,
    Proc, Extern, Const, Struct, Enum,
    Import, Pub, Match,
}

impl Display for Token<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Token::Bool(b)      => write!(f, "{}", b),
            Token::Int8(n)      => write!(f, "{}i8", n),
            Token::Int32(n)     => write!(f, "{}i32", n),
            Token::Int64(n)     => write!(f, "{}i64", n),
            Token::Uint8(n)     => write!(f, "{}u8", n),
            Token::Uint32(n)    => write!(f, "{}u32", n),
            Token::Uint64(n)    => write!(f, "{}u64", n),
            Token::Float32(n)   => write!(f, "{}f32", n),
            Token::Float64(n)   => write!(f, "{}f64", n),
            Token::Str(s)       => write!(f, "\"{:?}\"", s),
            Token::Var(s)       => write!(f, "{}", s),
            Token::BinaryOp(op) => write!(f, "{}", op),
            Token::UnaryOp(op)  => write!(f, "{}", op),
            Token::LParen       => write!(f, "("),
            Token::RParen       => write!(f, ")"),
            Token::LBrace       => write!(f, "{{"),
            Token::RBrace       => write!(f, "}}"),
            Token::LBracket     => write!(f, "["),
            Token::RBracket     => write!(f, "]"),
            Token::Dot          => write!(f, "."),
            Token::Comma        => write!(f, ","),
            Token::Semicolon    => write!(f, ";"),
            Token::Colon        => write!(f, ":"),
            Token::ColonColon   => write!(f, "::"),
            Token::Assign       => write!(f, "="),
            Token::At           => write!(f, "@"),
            Token::Arrow        => write!(f, "->"),
            Token::Let          => write!(f, "let"),
            Token::If           => write!(f, "if"),
            Token::Else         => write!(f, "else"),
            Token::Return       => write!(f, "return"),
            Token::While        => write!(f, "while"),
            Token::Break        => write!(f, "break"),
            Token::Continue     => write!(f, "continue"),
            Token::Proc         => write!(f, "proc"),
            Token::Extern       => write!(f, "extern"),
            Token::Const        => write!(f, "const"),
            Token::Struct       => write!(f, "struct"),
            Token::Enum         => write!(f, "enum"),
            Token::Import       => write!(f, "import"),
            Token::Pub          => write!(f, "pub"),
            Token::Match        => write!(f, "match"),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum UnaryOp { Neg, Not, Deref, AddrOf, }

impl Display for UnaryOp {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", match self {
            UnaryOp::Neg    => "-",
            UnaryOp::Not    => "!",
            UnaryOp::Deref  => "*",
            UnaryOp::AddrOf => "&",
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BinaryOp {
    Add, Sub, Mul, Div, Mod,
    Eq, Ne, Lt, Gt, Le, Ge,
    // logical (bool, short-circuiting)
    And, Or,
    // bitwise (integers)
    BitAnd, BitOr, BitXor, Shl, Shr,
}

impl Display for BinaryOp {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", match self {
            BinaryOp::Add => "+",
            BinaryOp::Sub => "-",
            BinaryOp::Mul => "*",
            BinaryOp::Div => "/",
            BinaryOp::Mod => "%",
            BinaryOp::Eq  => "==",
            BinaryOp::Ne  => "!=",
            BinaryOp::Lt  => "<",
            BinaryOp::Gt  => ">",
            BinaryOp::Le  => "<=",
            BinaryOp::Ge  => ">=",
            BinaryOp::And => "&&",
            BinaryOp::Or  => "||",
            BinaryOp::BitAnd => "&",
            BinaryOp::BitOr  => "|",
            BinaryOp::BitXor => "^",
            BinaryOp::Shl => "<<",
            BinaryOp::Shr => ">>",
        })
    }
}

/// A compile-time constant appearing in a type position - the `N` in `[T; N]` or
/// `simd<T, N>`. Either a concrete literal or an unresolved const generic
/// parameter. Like [`Type::Param`], a `Param` is abstract and must never survive
/// to codegen; monomorphization substitutes it away. Use [`ConstVal::expect_lit`]
/// at code-gen sites to assert that.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ConstVal<'a> {
    Lit(usize),
    /// An unresolved const generic parameter, e.g. the `N` in `[T; N]`. Produced
    /// by the parser; monomorphization substitutes it with a `Lit`.
    Param(&'a str),
}

impl<'a> ConstVal<'a> {
    /// The concrete literal value. Panics if a const param survived past
    /// monomorphization - mirrors the `Type::Param` "must not reach codegen"
    /// contract, so a bug surfaces loudly rather than miscompiling.
    pub fn expect_lit(&self) -> usize {
        match self {
            ConstVal::Lit(n) => *n,
            ConstVal::Param(name) => panic!("const param '{name}' survived to codegen"),
        }
    }
}

impl<'a> Display for ConstVal<'a> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            ConstVal::Lit(n) => write!(f, "{}", n),
            ConstVal::Param(name) => write!(f, "{}", name),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Type<'a> {
    Void, Bool,
    Int8, Int32, Int64,
    Uint8, Uint32, Uint64,
    Float32, Float64,
    Function {
        params: Vec<Type<'a>>,
        return_type: Box<Type<'a>>,
    },
    Pointer(Box<Self>),
    /// Fixed-size array type, e.g., `[T; N]`
    Array(Box<Self>, ConstVal<'a>),
    /// Slice type, e.g., `[T]`
    Slice(Box<Self>),
    Simd(Box<Self>, ConstVal<'a>),
    /// Static string slice (like `&'static str` in Rust)
    Str,
    /// A named struct type, with any generic arguments applied, e.g. `Vec2`
    /// (`args` empty), `Option<i32>` (one type arg) or `Buf<i32, 8>` (a type arg
    /// and a const arg - hence `GenericArg`, not `Type`). `args` is always empty
    /// after monomorphization: a generic struct type is rewritten to a concrete
    /// instance with a mangled `name` and no `args` (like generic functions). Use
    /// [`Type::plain_struct`] to build the common no-args case.
    Struct { name: &'a str, args: Vec<GenericArg<'a>> },
    /// A named enum, with any generic arguments applied (mirrors `Struct.args`;
    /// `args` is always empty after monomorphization - a generic enum type is
    /// rewritten to a concrete instance with a mangled `name`, like a generic
    /// struct). `repr` is the integer type its discriminant is stored as (default
    /// `i32`, or set by `@repr(<int>)`). `has_payload` is `false` for a field-less
    /// C-style enum - a bare scalar discriminant, zero overhead - and `true` for a
    /// data-carrying (ADT / sum-type) enum, which is an aggregate `{ tag: repr,
    /// payload: [P x i8] }` laid out via a synthetic struct of the same `name`
    /// (see the mid end's forward-declaration pass). The repr is baked into the
    /// type so scalar layout/codegen just delegate to it; `name` keeps enum
    /// values distinct from a plain integer for typechecking. Produced by
    /// typecheck when a `Struct(name)` resolves to a declared enum.
    Enum { name: &'a str, repr: Box<Self>, has_payload: bool, args: Vec<GenericArg<'a>> },
    /// A generic type parameter, e.g. `T`.
    /// Produced during typechecking by resolving a `Struct(name)` where the name
    /// matches a type param in scope. It is abstract and must never survive to
    /// codegen stage.
    Param(&'a str),
}

impl<'a> Type<'a> {
    /// A non-generic struct type (no type arguments) - the common case, and the
    /// only shape any stage after monomorphization ever produces.
    pub fn plain_struct(name: &'a str) -> Self {
        Type::Struct { name, args: Vec::new() }
    }

    pub fn is_numeric(&self) -> bool {
        matches!(self,
            Type::Int8 | Type::Int32 | Type::Int64
            | Type::Uint8 | Type::Uint32 | Type::Uint64
            | Type::Float32 | Type::Float64)
    }

    pub fn is_integer(&self) -> bool {
        matches!(self,
            Type::Int8 | Type::Int32 | Type::Int64
            | Type::Uint8 | Type::Uint32 | Type::Uint64)
    }

    pub fn is_numeric_or_numeric_simd(&self) -> bool {
        self.is_numeric() || matches!(self, Type::Simd(inner, _) if inner.is_numeric())
    }
}

impl<'a> Display for Type<'a> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        use Type::*;
        match self {
            Void => write!(f, "void"),
            Bool => write!(f, "bool"),
            Uint8 => write!(f, "u8"), Uint32 => write!(f, "u32"), Uint64 => write!(f, "u64"),
            Int8 => write!(f, "i8"), Int32 => write!(f, "i32"), Int64 => write!(f, "i64"),
            Float32 => write!(f, "f32"), Float64 => write!(f, "f64"),
            Function { params, return_type } => {
                let params_str = params.iter().map(|p| p.to_string()).collect::<Vec<_>>().join(", ");
                write!(f, "proc({}) {}", params_str, return_type)
            },
            Pointer(inner) => write!(f, "*{}", inner),
            Array(inner, size) => write!(f, "[{}; {}]", inner, size),
            Slice(inner) => write!(f, "[{}]", inner),
            Simd(inner, size) => write!(f, "simd[{}, {}]", inner, size),
            Str => write!(f, "str"),
            Struct { name, args } if args.is_empty() => write!(f, "{}", name),
            Struct { name, args } => {
                let args_str = args.iter().map(|a| a.to_string()).collect::<Vec<_>>().join(", ");
                write!(f, "{}<{}>", name, args_str)
            },
            Enum { name, args, .. } if args.is_empty() => write!(f, "{}", name),
            Enum { name, args, .. } => {
                let args_str = args.iter().map(|a| a.to_string()).collect::<Vec<_>>().join(", ");
                write!(f, "{}<{}>", name, args_str)
            },
            Param(name) => write!(f, "{}", name),
        }
    }
}

#[derive(Clone, Debug)]
pub enum ExprNode<'a> {
    Bool(bool),
    Int8(i8), Int32(i32), Int64(i64),
    Uint8(u8), Uint32(u32), Uint64(u64),
    Float32(f32), Float64(f64),
    /// String literal `"..."`. Holds the raw source text between the quotes.
    /// escape sequences are resolved later, during MIL lowering, e.g.
    /// `\n` ([\, n]) becomes a single byte 0x0A
    Str(&'a str),
    Var(&'a str),
    Slice(Vec<Expr<'a>>),

    Struct {
        name: &'a str,
        /// Turbofish generic arguments for a generic struct, e.g. the `i32` in
        /// `Option::<i32> { ... }` or the `i32, 8` in `Buf::<i32, 8> { ... }`.
        /// Empty for a non-generic struct literal.
        type_args: Vec<GenericArg<'a>>,
        fields: Vec<(&'a str, Expr<'a>)>,
    },
    Access {
        base: Box<Expr<'a>>,
        field: &'a str,
    },

    Index {
        slice: Box<Expr<'a>>,
        index: Box<Expr<'a>>,
    },
    Unary {
        op: UnaryOp,
        operand: Box<Expr<'a>>,
    },
    Binary {
        op: BinaryOp,
        left: Box<Expr<'a>>,
        right: Box<Expr<'a>>,
    },
    Call {
        func: Box<Expr<'a>>,
        /// Turbofish type/const arguments, e.g. `::<f32, 4>`. Empty for ordinary
        /// calls.
        type_args: Vec<GenericArg<'a>>,
        args: Vec<Expr<'a>>,
    },
}

impl<'a> Display for ExprNode<'a> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            ExprNode::Bool(val) => write!(f, "{}", val),
            ExprNode::Int8(val) => write!(f, "{}i8", val),
            ExprNode::Int32(val) => write!(f, "{}i32", val),
            ExprNode::Int64(val) => write!(f, "{}i64", val),
            ExprNode::Uint8(val) => write!(f, "{}u8", val),
            ExprNode::Uint32(val) => write!(f, "{}u32", val),
            ExprNode::Uint64(val) => write!(f, "{}u64", val),
            ExprNode::Float32(val) => write!(f, "{}f32", val),
            ExprNode::Float64(val) => write!(f, "{}f64", val),
            ExprNode::Str(s) => write!(f, "{:?}", s),
            ExprNode::Var(name) => write!(f, "{}", name),

            ExprNode::Slice(elements) => {
                let elements_str = elements.iter()
                    .map(|e| e.value.to_string())
                    .collect::<Vec<_>>()
                    .join(", ");
                write!(f, "[{}]", elements_str)
            },
            ExprNode::Access { base, field } => write!(f, "{}.{}", base.value, field),

            ExprNode::Struct { name, type_args, fields } => {
                let turbofish = if type_args.is_empty() {
                    String::new()
                } else {
                    format!("::<{}>", type_args.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(", "))
                };
                let fields_str = fields.iter()
                    .map(|(field_name, field_value)| format!("{}: {}", field_name, field_value.value))
                    .collect::<Vec<_>>()
                    .join(", ");
                write!(f, "{}{} {{ {} }}", name, turbofish, fields_str)
            },

            ExprNode::Index { slice, index } => write!(f, "{}[{}]", slice.value, index.value),
            ExprNode::Unary { op, operand } => write!(f, "({}{})", op, operand.value),
            ExprNode::Binary { op, left, right } => write!(f, "({} {} {})", left.value, op, right.value),
            ExprNode::Call { func, type_args, args } => {
                let turbofish = if type_args.is_empty() {
                    String::new()
                } else {
                    format!("::<{}>", type_args.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(", "))
                };
                let args_str = args.iter()
                    .map(|arg| arg.value.to_string())
                    .collect::<Vec<_>>()
                    .join(", ");
                write!(f, "{}{}({})", func.value, turbofish, args_str)
            }
        }
    }
}

pub type Expr<'a> = Metadata<ExprNode<'a>>;

/// A generic parameter declared in a function's generic list, e.g. the `T` and
/// `const N: u32` in `proc foo<T, const N: u32>(...)`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum GenericParam<'a> {
    /// A type parameter, e.g. `T`, optionally with trait bounds (`T: Display` or
    /// `T: A + B`). `bounds` lists the traits the concrete argument must
    /// implement; empty for an unbounded param. A bounded param's method calls
    /// resolve through the trait in the typechecker, and monomorphization picks
    /// the concrete impl (static dispatch).
    Type { name: &'a str, bounds: Vec<&'a str> },
    /// A compile-time constant parameter, e.g. `const N: u32`.
    Const(&'a str, Type<'a>),
}

impl<'a> Display for GenericParam<'a> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            GenericParam::Type { name, bounds } if bounds.is_empty() => write!(f, "{}", name),
            GenericParam::Type { name, bounds } => write!(f, "{}: {}", name, bounds.join(" + ")),
            GenericParam::Const(name, ty) => write!(f, "const {}: {}", name, ty),
        }
    }
}

/// A generic argument supplied at a call site via turbofish, e.g. the `f32` and
/// `4` in `simd_splat::<f32, 4>(v)`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum GenericArg<'a> {
    /// A type argument, e.g. `f32`, `[i32; 4]`, `*T`.
    Type(Type<'a>),
    /// A compile-time constant argument: either a literal (`4`) or a const
    /// generic parameter forwarded by name (the `N` in `simd_load::<f32, N>`).
    /// Note the parser can't tell a forwarded const param from a type - both are
    /// bare idents - so it emits those as `Type(Struct(name))`; typecheck and
    /// monomorphization reclassify them once the callee's kinds are known.
    Const(ConstVal<'a>),
}

impl<'a> Display for GenericArg<'a> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            GenericArg::Type(ty) => write!(f, "{}", ty),
            GenericArg::Const(cv) => write!(f, "{}", cv),
        }
    }
}

/// Renders a generic list as `<T, const N: u32>`, or the empty string when there
/// are no params.
fn fmt_generics(generics: &[GenericParam<'_>]) -> String {
    if generics.is_empty() {
        String::new()
    } else {
        format!("<{}>", generics.iter().map(|g| g.to_string()).collect::<Vec<_>>().join(", "))
    }
}

#[derive(Clone, Debug)]
pub enum StmtNode<'a> {
    Expr(Expr<'a>),
    Block(Vec<Stmt<'a>>),

    Declare {
        name: &'a str,
        ty: Type<'a>,
        value: Expr<'a>,
    },
    Assign {
        left: Expr<'a>,
        value: Expr<'a>,
    },
    If {
        condition: Expr<'a>,
        then_branch: Box<Stmt<'a>>,
        else_branch: Option<Box<Stmt<'a>>>,
    },
    While {
        condition: Expr<'a>,
        body: Box<Stmt<'a>>,
    },
    /// `match (scrutinee) { pattern => body ... }`. Field-less for now: patterns
    /// are enum variants, integer literals, or `_` (wildcard). Each arm's body is
    /// a single statement or a block. Typecheck enforces exhaustiveness.
    Match {
        scrutinee: Expr<'a>,
        arms: Vec<(Pattern<'a>, Box<Stmt<'a>>)>,
    },

    // TODO add label? (e.g. `continue 'label;`)
    Continue,
    Break,
    Return(Expr<'a>),
}

/// A `match` arm pattern.
#[derive(Clone, Debug)]
pub enum PatternNode<'a> {
    /// `_` - matches anything; the default arm. Also used as a field position in a
    /// `Variant` pattern to ignore that payload field.
    Wildcard,
    /// an integer-literal pattern, e.g. `5` or `-1`.
    Int(i64),
    /// a field-less enum-variant pattern, e.g. `Status::Continue` (stored joined).
    Path(&'a str),
    /// a data-carrying enum-variant pattern that destructures the payload, e.g.
    /// `Msg::Note(pitch, vel)`. `path` is the joined `Enum::Variant`; `fields` is
    /// one sub-pattern per payload field (`Bind` to name it, `Wildcard` to ignore).
    Variant { path: &'a str, fields: Vec<Pattern<'a>> },
    /// a struct-style variant pattern that destructures a named payload by field,
    /// e.g. `Msg::Cc { id, val }` or `Msg::Cc { id: x, val: _ }`. Each entry is
    /// `(field_name, sub_pattern)`; binding is by field name, so order is free.
    /// The shorthand `{ id }` desugars to `(id, Bind(id))` at parse time.
    StructVariant { path: &'a str, fields: Vec<(&'a str, Pattern<'a>)> },
    /// a binding introduced by a `Variant` field, e.g. the `pitch` in
    /// `Msg::Note(pitch, vel)`. Metadata-wrapped so each binding has a unique node
    /// id (its binding identity, mirroring how a `Declare` keys its local).
    Bind(&'a str),
}

pub type Pattern<'a> = Metadata<PatternNode<'a>>;

impl<'a> Display for PatternNode<'a> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            PatternNode::Wildcard => write!(f, "_"),
            PatternNode::Int(n) => write!(f, "{}", n),
            PatternNode::Path(s) => write!(f, "{}", s),
            PatternNode::Bind(name) => write!(f, "{}", name),
            PatternNode::Variant { path, fields } => {
                let inner = fields.iter().map(|p| p.value.to_string()).collect::<Vec<_>>().join(", ");
                write!(f, "{}({})", path, inner)
            }
            PatternNode::StructVariant { path, fields } => {
                let inner = fields.iter()
                    .map(|(name, p)| format!("{}: {}", name, p.value))
                    .collect::<Vec<_>>().join(", ");
                write!(f, "{} {{ {} }}", path, inner)
            }
        }
    }
}

impl<'a> Display for StmtNode<'a> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            StmtNode::Expr(expr) => write!(f, "{}", expr.value),
            StmtNode::Block(stmts) => {
                let stmts_str = stmts.iter().map(|stmt| format!("    {}\n", stmt.value)).collect::<String>();
                write!(f, "{{\n{}}}", stmts_str)
            },
            StmtNode::Declare { name, ty, value } => write!(f, "let {}: {} = {}", name, ty, value.value),
            StmtNode::Assign { left, value } => write!(f, "{} = {}", left.value, value.value),
            StmtNode::If { condition, then_branch, else_branch } => {
                let else_str = if let Some(else_branch) = else_branch {
                    format!(" else {}", else_branch.value)
                } else {
                    String::new()
                };
                write!(f, "if ({}) {}{}", condition.value, then_branch.value, else_str)
            },
            StmtNode::While { condition, body } => write!(f, "while ({}) {}", condition.value, body.value),
            StmtNode::Match { scrutinee, arms } => {
                let arms_str = arms.iter().map(|(p, body)| {
                    format!("    {} -> {}", p.value, body.value)
                }).collect::<Vec<_>>().join("\n");
                write!(f, "match ({}) {{\n{}\n}}", scrutinee.value, arms_str)
            },

            StmtNode::Continue => write!(f, "continue"),
            StmtNode::Break => write!(f, "break"),
            StmtNode::Return(expr) => write!(f, "return {}", expr.value),
        }
    }
}

pub type Stmt<'a> = Metadata<StmtNode<'a>>;

#[derive(Clone, Debug)]
pub struct AttributeNode<'a> {
    pub name: &'a str,
    pub value: Option<String>,
}

impl<'a> AttributeNode<'a> {
    pub fn new(name: &'a str, value: Option<String>) -> Self {
        Self { name, value }
    }

    pub fn is_true(&self, name: &'a str) -> bool {
        self.name == name && self.value.is_some() && self.value.as_deref() == Some("true")
    }

    pub fn is_false(&self, name: &'a str) -> bool {
        self.name == name && self.value.is_some() && self.value.as_deref() == Some("false")
    }
}

impl<'a> Display for AttributeNode<'a> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        if let Some(value) = &self.value {
            write!(f, "@{}({})", self.name, value)
        } else {
            write!(f, "@{}", self.name)
        }
    }
}

pub type Attribute<'a> = Metadata<AttributeNode<'a>>;

/// How a method takes `self`. `Associated` is no receiver at all - an associated
/// function like `Point::new`, called as `Point::new(...)`. `Value` is a by-value
/// `self`, `Pointer` is `*self` (a pointer receiver). Desugaring turns `Value`
/// into a leading `self: T` param and `Pointer` into `self: *T`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Receiver { Associated, Value, Pointer }

/// A method declared inside a struct/enum body or an `extend` block. This is a
/// purely front-end construct: the module resolver desugars every method into a
/// top-level [`TopLevelNode::Function`] named `Type$method` (prepending a `self`
/// param for `Value`/`Pointer` receivers) before any later stage runs, so
/// typecheck/mono/mil/codegen only ever see ordinary functions. `params` excludes
/// the receiver.
#[derive(Clone, Debug)]
pub struct MethodNode<'a> {
    pub is_pub: bool,
    pub attributes: Vec<Attribute<'a>>,
    pub receiver: Receiver,
    pub name: &'a str,
    pub generics: Vec<GenericParam<'a>>,
    pub params: Vec<(&'a str, Type<'a>)>,
    pub return_type: Type<'a>,
    pub body: Vec<Stmt<'a>>,
}

pub type Method<'a> = Metadata<MethodNode<'a>>;

/// One required method signature in a `trait` declaration: a method header with
/// no body, terminated by `;`. `params` excludes the receiver (recorded in
/// `receiver`). A `Self` in a param/return type refers to the implementing type;
/// the typechecker substitutes it (with the concrete type when checking
/// conformance, with the bounded type param when resolving a bounded call).
#[derive(Clone, Debug)]
pub struct TraitMethod<'a> {
    pub receiver: Receiver,
    pub name: &'a str,
    pub params: Vec<(&'a str, Type<'a>)>,
    pub return_type: Type<'a>,
}

/// A recorded `extend Target: Trait` conformance obligation, produced by the
/// module resolver (which desugars the block's methods to functions but keeps
/// this relation) and consumed by the typechecker, which verifies `Target`
/// implements every method of `Trait` and registers the impl so a `T: Trait`
/// bound can be checked at a generic call site. Names are final (post-mangling).
#[derive(Clone, Debug)]
pub struct ImplDecl<'a> {
    pub target: &'a str,
    pub trait_: &'a str,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub enum TopLevelNode<'a> {
    Function {
        name: &'a str,
        /// `true` if declared `pub`; controls whether other modules may import
        /// it. Default (no `pub`) is module-private. See `haven_front::module`.
        is_pub: bool,
        attributes: Vec<Attribute<'a>>,
        generics: Vec<GenericParam<'a>>,
        params: Vec<(&'a str, Type<'a>)>,
        return_type: Type<'a>,
        body: Vec<Stmt<'a>>,
    },
    Extern {
        name: &'a str,
        is_pub: bool,
        attributes: Vec<Attribute<'a>>,
        generics: Vec<GenericParam<'a>>,
        params: Vec<(&'a str, Type<'a>)>,
        return_type: Type<'a>,
    },

    Struct {
        name: &'a str,
        is_pub: bool,
        attributes: Vec<Attribute<'a>>,
        generics: Vec<GenericParam<'a>>,
        fields: Vec<(&'a str, Type<'a>)>,
    },

    /// A field-less, C-style enum: `enum Status { Continue, Sleep = 5, Error }`.
    /// Each variant is `(name, optional explicit discriminant, payload fields)`.
    /// Unspecified discriminants continue from the previous one + 1 (starting at
    /// 0), as in C. The discriminant's integer type is set by `@repr(<int>)` in
    /// `attributes` (default `i32`). A variant's payload is a list of `(field
    /// name, type)`: empty for a unit variant (`Stop`); for a tuple variant
    /// (`Note(u8, f32)`) the fields get synthesized names `"0"`, `"1"`, ...; for a
    /// struct variant (`Cc { id: u32 }`, Stage-3 Phase 2) the real names. A carried
    /// discriminant is only meaningful on a unit variant. `generics` (Stage-3
    /// Phase 3) declares type/const params in scope for every variant's payload
    /// field types, e.g. `enum Option<T> { None, Some(T) }`; empty for a plain enum.
    Enum {
        name: &'a str,
        is_pub: bool,
        attributes: Vec<Attribute<'a>>,
        generics: Vec<GenericParam<'a>>,
        variants: Vec<(&'a str, Option<i64>, Vec<(&'a str, Type<'a>)>)>,
    },

    /// A module-level constant, e.g. `const SR: f32 = 48000.0;`. The initializer
    /// must be a compile-time constant (literal or const struct literal); it is
    /// emitted as an LLVM `constant` global. `@export` gives it external linkage
    /// so a host can look the symbol up (see the CLAP `clap_entry` use case).
    Global {
        name: &'a str,
        is_pub: bool,
        attributes: Vec<Attribute<'a>>,
        ty: Type<'a>,
        value: Expr<'a>,
    },

    /// An `extend Type { ... }` (or `extend Type: Trait { ... }`) block adding
    /// methods to `target`. Inherent methods written directly in a struct/enum
    /// body are also parsed into one of these (with `trait_: None`). Trait
    /// conformance is not enforced in Stage 1 - `trait_` is recorded for later
    /// stages but otherwise ignored. The module resolver desugars every method
    /// into a top-level `Function` (see `lower_methods`) before typecheck, so no
    /// stage past front-end module resolution ever observes this variant.
    Extend {
        target: &'a str,
        trait_: Option<&'a str>,
        methods: Vec<Method<'a>>,
    },

    /// A `trait Name { proc m(*self) Ret; ... }` declaration: a set of required
    /// method signatures. Unlike `extend`, a trait is NOT desugared in the module
    /// resolver - it survives to the typechecker, which registers it, checks that
    /// every `extend T: Trait` conforms, and resolves a bounded type param's
    /// method calls through it. Monomorphization drops trait nodes (they emit no
    /// code); static dispatch falls out of substituting the concrete type and
    /// re-resolving the method call on the concrete instance.
    Trait {
        name: &'a str,
        is_pub: bool,
        methods: Vec<TraitMethod<'a>>,
    },
}

impl<'a> Display for TopLevelNode<'a> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            TopLevelNode::Function { name, is_pub, attributes, generics, params, return_type, body } => {
                let attrs_str = if attributes.is_empty() {
                    String::new()
                } else {
                    attributes.iter().map(|attr| attr.value.to_string()).collect::<Vec<_>>().join("\n") + "\n"
                };
                let pub_str = if *is_pub { "pub " } else { "" };
                let generics_str = fmt_generics(generics);
                let params_str = params.iter().map(|(name, ty)| format!("{}: {}", name, ty)).collect::<Vec<_>>().join(", ");
                let body_str = body.iter().map(|stmt| format!("    {}\n", stmt.value)).collect::<String>();

                write!(f, "{}{}proc {}{}({}) {} {{\n{}}}", attrs_str, pub_str, name, generics_str, params_str, return_type, body_str)
            },
            TopLevelNode::Extern { name, is_pub, attributes, generics, params, return_type } => {
                let attrs_str = if attributes.is_empty() {
                    String::new()
                } else {
                    attributes.iter().map(|attr| attr.value.to_string()).collect::<Vec<_>>().join("\n") + "\n"
                };
                let pub_str = if *is_pub { "pub " } else { "" };
                let generics_str = fmt_generics(generics);
                let params_str = params.iter().map(|(name, ty)| format!("{}: {}", name, ty)).collect::<Vec<_>>().join(", ");

                write!(f, "{}{}extern {}{}({}) {};", attrs_str, pub_str, name, generics_str, params_str, return_type)
            },
            TopLevelNode::Struct { name, is_pub, attributes, generics, fields } => {
                let attrs_str = if attributes.is_empty() {
                    String::new()
                } else {
                    attributes.iter().map(|attr| attr.value.to_string()).collect::<Vec<_>>().join("\n") + "\n"
                };
                let pub_str = if *is_pub { "pub " } else { "" };
                let generics_str = fmt_generics(generics);
                let fields_str = fields.iter().map(|(name, ty)| format!("    {}: {},\n", name, ty)).collect::<String>();

                write!(f, "{}{}struct {}{} {{\n{}}}", attrs_str, pub_str, name, generics_str, fields_str)
            },
            TopLevelNode::Global { name, is_pub, attributes, ty, value } => {
                let attrs_str = if attributes.is_empty() {
                    String::new()
                } else {
                    attributes.iter().map(|attr| attr.value.to_string()).collect::<Vec<_>>().join("\n") + "\n"
                };
                let pub_str = if *is_pub { "pub " } else { "" };

                write!(f, "{}{}const {}: {} = {};", attrs_str, pub_str, name, ty, value.value)
            },
            TopLevelNode::Enum { name, is_pub, attributes, generics, variants } => {
                let attrs_str = if attributes.is_empty() {
                    String::new()
                } else {
                    attributes.iter().map(|attr| attr.value.to_string()).collect::<Vec<_>>().join("\n") + "\n"
                };
                let pub_str = if *is_pub { "pub " } else { "" };
                let generics_str = fmt_generics(generics);
                let variants_str = variants.iter().map(|(vname, val, payload)| {
                    let payload_str = if payload.is_empty() {
                        String::new()
                    } else {
                        format!("({})", payload.iter().map(|(_, ty)| ty.to_string()).collect::<Vec<_>>().join(", "))
                    };
                    match val {
                        Some(v) => format!("    {}{} = {},\n", vname, payload_str, v),
                        None => format!("    {}{},\n", vname, payload_str),
                    }
                }).collect::<String>();

                write!(f, "{}{}enum {}{} {{\n{}}}", attrs_str, pub_str, name, generics_str, variants_str)
            },
            TopLevelNode::Extend { target, trait_, methods } => {
                let trait_str = match trait_ {
                    Some(t) => format!(": {}", t),
                    None => String::new(),
                };
                let methods_str = methods.iter().map(|m| {
                    let m = &m.value;
                    let recv = match m.receiver {
                        Receiver::Associated => String::new(),
                        Receiver::Value => "self".to_string(),
                        Receiver::Pointer => "*self".to_string(),
                    };
                    let sep = if !recv.is_empty() && !m.params.is_empty() { ", " } else { "" };
                    let params_str = m.params.iter()
                        .map(|(n, ty)| format!("{}: {}", n, ty)).collect::<Vec<_>>().join(", ");
                    format!("    proc {}({}{}{}) {} {{ ... }}\n", m.name, recv, sep, params_str, m.return_type)
                }).collect::<String>();
                write!(f, "extend {}{} {{\n{}}}", target, trait_str, methods_str)
            },
            TopLevelNode::Trait { name, is_pub, methods } => {
                let pub_str = if *is_pub { "pub " } else { "" };
                let methods_str = methods.iter().map(|m| {
                    let recv = match m.receiver {
                        Receiver::Associated => String::new(),
                        Receiver::Value => "self".to_string(),
                        Receiver::Pointer => "*self".to_string(),
                    };
                    let sep = if !recv.is_empty() && !m.params.is_empty() { ", " } else { "" };
                    let params_str = m.params.iter()
                        .map(|(n, ty)| format!("{}: {}", n, ty)).collect::<Vec<_>>().join(", ");
                    format!("    proc {}({}{}{}) {};\n", m.name, recv, sep, params_str, m.return_type)
                }).collect::<String>();
                write!(f, "{}trait {} {{\n{}}}", pub_str, name, methods_str)
            },
        }
    }
}

pub type TopLevel<'a> = Metadata<TopLevelNode<'a>>;

/// a module import, e.g. `import std/math` or `import std/math { sinf, cosf }`
///
/// kept out of `TopLevelNode` so the later stages (typecheck, mono, mil, ...)
/// never see imports: the module resolver eats every import, mangles + merges
/// the referenced modules, and hands those stages one flat program of concrete
/// items with no imports left
#[derive(Clone, Debug)]
pub struct Import<'a> {
    pub span: Span,
    /// path segments as written, e.g. `["std", "math"]` or `["utils", "foo"]`
    pub path: Vec<&'a str>,
    /// `None` = whole-module import (`import std/math`): every public symbol
    /// visible *only* qualified under the last path segment (`math::sinf`).
    /// `Some(list)` = selective (`import std/math { sinf }`): only those symbols,
    /// and visible *unqualified* (`sinf`).
    pub symbols: Option<Vec<&'a str>>,
}