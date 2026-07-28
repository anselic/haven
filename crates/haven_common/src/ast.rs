use std::fmt::{Display, Formatter};

use crate::defs::{DefId, TyHead};

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

/// A `::`-separated name exactly as written: `String`, `geo::Point`,
/// `Status::Ready`, `dsp::osc::Osc`.
///
/// The parser used to join these into one `&str` (via `Box::leak`) and every
/// consumer split them apart again — `split_once("::")`, which silently dropped
/// everything past the second segment. Keeping the segments means the *shape* of
/// a name survives parsing, so resolution can decide what each segment denotes
/// (module qualifier, type, value, enum variant) instead of guessing from a
/// string.
///
/// Pre-resolution only. Name resolution replaces every `Path` with what it
/// refers to; nothing downstream of `haven_front::module` should see one.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Path<'a> {
    /// At least one segment. `["geo", "Point"]` for `geo::Point`.
    pub segments: Vec<&'a str>,
}

impl<'a> Path<'a> {
    pub fn single(name: &'a str) -> Self {
        Path { segments: vec![name] }
    }

    /// The last segment — the name being referred to, ignoring any qualifiers.
    pub fn last(&self) -> &'a str {
        self.segments[self.segments.len() - 1]
    }

    /// The only segment, if this path is unqualified.
    pub fn as_single(&self) -> Option<&'a str> {
        match self.segments.as_slice() {
            [only] => Some(only),
            _ => None,
        }
    }

    /// The `(enum, variant)` of a two-segment path. After resolution every
    /// remaining qualified path is an enum variant, so this is the accessor the
    /// mid end uses — it replaces `split_once("::")`, which quietly treated
    /// `a::b::c` as `("a", "b::c")`.
    pub fn as_variant(&self) -> Option<(&'a str, &'a str)> {
        match self.segments.as_slice() {
            [e, v] => Some((*e, *v)),
            _ => None,
        }
    }

    /// This path with its head segment replaced — how resolution records that a
    /// qualified name's type or module part has been resolved.
    pub fn with_head(&self, head: &'a str) -> Self {
        let mut segments = self.segments.clone();
        segments[0] = head;
        Path { segments }
    }
}

impl<'a> Display for Path<'a> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.segments.join("::"))
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
    /// PRE-RESOLUTION ONLY: a named type as the parser saw it, before anything
    /// knows whether it denotes a struct, an enum or a type parameter — or even
    /// whether it exists. Name resolution rewrites every one of these into
    /// `Named` or `Param`, and errors if it can't; no stage after
    /// `haven_front::module` ever constructs or matches one.
    ///
    /// This is what used to be spelled `Struct { name }` in parser output, where
    /// "struct" was a lie roughly a third of the time and both typecheck and
    /// monomorphization had to re-disambiguate it independently.
    Path { path: Path<'a>, args: Vec<GenericArg<'a>> },
    /// FRONT + MID: a named type, identified by the definition it refers to.
    ///
    /// This is the resolved form of `Path`, and the only named type the resolver,
    /// the typechecker and monomorphization ever see. Whether it is a struct, an
    /// enum or an enum's synthetic payload struct is a property of the *definition*
    /// (`Defs::get(def).kind`), not of the type — which is the whole point: two
    /// modules may each declare a `Buf`, and a private `Option` in one module no
    /// longer reserves that name program-wide, because nothing compares names to
    /// decide whether two types are the same.
    ///
    /// `args` is always empty after monomorphization, which rewrites a generic use
    /// to a freshly minted instance `DefId` with no arguments.
    ///
    /// MIL lowering converts these to `Struct`/`Enum` (see `mil::ctx::lower_ty`);
    /// no stage after that seam sees a `Named`.
    Named { def: DefId, args: Vec<GenericArg<'a>> },
    /// A generic type parameter, e.g. `T`, and `Self` inside a trait method
    /// signature. Produced by name resolution for a path naming a type parameter
    /// of the enclosing item. It is abstract and must never survive to the
    /// codegen stage.
    Param(&'a str),
}

impl<'a> Type<'a> {
    /// A named type as written, before resolution.
    pub fn path(path: Path<'a>) -> Self {
        Type::Path { path, args: Vec::new() }
    }

    /// A resolved named type with no generic arguments - the common case, and
    /// the only shape that survives monomorphization.
    pub fn named(def: DefId) -> Self {
        Type::Named { def, args: Vec::new() }
    }

    /// The definition a resolved named type refers to, if it is one.
    pub fn def(&self) -> Option<DefId> {
        match self { Type::Named { def, .. } => Some(*def), _ => None }
    }

    /// Reject a [`Type::Path`] that reached a stage past name resolution. Every
    /// such site is a compiler bug, not a user error - resolution either rewrites
    /// a path or reports an unknown-type error, so nothing valid gets this far.
    pub fn unresolved(path: &Path<'a>) -> ! {
        panic!("unresolved type path `{path}` survived name resolution")
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
            Path { path, args } if args.is_empty() => write!(f, "{}", path),
            Path { path, args } => {
                let args_str = args.iter().map(|a| a.to_string()).collect::<Vec<_>>().join(", ");
                write!(f, "{}<{}>", path, args_str)
            },
            // a `Named` has no name to print without `Defs` in hand. Every
            // diagnostic that can reach one goes through `Context::show`, which
            // does have it; this fallback exists so `Debug`-ish uses and the
            // backend's `Display` keep working, and is deliberately ugly so a
            // missed site is obvious in output rather than merely wrong.
            Named { def, args } if args.is_empty() => write!(f, "#{}", def.0),
            Named { def, args } => {
                let args_str = args.iter().map(|a| a.to_string()).collect::<Vec<_>>().join(", ");
                write!(f, "#{}<{}>", def.0, args_str)
            },
            Param(name) => write!(f, "{}", name),
        }
    }
}

/// The bindings a successful [`unify`] produced: each of the pattern's free
/// parameters mapped to what the concrete type had in that position.
#[derive(Clone, Debug, Default)]
pub struct Unified<'a> {
    pub types: std::collections::HashMap<&'a str, Type<'a>>,
    pub consts: std::collections::HashMap<&'a str, ConstVal<'a>>,
}

/// Match `concrete` against `pattern`, whose free names are `params`, binding
/// each parameter to whatever `concrete` has in that position. `Vec<T>` against
/// `Vec<i32>` binds `T = i32`; `[T]` against `[[u8]]` binds `T = [u8]`.
///
/// This is what makes a structural `extend` dispatch: [`TyHead`] narrows a
/// receiver to one impl, and this recovers the arguments the head threw away.
/// One-directional by design - only the pattern may contain parameters, and a
/// parameter in `concrete` (an unsubstituted `T` inside a generic body) matches
/// nothing, which is correct: such a call is checked against the *bound*, not
/// against an impl.
///
/// A parameter appearing twice must bind consistently, so `extend Pair<T, T>`
/// rejects `Pair<i32, f32>`.
///
/// [`TyHead`]: crate::defs::TyHead
pub fn unify<'a>(
    pattern: &Type<'a>,
    concrete: &Type<'a>,
    params: &[&'a str],
    out: &mut Unified<'a>,
) -> bool {
    // a const position binds like a type one, but only a literal is concrete
    // enough to bind to.
    fn unify_cv<'a>(p: &ConstVal<'a>, c: &ConstVal<'a>, params: &[&'a str], out: &mut Unified<'a>) -> bool {
        match (p, c) {
            (ConstVal::Param(n), c) if params.contains(n) => match out.consts.get(n) {
                Some(prev) => prev == c,
                None => { out.consts.insert(n, c.clone()); true }
            },
            _ => p == c,
        }
    }
    match (pattern, concrete) {
        (Type::Param(n), c) if params.contains(n) => match out.types.get(n) {
            Some(prev) => prev == c,
            None => { out.types.insert(n, c.clone()); true }
        },
        (Type::Named { def: a, args: pa }, Type::Named { def: b, args: ca }) => {
            if a != b || pa.len() != ca.len() { return false; }
            pa.iter().zip(ca).all(|(p, c)| match (p, c) {
                (GenericArg::Type(p), GenericArg::Type(c)) => unify(p, c, params, out),
                (GenericArg::Const(p), GenericArg::Const(c)) => unify_cv(p, c, params, out),
                _ => false,
            })
        }
        (Type::Pointer(p), Type::Pointer(c)) | (Type::Slice(p), Type::Slice(c)) =>
            unify(p, c, params, out),
        (Type::Array(p, pn), Type::Array(c, cn)) | (Type::Simd(p, pn), Type::Simd(c, cn)) =>
            unify(p, c, params, out) && unify_cv(pn, cn, params, out),
        (Type::Function { params: pp, return_type: pr },
         Type::Function { params: cp, return_type: cr }) =>
            pp.len() == cp.len()
                && pp.iter().zip(cp).all(|(p, c)| unify(p, c, params, out))
                && unify(pr, cr, params, out),
        // scalars, `str`, `void`: no structure to descend into.
        (p, c) => p == c,
    }
}

/// A reference to a named type, in an expression or a pattern.
///
/// The parser records only what was written; `def` starts as
/// [`DefId::UNRESOLVED`] and name resolution replaces it with the identity of
/// the struct or enum the path names. `path` is kept afterwards for diagnostics
/// — and, for an `Enum::Variant` reference, its last segment is the variant
/// name, which stays a string because a variant is not a definition in its own
/// right: it has no symbol of its own, being a discriminant of its enum.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct NameRef<'a> {
    pub def: DefId,
    pub path: Path<'a>,
}

impl<'a> NameRef<'a> {
    /// As written by the parser, before resolution.
    pub fn new(path: Path<'a>) -> Self {
        NameRef { def: DefId::UNRESOLVED, path }
    }

    /// The variant name. Only meaningful once resolution has confirmed this is
    /// an `Enum::Variant` reference, which is the only two-segment form that
    /// survives it.
    pub fn variant(&self) -> &'a str { self.path.last() }
}

impl<'a> Display for NameRef<'a> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.path)
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
    /// An unqualified value reference: a local, a parameter, or (after
    /// resolution) a global or function under its resolved name.
    Var(&'a str),
    /// A qualified name, kept in segments rather than joined into one string.
    ///
    /// Before resolution this is any written path — `math::sinf`, `Point::new`,
    /// `Status::Ready` — and the parser makes no attempt to say which is which.
    /// Resolution rewrites it: a module-qualified value or associated function
    /// collapses to a `Var` under its resolved name, while an enum variant stays
    /// a two-segment `Path` whose first segment is now the enum's *resolved*
    /// name. So downstream, `Path` means exactly one thing — `Enum::Variant` —
    /// whose `def` is the enum and whose `variant()` is the variant.
    Path(NameRef<'a>),
    Slice(Vec<Expr<'a>>),

    Struct {
        /// The type being constructed: a one-segment path for a struct literal,
        /// or `[enum, variant]` for a struct-style variant literal. Either way
        /// resolution sets `def` to the named *type* — the struct, or the enum
        /// the variant belongs to.
        name: NameRef<'a>,
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
            ExprNode::Path(path) => write!(f, "{}", path),

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
    ///
    /// A bound names a trait, so it resolves to that trait's identity — which is
    /// what the conformance table is keyed by. Without that, a bound could only
    /// be checked by comparing trait *names*, and a `Display` declared in two
    /// modules would satisfy each other's bounds.
    Type { name: &'a str, bounds: Vec<NameRef<'a>> },
    /// A compile-time constant parameter, e.g. `const N: u32`.
    Const(&'a str, Type<'a>),
}

impl<'a> Display for GenericParam<'a> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            GenericParam::Type { name, bounds } if bounds.is_empty() => write!(f, "{}", name),
            GenericParam::Type { name, bounds } => {
                let bs = bounds.iter().map(|b| b.to_string()).collect::<Vec<_>>().join(" + ");
                write!(f, "{}: {}", name, bs)
            },
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
    /// a field-less enum-variant pattern, e.g. `Status::Continue`. Always the two
    /// segments `[enum, variant]`; resolution sets `def` to the enum.
    Path(NameRef<'a>),
    /// a data-carrying enum-variant pattern that destructures the payload, e.g.
    /// `Msg::Note(pitch, vel)`. `path` is `[enum, variant]`; `fields` is one
    /// sub-pattern per payload field (`Bind` to name it, `Wildcard` to ignore).
    Variant { path: NameRef<'a>, fields: Vec<Pattern<'a>> },
    /// a struct-style variant pattern that destructures a named payload by field,
    /// e.g. `Msg::Cc { id, val }` or `Msg::Cc { id: x, val: _ }`. Each entry is
    /// `(field_name, sub_pattern)`; binding is by field name, so order is free.
    /// The shorthand `{ id }` desugars to `(id, Bind(id))` at parse time.
    StructVariant { path: NameRef<'a>, fields: Vec<(&'a str, Pattern<'a>)> },
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
/// bound can be checked at a generic call site.
#[derive(Clone, Debug)]
pub struct ImplDecl<'a> {
    /// The implementing type as written, resolved: `i32`, `[T]`, `Vec<T>`. This
    /// is no longer a `DefId`, because a structural or primitive target has no
    /// definition to name — see [`TyHead`](crate::defs::TyHead).
    pub self_ty: Type<'a>,
    /// `self_ty`'s head, precomputed: what a candidate receiver is looked up by.
    pub head: TyHead,
    /// The impl's own type parameters (the `T` of `extend [T]`), inferred from
    /// the free names in `self_ty`. Empty for a fully concrete target.
    pub generics: Vec<GenericParam<'a>>,
    pub trait_: DefId,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub enum TopLevelNode<'a> {
    Function {
        name: &'a str,
        /// This item's identity, assigned by name resolution. Everything that
        /// needs to talk about this definition — its members, its instances, its
        /// fields, its conformances — keys on this rather than on `name`, which
        /// is only the symbol it happens to be emitted under.
        def: DefId,
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
        /// This item's identity, assigned by name resolution. Everything that
        /// needs to talk about this definition — its members, its instances, its
        /// fields, its conformances — keys on this rather than on `name`, which
        /// is only the symbol it happens to be emitted under.
        def: DefId,
        is_pub: bool,
        attributes: Vec<Attribute<'a>>,
        generics: Vec<GenericParam<'a>>,
        params: Vec<(&'a str, Type<'a>)>,
        return_type: Type<'a>,
    },

    Struct {
        name: &'a str,
        /// This item's identity, assigned by name resolution. Everything that
        /// needs to talk about this definition — its members, its instances, its
        /// fields, its conformances — keys on this rather than on `name`, which
        /// is only the symbol it happens to be emitted under.
        def: DefId,
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
        /// This item's identity, assigned by name resolution. Everything that
        /// needs to talk about this definition — its members, its instances, its
        /// fields, its conformances — keys on this rather than on `name`, which
        /// is only the symbol it happens to be emitted under.
        def: DefId,
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
        /// This item's identity, assigned by name resolution. Everything that
        /// needs to talk about this definition — its members, its instances, its
        /// fields, its conformances — keys on this rather than on `name`, which
        /// is only the symbol it happens to be emitted under.
        def: DefId,
        is_pub: bool,
        attributes: Vec<Attribute<'a>>,
        ty: Type<'a>,
        value: Expr<'a>,
    },

    /// An `extend Type { ... }` (or `extend Type: Trait { ... }`) block adding
    /// methods to `target`. Inherent methods written directly in a struct/enum
    /// body are also parsed into one of these (with `trait_: None`). The module
    /// resolver desugars every method into a top-level `Function` (see
    /// `lower_methods`) before typecheck, so no stage past front-end module
    /// resolution ever observes this variant.
    ///
    /// `target` is a full type, not a name: `i32`, `[T]` and `Vec<T>` are all
    /// extensible, so there is nothing to look up in a name table. Any type
    /// parameters it mentions are *inferred* rather than declared in a binder —
    /// see `lower_methods`'s `impl_generics`.
    Extend {
        target: Type<'a>,
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
        /// This item's identity, assigned by name resolution. Everything that
        /// needs to talk about this definition — its members, its instances, its
        /// fields, its conformances — keys on this rather than on `name`, which
        /// is only the symbol it happens to be emitted under.
        def: DefId,
        is_pub: bool,
        methods: Vec<TraitMethod<'a>>,
    },
}

impl<'a> Display for TopLevelNode<'a> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            TopLevelNode::Function { name, is_pub, attributes, generics, params, return_type, body, .. } => {
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
            TopLevelNode::Extern { name, is_pub, attributes, generics, params, return_type, .. } => {
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
            TopLevelNode::Struct { name, is_pub, attributes, generics, fields, .. } => {
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
            TopLevelNode::Global { name, is_pub, attributes, ty, value, .. } => {
                let attrs_str = if attributes.is_empty() {
                    String::new()
                } else {
                    attributes.iter().map(|attr| attr.value.to_string()).collect::<Vec<_>>().join("\n") + "\n"
                };
                let pub_str = if *is_pub { "pub " } else { "" };

                write!(f, "{}{}const {}: {} = {};", attrs_str, pub_str, name, ty, value.value)
            },
            TopLevelNode::Enum { name, is_pub, attributes, generics, variants, .. } => {
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
            TopLevelNode::Trait { name, is_pub, methods, .. } => {
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
    /// `pub import`: the imported symbols are also *re-exported*, so a module
    /// importing this one sees them as though they were declared here. Only
    /// meaningful on a selective import - a whole-module one binds a qualifier
    /// rather than any names, and re-exporting a qualifier needs module-level
    /// namespaces the resolver does not have yet.
    ///
    /// A re-export moves no code and mints no identity: the symbol keeps the
    /// definition, and so the emitted name, it already had. Only its visibility
    /// changes.
    pub is_pub: bool,
    /// `None` = whole-module import (`import std/math`): every public symbol
    /// visible *only* qualified under the last path segment (`math::sinf`).
    /// `Some(list)` = selective (`import std/math { sinf }`): only those symbols,
    /// and visible *unqualified* (`sinf`).
    pub symbols: Option<Vec<&'a str>>,
}