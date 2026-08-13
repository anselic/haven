use std::u64;
use chumsky::{
    input::MappedInput,
    pratt::*,
    prelude::*,
};
use haven_common::ast::*;
use haven_common::defs::DefId;

/// Field names for a tuple variant's payload: `Msg::Note(i32, i32)` gets fields
/// `"0"` and `"1"`, so a tuple variant and a struct-style variant share one
/// representation downstream. Indexing a fixed table keeps these `&'static str`
/// without leaking a fresh allocation per field, which is what the parser used
/// to do. A variant with more payload fields than this falls back to a leak.
/// Upper bound on a literal in const-argument position - an array length, a SIMD
/// lane count, a const generic argument. Literals lex as `i128` now that they
/// carry no width, so a size that would not survive the `as usize` narrowing has
/// to be rejected here rather than wrapping silently.
const MAX_CONST_ARG: i128 = u32::MAX as i128;

const TUPLE_FIELD_NAMES: [&str; 16] = [
    "0", "1", "2", "3", "4", "5", "6", "7",
    "8", "9", "10", "11", "12", "13", "14", "15",
];

fn tuple_field_name(i: usize) -> &'static str {
    TUPLE_FIELD_NAMES.get(i).copied()
        .unwrap_or_else(|| Box::leak(i.to_string().into_boxed_str()))
}

/// A `::`-separated name: one or more identifier segments, kept apart.
///
/// Written as a macro rather than a function because each use site has its own
/// `var` parser with its own inference-bound types, and spelling out a chumsky
/// parser's return type is far more noise than repeating four lines.
///
/// `repeated()` rewinds a partial match, so a trailing turbofish `::<...>` — not
/// an identifier — leaves the `::` for the caller's own alternative, exactly as
/// the previous `or_not()` form did.
macro_rules! path_of {
    ($var:expr) => {
        $var.map(|s| *s)
            .then(just(Token::ColonColon).ignore_then($var.map(|s| *s))
                .repeated().collect::<Vec<_>>())
            .map(|(head, rest): (&'src str, Vec<&'src str>)| {
                let mut segments = Vec::with_capacity(rest.len() + 1);
                segments.push(head);
                segments.extend(rest);
                Path { segments }
            })
    };
}

/// The payload tail of an `Enum::Variant` match pattern: positional `(a, b)` or
/// by-name `{ id, val }`. Only used to unify the two shapes under one `choice`
/// in the pattern parser before mapping to the corresponding `PatternNode`.
enum PatTail<'a> {
    Tuple(Vec<Pattern<'a>>),
    Struct(Vec<(&'a str, Pattern<'a>)>),
}

fn lexer<'a> (
    // a `FileId` is a Copy integer, so each token's span carries it for free -
    // this used to be an `Rc<str>` cloned per token, over a full filename.
    file: FileId,
)
-> impl Parser<
    'a,
    &'a str,
    Vec<Metadata<Token<'a>>>,
    extra::Err<Rich<'a, char>>,
> {
    macro_rules! try_parse_int {
        ($ty:ty, $val:expr, $span:expr) => {
            $val.parse::<$ty>()
                .map_err(|_| Rich::custom($span,
                    format!("failed to parse literal: {} does not fit in {}", $val, stringify!($ty))))
        };
    }

    // a suffix pins a literal's width exactly; without one the literal stays
    // width-less (`IntLit`/`FloatLit`) and the typechecker gives it the type its
    // context asks for, defaulting to 32 bits when there is none. That is why the
    // unsuffixed cases below do no range check - the target type is not known
    // yet, so `5000000000` is only too big once something asks for an `i32`.
    let float = text::int::<_, extra::Err<Rich<'a, char>>>(10)
        .then(just('.').then(text::digits(10)))
        .to_slice()
        .from_str::<f64>()
        .unwrapped()
        .then(just('f').or(just('F'))
            .ignore_then(text::int(10).from_str::<u32>().unwrapped())
            .or_not())
        .try_map(|(f, width), span| {
            match width {
                Some(32) => Ok(Token::Float32(f as f32)),
                Some(64) => Ok(Token::Float64(f)),
                None => Ok(Token::FloatLit(f)),
                Some(other) => Err(Rich::custom(span, format!("invalid float literal suffix 'f{other}'"))),
            }
        });

    let int = text::int(10)
        .to_slice()
        .then(
            just('i').or(just('I')).or(just('u')).or(just('U'))
                .map(|c| c.to_ascii_lowercase())
                .then(text::int(10).from_str::<u32>().unwrapped())
                .or_not()
        ).try_map(|(n, suffix): (&str, _), span| {
            match suffix {
                Some(('i', 8))  => try_parse_int!(i8, n, span).map(Token::Int8),
                Some(('i', 16)) => try_parse_int!(i16, n, span).map(Token::Int16),
                Some(('i', 32)) => try_parse_int!(i32, n, span).map(Token::Int32),
                Some(('i', 64)) => try_parse_int!(i64, n, span).map(Token::Int64),
                Some(('u', 8))  => try_parse_int!(u8, n, span).map(Token::Uint8),
                Some(('u', 16)) => try_parse_int!(u16, n, span).map(Token::Uint16),
                Some(('u', 32)) => try_parse_int!(u32, n, span).map(Token::Uint32),
                Some(('u', 64)) => try_parse_int!(u64, n, span).map(Token::Uint64),
                // `i128` is wide enough to hold every `i64` and `u64` the literal
                // could later be asked to be; anything past that has no possible
                // target type, so it is a lex error either way.
                None => try_parse_int!(i128, n, span).map(Token::IntLit),
                Some((other, width)) => Err(Rich::custom(span, format!("invalid integer literal suffix '{other}{width}'"))),
            }
        });

    // String literal: keep the raw inner text (escapes are resolved later in
    // MIL lowering). A backslash escapes the next char so that `\"` and `\\`
    // don't prematurely terminate the literal.
    let str_char = choice((
        just('\\').then(any()).ignored(), // escape pair, e.g. \n \" \\
        none_of("\\\"").ignored(),        // any ordinary char
    ));
    let str_ = str_char
        .clone()
        .repeated()
        .to_slice()
        .delimited_by(just('"'), just('"'))
        .map(Token::Str);

    // Interpolated string literal `f"...{expr}..."`. The leading `f` must sit
    // immediately against the opening quote; the inner text (including the
    // `{...}` holes) is captured raw exactly like `str_`, and the parser splits
    // and desugars it. Tried before `ident` so `f"..."` isn't lexed as the
    // variable `f` followed by a string; an ordinary ident like `foo` fails the
    // quote and backtracks to `ident`.
    let fstr = just('f')
        .ignore_then(
            str_char
                .repeated()
                .to_slice()
                .delimited_by(just('"'), just('"')),
        )
        .map(Token::FStr);

    let ident = text::ascii::ident().map(|ident: &str| match ident {
        "true"     => Token::Bool(true),
        "false"    => Token::Bool(false),
        "let"      => Token::Let,
        "if"       => Token::If,
        "else"     => Token::Else,
        "return"   => Token::Return,
        "while"    => Token::While,
        "for"      => Token::For,
        "break"    => Token::Break,
        "continue" => Token::Continue,
        "proc"     => Token::Proc,
        "extern"   => Token::Extern,
        "const"    => Token::Const,
        "struct"   => Token::Struct,
        "enum"     => Token::Enum,
        "match"    => Token::Match,
        "import"   => Token::Import,
        "pub"      => Token::Pub,
        _ => Token::Var(ident),
    });

    // chumsky have a built-in arity limit for choice (26, A-Z) so split these up
    let math = choice((
        just('|').then(just('|')).to(Token::BinaryOp(BinaryOp::Or)),
        just('&').then(just('&')).to(Token::BinaryOp(BinaryOp::And)),
        just('<').then(just('=')).to(Token::BinaryOp(BinaryOp::Le)),
        just('>').then(just('=')).to(Token::BinaryOp(BinaryOp::Ge)),
        just('=').then(just('=')).to(Token::BinaryOp(BinaryOp::Eq)),
        just('!').then(just('=')).to(Token::BinaryOp(BinaryOp::Ne)),
        just('-').then(just('>')).to(Token::Arrow),

        just('^').to(Token::BinaryOp(BinaryOp::BitXor)),
        just('|').to(Token::BinaryOp(BinaryOp::BitOr)),
        just('+').to(Token::BinaryOp(BinaryOp::Add)),
        just('-').to(Token::BinaryOp(BinaryOp::Sub)), // Map to unary neg in parsing
        just('*').to(Token::BinaryOp(BinaryOp::Mul)), // This too, deref in prefix position
        just('/').to(Token::BinaryOp(BinaryOp::Div)),
        just('%').to(Token::BinaryOp(BinaryOp::Mod)),
        just('<').to(Token::BinaryOp(BinaryOp::Lt)),
        just('>').to(Token::BinaryOp(BinaryOp::Gt)),

        just('!').to(Token::UnaryOp(UnaryOp::Not)),
        // one token, like `*`/Mul: infix bitwise-and, or prefix address-of.
        just('&').to(Token::BinaryOp(BinaryOp::BitAnd)),
    ));

    let delim = choice((
        just('(').to(Token::LParen),
        just(')').to(Token::RParen),
        just('{').to(Token::LBrace),
        just('}').to(Token::RBrace),
        just('[').to(Token::LBracket),
        just(']').to(Token::RBracket),

        just('.').to(Token::Dot),
        just(',').to(Token::Comma),
        just(';').to(Token::Semicolon),
        just(':').then(just(':')).to(Token::ColonColon), // before `:`
        just(':').to(Token::Colon),
        just('=').to(Token::Assign),
        just('@').to(Token::At),
    ));

    let token = float
        .or(int)
        .or(str_)
        .or(fstr)
        .or(ident)
        .or(math)
        .or(delim);

    let comment = just("//")
        .then(any().and_is(just('\n').not()).repeated())
        .padded();

    token
        .map_with(move |tok, e| {
            let input_span: SimpleSpan = e.span();

            Metadata::new(
                tok,
                Span::new(file, input_span.start, input_span.end),
            )
        })
        .padded_by(comment.repeated())
        .padded()
        .recover_with(skip_then_retry_until(any().ignored(), end()))
        .repeated()
        .collect()
}

pub fn lex<'a>(file: FileId, source: &'a str) -> (
    Option<Vec<Metadata<Token<'a>>>>,
    Vec<chumsky::error::Rich<'a, char, Span>>
) {
    let (tks, errs) = lexer(file)
        .parse(source)
        .into_output_errors();

    (tks, errs.into_iter()
        .map(|e| {
            e.map_span(|simple_span| {
                Span::new(file, simple_span.start, simple_span.end)
            })
        })
        .collect())
}

// --- f-string desugaring -----------------------------------------------------
//
// `f"a {x} b"` is lexed as one `Token::FStr` holding the raw inner text, then
// expanded here (at parse time) into ordinary `String`-building calls, so every
// downstream pass sees only plain AST and needs no f-string arm:
//
//     String::new().fstr_lit("a ").fstr_val(x.display()).fstr_lit(" b")
//
// `fstr_lit`/`fstr_val` are `String` methods (see std/string) that thread the
// owned accumulator through by value; `x.display()` does the per-type rendering
// via the `Display` trait. `String` and `Display` are both re-exported by the
// prelude, so `f"..."` needs no import. Interpolations are deliberately
// restricted to a variable, a `.field` access chain, or a literal (optionally
// negated) - any real computation must be bound to a `let` first, keeping the
// work out of the string.

/// One piece of a split f-string: literal text, or a built interpolation expr.
enum FStrPart<'a> {
    Lit(&'a str),
    Interp(Expr<'a>),
}

fn leak_str(s: String) -> &'static str {
    Box::leak(s.into_boxed_str())
}

/// Expand an f-string's raw inner text into the desugared `String`-building
/// `ExprNode`. Every synthesized node borrows the whole literal's `span`.
fn expand_fstring<'src>(raw: &'src str, span: Span) -> Result<ExprNode<'src>, String> {
    let mk = |node: ExprNode<'src>| Metadata::new(node, span);
    let path = |segments: Vec<&'src str>| {
        Metadata::new(ExprNode::Path(NameRef::new(Path { segments })), span)
    };
    let call = |func: Expr<'src>, args: Vec<Expr<'src>>| ExprNode::Call {
        func: Box::new(func),
        type_args: Vec::new(),
        args,
    };
    // a method call `recv.name(arg)`, used for the `String` chaining helpers and
    // for `expr.display()`.
    let method = |recv: Expr<'src>, name: &'src str, args: Vec<Expr<'src>>| {
        ExprNode::Call {
            func: Box::new(Metadata::new(
                ExprNode::Access { base: Box::new(recv), field: name },
                span,
            )),
            type_args: Vec::new(),
            args,
        }
    };

    // seed: `String::new()`. String and Display are both re-exported by the
    // prelude, and the `fstr_lit`/`fstr_val` chaining methods live on String, so
    // the whole expansion resolves with no import at the use site.
    let mut acc: Expr<'src> = mk(call(path(vec!["String", "new"]), Vec::new()));

    for part in split_fstring(raw, span)? {
        acc = match part {
            FStrPart::Lit(text) => {
                let chunk = mk(ExprNode::Str(text));
                mk(method(acc, "fstr_lit", vec![chunk]))
            }
            FStrPart::Interp(expr) => {
                // `expr.display()` -> owned String, then appended (and consumed).
                let rendered = mk(method(expr, "display", Vec::new()));
                mk(method(acc, "fstr_val", vec![rendered]))
            }
        };
    }

    Ok(acc.value)
}

/// Split raw f-string text into literal chunks and interpolations. `{{`/`}}`
/// escape literal braces; a `\`-escape is passed through verbatim (resolved with
/// the rest of the string's escapes during MIL lowering).
fn split_fstring<'src>(raw: &'src str, span: Span) -> Result<Vec<FStrPart<'src>>, String> {
    let mut parts = Vec::new();
    let mut lit = String::new();
    let mut chars = raw.char_indices().peekable();

    while let Some((idx, c)) = chars.next() {
        match c {
            '{' => {
                if matches!(chars.peek(), Some((_, '{'))) {
                    chars.next();
                    lit.push('{');
                    continue;
                }
                if !lit.is_empty() {
                    parts.push(FStrPart::Lit(leak_str(std::mem::take(&mut lit))));
                }
                let start = idx + 1;
                let mut end = None;
                for (j, cj) in chars.by_ref() {
                    if cj == '}' {
                        end = Some(j);
                        break;
                    }
                    if cj == '{' {
                        return Err("nested '{' in f-string interpolation".to_string());
                    }
                }
                let end = end.ok_or_else(||
                    "unterminated '{' in f-string; expected a closing '}'".to_string())?;
                parts.push(FStrPart::Interp(build_interp(&raw[start..end], span)?));
            }
            '}' => {
                if matches!(chars.peek(), Some((_, '}'))) {
                    chars.next();
                    lit.push('}');
                    continue;
                }
                return Err("unmatched '}' in f-string; write '}}' for a literal brace".to_string());
            }
            '\\' => {
                lit.push('\\');
                if let Some((_, n)) = chars.next() {
                    lit.push(n);
                }
            }
            _ => lit.push(c),
        }
    }

    if !lit.is_empty() {
        parts.push(FStrPart::Lit(leak_str(lit)));
    }
    Ok(parts)
}

fn is_numeric(t: &Token) -> bool {
    matches!(t,
        Token::Int8(_) | Token::Int16(_) | Token::Int32(_) | Token::Int64(_) |
        Token::Uint8(_) | Token::Uint16(_) | Token::Uint32(_) | Token::Uint64(_) |
        Token::Float32(_) | Token::Float64(_) |
        Token::IntLit(_) | Token::FloatLit(_))
}

fn literal_node<'src>(t: &Token<'src>) -> Option<ExprNode<'src>> {
    Some(match *t {
        Token::Bool(b) => ExprNode::Bool(b),
        Token::Int8(n) => ExprNode::Int8(n),
        Token::Int16(n) => ExprNode::Int16(n),
        Token::Int32(n) => ExprNode::Int32(n),
        Token::Int64(n) => ExprNode::Int64(n),
        Token::Uint8(n) => ExprNode::Uint8(n),
        Token::Uint16(n) => ExprNode::Uint16(n),
        Token::Uint32(n) => ExprNode::Uint32(n),
        Token::Uint64(n) => ExprNode::Uint64(n),
        Token::Float32(f) => ExprNode::Float32(f),
        Token::Float64(f) => ExprNode::Float64(f),
        Token::IntLit(n) => ExprNode::IntLit(n),
        Token::FloatLit(f) => ExprNode::FloatLit(f),
        Token::Str(s) => ExprNode::Str(s),
        _ => return None,
    })
}

/// Build the restricted interpolation expression from one `{...}` body: a
/// variable, a `.field` access chain, or a (possibly negated) literal.
fn build_interp<'src>(inner: &'src str, span: Span) -> Result<Expr<'src>, String> {
    let (toks, errs) = lex(span.file, inner);
    if !errs.is_empty() {
        return Err(format!("invalid f-string interpolation `{}`", inner.trim()));
    }
    let toks: Vec<Token<'src>> = toks.unwrap_or_default().into_iter().map(|m| m.value).collect();
    let mk = |node: ExprNode<'src>| Metadata::new(node, span);
    let unsupported = || Err(format!(
        "f-string interpolation `{}` must be a variable, field access, or literal - \
         bind a complex expression to a `let` first", inner.trim()));

    match toks.as_slice() {
        [] => Err("empty f-string interpolation `{}`".to_string()),
        [t] if literal_node(t).is_some() => Ok(mk(literal_node(t).unwrap())),
        [Token::BinaryOp(BinaryOp::Sub), t] if is_numeric(t) => Ok(mk(ExprNode::Unary {
            op: UnaryOp::Neg,
            operand: Box::new(mk(literal_node(t).unwrap())),
        })),
        [Token::Var(head), rest @ ..] => {
            let mut expr = mk(ExprNode::Path(NameRef::new(Path { segments: vec![*head] })));
            let mut it = rest.iter();
            while let Some(t) = it.next() {
                match (t, it.next()) {
                    (Token::Dot, Some(Token::Var(field))) => {
                        expr = mk(ExprNode::Access { base: Box::new(expr), field: *field });
                    }
                    _ => return unsupported(),
                }
            }
            Ok(expr)
        }
        _ => unsupported(),
    }
}

fn parse_expr<'tks, 'src: 'tks>()
-> impl Parser<
    'tks,
    MappedInput<'tks, Token<'src>, Span, &'tks [Metadata<Token<'src>>]>,
    Expr<'src>,
    extra::Err<Rich<'tks, Token<'src>, Span>>,
> {
    recursive(|expr| {
        let var = select_ref! { Token::Var(ident) => ident };

        // `::<T, N>` — the turbofish, shared by every site that takes one. The
        // leading `::` is what disambiguates the angle brackets from the `<`/`>`
        // comparison operators, so it is part of the production rather than the
        // caller's job; the sites differ only in what may follow.
        let turbofish = just(Token::ColonColon)
            .ignore_then(
                choice((
                    select! { Token::IntLit(n) => n }.try_map(|n, span| {
                        if !(0..=MAX_CONST_ARG).contains(&n) {
                            Err(Rich::custom(span, format!("const turbofish argument must be between 0 and {MAX_CONST_ARG}")))
                        } else {
                            Ok(GenericArg::Const(ConstVal::Lit(n as usize)))
                        }
                    }),
                    // a bare ident is ambiguous between a type and a forwarded
                    // const param; it parses as a type and is reclassified
                    // downstream once the callee's or type's kinds are known.
                    parse_type().map(GenericArg::Type),
                ))
                .separated_by(just(Token::Comma))
                .allow_trailing()
                .collect::<Vec<_>>()
                .delimited_by(
                    just(Token::BinaryOp(BinaryOp::Lt)),
                    just(Token::BinaryOp(BinaryOp::Gt))),
            )
            .boxed();

        // `(a, b, c)` — a call's argument list.
        let call_args = expr.clone()
            .separated_by(just(Token::Comma))
            .allow_trailing()
            .collect::<Vec<_>>()
            .delimited_by(just(Token::LParen), just(Token::RParen))
            .boxed();

        macro_rules! una {
            // Separate $from and $op because some operators (like '-') can be
            // both unary and binary
            ($from:expr, $op:expr, $precedence:expr) => {
                prefix($precedence, just($from), |_, v, e|
                    Metadata::new(
                        ExprNode::Unary {
                            op: $op,
                            operand: Box::new(v),
                        },
                        e.span(),
                    )
                )
            };
        }

        macro_rules! bin {
            ($op:expr, $precedence:expr) => {
                infix(left($precedence), just(Token::BinaryOp($op)), |x, _, y, e|
                    Metadata::new(
                        ExprNode::Binary {
                            op: $op,
                            left: Box::new(x),
                            right: Box::new(y),
                        },
                        e.span(),
                    )
                )
            };
        }

        // A shift operator (`<<`/`>>`) is two adjacent `<`/`>` tokens rather than a
        // dedicated token, so nested generics like `Vec<Option<*T>>` keep closing
        // on single `>`s. $tok is the single-angle op to double up; must be listed
        // before the matching `bin!(Lt/Gt)` so two angles are tried before one.
        macro_rules! shift {
            ($tok:expr, $op:expr, $precedence:expr) => {
                infix(left($precedence),
                    just(Token::BinaryOp($tok)).ignore_then(just(Token::BinaryOp($tok))),
                    |x, _, y, e|
                    Metadata::new(
                        ExprNode::Binary {
                            op: $op,
                            left: Box::new(x),
                            right: Box::new(y),
                        },
                        e.span(),
                    )
                )
            };
        }

        choice((
            select_ref! {
                Token::Bool(b)    => ExprNode::Bool(*b),
                Token::Int8(i)    => ExprNode::Int8(*i),
                Token::Int16(i)   => ExprNode::Int16(*i),
                Token::Int32(i)   => ExprNode::Int32(*i),
                Token::Int64(i)   => ExprNode::Int64(*i),
                Token::Uint8(u)   => ExprNode::Uint8(*u),
                Token::Uint16(u)  => ExprNode::Uint16(*u),
                Token::Uint32(u)  => ExprNode::Uint32(*u),
                Token::Uint64(u)  => ExprNode::Uint64(*u),
                Token::Float32(f) => ExprNode::Float32(*f),
                Token::Float64(f) => ExprNode::Float64(*f),
                Token::IntLit(n)  => ExprNode::IntLit(*n),
                Token::FloatLit(f)=> ExprNode::FloatLit(*f),
                Token::Str(s)     => ExprNode::Str(*s),
            },

            // an interpolated string `f"...{expr}..."`, desugared to String-
            // building calls right here so nothing downstream sees an f-string.
            select_ref! { Token::FStr(raw) => *raw }
                .try_map(|raw, span| expand_fstring(raw, span)
                    .map_err(|m| Rich::custom(span, m))),

            // Struct init, with an optional qualifier and an optional turbofish
            // for generic structs: `S { f: v }`, `geo::Point { f: v }`,
            // `Option::<i32> { f: v }`, or `mod::Option::<i32> { f: v }`. A first
            // `::Name` is a path segment (module qualifier, or the enum of a
            // struct-style variant literal); a `::<...>` is the generic turbofish,
            // disambiguating `<`/`>` from comparison operators as the call
            // turbofish does. The segment alternative rejects `::<` (not an ident)
            // and backtracks, so the turbofish still sees it.
            path_of!(var)
                .then(turbofish.clone().or_not().map(|t| t.unwrap_or_default()))
                .then(
                    // `field: value`, or the shorthand `field` (equivalent to
                    // `field: field` - a variable of the same name in scope).
                    var.map_with(|s, e| (*s, e.span()))
                        .then(just(Token::Colon).ignore_then(expr.clone()).or_not())
                        .map(|((name, span), value)| {
                            let value = value.unwrap_or_else(|| Metadata::new(
                                ExprNode::Path(NameRef::new(Path { segments: vec![name] })),
                                span,
                            ));
                            (name, value)
                        })
                        .separated_by(just(Token::Comma))
                        .allow_trailing()
                        .collect::<Vec<_>>()
                        .delimited_by(just(Token::LBrace), just(Token::RBrace))
                )
                .map(|((name, type_args), fields)| ExprNode::Struct {
                    name: NameRef::new(name),
                    type_args,
                    fields,
                }),

            // An associated function reached through a *generic* type, with the
            // turbofish on the type rather than the call: `Buf::<i32>::make()`,
            // `mod::Buf::<i32>::make()`, `Buf::<i32>::make::<U>()`.
            //
            // The whole call is built here, arguments included, because the
            // turbofish sits in the middle of the name: `path_of!` stops at
            // `Buf` (`::<` is not a segment), so a postfix operator would have
            // to graft the trailing segment back onto whatever it was applied
            // to - including expressions that are not names at all. Requiring
            // the leading path syntactically means that case cannot arise.
            //
            // Both turbofishes land in one `type_args` list, in written order.
            // That is exactly right: `extend` desugars each method to a function
            // whose generics are the impl's followed by the method's own, so
            // `Buf::<i32>::make::<U>()` is just `Buf$make::<i32, U>()`.
            path_of!(var)
                .then(turbofish.clone())
                .then(just(Token::ColonColon).ignore_then(var.map(|s| *s)))
                .then(turbofish.clone().or_not().map(|t| t.unwrap_or_default()))
                .then(call_args.clone())
                .map_with(|((((mut path, mut type_args), assoc), own_args), args), e| {
                    path.segments.push(assoc);
                    type_args.extend(own_args);
                    ExprNode::Call {
                        func: Box::new(Metadata::new(
                            ExprNode::Path(NameRef::new(path)), e.span())),
                        type_args,
                        args,
                    }
                }),

            // a variable, or a qualified ref `qualifier::symbol` (`math::sinf`,
            // `Point::new`, `Status::Ready`). The parser doesn't try to tell those
            // apart - it just records the segments and lets the module resolver
            // decide. A `::symbol` is only taken when followed by an ident, so a
            // turbofish `::<...>` is left for the call postfix.
            path_of!(var).map(|p| ExprNode::Path(NameRef::new(p))),
            expr.clone()
                .separated_by(just(Token::Comma))
                .allow_leading()
                .allow_trailing()
                .collect::<Vec<_>>()
                .delimited_by(just(Token::LBracket), just(Token::RBracket))
                .map(|inner| ExprNode::Slice(inner))
        ))

        .map_with(|node, e| {
            Metadata::new(
                node,
                e.span(),
            )
        })
        .boxed()
        .or(expr.clone().delimited_by(just(Token::LParen), just(Token::RParen)))

        .pratt((
            postfix(
                200,
                just(Token::Dot).ignore_then(var),
                |base, field: &&str, e| {
                    Metadata::new(
                        ExprNode::Access {
                            base: Box::new(base),
                            field: *field,
                        },
                        e.span(),
                    )
                }
            ),

            // calls, with an optional turbofish `::<T, N>` before the args.
            // `::` disambiguates from the `<`/`>` comparison operators.
            postfix(
                300,
                turbofish.clone()
                    .or_not()
                    .map(|t| t.unwrap_or_default())
                    .then(call_args.clone())
                    .boxed(),
                |func, (type_args, args), e|
                Metadata::new(
                    ExprNode::Call {
                        func: Box::new(func),
                        type_args,
                        args,
                    },
                    e.span(),
                ),
            ),

            // A generic function taken *by value* (a function pointer):
            // `entry_init::<Gain>` with no call following. `path_of!` rewinds the
            // trailing `::<...>`, and the call postfix above fails without a `(`,
            // so this postfix (listed after it, same precedence) claims the bare
            // turbofish. Only a name can carry one, so the base is always a `Path`.
            postfix(
                300,
                turbofish.clone(),
                |base: Expr<'src>, type_args, e| {
                    let name = match base.value {
                        ExprNode::Path(nr) => nr,
                        // unreachable in practice: only a path can precede `::<`.
                        _ => NameRef::new(Path { segments: Vec::new() }),
                    };
                    Metadata::new(ExprNode::FnRef { name, type_args }, e.span())
                },
            ),

            // Index
            postfix(
                290,
                expr.clone().delimited_by(just(Token::LBracket), just(Token::RBracket)),
                |slice, index, e| Metadata::new(
                    ExprNode::Index {
                        slice: Box::new(slice),
                        index: Box::new(index),
                    },
                    e.span(),
                ),
            ),

            una!(Token::BinaryOp(BinaryOp::Sub), UnaryOp::Neg, 190),
            una!(Token::UnaryOp(UnaryOp::Not), UnaryOp::Not, 190),
            una!(Token::BinaryOp(BinaryOp::Mul), UnaryOp::Deref, 190),
            una!(Token::BinaryOp(BinaryOp::BitAnd), UnaryOp::AddrOf, 190),

            bin!(BinaryOp::Mul, 180),
            bin!(BinaryOp::Div, 180),
            bin!(BinaryOp::Mod, 180),

            bin!(BinaryOp::Add, 170),
            bin!(BinaryOp::Sub, 170),

            // shifts bind tighter than comparison (C order). Listed before the
            // `<`/`>` comparisons so `<<`/`>>` win over a single angle bracket.
            shift!(BinaryOp::Lt, BinaryOp::Shl, 165),
            shift!(BinaryOp::Gt, BinaryOp::Shr, 165),

            bin!(BinaryOp::Lt,  160),
            bin!(BinaryOp::Gt,  160),
            bin!(BinaryOp::Le,  160),
            bin!(BinaryOp::Ge,  160),

            bin!(BinaryOp::Eq,  150),
            bin!(BinaryOp::Ne,  150),

            // bitwise, in C precedence: & tighter than ^ tighter than |
            bin!(BinaryOp::BitAnd, 145),
            bin!(BinaryOp::BitXor, 140),
            bin!(BinaryOp::BitOr,  135),

            bin!(BinaryOp::And, 130),
            bin!(BinaryOp::Or,  120),
        ))
    })
}

/// A size in a `[T; N]` / `simd<T, N>` type position, before it's validated into
/// a [`ConstVal`]. A literal keeps its raw `i32` so the bound checks (`> 0`,
/// `1..=64`) can run and report the offending value; an identifier is a const
/// generic parameter reference, validated later in typecheck.
enum SizeArg<'a> {
    Lit(i128),
    Param(&'a str),
}

/// One argument inside an angle-bracket list in type position, e.g. the parts of
/// `simd<f32, 4>` or `Option<i32>`. The parser can't yet tell a size from a type
/// param (both are bare idents), so a bare int is a [`GenArg::Size`] and anything
/// else parses as a [`GenArg::Ty`]; the dispatch in `parse_type` reinterprets
/// them per the head name (`simd` wants `<type, size>`; a struct wants types).
enum GenArg<'a> {
    Size(i128),
    Ty(Type<'a>),
}

// TODO: no fn-type production yet (fn-as-value), and no user generic-type
// production either, only the hardcoded `simd<T,N>` handles `name<...>`. so a
// generic type in type position (`x: List<T>`, nested turbofish `f::<Vec<i32>>`)
// won't parse. mangle_ty in mono.rs already collapses fn types to "fn", so once
// this lands watch the mangler collision noted there.
fn parse_type<'tks, 'src: 'tks>()
-> impl Parser<
    'tks,
    MappedInput<'tks, Token<'src>, Span, &'tks [Metadata<Token<'src>>]>,
    Type<'src>,
    extra::Err<Rich<'tks, Token<'src>, Span>>,
> {
    recursive(|ty| {
        let var = select_ref! { Token::Var(ident) => ident };

        choice((
            // `!` - the never/bottom type, as a return type of a diverging proc
            // (`proc panic() ! { abort(...) }`). The `!` token starts no other
            // type, so this alternative is unambiguous.
            just(Token::UnaryOp(UnaryOp::Not)).to(Type::Never),
            // proc(T1, T2) R
            // starts no other type, so this alternative is unambiguous
            just(Token::Proc)
                .ignore_then(
                    ty.clone()
                        .separated_by(just(Token::Comma))
                        .allow_trailing()
                        .collect::<Vec<_>>()
                        .delimited_by(just(Token::LParen), just(Token::RParen))
                )
                .then(ty.clone().or_not())
                .map(|(params, ret)| Type::Function {
                    params,
                    return_type: Box::new(ret.unwrap_or(Type::Void)),
                }),
            // [T; N]
            just(Token::LBracket)
                .ignore_then(ty.clone())
                .then_ignore(just(Token::Semicolon))
                .then(select! {
                    Token::IntLit(x) => SizeArg::Lit(x),
                    Token::Var(n) => SizeArg::Param(n),
                })
                .then_ignore(just(Token::RBracket))
                .try_map(|(inner, size), span| match size {
                    SizeArg::Lit(x) if x > 0 && x <= MAX_CONST_ARG => Ok(Type::Array(Box::new(inner), ConstVal::Lit(x as usize))),
                    SizeArg::Lit(x) => Err(Rich::custom(span, format!("invalid array size parameter: {x} (must be between 1 and {MAX_CONST_ARG})"))),
                    SizeArg::Param(n) => Ok(Type::Array(Box::new(inner), ConstVal::Param(n))),
                }),
            // [T]
            just(Token::LBracket)
                .ignore_then(ty.clone())
                .then_ignore(just(Token::RBracket))
                .map(|inner| Type::Slice(Box::new(inner))),
            // a named type, optionally with angle-bracket arguments:
            //   scalar/struct: `i32`, `Vec2`
            //   qualified struct: `geo::Point` (from a whole-module import)
            //   simd:          `simd<f32, 4>`  (element type + lane count)
            //   generic struct: `Option<i32>`, `Pair<K, V>`
            // `<` is unambiguous here - type position has no comparison operators.
            // a leading `qualifier::` stays a separate segment, resolved by the
            // module resolver.
            path_of!(var)
            .then(
                choice((
                    select! { Token::IntLit(x) => GenArg::Size(x) },
                    ty.clone().map(GenArg::Ty),
                ))
                .separated_by(just(Token::Comma))
                .allow_trailing()
                .at_least(1)
                .collect::<Vec<_>>()
                .delimited_by(
                    just(Token::BinaryOp(BinaryOp::Lt)),
                    just(Token::BinaryOp(BinaryOp::Gt)))
                .or_not())
            .try_map(|(path, args), span| {
                // the built-in names are keywords, so they only ever appear
                // unqualified - `geo::i32` names a type called `i32` in `geo`.
                let scalar = match path.as_single() {
                    Some("void") => Some(Type::Void),
                    Some("bool") => Some(Type::Bool),
                    Some("i8")   => Some(Type::Int8),
                    Some("i16")  => Some(Type::Int16),
                    Some("i32")  => Some(Type::Int32),
                    Some("i64")  => Some(Type::Int64),
                    Some("u8")   => Some(Type::Uint8),
                    Some("u16")  => Some(Type::Uint16),
                    Some("u32")  => Some(Type::Uint32),
                    Some("u64")  => Some(Type::Uint64),
                    Some("f32")  => Some(Type::Float32),
                    Some("f64")  => Some(Type::Float64),
                    Some("str")  => Some(Type::Str),
                    _ => None,
                };
                let Some(args) = args else {
                    return Ok(scalar.unwrap_or(Type::Path { path, args: Vec::new() }));
                };
                // `simd<element, lanes>`: exactly a type then a size.
                if path.as_single() == Some("simd") {
                    if args.len() != 2 {
                        return Err(Rich::custom(span, format!("simd<...> takes 2 arguments (element type, lane count), got {}", args.len())));
                    }
                    let mut it = args.into_iter();
                    let elem = match it.next().unwrap() {
                        GenArg::Ty(t) => t,
                        GenArg::Size(_) => return Err(Rich::custom(span, "simd<...> element (first argument) must be a type")),
                    };
                    // the lane count is a literal, or a bare ident naming a
                    // const generic param (parsed as an unqualified named type).
                    let size = match it.next().unwrap() {
                        GenArg::Size(x) if x > 0 && x <= 64 => ConstVal::Lit(x as usize),
                        GenArg::Size(x) => return Err(Rich::custom(span, format!("invalid SIMD size parameter: {x} (must be between 1 and 64)"))),
                        GenArg::Ty(Type::Path { ref path, ref args }) if args.is_empty() && path.as_single().is_some() =>
                            ConstVal::Param(path.as_single().unwrap()),
                        GenArg::Ty(_) => return Err(Rich::custom(span, "simd<...> lane count (second argument) must be an integer or a const parameter")),
                    };
                    return Ok(Type::Simd(Box::new(elem), size));
                }
                // any other head is a generic named type. arguments are types or
                // const values (`Buf<i32, 8>`); a bare ident stays a type and is
                // reclassified downstream if the type declares it `const`.
                let mut gargs = Vec::with_capacity(args.len());
                for a in args {
                    gargs.push(match a {
                        GenArg::Ty(t) => GenericArg::Type(t),
                        GenArg::Size(x) if (0..=MAX_CONST_ARG).contains(&x) => GenericArg::Const(ConstVal::Lit(x as usize)),
                        GenArg::Size(x) => return Err(Rich::custom(span, format!("const argument '{x}' in generic type '{path}<...>' must be between 0 and {MAX_CONST_ARG}"))),
                    });
                }
                Ok(Type::Path { path, args: gargs })
            })
        ))
        .boxed()
        .pratt((
            // [T]
            // prefix(1, just(Token::LBracket).then(just(Token::RBracket)), |_, t, _| Type::Slice(Box::new(t))),
            // *T
            prefix(1, just(Token::BinaryOp(BinaryOp::Mul)), |_, t, _| Type::Pointer(Box::new(t))),
        ))
        .labelled("type")
    })
}

/// One postfix step of a `for`-iterand place expression: `.field` or `[index]`.
enum PlacePostfix<'a> {
    Field(&'a str),
    Index(Expr<'a>),
}

fn parse_stmt<'tks, 'src: 'tks>()
-> impl Parser<
    'tks,
    MappedInput<'tks, Token<'src>, Span, &'tks [Metadata<Token<'src>>]>,
    Stmt<'src>,
    extra::Err<Rich<'tks, Token<'src>, Span>>,
> {
    recursive(|stmt| {
        let var = select_ref! { Token::Var(ident) => ident };

        let single_stmt_or_block = stmt.clone()
            .repeated()
            .collect::<Vec<_>>()
            .delimited_by(just(Token::LBrace), just(Token::RBrace))
            .map_with(|node, e| Metadata::new(
                StmtNode::Block(node),
                e.span(),
            ))
            .or(stmt.clone());

        let declare = just(Token::Let)
            .ignore_then(var)
            .then_ignore(just(Token::Colon))
            .then(parse_type())
            .then_ignore(just(Token::Assign))
            .then(parse_expr())
            .then_ignore(just(Token::Semicolon))
            .map(|((name, ty), value)| StmtNode::Declare {
                name,
                ty,
                value,
            });

        let assign_or_expr = parse_expr()
            .then(just(Token::Assign)
                .ignore_then(parse_expr())
                .or_not())
            .then_ignore(just(Token::Semicolon))
            .map(|(left, value)| match value {
                Some(value) => StmtNode::Assign { left, value },
                None => StmtNode::Expr(left),
            });

        let if_ = just(Token::If)
            .ignore_then(parse_expr()
                .delimited_by(just(Token::LParen), just(Token::RParen)))
            .then(single_stmt_or_block.clone())
            .map(|(condition, then_branch)| StmtNode::If {
                condition,
                then_branch: Box::new(then_branch),
                else_branch: None,
            });

        let if_else = just(Token::If)
            .ignore_then(parse_expr()
                .delimited_by(just(Token::LParen), just(Token::RParen)))
            .then(single_stmt_or_block.clone())
            .then(just(Token::Else)
                .ignore_then(single_stmt_or_block.clone())
            )
            .map(|((condition, then_branch), else_branch)| StmtNode::If {
                condition,
                then_branch: Box::new(then_branch),
                else_branch: Some(Box::new(else_branch)),
            });

        let while_ = just(Token::While)
            .ignore_then(parse_expr()
                .delimited_by(just(Token::LParen), just(Token::RParen)))
            .then(single_stmt_or_block.clone())
            .map(|(condition, body)| StmtNode::While {
                condition,
                body: Box::new(body),
            });

        // The `for` iterand is a *place* expression: a variable followed by any
        // chain of `.field` / `[index]`. Restricting the grammar here (rather than
        // parsing a full expression) does double duty: it structurally forbids the
        // `Name { ... }` struct-literal reading of `for x in it { ... }` — the same
        // ambiguity `if`/`while`/`match` avoid with parentheses — and it enforces
        // that the iterand is a stable place, since the desugar re-evaluates it as
        // the `.next()` receiver every iteration.
        let place = var
            .map_with(|s, e| Metadata::new(ExprNode::Var(*s), e.span()))
            .then(
                choice((
                    just(Token::Dot).ignore_then(var.map(|s| *s)).map(PlacePostfix::Field),
                    parse_expr()
                        .delimited_by(just(Token::LBracket), just(Token::RBracket))
                        .map(PlacePostfix::Index),
                )).repeated().collect::<Vec<_>>()
            )
            .map_with(|(base, ops), e| {
                let span = e.span();
                ops.into_iter().fold(base, |base, op| match op {
                    PlacePostfix::Field(field) => Metadata::new(
                        ExprNode::Access { base: Box::new(base), field }, span),
                    PlacePostfix::Index(index) => Metadata::new(
                        ExprNode::Index { slice: Box::new(base), index: Box::new(index) }, span),
                })
            });

        // A trailing `(...)` means the iterand is a call (`mk()`, `v.iter()`) — a
        // fresh iterator each time the loop re-evaluates it. Catch it here for a
        // targeted message instead of a bare "unexpected `(`" from the body parser.
        let iterand = place
            .then(
                parse_expr()
                    .separated_by(just(Token::Comma))
                    .allow_trailing()
                    .collect::<Vec<_>>()
                    .delimited_by(just(Token::LParen), just(Token::RParen))
                    .or_not(),
            )
            .try_map(|(place, call), span| match call {
                None => Ok(place),
                Some(_) => Err(Rich::custom(span, "the `for` iterator cannot be a call \
                    like `v.iter()`; bind it to a `let` first, then `for x in it`")),
            });

        // `for x in <place> <body>`, desugared here into the `Iterator` protocol:
        //
        //     while (true) {
        //         match (<place>.next()) {
        //             Option::Some(x) -> <body>
        //             Option::None    -> break;
        //         }
        //     }
        //
        // `Option` is left unqualified: it resolves to whichever `Option` is in
        // scope (std's, or a module's own), matching the enum the iterator's `next`
        // actually returns.
        let for_ = just(Token::For)
            .ignore_then(var.map(|s| *s))
            .then_ignore(select_ref! { Token::Var(s) if *s == "in" => () })
            .then(iterand)
            .then(single_stmt_or_block.clone())
            .map_with(|((loop_var, iter), body), e| {
                let span = e.span();
                let opt_pat = |variant, fields| Metadata::new(
                    PatternNode::Variant {
                        path: NameRef::new(Path { segments: vec!["Option", variant] }),
                        fields,
                    },
                    span,
                );
                let some_arm = (
                    opt_pat("Some", vec![Metadata::new(PatternNode::Bind(loop_var), span)]),
                    Box::new(body),
                );
                let none_arm = (
                    Metadata::new(PatternNode::Path(
                        NameRef::new(Path { segments: vec!["Option", "None"] })), span),
                    Box::new(Metadata::new(StmtNode::Break, span)),
                );
                // `<iter>.next()`
                let next_call = Metadata::new(ExprNode::Call {
                    func: Box::new(Metadata::new(
                        ExprNode::Access { base: Box::new(iter), field: "next" }, span)),
                    type_args: Vec::new(),
                    args: Vec::new(),
                }, span);
                let match_stmt = Metadata::new(StmtNode::Match {
                    scrutinee: next_call,
                    arms: vec![some_arm, none_arm],
                }, span);
                StmtNode::While {
                    condition: Metadata::new(ExprNode::Bool(true), span),
                    body: Box::new(match_stmt),
                }
            });

        // a match pattern: `Enum::Variant`, a data-variant destructure
        // `Enum::Variant(binding, ...)`, an integer literal (opt. negative), or
        // `_`. A binding field is a name (`Bind`) or `_` (ignore the field).
        let int_pat = just(Token::BinaryOp(BinaryOp::Sub)).or_not()
            .then(select_ref! {
                Token::Int8(n)   => *n as i128,
                Token::Int16(n)  => *n as i128,
                Token::Int32(n)  => *n as i128,
                Token::Int64(n)  => *n as i128,
                Token::Uint8(n)  => *n as i128,
                Token::Uint16(n) => *n as i128,
                Token::Uint32(n) => *n as i128,
                Token::Uint64(n) => *n as i128,
                Token::IntLit(n) => *n,
            })
            .try_map(|(neg, n), span| {
                let n = if neg.is_some() { -n } else { n };
                i64::try_from(n)
                    .map(PatternNode::Int)
                    .map_err(|_| Rich::custom(span, format!("integer pattern {n} is out of range")))
            });
        // each payload sub-pattern is Metadata-wrapped so a `Bind` carries a
        // unique node id (its binding identity, like a `Declare`'s local).
        let field_pat = var.map_with(|s, e| {
            let node = if *s == "_" { PatternNode::Wildcard } else { PatternNode::Bind(*s) };
            Metadata::new(node, e.span())
        });
        let variant_tail = field_pat.clone()
            .separated_by(just(Token::Comma))
            .allow_trailing()
            .at_least(1)
            .collect::<Vec<_>>()
            .delimited_by(just(Token::LParen), just(Token::RParen));
        // a struct-style destructure field: `field` (shorthand, binds `field`),
        // `field: name`, or `field: _`. The shorthand's `Bind` gets its own node id
        // (binding identity) from the field-name span.
        let struct_field_pat = var.map_with(|s, e| (*s, e.span()))
            .then(just(Token::Colon).ignore_then(field_pat.clone()).or_not())
            .map(|((fname, fspan), sub)| {
                let pat = sub.unwrap_or_else(|| Metadata::new(PatternNode::Bind(fname), fspan));
                (fname, pat)
            });
        let struct_variant_tail = struct_field_pat
            .separated_by(just(Token::Comma))
            .allow_trailing()
            .at_least(1)
            .collect::<Vec<_>>()
            .delimited_by(just(Token::LBrace), just(Token::RBrace));
        // after `Enum::Variant`, a `(...)` tail destructures positionally, a `{...}`
        // tail destructures by field name, and neither is a field-less `Path`.
        let path_pat = path_of!(var)
            .filter(|p: &Path| p.segments.len() >= 2)
            .then(choice((
                variant_tail.map(PatTail::Tuple),
                struct_variant_tail.map(PatTail::Struct),
            )).or_not())
            .map(|(path, tail)| {
                let path = NameRef::new(path);
                match tail {
                    Some(PatTail::Tuple(fields))  => PatternNode::Variant { path, fields },
                    Some(PatTail::Struct(fields)) => PatternNode::StructVariant { path, fields },
                    None => PatternNode::Path(path),
                }
            });
        let wild_pat = var.try_map(|s, span| if *s == "_" {
            Ok(PatternNode::Wildcard)
        } else {
            Err(Rich::custom(span, "expected `_`, an enum variant `Enum::Variant`, or an integer literal in a match pattern"))
        });
        let pattern = choice((path_pat, int_pat, wild_pat))
            .map_with(|p, e| Metadata::new(p, e.span()));

        // `match (scrutinee) { pattern => body ... }`. Parens on the scrutinee
        // mirror `if`/`while` and avoid the `Name { ... }` struct-literal ambiguity.
        let match_ = just(Token::Match)
            .ignore_then(parse_expr().delimited_by(just(Token::LParen), just(Token::RParen)))
            .then(
                pattern
                    .then_ignore(just(Token::Arrow))
                    .then(single_stmt_or_block.clone().map(Box::new))
                    .repeated()
                    .at_least(1)
                    .collect::<Vec<_>>()
                    .delimited_by(just(Token::LBrace), just(Token::RBrace))
            )
            .map(|(scrutinee, arms)| StmtNode::Match { scrutinee, arms });

        let contbreak = choice((
            just(Token::Continue).to(StmtNode::Continue),
            just(Token::Break).to(StmtNode::Break),
        )).then_ignore(just(Token::Semicolon));

        let return_ = just(Token::Return)
            .ignore_then(parse_expr())
            .then_ignore(just(Token::Semicolon))
            .map(StmtNode::Return);

        declare
            .or(assign_or_expr)
            .or(if_else)
            .or(if_)
            .or(while_)
            .or(for_)
            .or(match_)
            .or(contbreak)
            .or(return_)
            .map_with(|node, e| {
                Metadata::new(
                    node,
                    e.span(),
                )
            })
            .boxed()
    })
}

/// Everything in an attribute after the sigil: `name` and an optional
/// `(value)`. Shared so that the item and module spellings cannot drift into
/// accepting different values.
fn attribute_tail<'tks, 'src: 'tks>()
-> impl Parser<
    'tks,
    MappedInput<'tks, Token<'src>, Span, &'tks [Metadata<Token<'src>>]>,
    AttributeNode<'src>,
    extra::Err<Rich<'tks, Token<'src>, Span>>,
> {
    select_ref! { Token::Var(ident) => ident }
        .then(
            just(Token::LParen)
                .ignore_then(select_ref! {
                    Token::Var(s) => s.to_string(),
                    Token::Bool(b) => if *b { "true" } else { "false" }.to_string(),
                    Token::IntLit(i) => i.to_string(),
                    // Token::Str(s) => s.to_string()
                })
                .then_ignore(just(Token::RParen))
                .or_not()
        )
        .map(|(name, value)| AttributeNode::new(name, value))
        .boxed()
}

/// An attribute on the item that follows it: `@name` or `@name(value)`.
fn parse_attribute<'tks, 'src: 'tks>()
-> impl Parser<
    'tks,
    MappedInput<'tks, Token<'src>, Span, &'tks [Metadata<Token<'src>>]>,
    Attribute<'src>,
    extra::Err<Rich<'tks, Token<'src>, Span>>,
> {
    just(Token::At)
        .ignore_then(attribute_tail())
        .map_with(|attr, e| Metadata::new(attr, e.span()))
        .boxed()
}

/// An attribute on the *enclosing module*: `@!name`. Says something about the
/// file it appears in rather than about any declaration, which is what the `!`
/// marks - an ordinary `@name` sitting at file scope would silently attach to
/// whatever item came next, so the two spellings have to be distinguishable
/// before the file's items are even known.
///
/// It parses anywhere an item may appear. Convention is the top of the file,
/// but position carries no meaning: the statement is about the whole module.
fn parse_mod_attribute<'tks, 'src: 'tks>()
-> impl Parser<
    'tks,
    MappedInput<'tks, Token<'src>, Span, &'tks [Metadata<Token<'src>>]>,
    Attribute<'src>,
    extra::Err<Rich<'tks, Token<'src>, Span>>,
> {
    just(Token::At)
        .ignore_then(just(Token::UnaryOp(UnaryOp::Not)))
        .ignore_then(attribute_tail())
        .map_with(|attr, e| Metadata::new(attr, e.span()))
        .boxed()
}

/// A method inside a struct/enum body or an `extend` block:
/// `[attrs] [pub] proc name[<generics>](receiver?, params...) [RetType] { body }`.
/// The receiver is `self` (by value), `*self` (by pointer), or absent (an
/// associated function). `proc` heads every method, so it cleanly delimits methods
/// from the comma-separated fields/variants that precede them in a type body.
fn parse_method<'tks, 'src: 'tks>()
-> impl Parser<
    'tks,
    MappedInput<'tks, Token<'src>, Span, &'tks [Metadata<Token<'src>>]>,
    Method<'src>,
    extra::Err<Rich<'tks, Token<'src>, Span>>,
> {
    let var = select_ref! { Token::Var(ident) => ident };

    let generic_param = choice((
        just(Token::Const)
            .ignore_then(var.map(|s| *s))
            .then_ignore(just(Token::Colon))
            .then(parse_type())
            .map(|(name, ty)| GenericParam::Const(name, ty)),
        // a bare type param, optionally with trait bounds: `T`, `T: Display`,
        // `T: A + B`. bounds are trait names joined by `+` (the `Add` op token).
        var.map(|s| *s)
            .then(
                just(Token::Colon)
                    .ignore_then(
                        var.map(|s| NameRef::new(Path::single(*s)))
                            .separated_by(just(Token::BinaryOp(BinaryOp::Add)))
                            .at_least(1)
                            .collect::<Vec<_>>())
                    .or_not())
            .map(|(name, bounds)| GenericParam::Type { name, bounds: bounds.unwrap_or_default() }),
    ));
    let generics = generic_param
        .separated_by(just(Token::Comma))
        .allow_trailing()
        .collect::<Vec<_>>()
        .delimited_by(
            just(Token::BinaryOp(BinaryOp::Lt)),
            just(Token::BinaryOp(BinaryOp::Gt)))
        .or_not()
        .map(|g| g.unwrap_or_default());

    let normal_param = var.map(|s| *s)
        .then_ignore(just(Token::Colon))
        .then(parse_type())
        .boxed();

    // `self` or `*self`. A guard keeps the token uncommitted when it isn't
    // literally `self`, so the associated-function alternative below backtracks
    // cleanly on a normal first param like `(x: i32)`.
    let self_recv = just(Token::BinaryOp(BinaryOp::Mul)).or_not()
        .then(select_ref! { Token::Var(s) if *s == "self" => () })
        .map(|(star, _)| if star.is_some() { Receiver::Pointer } else { Receiver::Value });

    let params_inner = choice((
        self_recv
            .then(just(Token::Comma).ignore_then(normal_param.clone()).repeated().collect::<Vec<_>>())
            .map(|(recv, ps)| (recv, ps)),
        normal_param
            .separated_by(just(Token::Comma))
            .allow_trailing()
            .collect::<Vec<_>>()
            .map(|ps| (Receiver::Associated, ps)),
    ))
        .delimited_by(just(Token::LParen), just(Token::RParen));

    parse_attribute()
        .repeated().collect::<Vec<_>>()
        .then(just(Token::Pub).or_not().map(|o| o.is_some()))
        .then_ignore(just(Token::Proc))
        .then(var.map(|s| *s))
        .then(generics)
        .then(params_inner)
        .then(parse_type().or_not().map(|t| t.unwrap_or(Type::Void)))
        .then(
            parse_stmt()
                .repeated()
                .collect::<Vec<_>>()
                .delimited_by(just(Token::LBrace), just(Token::RBrace))
        )
        .map_with(|((((((attributes, is_pub), name), generics), (receiver, params)), return_type), body), e|
            Metadata::new(
                MethodNode { is_pub, attributes, receiver, name, generics, params, return_type, body },
                e.span(),
            ))
        .boxed()
}

/// One required method signature inside a `trait` declaration:
/// `proc name(receiver?, params...) [RetType];` - a method header terminated by
/// `;` instead of a `{ body }`. Mirrors `parse_method`'s receiver/param shapes
/// but omits attributes, `pub`, generics and the body (a trait states signatures
/// only; conformance and default methods are the typechecker's / a later stage's
/// concern).
fn parse_trait_method<'tks, 'src: 'tks>()
-> impl Parser<
    'tks,
    MappedInput<'tks, Token<'src>, Span, &'tks [Metadata<Token<'src>>]>,
    TraitMethod<'src>,
    extra::Err<Rich<'tks, Token<'src>, Span>>,
> {
    let var = select_ref! { Token::Var(ident) => ident };

    let normal_param = var.map(|s| *s)
        .then_ignore(just(Token::Colon))
        .then(parse_type())
        .boxed();

    let self_recv = just(Token::BinaryOp(BinaryOp::Mul)).or_not()
        .then(select_ref! { Token::Var(s) if *s == "self" => () })
        .map(|(star, _)| if star.is_some() { Receiver::Pointer } else { Receiver::Value });

    let params_inner = choice((
        self_recv
            .then(just(Token::Comma).ignore_then(normal_param.clone()).repeated().collect::<Vec<_>>())
            .map(|(recv, ps)| (recv, ps)),
        normal_param
            .separated_by(just(Token::Comma))
            .allow_trailing()
            .collect::<Vec<_>>()
            .map(|ps| (Receiver::Associated, ps)),
    ))
        .delimited_by(just(Token::LParen), just(Token::RParen));

    just(Token::Proc)
        .ignore_then(var.map(|s| *s))
        .then(params_inner)
        .then(parse_type().or_not().map(|t| t.unwrap_or(Type::Void)))
        .then_ignore(just(Token::Semicolon))
        .map(|((name, (receiver, params)), return_type)|
            TraitMethod { receiver, name, params, return_type })
        .boxed()
}

/// One item inside a `trait` body: an associated-type requirement (`type Item;`)
/// or a required method signature. Partitioned into the trait node's
/// `assoc_types` / `methods` after parsing.
enum TraitItem<'a> {
    Assoc(&'a str),
    Method(TraitMethod<'a>),
}

/// One item inside an `extend` body: an associated-type binding (`type Item =
/// i32;`) or a method definition. Partitioned into the extend node's
/// `assoc_bindings` / `methods` after parsing.
enum ExtendItem<'a> {
    Assoc(&'a str, Type<'a>),
    Method(Method<'a>),
}

fn parse_toplevel<'tks, 'src: 'tks>()
-> impl Parser<
    'tks,
    MappedInput<'tks, Token<'src>, Span, &'tks [Metadata<Token<'src>>]>,
    Vec<TopLevel<'src>>,
    extra::Err<Rich<'tks, Token<'src>, Span>>,
> {
    let var = select_ref! { Token::Var(ident) => ident };

    // a single generic parameter: either `const N: u32` or a bare type param `T`
    let generic_param = choice((
        just(Token::Const)
            .ignore_then(var.map(|s| *s))
            .then_ignore(just(Token::Colon))
            .then(parse_type())
            .map(|(name, ty)| GenericParam::Const(name, ty)),
        // a bare type param, optionally with trait bounds: `T`, `T: Display`,
        // `T: A + B`. bounds are trait names joined by `+` (the `Add` op token).
        var.map(|s| *s)
            .then(
                just(Token::Colon)
                    .ignore_then(
                        var.map(|s| NameRef::new(Path::single(*s)))
                            .separated_by(just(Token::BinaryOp(BinaryOp::Add)))
                            .at_least(1)
                            .collect::<Vec<_>>())
                    .or_not())
            .map(|(name, bounds)| GenericParam::Type { name, bounds: bounds.unwrap_or_default() }),
    ));

    // optional `<T, const N: u32, ...>` list following a function name
    let generics = generic_param
        .separated_by(just(Token::Comma))
        .allow_trailing()
        .collect::<Vec<_>>()
        .delimited_by(
            just(Token::BinaryOp(BinaryOp::Lt)),
            just(Token::BinaryOp(BinaryOp::Gt)))
        .or_not()
        .map(|g| g.unwrap_or_default())
        .boxed();

    // every top-level item shares the header `[attrs] [pub] <keyword>`: any
    // attributes, then an optional `pub` marker, then the item keyword. absent
    // `pub` => module-private. yields `(attributes, is_pub)`.
    let item_header = parse_attribute()
        .repeated().collect::<Vec<_>>()
        .then(just(Token::Pub).or_not().map(|o| o.is_some()))
        .boxed();

    let function = item_header.clone()
        .then_ignore(just(Token::Proc))
        .then(var)
        .then(generics.clone())
        .then(
            var.map(|s| *s)
                .then_ignore(just(Token::Colon))
                .then(parse_type())
                .separated_by(just(Token::Comma))
                .allow_trailing()
                .collect::<Vec<_>>()
                .delimited_by(just(Token::LParen), just(Token::RParen))
        )
        .then(parse_type().or_not().map(|t| t.unwrap_or(Type::Void)))
        .then(
            parse_stmt()
                // .separated_by(just(Token::Semicolon))
                // .allow_leading()
                // .allow_trailing()
                .repeated()
                .collect::<Vec<_>>()
                .delimited_by(just(Token::LBrace), just(Token::RBrace))
        )
        .map(|((((((attributes, is_pub), name), generics), params), return_type), body)| (TopLevelNode::Function {
            name,
            def: DefId::UNRESOLVED,
            is_pub,
            attributes,
            generics,
            params,
            return_type,
            body,
        }, Vec::new()));

    let extern_ = item_header.clone()
        .then_ignore(just(Token::Extern))
        .then(var)
        .then(generics.clone())
        .then(
            var.map(|s| *s)
                .then_ignore(just(Token::Colon))
                .then(parse_type())
                .separated_by(just(Token::Comma))
                .allow_trailing()
                .collect::<Vec<_>>()
                .delimited_by(just(Token::LParen), just(Token::RParen))
        )
        .then(parse_type().or_not().map(|t| t.unwrap_or(Type::Void)))
        .then_ignore(just(Token::Semicolon))
        .map(|(((((attributes, is_pub), name), generics), params), return_type)| (TopLevelNode::Extern {
            name,
            def: DefId::UNRESOLVED,
            is_pub,
            attributes,
            generics,
            params,
            return_type,
        }, Vec::new()));

    // struct body: comma-separated fields first, then zero or more methods. `proc`
    // starts every method and can't start a field, so it's an unambiguous delimiter
    // (methods must follow all fields - the roadmap's "fields first, then methods").
    let struct_ = item_header.clone()
        .then_ignore(just(Token::Struct))
        .then(var)
        .then(generics.clone())
        .then(
            var.map(|s| *s)
                .then_ignore(just(Token::Colon))
                .then(parse_type())
                .separated_by(just(Token::Comma))
                .allow_trailing()
                .collect::<Vec<_>>()
                .then(parse_method().repeated().collect::<Vec<_>>())
                .delimited_by(just(Token::LBrace), just(Token::RBrace))
        )
        .map(|((((attributes, is_pub), name), generics), (fields, methods))| (TopLevelNode::Struct {
            name,
            def: DefId::UNRESOLVED,
            is_pub,
            attributes,
            generics,
            fields,
        }, methods));

    // module-level constant: `[attrs] [pub] const NAME: Type = <expr>;`
    let global = item_header.clone()
        .then_ignore(just(Token::Const))
        .then(var.map(|s| *s))
        .then_ignore(just(Token::Colon))
        .then(parse_type())
        .then_ignore(just(Token::Assign))
        .then(parse_expr())
        .then_ignore(just(Token::Semicolon))
        .map(|((((attributes, is_pub), name), ty), value)| (TopLevelNode::Global {
            name,
            def: DefId::UNRESOLVED,
            is_pub,
            attributes,
            ty,
            value,
        }, Vec::new()));

    // an enum variant: a name, an optional tuple payload `(T, U, ...)`, and an
    // optional explicit `= <int>` discriminant (a leading `-` is allowed for
    // negative discriminants). Tuple-payload field names are synthesized `"0"`,
    // `"1"`, ...; a discriminant is only meaningful on a unit (payload-less)
    // variant, which typecheck enforces.
    let enum_payload = parse_type()
        .separated_by(just(Token::Comma))
        .allow_trailing()
        .at_least(1)
        .collect::<Vec<_>>()
        .delimited_by(just(Token::LParen), just(Token::RParen))
        .map(|tys| tys.into_iter().enumerate()
            .map(|(i, ty)| (tuple_field_name(i), ty))
            .collect::<Vec<(&str, Type)>>());
    // a struct-style variant payload `{ field: T, ... }`: field names are kept as
    // written (unlike the tuple form's synthesized "0", "1", ...). Reuses the same
    // synthetic-struct backing; the real names just flow into the payload struct.
    let enum_struct_payload = var.map(|s| *s)
        .then_ignore(just(Token::Colon))
        .then(parse_type())
        .separated_by(just(Token::Comma))
        .allow_trailing()
        .at_least(1)
        .collect::<Vec<(&str, Type)>>()
        .delimited_by(just(Token::LBrace), just(Token::RBrace));
    let enum_discriminant = just(Token::Assign)
        .ignore_then(just(Token::BinaryOp(BinaryOp::Sub)).or_not())
        .then(select_ref! {
            Token::Int8(n)   => *n as i128,
            Token::Int16(n)  => *n as i128,
            Token::Int32(n)  => *n as i128,
            Token::Int64(n)  => *n as i128,
            Token::Uint8(n)  => *n as i128,
            Token::Uint16(n) => *n as i128,
            Token::Uint32(n) => *n as i128,
            Token::Uint64(n) => *n as i128,
            Token::IntLit(n) => *n,
        })
        .try_map(|(neg, n), span| {
            let n = if neg.is_some() { -n } else { n };
            i64::try_from(n)
                .map_err(|_| Rich::custom(span, format!("enum discriminant {n} is out of range")))
        });
    let enum_variant = var.map(|s| *s)
        .then(choice((enum_payload, enum_struct_payload)).or_not().map(|p| p.unwrap_or_default()))
        .then(enum_discriminant.or_not())
        .map(|((name, payload), disc)| (name, disc, payload));

    // enum body: comma-separated variants first, then zero or more methods, same
    // `proc`-delimits-methods rule as structs.
    let enum_ = item_header.clone()
        .then_ignore(just(Token::Enum))
        .then(var.map(|s| *s))
        .then(generics.clone())
        .then(
            enum_variant
                .separated_by(just(Token::Comma))
                .allow_trailing()
                .collect::<Vec<_>>()
                .then(parse_method().repeated().collect::<Vec<_>>())
                .delimited_by(just(Token::LBrace), just(Token::RBrace))
        )
        .map(|((((attributes, is_pub), name), generics), (variants, methods))| (TopLevelNode::Enum {
            name,
            def: DefId::UNRESOLVED,
            is_pub,
            attributes,
            generics,
            variants,
        }, methods));

    // `extend Type { methods }` or `extend Type: Trait { methods }`. `extend` isn't
    // a reserved keyword (it lexes as a `Var`), so match it by text.
    //
    // The target is a full type rather than a name, which is what admits `i32`,
    // `[T]` and `Vec<T>` alongside `Point`. The type grammar stops before the
    // `:` and the `{`, so neither the trait nor the body needs a delimiter to
    // separate it from the target.
    //
    // The target binds its type parameters implicitly, so there is no binder to
    // write a bound on and a `where` clause is the only place one can go:
    // `extend Vec<T>: Display where T: Display { ... }`. Like `extend` itself,
    // `where` lexes as a `Var` and is matched by text. Only the bounded form
    // parses - a bare `where T` would say nothing - which is also why the clause
    // reuses `GenericParam::Type` rather than earning a node of its own.
    let where_bounds = select_ref! { Token::Var(s) if *s == "where" => () }
        .ignore_then(
            var.map(|s| *s)
                .then_ignore(just(Token::Colon))
                .then(
                    var.map(|s| NameRef::new(Path::single(*s)))
                        .separated_by(just(Token::BinaryOp(BinaryOp::Add)))
                        .at_least(1)
                        .collect::<Vec<_>>())
                .map(|(name, bounds)| GenericParam::Type { name, bounds })
                .separated_by(just(Token::Comma))
                .allow_trailing()
                .at_least(1)
                .collect::<Vec<_>>())
        .or_not()
        .map(|w| w.unwrap_or_default());

    // an `extend` body item: an associated-type binding `type Item = Ty;` (tried
    // first, since it is the only one that leads with `type`) or a method.
    let assoc_binding = select_ref! { Token::Var(s) if *s == "type" => () }
        .ignore_then(var.map(|s| *s))
        .then_ignore(just(Token::Assign))
        .then(parse_type())
        .then_ignore(just(Token::Semicolon))
        .map(|(n, ty)| ExtendItem::Assoc(n, ty));
    let extend_item = choice((
        assoc_binding,
        parse_method().map(ExtendItem::Method),
    ));
    let extend_ = select_ref! { Token::Var(s) if *s == "extend" => () }
        .ignore_then(parse_type())
        .then(just(Token::Colon).ignore_then(var.map(|s| *s)).or_not())
        .then(where_bounds)
        .then(
            extend_item
                .repeated()
                .collect::<Vec<_>>()
                .delimited_by(just(Token::LBrace), just(Token::RBrace))
        )
        .map(|(((target, trait_), where_bounds), items)| {
            let mut assoc_bindings = Vec::new();
            let mut methods = Vec::new();
            for it in items {
                match it {
                    ExtendItem::Assoc(n, ty) => assoc_bindings.push((n, ty)),
                    ExtendItem::Method(m) => methods.push(m),
                }
            }
            (TopLevelNode::Extend { target, trait_, where_bounds, assoc_bindings, methods }, Vec::new())
        });

    // `[attrs] [pub] trait Name { proc m(...) Ret; ... }`. Like `extend`, `trait`
    // lexes as a `Var` (not a reserved keyword), so it is matched by text rather
    // than by a token - but the header before it is the same one every other item
    // has, so `item_header` covers it. `pub` is real: trait names are
    // module-namespaced like struct and enum names, so a private trait is
    // invisible to other modules rather than merely un-importable.
    // a `trait` body item: an associated-type requirement `type Item;` (tried
    // first, since it is the only one that leads with `type`) or a method sig.
    let assoc_decl = select_ref! { Token::Var(s) if *s == "type" => () }
        .ignore_then(var.map(|s| *s))
        .then_ignore(just(Token::Semicolon))
        .map(TraitItem::Assoc);
    let trait_item = choice((
        assoc_decl,
        parse_trait_method().map(TraitItem::Method),
    ));
    let trait_ = item_header.clone()
        .then_ignore(select_ref! { Token::Var(s) if *s == "trait" => () })
        .then(var.map(|s| *s))
        .then(
            trait_item
                .repeated()
                .collect::<Vec<_>>()
                .delimited_by(just(Token::LBrace), just(Token::RBrace))
        )
        .map(|(((attributes, is_pub), name), items)| {
            let mut assoc_types = Vec::new();
            let mut methods = Vec::new();
            for it in items {
                match it {
                    TraitItem::Assoc(n) => assoc_types.push(n),
                    TraitItem::Method(m) => methods.push(m),
                }
            }
            (TopLevelNode::Trait {
                name, def: DefId::UNRESOLVED, is_pub, attributes, assoc_types, methods,
            }, Vec::new())
        });

    choice((
        function,
        extern_,
        struct_,
        enum_,
        extend_,
        trait_,
        global,
    ))
        .map_with(|(node, methods), e| {
            let span = e.span();
            // inherent methods on a struct/enum body become a synthesized `Extend`
            // targeting that type, emitted right after the type node.
            let mut out = vec![Metadata::new(node, span.clone())];
            if !methods.is_empty() {
                // the target is the type *applied to its own parameters*, so a
                // generic type's inherent methods desugar exactly as if the
                // author had written `extend Vec<T> { ... }` out of line - the
                // block's `self` is a `Vec<T>`, not a bare `Vec`.
                let (name, generics) = match &out[0].value {
                    TopLevelNode::Struct { name, generics, .. }
                    | TopLevelNode::Enum { name, generics, .. } => (*name, generics),
                    _ => unreachable!("only struct/enum bodies carry inherent methods"),
                };
                let target = Type::Path {
                    path: Path::single(name),
                    args: generics.iter().map(|g| match g {
                        GenericParam::Type { name, .. } =>
                            GenericArg::Type(Type::path(Path::single(name))),
                        GenericParam::Const(name, _) => GenericArg::Const(ConstVal::Param(name)),
                    }).collect(),
                };
                out.push(Metadata::new(
                    TopLevelNode::Extend {
                        target,
                        trait_: None,
                        // a bound on the type's own parameter (`struct Pair<T:
                        // Display>`) binds its inherent methods too - it is the
                        // same clause an out-of-line `extend Pair<T> where T:
                        // Display` would have to spell out.
                        where_bounds: generics.iter()
                            .filter(|g| matches!(g,
                                GenericParam::Type { bounds, .. } if !bounds.is_empty()))
                            .cloned()
                            .collect(),
                        // inherent methods carry no trait, hence no bindings.
                        assoc_bindings: Vec::new(),
                        methods,
                    },
                    span,
                ));
            }
            out
        })
        .boxed()
}

fn parse_import<'tks, 'src: 'tks>()
-> impl Parser<
    'tks,
    MappedInput<'tks, Token<'src>, Span, &'tks [Metadata<Token<'src>>]>,
    Import<'src>,
    extra::Err<Rich<'tks, Token<'src>, Span>>,
> {
    let var = select_ref! { Token::Var(ident) => *ident };

    // `import seg/seg/...` optionally followed by `{ sym, sym, ... }`. the path
    // separator reuses the `/` (division) token; unambiguous here since an import
    // never contains an expression.
    // TODO: path segments are `var` only, so a segment that lexes to a keyword
    // (`import std/const`) won't parse. and `{}` is `.at_least(1)`, so an empty
    // selective import is a hard error rather than a no-op.
    just(Token::Pub).or_not().map(|p| p.is_some())
        .then_ignore(just(Token::Import))
        .then(
            var.separated_by(just(Token::BinaryOp(BinaryOp::Div)))
                .at_least(1)
                .collect::<Vec<_>>()
        )
        .then(
            var.separated_by(just(Token::Comma))
                .allow_trailing()
                .at_least(1)
                .collect::<Vec<_>>()
                .delimited_by(just(Token::LBrace), just(Token::RBrace))
                .or_not()
        )
        .map_with(|((is_pub, path), symbols), e| Import { span: e.span(), path, symbols, is_pub })
        .boxed()
}

/// One item at file scope: a module attribute, an `import`, or a real top-level
/// definition. parsed from the same stream and partitioned by `parse`.
enum FileItem<'a> {
    ModAttr(Attribute<'a>),
    Import(Import<'a>),
    // one source item can expand to several top-levels: a struct/enum with a body
    // of methods yields the type node plus a synthesized `Extend`.
    Items(Vec<TopLevel<'a>>),
}

/// Parse one file into its module attributes, its imports, and its items.
pub fn parse<'a>(file: FileId, len: usize, tokens: &'a [Metadata<Token<'a>>]) -> (
    Option<(Vec<Attribute<'a>>, Vec<Import<'a>>, Vec<TopLevel<'a>>)>,
    Vec<chumsky::error::Rich<'a, Token<'a>, Span>>,
) {
    let (out, errs) = choice((
            // first: `@!` is the only file-scope construct starting with two
            // fixed tokens, so trying it here costs nothing and keeps a module
            // attribute from being offered to `parse_toplevel` as a malformed
            // item header.
            parse_mod_attribute().map(FileItem::ModAttr),
            parse_import().map(FileItem::Import),
            parse_toplevel().map(FileItem::Items),
        ))
        .repeated()
        .collect::<Vec<_>>()
        .parse(
            tokens
            .map(Span::new(file, len, len),
                |Metadata { value: t, span: s, .. }| (t, s),
            ))
        .into_output_errors();

    let split = out.map(|items| {
        let mut mod_attrs = Vec::new();
        let mut imports = Vec::new();
        let mut tops = Vec::new();
        for item in items {
            match item {
                FileItem::ModAttr(a) => mod_attrs.push(a),
                FileItem::Import(i) => imports.push(i),
                FileItem::Items(ts) => tops.extend(ts),
            }
        }
        (mod_attrs, imports, tops)
    });

    (split, errs)
}
