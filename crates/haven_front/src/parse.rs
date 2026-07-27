use std::{rc::Rc, u64};
use chumsky::{
    input::MappedInput,
    pratt::*,
    prelude::*,
};
use haven_common::ast::*;

/// Leak a String to get a `&'static str` (coerces to the source lifetime). used
/// for the few synthetic identifiers the parser has to mint, e.g. the joined
/// `qualifier::symbol` of a qualified ref. compiler is single-shot so leaking a
/// bounded number of names is fine.
// TODO: the rest of the pipeline now threads am arena for exactly this kind of
// synthetic name (see module.rs / mono.rs). this leak predates that and should
// probably use the arena too instead of leaking for the whole process.
fn leak(s: String) -> &'static str {
    Box::leak(s.into_boxed_str())
}

/// The payload tail of an `Enum::Variant` match pattern: positional `(a, b)` or
/// by-name `{ id, val }`. Only used to unify the two shapes under one `choice`
/// in the pattern parser before mapping to the corresponding `PatternNode`.
enum PatTail<'a> {
    Tuple(Vec<Pattern<'a>>),
    Struct(Vec<(&'a str, Pattern<'a>)>),
}

fn lexer<'a> (
    // Rc should be cheaper to clone per token
    file_path: Rc<str>,
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

    // defaults ints and floats to 32 bits and use suffixes to disambiguate
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
                None => Ok(Token::Float32(f as f32)),
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
                Some(('i', 32)) => try_parse_int!(i32, n, span).map(Token::Int32),
                Some(('i', 64)) => try_parse_int!(i64, n, span).map(Token::Int64),
                Some(('u', 8))  => try_parse_int!(u8, n, span).map(Token::Uint8),
                Some(('u', 32)) => try_parse_int!(u32, n, span).map(Token::Uint32),
                Some(('u', 64)) => try_parse_int!(u64, n, span).map(Token::Uint64),
                None => try_parse_int!(i32, n, span).map(Token::Int32),
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
        .repeated()
        .to_slice()
        .delimited_by(just('"'), just('"'))
        .map(Token::Str);

    let ident = text::ascii::ident().map(|ident: &str| match ident {
        "true"     => Token::Bool(true),
        "false"    => Token::Bool(false),
        "let"      => Token::Let,
        "if"       => Token::If,
        "else"     => Token::Else,
        "return"   => Token::Return,
        "while"    => Token::While,
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
                Span::new(
                    file_path.to_string(),
                    input_span.start,
                    input_span.end,
                ),
            )
        })
        .padded_by(comment.repeated())
        .padded()
        .recover_with(skip_then_retry_until(any().ignored(), end()))
        .repeated()
        .collect()
}

pub fn lex<'a>(file_path: &'a str, source: &'a str) -> (
    Option<Vec<Metadata<Token<'a>>>>,
    Vec<chumsky::error::Rich<'a, char, Span>>
) {
    let (tks, errs) = lexer(std::rc::Rc::from(file_path))
        .parse(source)
        .into_output_errors();

    (tks, errs.into_iter()
        .map(|e| {
            e.map_span(|simple_span| {
                Span::new(
                    file_path.to_string(),
                    simple_span.start,
                    simple_span.end
                )
            })
        })
        .collect())
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
                Token::Int32(i)   => ExprNode::Int32(*i),
                Token::Int64(i)   => ExprNode::Int64(*i),
                Token::Uint8(u)   => ExprNode::Uint8(*u),
                Token::Uint32(u)  => ExprNode::Uint32(*u),
                Token::Uint64(u)  => ExprNode::Uint64(*u),
                Token::Float32(f) => ExprNode::Float32(*f),
                Token::Float64(f) => ExprNode::Float64(*f),
                Token::Str(s)     => ExprNode::Str(*s),
            },

            // Struct init, with an optional qualifier and an optional turbofish
            // for generic structs: `S { f: v }`, `geo::Point { f: v }`,
            // `Option::<i32> { f: v }`, or `mod::Option::<i32> { f: v }`. A first
            // `::Name` is the module qualifier (folded into `"qual::Name"`); a
            // `::<...>` is the generic turbofish, disambiguating `<`/`>` from
            // comparison operators as the call turbofish does. The qualifier
            // alternative rejects `::<` (not an ident) and backtracks, so the
            // turbofish still sees it.
            var.map(|s| *s)
                .then(just(Token::ColonColon).ignore_then(var.map(|s| *s)).or_not())
                .map(|(a, b)| -> &str {
                    match b {
                        Some(sym) => leak(format!("{}::{}", a, sym)),
                        None => a,
                    }
                })
                .then(
                    just(Token::ColonColon)
                        .ignore_then(
                            choice((
                                select! { Token::Int32(n) => n }.try_map(|n, span| {
                                    if n < 0 {
                                        Err(Rich::custom(span, "const turbofish argument must be non-negative"))
                                    } else {
                                        Ok(GenericArg::Const(ConstVal::Lit(n as usize)))
                                    }
                                }),
                                // a bare ident is ambiguous between a type and a
                                // forwarded const param; parses as a type, reclassified
                                // downstream once the struct's kinds are known.
                                parse_type().map(GenericArg::Type),
                            ))
                                .separated_by(just(Token::Comma))
                                .allow_trailing()
                                .collect::<Vec<_>>()
                                .delimited_by(
                                    just(Token::BinaryOp(BinaryOp::Lt)),
                                    just(Token::BinaryOp(BinaryOp::Gt))),
                        )
                        .or_not()
                        .map(|t| t.unwrap_or_default())
                )
                .then(
                    var.map(|s| *s)
                        .then_ignore(just(Token::Colon))
                        .then(expr.clone())
                        .separated_by(just(Token::Comma))
                        .allow_trailing()
                        .collect::<Vec<_>>()
                        .delimited_by(just(Token::LBrace), just(Token::RBrace))
                )
                .map(|((name, type_args), fields)| ExprNode::Struct {
                    name,
                    type_args,
                    fields,
                }),

            // a variable, or a module-qualified ref `qualifier::symbol`
            // (e.g. `math::sinf`). the qualified form is one `Var` holding the
            // joined `"qualifier::symbol"` string; the module resolver splits and
            // rewrites it before any later stage. the `::symbol` is optional and
            // only taken when followed by an ident, so a turbofish `::<...>` is
            // left for the call postfix.
            // TODO: only one `::` segment - `a::b::c` leaves `::c` dangling.
            var.then(just(Token::ColonColon).ignore_then(var).or_not())
                .map(|(a, b)| match b {
                    Some(sym) => ExprNode::Var(leak(format!("{}::{}", a, sym))),
                    None => ExprNode::Var(*a),
                }),
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
                just(Token::ColonColon)
                    .ignore_then(
                        choice((
                            select! { Token::Int32(n) => n }.try_map(|n, span| {
                                if n < 0 {
                                    Err(Rich::custom(span, "const turbofish argument must be non-negative"))
                                } else {
                                    Ok(GenericArg::Const(ConstVal::Lit(n as usize)))
                                }
                            }),
                            // a bare ident here (e.g. `N`) is ambiguous between a type
                            // and a forwarded const param; it parses as a type and gets
                            // reclassified downstream once the callee's kinds are known.
                            parse_type().map(GenericArg::Type),
                        ))
                        .separated_by(just(Token::Comma))
                        .allow_trailing()
                        .collect::<Vec<_>>()
                        .delimited_by(
                            just(Token::BinaryOp(BinaryOp::Lt)),
                            just(Token::BinaryOp(BinaryOp::Gt))),
                    )
                    .or_not()
                    .map(|t| t.unwrap_or_default())
                    .then(
                        expr.clone()
                            .separated_by(just(Token::Comma))
                            .allow_trailing()
                            .collect::<Vec<_>>()
                            .delimited_by(just(Token::LParen), just(Token::RParen)),
                    )
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
    Lit(i32),
    Param(&'a str),
}

/// One argument inside an angle-bracket list in type position, e.g. the parts of
/// `simd<f32, 4>` or `Option<i32>`. The parser can't yet tell a size from a type
/// param (both are bare idents), so a bare int is a [`GenArg::Size`] and anything
/// else parses as a [`GenArg::Ty`]; the dispatch in `parse_type` reinterprets
/// them per the head name (`simd` wants `<type, size>`; a struct wants types).
enum GenArg<'a> {
    Size(i32),
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
                    Token::Int32(x) => SizeArg::Lit(x),
                    Token::Var(n) => SizeArg::Param(n),
                })
                .then_ignore(just(Token::RBracket))
                .try_map(|(inner, size), span| match size {
                    SizeArg::Lit(x) if x > 0 => Ok(Type::Array(Box::new(inner), ConstVal::Lit(x as usize))),
                    SizeArg::Lit(x) => Err(Rich::custom(span, format!("invalid array size parameter: {x} (must be greater than 0)"))),
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
            // a leading `qualifier::` is folded into the joined `"qual::Name"`
            // name, split + resolved by the module resolver.
            var.map(|s| *s)
                .then(just(Token::ColonColon).ignore_then(var.map(|s| *s)).or_not())
                .map(|(a, b)| -> &str {
                    match b {
                        Some(sym) => leak(format!("{}::{}", a, sym)),
                        None => a,
                    }
                })
            .then(
                choice((
                    select! { Token::Int32(x) => GenArg::Size(x) },
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
            .try_map(|(name, args), span| {
                let Some(args) = args else {
                    return Ok(match name {
                        "void" => Type::Void,
                        "bool" => Type::Bool,
                        "i8"   => Type::Int8,
                        "i32"  => Type::Int32,
                        "i64"  => Type::Int64,
                        "u8"   => Type::Uint8,
                        "u32"  => Type::Uint32,
                        "u64"  => Type::Uint64,
                        "f32"  => Type::Float32,
                        "f64"  => Type::Float64,
                        "str"  => Type::Str,
                        other  => Type::Struct { name: other, args: Vec::new() },
                    });
                };
                match name {
                    // `simd<element, lanes>`: exactly a type then a size.
                    "simd" => {
                        if args.len() != 2 {
                            return Err(Rich::custom(span, format!("simd<...> takes 2 arguments (element type, lane count), got {}", args.len())));
                        }
                        let mut it = args.into_iter();
                        let elem = match it.next().unwrap() {
                            GenArg::Ty(t) => t,
                            GenArg::Size(_) => return Err(Rich::custom(span, "simd<...> element (first argument) must be a type")),
                        };
                        // the lane count is a literal, or a bare ident naming a
                        // const generic param (parsed as a no-arg struct type).
                        let size = match it.next().unwrap() {
                            GenArg::Size(x) if x > 0 && x <= 64 => ConstVal::Lit(x as usize),
                            GenArg::Size(x) => return Err(Rich::custom(span, format!("invalid SIMD size parameter: {x} (must be between 1 and 64)"))),
                            GenArg::Ty(Type::Struct { name, args }) if args.is_empty() => ConstVal::Param(name),
                            GenArg::Ty(_) => return Err(Rich::custom(span, "simd<...> lane count (second argument) must be an integer or a const parameter")),
                        };
                        Ok(Type::Simd(Box::new(elem), size))
                    }
                    // any other head is a generic struct. arguments are types or
                    // const values (`Buf<i32, 8>`); a bare ident stays a type and is
                    // reclassified downstream if the struct declares it `const`.
                    other => {
                        let mut gargs = Vec::with_capacity(args.len());
                        for a in args {
                            gargs.push(match a {
                                GenArg::Ty(t) => GenericArg::Type(t),
                                GenArg::Size(x) if x >= 0 => GenericArg::Const(ConstVal::Lit(x as usize)),
                                GenArg::Size(x) => return Err(Rich::custom(span, format!("const argument '{x}' in generic type '{other}<...>' must be non-negative"))),
                            });
                        }
                        Ok(Type::Struct { name: other, args: gargs })
                    }
                }
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

        // a match pattern: `Enum::Variant`, a data-variant destructure
        // `Enum::Variant(binding, ...)`, an integer literal (opt. negative), or
        // `_`. A binding field is a name (`Bind`) or `_` (ignore the field).
        let int_pat = just(Token::BinaryOp(BinaryOp::Sub)).or_not()
            .then(select_ref! {
                Token::Int8(n)   => *n as i64,
                Token::Int32(n)  => *n as i64,
                Token::Int64(n)  => *n,
                Token::Uint8(n)  => *n as i64,
                Token::Uint32(n) => *n as i64,
                Token::Uint64(n) => *n as i64,
            })
            .map(|(neg, n)| PatternNode::Int(if neg.is_some() { -n } else { n }));
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
        let path_pat = var.then_ignore(just(Token::ColonColon)).then(var)
            .then(choice((
                variant_tail.map(PatTail::Tuple),
                struct_variant_tail.map(PatTail::Struct),
            )).or_not())
            .map(|((a, b), tail)| {
                let path = leak(format!("{}::{}", a, b));
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

fn parse_attribute<'tks, 'src: 'tks>()
-> impl Parser<
    'tks,
    MappedInput<'tks, Token<'src>, Span, &'tks [Metadata<Token<'src>>]>,
    Attribute<'src>,
    extra::Err<Rich<'tks, Token<'src>, Span>>,
> {
    just(Token::At)
        .ignore_then(select_ref! { Token::Var(ident) => ident })
        .then(
            just(Token::LParen)
                .ignore_then(select_ref! {
                    Token::Var(s) => s.to_string(),
                    Token::Bool(b) => if *b { "true" } else { "false" }.to_string(),
                    Token::Int32(i) => i.to_string(),
                    // Token::Str(s) => s.to_string()
                })
                .then_ignore(just(Token::RParen))
                .or_not()
        )
        .map(|(name, value)| AttributeNode::new(name, value))
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
                        var.map(|s| *s)
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
                        var.map(|s| *s)
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
            .map(|(i, ty)| (leak(i.to_string()), ty))
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
            Token::Int8(n)   => *n as i64,
            Token::Int32(n)  => *n as i64,
            Token::Int64(n)  => *n,
            Token::Uint8(n)  => *n as i64,
            Token::Uint32(n) => *n as i64,
            Token::Uint64(n) => *n as i64,
        })
        .map(|(neg, n)| if neg.is_some() { -n } else { n });
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
            is_pub,
            attributes,
            generics,
            variants,
        }, methods));

    // `extend Type { methods }` or `extend Type: Trait { methods }`. `extend` isn't
    // a reserved keyword (it lexes as a `Var`), so match it by text. The trait name
    // is recorded but unchecked in Stage 1.
    let extend_ = select_ref! { Token::Var(s) if *s == "extend" => () }
        .ignore_then(var.map(|s| *s))
        .then(just(Token::Colon).ignore_then(var.map(|s| *s)).or_not())
        .then(
            parse_method()
                .repeated()
                .collect::<Vec<_>>()
                .delimited_by(just(Token::LBrace), just(Token::RBrace))
        )
        .map(|((target, trait_), methods)| (TopLevelNode::Extend { target, trait_, methods }, Vec::new()));

    // `[pub] trait Name { proc m(...) Ret; ... }`. Like `extend`, `trait` lexes as
    // a `Var` (not a reserved keyword), so match it by text - which is why this
    // can't use the shared `item_header` combinator (that one ends with a real
    // keyword token). The `pub` marker is spelled out here instead; attributes on
    // a trait are still not accepted, since `TopLevelNode::Trait` has nowhere to
    // put them. Trait names are kept stable (unmangled) across modules, like enum
    // names, so `pub` currently gates only whether another module can *import* the
    // name - a bound `T: Trait` still resolves globally. See the module-system
    // redesign notes.
    let trait_ = just(Token::Pub).or_not().map(|o| o.is_some())
        .then_ignore(select_ref! { Token::Var(s) if *s == "trait" => () })
        .then(var.map(|s| *s))
        .then(
            parse_trait_method()
                .repeated()
                .collect::<Vec<_>>()
                .delimited_by(just(Token::LBrace), just(Token::RBrace))
        )
        .map(|((is_pub, name), methods)| (TopLevelNode::Trait { name, is_pub, methods }, Vec::new()));

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
                let target = match &out[0].value {
                    TopLevelNode::Struct { name, .. } | TopLevelNode::Enum { name, .. } => *name,
                    _ => unreachable!("only struct/enum bodies carry inherent methods"),
                };
                out.push(Metadata::new(
                    TopLevelNode::Extend { target, trait_: None, methods },
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
    just(Token::Import)
        .ignore_then(
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
        .map_with(|(path, symbols), e| Import { span: e.span(), path, symbols })
        .boxed()
}

/// One item at file scope: an `import` or a real top-level definition. parsed
/// from the same stream and partitioned by `parse`.
enum FileItem<'a> {
    Import(Import<'a>),
    // one source item can expand to several top-levels: a struct/enum with a body
    // of methods yields the type node plus a synthesized `Extend`.
    Items(Vec<TopLevel<'a>>),
}

pub fn parse<'a>(file_path: String, len: usize, tokens: &'a [Metadata<Token<'a>>]) -> (
    Option<(Vec<Import<'a>>, Vec<TopLevel<'a>>)>,
    Vec<chumsky::error::Rich<'a, Token<'a>, Span>>,
) {
    let (out, errs) = choice((
            parse_import().map(FileItem::Import),
            parse_toplevel().map(FileItem::Items),
        ))
        .repeated()
        .collect::<Vec<_>>()
        .parse(
            tokens
            .map(Span::new(file_path, len, len),
                |Metadata { value: t, span: s, .. }| (t, s),
            ))
        .into_output_errors();

    let split = out.map(|items| {
        let mut imports = Vec::new();
        let mut tops = Vec::new();
        for item in items {
            match item {
                FileItem::Import(i) => imports.push(i),
                FileItem::Items(ts) => tops.extend(ts),
            }
        }
        (imports, tops)
    });

    (split, errs)
}
