use haven_common::ast::*;
use haven_common::defs::{Defs, MemberTable};
use crate::intrinsics::Intrinsic;
use crate::mono::concrete_method_name;
use std::collections::{HashMap, HashSet};

// Runtime-safety (`@alloc(false)`) checking.
//
// "clean" means a function performs no heap allocation, transitively. The
// analysis is a greatest fixpoint over the call graph:
//
//   * extern    -> clean iff annotated `@alloc(false)` (we trust the user);
//                  an unannotated extern is an allocating leaf.
//   * intrinsic -> always clean (dropped during call collection).
//   * function  -> clean iff every callee is clean.
//
// We can't do this in a single definition-order pass, modules are flattened with
// imports appearing *after* the module that imports them, so a callee can be
// defined later in the program than its caller. The fixpoint starts every function
// optimistically clean and propagates dirtiness until it stabilizes, which is
// order-independent and handles (mutual) recursion correctly.

/// Map from a callable's final (post-mono) name to whether it is known clean.
type CleanMap<'a> = HashMap<&'a str, bool>;

/// Sentinel "callee" for an indirect call through a function pointer. Its target
/// isn't statically known, so we can't prove it clean - it never appears in the
/// clean map, so any function that makes one is forced dirty. Not a valid
/// identifier, so it can't collide with a real callable's name.
const INDIRECT_CALLEE: &str = "<indirect call>";

/// The dispatch information a method call needs to be resolved to its target: the
/// post-mono inferred type of every expression (to find a receiver's type) and
/// the member table (to dispatch that type + method name to a concrete function).
struct Resolve<'p, 'a> {
    node_types: &'p HashMap<usize, Type<'a>>,
    members: &'p MemberTable<'a>,
}

/// What the callee position of a `Call` resolves to for the call graph.
enum Callee<'a> {
    /// Not a call edge at all: an intrinsic or an enum-variant constructor.
    None,
    /// A statically-known target, by its final name.
    Named(&'a str),
    /// Target unknown (a fn-pointer, or a method we couldn't resolve): forced dirty.
    Indirect,
}

/// Classify a `Call`'s callee expression. A bare name is a direct call (unless a
/// local of that name shadows it - then it's a value, i.e. indirect); a path is an
/// enum constructor (no call); and `recv.m(..)` is a method call, resolved through
/// the member table to the concrete function it dispatches to so it becomes a real
/// graph edge rather than an opaque indirect one.
fn classify_callee<'a>(func: &Expr<'a>, locals: &[&'a str], r: &Resolve<'_, 'a>) -> Callee<'a> {
    match &func.value {
        ExprNode::Var(name) if !locals.contains(name) => {
            if Intrinsic::lookup(name).is_some() { Callee::None } else { Callee::Named(name) }
        }
        ExprNode::Path(_) => Callee::None,
        ExprNode::Access { base, field } => {
            match r.node_types.get(&base.id)
                .and_then(|ty| concrete_method_name(r.members, ty, field))
            {
                Some(callee) => Callee::Named(callee),
                None => Callee::Indirect,
            }
        }
        _ => Callee::Indirect,
    }
}

/// Returns the immediate calls in this expression whose callee is dirty.
// `locals` is the stack of param/`let` names in scope at this point (see the
// rewriter in module.rs for the same shape). a called name that's shadowed by a
// local isn't the top-level symbol of that name: it's an indirect call through a
// value, so we route it to INDIRECT_CALLEE instead of the clean-map lookup.
fn dirty_calls_expr<'a>(clean: &CleanMap<'a>, locals: &[&'a str], r: &Resolve<'_, 'a>, e: &Expr<'a>) -> Vec<(&'a str, Span)> {
    match &e.value {
        ExprNode::Call { func, args, .. } => {
            // dirty calls nested in the arguments, first
            let mut dirty: Vec<(&'a str, Span)> = args.iter()
                .flat_map(|a| dirty_calls_expr(clean, locals, r, a))
                .collect();
            // then dirty calls in a method call's receiver (`recv.m()`'s `recv`).
            if let ExprNode::Access { base, .. } = &func.value {
                dirty.extend(dirty_calls_expr(clean, locals, r, base));
            }

            // then the callee itself (intrinsics/constructors are always clean).
            match classify_callee(func, locals, r) {
                Callee::None => {}
                Callee::Named(name) => {
                    if !clean.get(name).copied().unwrap_or(false) {
                        dirty.push((name, func.span.clone()));
                    }
                }
                Callee::Indirect => dirty.push((INDIRECT_CALLEE, func.span.clone())),
            }
            dirty
        }
        ExprNode::Binary { left, right, .. } => {
            let mut dirty = dirty_calls_expr(clean, locals, r, left);
            dirty.extend(dirty_calls_expr(clean, locals, r, right));
            dirty
        }
        ExprNode::Unary { operand, .. } => dirty_calls_expr(clean, locals, r, operand),
        ExprNode::Index { slice, index } => {
            let mut dirty = dirty_calls_expr(clean, locals, r, slice);
            dirty.extend(dirty_calls_expr(clean, locals, r, index));
            dirty
        }
        // literals & variable reads are always clean
        _ => vec![],
    }
}

fn dirty_calls_stmt<'a>(clean: &CleanMap<'a>, locals: &mut Vec<&'a str>, r: &Resolve<'_, 'a>, s: &Stmt<'a>) -> Vec<(&'a str, Span)> {
    match &s.value {
        StmtNode::Expr(e) => dirty_calls_expr(clean, locals, r, e),
        StmtNode::Block(block) => {
            let mark = locals.len();
            let mut dirty = Vec::new();
            for s in block { dirty.extend(dirty_calls_stmt(clean, locals, r, s)); }
            locals.truncate(mark);
            dirty
        }

        StmtNode::Declare { name, value, .. } => {
            let dirty = dirty_calls_expr(clean, locals, r, value); // before binding
            locals.push(name);
            dirty
        }
        StmtNode::Assign { value, .. } => dirty_calls_expr(clean, locals, r, value),

        StmtNode::If { condition, then_branch, else_branch, .. } => {
            let mut dirty = dirty_calls_expr(clean, locals, r, condition);
            dirty.extend(dirty_calls_stmt(clean, locals, r, then_branch));
            if let Some(b) = else_branch {
                dirty.extend(dirty_calls_stmt(clean, locals, r, b));
            }
            dirty
        }

        StmtNode::While { condition, body } => {
            let mut dirty = dirty_calls_expr(clean, locals, r, condition);
            dirty.extend(dirty_calls_stmt(clean, locals, r, body));
            dirty
        }

        StmtNode::Match { scrutinee, arms } => {
            let mut dirty = dirty_calls_expr(clean, locals, r, scrutinee);
            for (_pat, body) in arms {
                dirty.extend(dirty_calls_stmt(clean, locals, r, body));
            }
            dirty
        }

        StmtNode::Break | StmtNode::Continue | StmtNode::Return(None) => vec![],
        StmtNode::Return(Some(e)) => dirty_calls_expr(clean, locals, r, e),
    }
}

fn collect_calls_expr<'a>(calls: &mut HashSet<&'a str>, locals: &[&'a str], r: &Resolve<'_, 'a>, e: &Expr<'a>) {
    match &e.value {
        ExprNode::Call { func, args, .. } => {
            match classify_callee(func, locals, r) {
                // intrinsics and enum constructors call nothing, so they're not
                // graph edges.
                Callee::None => {}
                Callee::Named(name) => { calls.insert(name); }
                // fn-pointer or unresolved callee: record the sentinel so the
                // enclosing function is forced dirty.
                Callee::Indirect => { calls.insert(INDIRECT_CALLEE); }
            }
            // a method call's receiver (`recv.m()`'s `recv`) can contain calls too.
            if let ExprNode::Access { base, .. } = &func.value {
                collect_calls_expr(calls, locals, r, base);
            }
            for arg in args {
                collect_calls_expr(calls, locals, r, arg);
            }
        }
        ExprNode::Binary { left, right, .. } => {
            collect_calls_expr(calls, locals, r, left);
            collect_calls_expr(calls, locals, r, right);
        }
        ExprNode::Unary { operand, .. } => collect_calls_expr(calls, locals, r, operand),
        ExprNode::Index { slice, index } => {
            collect_calls_expr(calls, locals, r, slice);
            collect_calls_expr(calls, locals, r, index);
        }
        _ => {}
    }
}

fn collect_calls_stmt<'a>(calls: &mut HashSet<&'a str>, locals: &mut Vec<&'a str>, r: &Resolve<'_, 'a>, s: &Stmt<'a>) {
    match &s.value {
        StmtNode::Expr(e) => collect_calls_expr(calls, locals, r, e),
        StmtNode::Block(stmts) => {
            let mark = locals.len();
            for s in stmts { collect_calls_stmt(calls, locals, r, s); }
            locals.truncate(mark);
        }
        StmtNode::Declare { name, value, .. } => {
            collect_calls_expr(calls, locals, r, value); // before binding
            locals.push(name);
        }
        StmtNode::Assign { value, .. } => collect_calls_expr(calls, locals, r, value),
        StmtNode::If { condition, then_branch, else_branch, .. } => {
            collect_calls_expr(calls, locals, r, condition);
            collect_calls_stmt(calls, locals, r, then_branch);
            if let Some(b) = else_branch { collect_calls_stmt(calls, locals, r, b); }
        }
        StmtNode::While { condition, body } => {
            collect_calls_expr(calls, locals, r, condition);
            collect_calls_stmt(calls, locals, r, body);
        }
        StmtNode::Match { scrutinee, arms } => {
            collect_calls_expr(calls, locals, r, scrutinee);
            for (_pat, body) in arms {
                let mark = locals.len();
                collect_calls_stmt(calls, locals, r, body);
                locals.truncate(mark);
            }
        }
        StmtNode::Return(Some(e)) => collect_calls_expr(calls, locals, r, e),
        StmtNode::Return(None) | StmtNode::Break | StmtNode::Continue => {}
    }
}

/// Compute the clean/dirty status of every callable via a fixpoint.
fn compute_clean<'a>(program: &[TopLevel<'a>], r: &Resolve<'_, 'a>) -> CleanMap<'a> {
    let mut clean: CleanMap<'a> = HashMap::new();

    // leaves: externs are clean iff annotated, functions start optimistically
    // clean and the call graph records who they call
    let mut calls: HashMap<&'a str, HashSet<&'a str>> = HashMap::new();
    for node in program {
        match &node.value {
            TopLevelNode::Extern { name, attributes, .. } => {
                let is_clean = attributes.iter().any(|a| a.value.is_false("alloc"));
                clean.insert(name, is_clean);
            }
            TopLevelNode::Function { name, params, body, .. } => {
                let mut callees = HashSet::new();
                let mut locals: Vec<&'a str> = params.iter().map(|(p, _)| *p).collect();
                for s in body { collect_calls_stmt(&mut callees, &mut locals, r, s); }
                calls.insert(name, callees);
                clean.insert(name, true);
            }
            TopLevelNode::Struct { .. } | TopLevelNode::Global { .. }
            | TopLevelNode::Enum { .. } => {}
            TopLevelNode::Trait { .. } => unreachable!("traits dropped in monomorphization"),
            TopLevelNode::Extend { .. } => unreachable!("extend desugared before safecheck"),
        }
    }

    // propagate dirtiness until stable. a function goes dirty as soon as any of
    // its callees is dirty (or unknown, which we treat conservatively as dirty)
    loop {
        let mut changed = false;
        for (&fname, callees) in &calls {
            if clean.get(fname).copied() == Some(false) { continue; }
            let is_dirty = callees.iter()
                .any(|c| !clean.get(c).copied().unwrap_or(false));
            if is_dirty {
                clean.insert(fname, false);
                changed = true;
            }
        }
        if !changed { break; }
    }

    clean
}

pub fn alloc_check_program<'a>(
    program: &[TopLevel<'a>],
    defs: &Defs<'a>,
    node_types: &HashMap<usize, Type<'a>>,
) -> Result<(), Vec<Error>> {
    let r = Resolve { node_types, members: defs.members() };
    let clean = compute_clean(program, &r);

    // mono rewrites generic calls to their mangled instance name; prefer the
    // friendly spelling it recorded (`alloc::<Vec2>`) over `std.alloc$alloc$Vec2`
    let show = |n: &'a str| defs.show_symbol(n);

    // report each `@alloc(false)` function that came out dirty, pointing at the
    // offending immediate calls in its body. dirtiness always propagates through
    // at least one immediate callee, so a dirty function has >=1 span to blame
    let mut errors: Vec<Error> = Vec::new();
    for node in program {
        let TopLevelNode::Function { name, attributes, params, body, .. } = &node.value else { continue };
        let marked = attributes.iter().any(|attr| attr.value.is_false("alloc"));
        if !marked || clean.get(name).copied().unwrap_or(false) {
            continue;
        }

        let mut locals: Vec<&'a str> = params.iter().map(|(p, _)| *p).collect();
        let mut blamed: Vec<(&'a str, Span)> = Vec::new();
        for s in body { blamed.extend(dirty_calls_stmt(&clean, &mut locals, &r, s)); }
        for (callee, span) in blamed {
            errors.push(Error::new(span, format!(
                "'{}' is marked as @alloc(false) but may allocate", show(name)))
                .with_label(span, format!("calls '{}', which may allocate", show(callee))));
        }
    }

    if errors.is_empty() { Ok(()) } else { Err(errors) }
}
