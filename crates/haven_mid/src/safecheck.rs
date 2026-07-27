use haven_common::ast::*;
use haven_common::defs::Defs;
use crate::intrinsics::Intrinsic;
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

/// Returns the immediate calls in this expression whose callee is dirty.
// `locals` is the stack of param/`let` names in scope at this point (see the
// rewriter in module.rs for the same shape). a called name that's shadowed by a
// local isn't the top-level symbol of that name: it's an indirect call through a
// value, so we route it to INDIRECT_CALLEE instead of the clean-map lookup.
fn dirty_calls_expr<'a>(clean: &CleanMap<'a>, locals: &[&'a str], e: &Expr<'a>) -> Vec<(&'a str, Span)> {
    match &e.value {
        ExprNode::Call { func, args, .. } => {
            // dirty calls nested in the arguments, first
            let mut dirty: Vec<(&'a str, Span)> = args.iter()
                .flat_map(|a| dirty_calls_expr(clean, locals, a))
                .collect();

            // then the callee itself (intrinsics are always clean).
            match &func.value {
                ExprNode::Var(name) if !locals.contains(name) => {
                    if Intrinsic::lookup(name).is_none()
                        && !clean.get(name).copied().unwrap_or(false)
                    {
                        dirty.push((*name, func.span.clone()));
                    }
                }
                // a data-enum constructor `Enum::Variant(...)`: not a real call,
                // just stores into stack storage. Always clean.
                ExprNode::Path(_) => {}
                // local fn-pointer or any computed callee: target unknown, dirty
                _ => dirty.push((INDIRECT_CALLEE, func.span.clone())),
            }
            dirty
        }
        ExprNode::Binary { left, right, .. } => {
            let mut dirty = dirty_calls_expr(clean, locals, left);
            dirty.extend(dirty_calls_expr(clean, locals, right));
            dirty
        }
        ExprNode::Unary { operand, .. } => dirty_calls_expr(clean, locals, operand),
        ExprNode::Index { slice, index } => {
            let mut dirty = dirty_calls_expr(clean, locals, slice);
            dirty.extend(dirty_calls_expr(clean, locals, index));
            dirty
        }
        // literals & variable reads are always clean
        _ => vec![],
    }
}

fn dirty_calls_stmt<'a>(clean: &CleanMap<'a>, locals: &mut Vec<&'a str>, s: &Stmt<'a>) -> Vec<(&'a str, Span)> {
    match &s.value {
        StmtNode::Expr(e) => dirty_calls_expr(clean, locals, e),
        StmtNode::Block(block) => {
            let mark = locals.len();
            let mut dirty = Vec::new();
            for s in block { dirty.extend(dirty_calls_stmt(clean, locals, s)); }
            locals.truncate(mark);
            dirty
        }

        StmtNode::Declare { name, value, .. } => {
            let dirty = dirty_calls_expr(clean, locals, value); // before binding
            locals.push(name);
            dirty
        }
        StmtNode::Assign { value, .. } => dirty_calls_expr(clean, locals, value),

        StmtNode::If { condition, then_branch, else_branch, .. } => {
            let mut dirty = dirty_calls_expr(clean, locals, condition);
            dirty.extend(dirty_calls_stmt(clean, locals, then_branch));
            if let Some(b) = else_branch {
                dirty.extend(dirty_calls_stmt(clean, locals, b));
            }
            dirty
        }

        StmtNode::While { condition, body } => {
            let mut dirty = dirty_calls_expr(clean, locals, condition);
            dirty.extend(dirty_calls_stmt(clean, locals, body));
            dirty
        }

        StmtNode::Match { scrutinee, arms } => {
            let mut dirty = dirty_calls_expr(clean, locals, scrutinee);
            for (_pat, body) in arms {
                dirty.extend(dirty_calls_stmt(clean, locals, body));
            }
            dirty
        }

        StmtNode::Break | StmtNode::Continue => vec![],
        StmtNode::Return(e) => dirty_calls_expr(clean, locals, e),
    }
}

fn collect_calls_expr<'a>(calls: &mut HashSet<&'a str>, locals: &[&'a str], e: &Expr<'a>) {
    match &e.value {
        ExprNode::Call { func, args, .. } => {
            match &func.value {
                ExprNode::Var(name) if !locals.contains(name) => {
                    if Intrinsic::lookup(name).is_none() {
                        calls.insert(*name);
                    }
                }
                // skip data-enum constructors (`Enum::Variant(...)`): they call
                // nothing (see `dirty_calls_expr`), so they're not graph edges.
                ExprNode::Path(_) => {}
                // indirect call (local fn pointer or computed callee): record the
                // sentinel so the enclosing function is forced dirty
                _ => { calls.insert(INDIRECT_CALLEE); }
            }
            for arg in args {
                collect_calls_expr(calls, locals, arg);
            }
        }
        ExprNode::Binary { left, right, .. } => {
            collect_calls_expr(calls, locals, left);
            collect_calls_expr(calls, locals, right);
        }
        ExprNode::Unary { operand, .. } => collect_calls_expr(calls, locals, operand),
        ExprNode::Index { slice, index } => {
            collect_calls_expr(calls, locals, slice);
            collect_calls_expr(calls, locals, index);
        }
        _ => {}
    }
}

fn collect_calls_stmt<'a>(calls: &mut HashSet<&'a str>, locals: &mut Vec<&'a str>, s: &Stmt<'a>) {
    match &s.value {
        StmtNode::Expr(e) => collect_calls_expr(calls, locals, e),
        StmtNode::Block(stmts) => {
            let mark = locals.len();
            for s in stmts { collect_calls_stmt(calls, locals, s); }
            locals.truncate(mark);
        }
        StmtNode::Declare { name, value, .. } => {
            collect_calls_expr(calls, locals, value); // before binding
            locals.push(name);
        }
        StmtNode::Assign { value, .. } => collect_calls_expr(calls, locals, value),
        StmtNode::If { condition, then_branch, else_branch, .. } => {
            collect_calls_expr(calls, locals, condition);
            collect_calls_stmt(calls, locals, then_branch);
            if let Some(b) = else_branch { collect_calls_stmt(calls, locals, b); }
        }
        StmtNode::While { condition, body } => {
            collect_calls_expr(calls, locals, condition);
            collect_calls_stmt(calls, locals, body);
        }
        StmtNode::Match { scrutinee, arms } => {
            collect_calls_expr(calls, locals, scrutinee);
            for (_pat, body) in arms {
                let mark = locals.len();
                collect_calls_stmt(calls, locals, body);
                locals.truncate(mark);
            }
        }
        StmtNode::Return(e) => collect_calls_expr(calls, locals, e),
        StmtNode::Break | StmtNode::Continue => {}
    }
}

/// Compute the clean/dirty status of every callable via a fixpoint.
fn compute_clean<'a>(program: &[TopLevel<'a>]) -> CleanMap<'a> {
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
                for s in body { collect_calls_stmt(&mut callees, &mut locals, s); }
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
) -> Result<(), Vec<Error>> {
    let clean = compute_clean(program);

    // mono rewrites generic calls to their mangled instance name; prefer the
    // friendly spelling it recorded (`alloc::<Vec2>`) over `std.alloc$alloc$Vec2`
    let show = |n: &'a str| defs.show(n);

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
        for s in body { blamed.extend(dirty_calls_stmt(&clean, &mut locals, s)); }
        for (callee, span) in blamed {
            errors.push(Error::new(span, format!(
                "Function '{}' is marked as @alloc(false) but calls '{}', which may allocate.",
                show(name),
                show(callee),
            )));
        }
    }

    if errors.is_empty() { Ok(()) } else { Err(errors) }
}
