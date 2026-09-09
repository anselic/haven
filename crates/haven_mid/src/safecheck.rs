use haven_common::ast::*;
use haven_common::defs::{Defs, Instances, MemberTable};
use crate::intrinsics::Intrinsic;
use crate::mono::concrete_method_name;
use std::collections::{HashMap, HashSet, VecDeque};

// `@alloc(false)` checking. "clean" = never allocates, even through callees.
//
//   * extern    -> clean only if marked `@alloc(false)`; we trust the mark.
//   * intrinsic -> always clean.
//   * function  -> clean only if every callee is clean.
//
// Why a fixpoint, not one pass: modules are flattened, so a callee can come
// after its caller. Start everything clean, spread dirtiness until it settles.
// Order stops mattering, and recursion just works.

/// Keyed by each callable's final, post-mono name.
type CleanMap<'a> = HashMap<&'a str, bool>;

/// A call through a function pointer has no known target, so we cannot prove
/// it clean. This stands in for one. Any function that makes such a call turns
/// dirty. The name is not a valid identifier on purpose: it cannot collide
/// with a real callable.
const INDIRECT_CALLEE: &str = "<indirect call>";

/// Everything needed to resolve a `recv.m()` to the function it calls.
struct Resolve<'p, 'a> {
    node_types: &'p HashMap<usize, Type<'a>>,
    members: &'p MemberTable<'a>,
    /// This pass runs after mono, but the member table is keyed on templates.
    /// Without this, a monomorphized receiver would not find its impl.
    instances: &'p Instances<'a>,
}

/// What the callee position of a `Call` resolves to for the call graph.
enum Callee<'a> {
    /// An intrinsic or enum constructor: not a call edge at all.
    None,
    Named(&'a str),
    /// A fn-pointer or unresolved method. Forced dirty: we cannot see its body.
    Indirect,
}

/// A bare name is not always a direct call: a local can shadow the
/// function, which makes the callee a value.
/// Method calls go through the member table so they land on a concrete
/// function. Without that we would emit a dynamic edge and lose the
/// call graph.
fn classify_callee<'a>(func: &Expr<'a>, locals: &[&'a str], r: &Resolve<'_, 'a>) -> Callee<'a> {
    match &func.value {
        ExprNode::Var(name) if !locals.contains(name) => {
            if Intrinsic::lookup(name).is_some() { Callee::None } else { Callee::Named(name) }
        }
        ExprNode::Path(_) => Callee::None,
        ExprNode::Access { base, field } => {
            match r.node_types.get(&base.id)
                .and_then(|ty| concrete_method_name(r.members, r.instances, ty, field))
            {
                Some(callee) => Callee::Named(callee),
                None => Callee::Indirect,
            }
        }
        _ => Callee::Indirect,
    }
}

// `locals`: param and `let` names in scope, so a shadowed name routes to
// INDIRECT_CALLEE instead of a clean-map lookup.
fn dirty_calls_expr<'a>(clean: &CleanMap<'a>, locals: &[&'a str], r: &Resolve<'_, 'a>, e: &Expr<'a>) -> Vec<(&'a str, Span)> {
    match &e.value {
        ExprNode::Call { func, args, .. } => {
            let mut dirty: Vec<(&'a str, Span)> = args.iter()
                .flat_map(|a| dirty_calls_expr(clean, locals, r, a))
                .collect();
            // a method call's receiver can hide calls too.
            if let ExprNode::Access { base, .. } = &func.value {
                dirty.extend(dirty_calls_expr(clean, locals, r, base));
            }

            match classify_callee(func, locals, r) {
                Callee::None => {}
                Callee::Named(name) => {
                    if !clean.get(name).copied().unwrap_or(false) {
                        dirty.push((name, func.span));
                    }
                }
                Callee::Indirect => dirty.push((INDIRECT_CALLEE, func.span)),
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
        // no other expression kind contains a call
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
                Callee::None => {}
                Callee::Named(name) => { calls.insert(name); }
                Callee::Indirect => { calls.insert(INDIRECT_CALLEE); }
            }
            // a method call's receiver can hide calls too.
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

/// Keyed by caller. An absent name has no body here - an extern, or an
/// unresolved callee - which is what makes it a leaf in a blame chain.
type CallGraph<'a> = HashMap<&'a str, HashSet<&'a str>>;

/// Also returns the call graph, so a violation can be traced down to the
/// leaf that really allocates.
fn compute_clean<'a>(program: &[TopLevel<'a>], r: &Resolve<'_, 'a>)
-> (CleanMap<'a>, CallGraph<'a>) {
    let mut clean: CleanMap<'a> = HashMap::new();

    // extern: clean only if marked. function: assume clean, record its callees.
    let mut calls: CallGraph<'a> = HashMap::new();
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
            TopLevelNode::Alias { .. } => unreachable!("aliases expanded before safecheck"),
            TopLevelNode::Extend { .. } => unreachable!("extend desugared before safecheck"),
        }
    }

    // spread dirtiness until stable. an unknown callee counts as dirty.
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

    (clean, calls)
}

/// The shortest chain from `start` to the leaf that makes it dirty, e.g.
/// `Serial$tick -> Bad$tick -> Vec$with_capacity -> malloc`.
///
/// Why bother: dirtiness spreads upward, so the callee we first blame usually
/// allocates nothing itself - it is a generic wrapper. Pointing only at it
/// blames the one innocent function. This walks down to the real culprit.
///
/// Breadth-first for the shortest chain; callees visited in name order, so the
/// same program always reports the same one.
fn blame_chain<'a>(calls: &CallGraph<'a>, clean: &CleanMap<'a>, start: &'a str) -> Vec<&'a str> {
    fn path_to<'a>(prev: &HashMap<&'a str, &'a str>, start: &'a str, end: &'a str) -> Vec<&'a str> {
        let mut out = vec![end];
        let mut cur = end;
        while cur != start {
            match prev.get(cur) {
                Some(p) => { out.push(p); cur = p; }
                None => break,
            }
        }
        out.reverse();
        out
    }

    let mut prev: HashMap<&'a str, &'a str> = HashMap::new();
    let mut seen: HashSet<&'a str> = HashSet::new();
    let mut queue: VecDeque<&'a str> = VecDeque::new();
    seen.insert(start);
    queue.push_back(start);

    while let Some(cur) = queue.pop_front() {
        // no body here: an unmarked extern, the indirect-call sentinel, or an
        // unknown symbol. this is the leaf that allocates.
        let Some(callees) = calls.get(cur) else { return path_to(&prev, start, cur) };
        let mut dirty: Vec<&'a str> = callees.iter().copied()
            .filter(|c| !clean.get(c).copied().unwrap_or(false))
            .collect();
        // the fixpoint cannot make a dirty function with no dirty callee. treat
        // it as a leaf anyway, so this stays total instead of looping.
        if dirty.is_empty() { return path_to(&prev, start, cur); }
        dirty.sort_unstable();
        for d in dirty {
            if seen.insert(d) { prev.insert(d, cur); queue.push_back(d); }
        }
    }
    vec![start]
}

pub fn alloc_check_program<'a>(
    program: &[TopLevel<'a>],
    defs: &Defs<'a>,
    node_types: &HashMap<usize, Type<'a>>,
) -> Result<(), Vec<Error>> {
    let r = Resolve { node_types, members: defs.members(), instances: defs.instances() };
    let (clean, calls) = compute_clean(program, &r);

    // mono mangles generic call names. prefer the friendly spelling it
    // recorded (`alloc::<Vec2>`) over `std.alloc$alloc$Vec2`.
    let show = |n: &'a str| defs.show_symbol(n);

    // a dirty `@alloc(false)` function always has at least one dirty immediate
    // callee to blame: dirtiness only ever arrives through one.
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
            let err = Error::new(span, format!(
                "'{}' is marked as @alloc(false) but may allocate", show(name)))
                .with_label(span, format!("calls '{}', which may allocate", show(callee)));
            // the immediate callee is just the entry point; name where the
            // allocation really is.
            let chain = blame_chain(&calls, &clean, callee);
            errors.push(if chain.len() > 1 {
                let rendered = chain.iter().map(|n| match *n {
                    INDIRECT_CALLEE => INDIRECT_CALLEE.to_string(),
                    other => format!("'{}'", show(other)),
                }).collect::<Vec<_>>().join(" -> ");
                err.with_note(format!("allocation reaches it through: {}", rendered))
            } else {
                err
            });
        }
    }

    if errors.is_empty() { Ok(()) } else { Err(errors) }
}
