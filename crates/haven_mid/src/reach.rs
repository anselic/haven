//! Reachability pruning: drop every function and extern that nothing can call.
//!
//! The module resolver puts the whole prelude, and every imported module, into
//! one flat program. So a ten-line `main` pulls all of `std` through mono,
//! typecheck, MIL and LLVM emission. clang's `-O3` removes the unused
//! `internal` functions from the binary anyway, so this pass does not change
//! the output. It is here so that the rest of the pipeline (and clang's parser)
//! never sees the dead code, and so the emitted `.ll` stays (somewhat) readable.
//!
//! The pass runs after ownership and before MIL. Ownership is the last pass
//! that adds new calls: the `delete`s it inserts. A destructor can be reachable
//! only through one of those calls, and we must still mark it, so the walk has
//! to see them.
//!
//! The analysis is a worklist over names. After mono, a reference to a function
//! has one of two shapes. Both of them name the emitted symbol directly:
//!
//! - `ExprNode::Var(symbol)`. This is the shape for a call, and also for a
//!   function used as a value, for example a function pointer kept in a local.
//!   We mark every `Var` that names a callable, not only the ones in call
//!   position. That is what makes address-taken functions roots instead of
//!   false negatives.
//! - a method call on a receiver. Its target is in `Context::method_calls`.
//!
//! If we keep too much, nothing bad happens, because clang removes it. If we
//! keep too little, we get a missing symbol at link time. So when a name is not
//! clear, for example a local that shadows a function, we keep the function.

use std::collections::{HashMap, HashSet};

use haven_common::ast::{Expr, ExprNode, Stmt, StmtNode, TopLevel, TopLevelNode};

use crate::typecheck::Context;

/// Remove from `program` every `Function` and `Extern` that is not reachable
/// from an `@export`ed function. Keep types, globals, and everything else.
/// They are cheap to emit, and the backend can need them for layout anyway.
pub fn prune_unreachable<'a>(program: &mut Vec<TopLevel<'a>>, cx: &Context<'a>) {
    // every callable, by the name it is emitted with, so we can tell if a
    // `Var` names a function
    let callables: HashMap<&'a str, usize> = program.iter().enumerate()
        .filter_map(|(i, item)| match &item.value {
            TopLevelNode::Function { name, .. }
            | TopLevelNode::Extern { name, .. } => Some((*name, i)),
            _ => None,
        })
        .collect();

    let mut reachable: HashSet<&'a str> = HashSet::new();
    let mut worklist: Vec<&'a str> = Vec::new();

    let mut walker = Walker {
        cx,
        callables: &callables,
        reachable: &mut reachable,
        worklist: &mut worklist,
    };

    // roots: everything with external linkage. Also every global initializer,
    // because a constant can hold a function pointer.
    for item in program.iter() {
        match &item.value {
            TopLevelNode::Function { name, attributes, .. }
                if attributes.iter().any(|a| a.value.name == "export") =>
            {
                walker.mark(name);
            }
            TopLevelNode::Global { value, .. } => walker.expr(value),
            _ => {}
        }
    }

    while let Some(name) = walker.worklist.pop() {
        // an extern has no body. we marked it, and that is all it needs.
        if let TopLevelNode::Function { body, .. } = &program[callables[name]].value {
            for stmt in body { walker.stmt(stmt); }
        }
    }

    program.retain(|item| match &item.value {
        TopLevelNode::Function { name, .. } | TopLevelNode::Extern { name, .. } => reachable.contains(name),
        _ => true,
    });
}

struct Walker<'w, 'a> {
    cx: &'w Context<'a>,
    callables: &'w HashMap<&'a str, usize>,
    reachable: &'w mut HashSet<&'a str>,
    worklist: &'w mut Vec<&'a str>,
}

impl<'w, 'a> Walker<'w, 'a> {
    fn mark(&mut self, name: &'a str) {
        if self.reachable.insert(name) {
            self.worklist.push(name);
        }
    }

    fn stmt(&mut self, stmt: &Stmt<'a>) {
        match &stmt.value {
            StmtNode::Expr(e) => self.expr(e),
            StmtNode::Block(stmts) => for s in stmts { self.stmt(s) },
            StmtNode::Declare { value, .. } => self.expr(value),
            StmtNode::Assign { left, value } => { self.expr(left); self.expr(value); }
            StmtNode::If { condition, then_branch, else_branch } => {
                self.expr(condition);
                self.stmt(then_branch);
                if let Some(e) = else_branch { self.stmt(e); }
            }
            StmtNode::While { condition, body } => { self.expr(condition); self.stmt(body); }
            StmtNode::Match { scrutinee, arms } => {
                self.expr(scrutinee);
                for (_, body) in arms { self.stmt(body); }
            }
            StmtNode::Return(value) => if let Some(e) = value { self.expr(e) },
            StmtNode::Continue | StmtNode::Break => {}
        }
    }

    fn expr(&mut self, expr: &Expr<'a>) {
        match &expr.value {
            // the only place a function is named. We do not check if this `Var`
            // is really a local. If a local shadows a function, we only keep a
            // function that clang would have removed anyway.
            ExprNode::Var(name) => {
                if self.callables.contains_key(name) { self.mark(name); }
            }
            ExprNode::Call { func, args, .. } => {
                // the callee of a method call is `recv.method` (an `Access`).
                // Only the typechecker knows the target it desugars to.
                if let Some(mc) = self.cx.method_calls.get(&expr.id) {
                    self.mark(mc.target);
                }
                self.expr(func);
                for a in args { self.expr(a); }
            }
            ExprNode::Struct { fields, .. } => for (_, e) in fields { self.expr(e) },
            ExprNode::Slice(elements) => for e in elements { self.expr(e) },
            ExprNode::Access { base, .. } => self.expr(base),
            ExprNode::Index { slice, index } => { self.expr(slice); self.expr(index); }
            ExprNode::Unary { operand, .. } => self.expr(operand),
            ExprNode::Binary { left, right, .. } => { self.expr(left); self.expr(right); }
            ExprNode::FnRef { .. } => unreachable!("FnRef eliminated by monomorphization"),
            ExprNode::Bool(_)
            | ExprNode::Int8(_) | ExprNode::Int16(_) | ExprNode::Int32(_) | ExprNode::Int64(_)
            | ExprNode::Uint8(_) | ExprNode::Uint16(_) | ExprNode::Uint32(_) | ExprNode::Uint64(_)
            | ExprNode::Float32(_) | ExprNode::Float64(_)
            | ExprNode::IntLit(_) | ExprNode::FloatLit(_)
            | ExprNode::Str(_) | ExprNode::Path(_) => {}
        }
    }
}
