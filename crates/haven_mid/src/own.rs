//! Ownership: move checking and automatic `delete` insertion.
//!
//! A type *owns* something when it implements the prelude's `Delete` lang item,
//! or when one of its fields does. Every other type is `Copy`: a bitwise copy of
//! it is as good as the original, which is what every value in the language was
//! before this pass existed. So ownership is entirely opt-in - a program that
//! never writes `extend T: Delete` is unaffected by anything here.
//!
//! For an owning type two things change, and they define each other:
//!
//!   * **moves.** Assigning or passing the value by value transfers ownership;
//!     the source is dead afterwards and using it is an error. Without this,
//!     `let b = a;` on a heap-owning struct silently makes two owners of one
//!     allocation - the aliasing bug in `sketch_uaf.hv`.
//!   * **destruction.** The owner's `delete` is called for it: at the end of the
//!     scope that owns it, before it is overwritten, and on the way out of a
//!     `return`. Exactly once, and never on a value that has been moved away -
//!     which is precisely what move checking establishes.
//!
//! ## where this runs
//!
//! After monomorphization *and* after the second typecheck, for the same reason
//! trait dispatch resolves there: every type is concrete, so "is this `Copy`?"
//! has an answer without a `Copy` bound on every generic parameter, and
//! `node_types` already says what every expression's type is. Generic templates
//! are skipped - they emit no code, exactly as in MIL lowering.
//!
//! The `delete` calls this pass synthesizes are ordinary AST calls, and it
//! records their types and bindings into the typecheck context as it builds
//! them, so MIL lowering can lower them with no further typecheck pass.
//!
//! ## what is deliberately not handled
//!
//!   * **conditional drops.** A value moved on one path but not another would
//!     need a runtime drop flag. Rather than leak or double-free, this is a
//!     clean error at the point the drop would be emitted.
//!   * **partial moves.** Moving one field out of an owning value is rejected;
//!     only whole locals move.
//!   * **enum payloads and array elements** are never dropped automatically
//!     (that needs a tag switch and a loop respectively). Such a type is still
//!     correctly non-`Copy`, so it moves rather than aliases - it just leaks
//!     unless its enclosing type implements `Delete` itself.

use std::collections::{HashMap, HashSet};

use haven_common::ast::*;
use haven_common::defs::{DefId, TyHead};
use haven_common::layout::TypeTable;

use crate::typecheck::{Context, EnumDef, RecvAdjust};

/// The one method a `Delete` impl provides.
const DELETE_METHOD: &str = "delete";

/// Name given to the temporary that holds a return value while the returning
/// scope's owners are destroyed. Not a legal source identifier, so it cannot
/// collide with a user's local.
const RET_TEMP: &str = "$ret";

/// One `delete` call needed to destroy a value in place: the chain of fields to
/// walk from the owner (empty when the owner implements `Delete` itself), and
/// the function to call on the address of what that chain reaches.
#[derive(Clone, Debug)]
struct DropPath<'a> {
    /// `(field name, type of the access)`, outermost first.
    steps: Vec<(&'a str, Type<'a>)>,
    /// Emitted name of the `delete` to call.
    target: &'a str,
}

/// Which types own something, and how to destroy them.
struct Model<'a> {
    types: TypeTable<'a>,
    enums: HashMap<DefId, EnumDef<'a>>,
    /// Types with a `Delete` impl of their own -> the emitted name of their
    /// `delete`. A type not in here may still own something *through a field*.
    deletes: HashMap<DefId, &'a str>,
    /// Every `delete` symbol, so an explicit call to one can be rejected.
    delete_fns: HashSet<&'a str>,
}

impl<'a> Model<'a> {
    /// Whether a value of this type can be duplicated by copying its bits.
    ///
    /// The recursion terminates because the only way to be non-`Copy` is to
    /// reach a `Delete` impl, and a type cannot contain itself by value.
    fn is_copy(&self, ty: &Type<'a>) -> bool {
        match ty {
            Type::Named { def, .. } => {
                if self.deletes.contains_key(def) { return false; }
                // an enum's payload fields, not the `{ $tag, $payload }`
                // aggregate - the blob is opaque bytes and would look `Copy`.
                if let Some(e) = self.enums.get(def) {
                    return e.payloads.values().flatten().all(|(_, t)| self.is_copy(t));
                }
                match self.types.get(def) {
                    Some(info) => info.fields.iter().all(|(_, t)| self.is_copy(t)),
                    None => true,
                }
            }
            Type::Array(inner, _) => self.is_copy(inner),
            // primitives, pointers, `str`, slices, SIMD vectors and function
            // pointers are all plain values: a pointer *to* an owner is a
            // borrow, and copying it transfers nothing.
            _ => true,
        }
    }

    /// The `delete` calls that destroy a value of `ty` held at `steps` so far.
    /// A type implementing `Delete` is destroyed by its own `delete`, and its
    /// fields are that method's business, so the walk stops there.
    fn drop_paths(&self, ty: &Type<'a>, steps: &mut Vec<(&'a str, Type<'a>)>, out: &mut Vec<DropPath<'a>>) {
        let Type::Named { def, .. } = ty else { return };
        if let Some(target) = self.deletes.get(def) {
            out.push(DropPath { steps: steps.clone(), target });
            return;
        }
        // an enum's fields are its tag and an opaque payload blob; destroying
        // the live variant would need a switch on the tag (see module docs).
        if self.enums.contains_key(def) { return; }
        let Some(info) = self.types.get(def) else { return };
        for (fname, fty) in info.fields.clone() {
            steps.push((fname, fty.clone()));
            self.drop_paths(&fty, steps, out);
            steps.pop();
        }
    }

    fn drops_for(&self, ty: &Type<'a>) -> Vec<DropPath<'a>> {
        let mut out = Vec::new();
        self.drop_paths(ty, &mut Vec::new(), &mut out);
        out
    }
}

/// What has happened to a binding's value at some program point. A binding not
/// in the flow map is live.
#[derive(Clone, Debug)]
enum State {
    /// Moved away on every path reaching here.
    Moved(Span),
    /// Moved on some paths but not others - the case a runtime drop flag would
    /// cover. Using it is an error, and so is dropping it.
    Maybe(Span),
}

impl State {
    fn span(&self) -> Span {
        match self { State::Moved(s) | State::Maybe(s) => *s }
    }
}

type Flow<'a> = HashMap<Binding<'a>, State>;

/// A binding that may need destroying when its scope ends.
struct Slot<'a> {
    binding: Binding<'a>,
    name: &'a str,
    ty: Type<'a>,
    drops: Vec<DropPath<'a>>,
}

/// Where a place expression lives, for deciding whether consuming it is a move.
enum Root<'a> {
    /// The whole of a local or parameter: moving it is allowed.
    Whole(Binding<'a>),
    /// A field of one: a partial move, which is not supported.
    Field(Binding<'a>),
    /// Behind a pointer or an index - not ours to move out of.
    Borrowed,
    /// A freshly produced value (a call result, a literal): consuming it just
    /// takes ownership of something nobody else names.
    Temp,
}

struct Checker<'a, 'c> {
    model: &'c Model<'a>,
    cx: &'c mut Context<'a>,
    errors: Vec<Error>,
    /// Innermost-last stack of scopes, each listing what it owns in declaration
    /// order. Scope 0 is the function's parameters plus its top-level locals.
    scopes: Vec<Vec<Slot<'a>>>,
    /// `scopes.len()` at the point each enclosing loop was entered, so `break`
    /// and `continue` know how many scopes they leave.
    loops: Vec<usize>,
    moved: Flow<'a>,
    ret_ty: Type<'a>,
}

impl<'a, 'c> Checker<'a, 'c> {
    fn error(&mut self, span: Span, msg: String) {
        self.errors.push(Error::new(span, msg));
    }

    fn ty_of(&self, e: &Expr<'a>) -> Option<Type<'a>> {
        self.cx.node_types.get(&e.id).cloned()
    }

    fn slot(&self, b: Binding<'a>) -> Option<&Slot<'a>> {
        self.scopes.iter().flatten().find(|s| s.binding == b)
    }

    fn find_slot(&self, b: Binding<'a>) -> Option<(usize, usize)> {
        self.scopes.iter().enumerate().find_map(|(i, sc)| {
            sc.iter().position(|s| s.binding == b).map(|j| (i, j))
        })
    }

    /// How a binding reads in a diagnostic.
    fn binding_name(&self, b: Binding<'a>) -> &'a str {
        match b {
            Binding::Param(name) => name,
            Binding::Local(_) => self.slot(b).map_or("<local>", |s| s.name),
        }
    }

    // --- analysis

    fn root(&self, e: &Expr<'a>) -> Root<'a> {
        match &e.value {
            ExprNode::Var(_) => match self.cx.resolved.get(&e.id) {
                Some(b) => Root::Whole(*b),
                // a global or a bare function name: not a binding we track.
                None => Root::Temp,
            },
            ExprNode::Access { base, .. } => {
                // `p.field` through a pointer reaches someone else's storage,
                // not a part of the local `p`.
                if matches!(self.ty_of(base), Some(Type::Pointer(_))) {
                    return Root::Borrowed;
                }
                match self.root(base) {
                    Root::Whole(b) | Root::Field(b) => Root::Field(b),
                    other => other,
                }
            }
            ExprNode::Index { .. } | ExprNode::Unary { op: UnaryOp::Deref, .. } => Root::Borrowed,
            _ => Root::Temp,
        }
    }

    /// Reading a binding: an error if its value has been moved away.
    fn read(&mut self, b: Binding<'a>, span: Span) {
        let Some(state) = self.moved.get(&b).cloned() else { return };
        let name = self.binding_name(b);
        let msg = match state {
            State::Moved(_) => format!(
                "use of '{}' after its value was moved out of it", name),
            State::Maybe(_) => format!(
                "use of '{}', whose value is moved away on some paths reaching here", name),
        };
        self.error(span, msg);
        // report the first offending use only. the value really is gone, but
        // repeating the message at every later use buries the one that matters.
        self.moved.remove(&b);
    }

    /// Walk an expression for the values it reads and the arguments it consumes.
    fn visit(&mut self, e: &Expr<'a>) {
        match &e.value {
            ExprNode::Var(_) => {
                if let Some(b) = self.cx.resolved.get(&e.id).copied() {
                    self.read(b, e.span);
                }
            }
            ExprNode::Access { base, .. } => self.visit(base),
            ExprNode::Index { slice, index } => { self.visit(slice); self.visit(index); }
            ExprNode::Unary { operand, .. } => self.visit(operand),
            ExprNode::Binary { left, right, .. } => { self.visit(left); self.visit(right); }
            // a literal takes ownership of whatever is put into it.
            ExprNode::Struct { fields, .. } => {
                for (_, f) in fields { self.consume(f); }
            }
            ExprNode::Slice(elements) => {
                for x in elements { self.consume(x); }
            }
            ExprNode::Call { func, args, .. } => {
                if let Some(mc) = self.cx.method_calls.get(&e.id).cloned() {
                    let ExprNode::Access { base, .. } = &func.value else {
                        unreachable!("method call callee is always a field access")
                    };
                    match mc.adjust {
                        // `&recv`: a borrow, the receiver keeps its value.
                        RecvAdjust::AddrOf => { self.visit(base); self.borrowed_temp(base); }
                        // a by-value `self` takes the receiver with it.
                        RecvAdjust::AsIs => self.consume(base),
                    }
                    if self.model.delete_fns.contains(mc.target) {
                        self.error(e.span, format!(
                            "'{}' is a destructor and is called for you when the value is \
                             destroyed; calling it here would release the resource twice",
                            DELETE_METHOD));
                    }
                } else {
                    self.visit(func);
                }
                for a in args { self.consume(a); }
            }
            _ => {}
        }
    }

    /// A `*self` method called on a temporary: `make_buf().len()`.
    ///
    /// Lowering gives the temporary a slot to be borrowed from, which is all a
    /// `Copy` receiver needs. An owning one is different: the slot is not a
    /// binding, so no scope lists it and nothing ever calls its `delete` - the
    /// resource would leak, silently and every time. That is the one thing this
    /// pass exists to prevent, so require a `let` instead of accepting it.
    fn borrowed_temp(&mut self, base: &Expr<'a>) {
        // a bare name reaches `Temp` only when it is a module-level global.
        // Lowering does copy one of those, but a constant's resource is static
        // and was never acquired by this call, so there is nothing here to leak.
        if matches!(base.value, ExprNode::Var(_)) { return; }
        if !matches!(self.root(base), Root::Temp) { return; }
        let Some(ty) = self.ty_of(base) else { return };
        if self.model.is_copy(&ty) { return; }
        let shown = self.cx.show(&ty);
        self.error(base.span, format!(
            "cannot call a method on this temporary: `{}` owns a resource, and a \
             temporary belongs to no scope, so its '{}' would never run and the \
             resource would leak. Bind it with a `let` first, then call the \
             method on that",
            shown, DELETE_METHOD));
    }

    /// An expression in a position that takes its value: a `let` initializer, an
    /// argument, a field of a literal, the operand of `return`.
    fn consume(&mut self, e: &Expr<'a>) {
        self.visit(e);
        let Some(ty) = self.ty_of(e) else { return };
        if self.model.is_copy(&ty) { return; }
        match self.root(e) {
            Root::Whole(b) => { self.moved.insert(b, State::Moved(e.span)); }
            Root::Field(b) => {
                let name = self.binding_name(b);
                let shown = self.cx.show(&ty);
                self.error(e.span, format!(
                    "cannot move a '{}' out of a field of '{}': it owns a resource, and moving \
                     one field out of a value would leave the rest without an owner. Borrow it \
                     with '&' instead", shown, name));
            }
            Root::Borrowed => {
                let shown = self.cx.show(&ty);
                self.error(e.span, format!(
                    "cannot move a '{}' out of a pointer or an element: it owns a resource, and \
                     the place it is moved out of would be left invalid. Borrow it with '&' \
                     instead", shown));
            }
            // a value nobody else names; ownership travels with it.
            Root::Temp => {}
        }
    }

    /// Whether `e` reads binding `b` anywhere.
    fn reads(&self, e: &Expr<'a>, b: Binding<'a>) -> bool {
        let mut found = false;
        self.walk_reads(e, &mut |x| if x == b { found = true; });
        found
    }

    /// The first binding `e` reads that is owned by a scope about to be unwound.
    fn first_dropped_read(&self, e: &Expr<'a>) -> Option<Binding<'a>> {
        let owned: HashSet<Binding<'a>> = self.scopes.iter().flatten()
            .filter(|s| !self.moved.contains_key(&s.binding))
            .map(|s| s.binding)
            .collect();
        let mut hit = None;
        self.walk_reads(e, &mut |b| if hit.is_none() && owned.contains(&b) { hit = Some(b); });
        hit
    }

    fn walk_reads(&self, e: &Expr<'a>, f: &mut impl FnMut(Binding<'a>)) {
        if let Some(b) = self.cx.resolved.get(&e.id).copied() { f(b); }
        match &e.value {
            ExprNode::Access { base, .. } => self.walk_reads(base, f),
            ExprNode::Index { slice, index } => { self.walk_reads(slice, f); self.walk_reads(index, f); }
            ExprNode::Unary { operand, .. } => self.walk_reads(operand, f),
            ExprNode::Binary { left, right, .. } => { self.walk_reads(left, f); self.walk_reads(right, f); }
            ExprNode::Struct { fields, .. } => for (_, x) in fields { self.walk_reads(x, f) },
            ExprNode::Slice(elements) => for x in elements { self.walk_reads(x, f) },
            ExprNode::Call { func, args, .. } => {
                self.walk_reads(func, f);
                for a in args { self.walk_reads(a, f); }
            }
            _ => {}
        }
    }

    // --- scopes and destruction

    fn declare(&mut self, binding: Binding<'a>, name: &'a str, ty: Type<'a>) {
        // re-entering a scope re-declares; the binding is live again.
        self.moved.remove(&binding);
        let drops = self.model.drops_for(&ty);
        if drops.is_empty() { return; }
        self.scopes.last_mut().unwrap().push(Slot { binding, name, ty, drops });
    }

    /// The `delete` calls destroying one slot, or nothing if its value has been
    /// moved away. A value moved on only some paths is the case that cannot be
    /// decided statically, and is reported here rather than guessed at.
    fn destroy(&mut self, scope: usize, index: usize, span: Span) -> Vec<Stmt<'a>> {
        let (binding, name, ty, drops) = {
            let s = &self.scopes[scope][index];
            (s.binding, s.name, s.ty.clone(), s.drops.clone())
        };
        match self.moved.get(&binding) {
            Some(State::Moved(_)) => return Vec::new(),
            Some(State::Maybe(at)) => {
                let at = *at;
                self.error(at, format!(
                    "'{}' owns a resource and is moved away here, but not on every path that \
                     reaches the end of its scope, so whether it still needs destroying is not \
                     decidable at compile time. Move it on every path, or on none",
                    name));
                return Vec::new();
            }
            None => {}
        }
        drops.iter()
            .map(|p| self.delete_call(binding, name, &ty, p, span))
            .collect()
    }

    /// Destroy everything owned by scopes `from ..`, innermost and
    /// latest-declared first.
    fn unwind(&mut self, from: usize, span: Span) -> Vec<Stmt<'a>> {
        let mut out = Vec::new();
        for scope in (from..self.scopes.len()).rev() {
            for index in (0..self.scopes[scope].len()).rev() {
                let stmts = self.destroy(scope, index, span);
                out.extend(stmts);
            }
        }
        out
    }

    // --- synthesis
    //
    // Every node built here is registered in the typecheck context as it is
    // made, because nothing typechecks the program again after this pass: MIL
    // lowering reads `node_types` for the call's signature and `resolved` for
    // the owner's storage slot, and both have to be there.

    fn expr(&mut self, node: ExprNode<'a>, ty: Type<'a>, span: Span) -> Expr<'a> {
        let e = Metadata::new(node, span);
        self.cx.node_types.insert(e.id, ty);
        e
    }

    /// `delete(&owner.field...)` as a statement.
    fn delete_call(&mut self, binding: Binding<'a>, name: &'a str, ty: &Type<'a>,
                   path: &DropPath<'a>, span: Span) -> Stmt<'a> {
        let root = self.expr(ExprNode::Var(name), ty.clone(), span);
        self.cx.resolved.insert(root.id, binding);

        let mut place = root;
        let mut place_ty = ty.clone();
        for (field, fty) in path.steps.iter() {
            place = self.expr(
                ExprNode::Access { base: Box::new(place), field: *field },
                fty.clone(), span);
            place_ty = fty.clone();
        }

        let ptr_ty = Type::Pointer(Box::new(place_ty));
        let addr = self.expr(
            ExprNode::Unary { op: UnaryOp::AddrOf, operand: Box::new(place) },
            ptr_ty.clone(), span);
        // the callee's type is what MIL reads to coerce the argument; a
        // destructor is always `proc(*T) void`.
        let func = self.expr(ExprNode::Var(path.target), Type::Function {
            params: vec![ptr_ty],
            return_type: Box::new(Type::Void),
        }, span);
        let call = self.expr(ExprNode::Call {
            func: Box::new(func), type_args: Vec::new(), args: vec![addr],
        }, Type::Void, span);
        Metadata::new(StmtNode::Expr(call), span)
    }

    // --- statements

    /// Process one statement, appending it (and any destruction around it) to
    /// `out`. Returns whether control definitely leaves here.
    fn one(&mut self, stmt: Stmt<'a>, out: &mut Vec<Stmt<'a>>) -> bool {
        let Metadata { span, id, value } = stmt;

        match value {
            StmtNode::Expr(e) => {
                self.visit(&e);
                // a produced owner that is never bound has no owner to destroy it.
                if let Some(ty) = self.ty_of(&e) {
                    if !self.model.is_copy(&ty) && matches!(self.root(&e), Root::Temp) {
                        let shown = self.cx.show(&ty);
                        self.error(span, format!(
                            "this '{}' owns a resource but is discarded without an owner, so it \
                             would never be destroyed. Bind it with `let`", shown));
                    }
                }
                out.push(Metadata { span, id, value: StmtNode::Expr(e) });
                false
            }

            StmtNode::Block(stmts) => {
                let (stmts, diverged) = self.scoped(stmts, span);
                out.push(Metadata { span, id, value: StmtNode::Block(stmts) });
                diverged
            }

            StmtNode::Declare { name, ty, value } => {
                self.consume(&value);
                out.push(Metadata { span, id, value: StmtNode::Declare { name, ty: ty.clone(), value } });
                // the binding identity is the `Declare`'s node id, matching what
                // name resolution recorded for every use of it.
                self.declare(Binding::Local(id), name, ty);
                false
            }

            StmtNode::Assign { left, value } => {
                self.consume(&value);
                match self.root(&left) {
                    // overwriting a whole local: its old value, if it still has
                    // one, has to be destroyed first.
                    Root::Whole(b) => {
                        if self.slot(b).is_some() {
                            if self.reads(&value, b) {
                                let name = self.binding_name(b);
                                self.error(span, format!(
                                    "cannot overwrite '{}': it owns a resource that must be \
                                     destroyed first, but the new value reads it. Bind the new \
                                     value with `let` before assigning", name));
                            } else {
                                let (i, j) = self.find_slot(b).unwrap();
                                let drops = self.destroy(i, j, span);
                                out.extend(drops);
                            }
                        }
                        // whatever it held before, it holds a fresh value now.
                        self.moved.remove(&b);
                    }
                    _ => {
                        // the left side is read: `x.f = ...` needs a live `x`.
                        self.visit(&left);
                        if let Some(ty) = self.ty_of(&left) {
                            if !self.model.is_copy(&ty) {
                                let shown = self.cx.show(&ty);
                                self.error(span, format!(
                                    "cannot overwrite a '{}' in place: it owns a resource, and the \
                                     value being replaced would never be destroyed", shown));
                            }
                        }
                    }
                }
                out.push(Metadata { span, id, value: StmtNode::Assign { left, value } });
                false
            }

            StmtNode::If { condition, then_branch, else_branch } => {
                self.visit(&condition);
                let entry = self.moved.clone();
                let (then_branch, then_div) = self.branch(then_branch);
                let then_state = std::mem::replace(&mut self.moved, entry.clone());
                let (else_branch, else_div) = match else_branch {
                    Some(b) => { let (b, d) = self.branch(b); (Some(b), d) }
                    None => (None, false),
                };
                let else_state = std::mem::replace(&mut self.moved, entry.clone());
                self.moved = merge(&entry, then_state, then_div, else_state, else_div);
                out.push(Metadata { span, id, value: StmtNode::If { condition, then_branch, else_branch } });
                then_div && else_div
            }

            StmtNode::While { condition, body } => {
                self.visit(&condition);
                let entry = self.moved.clone();
                let depth = self.scopes.len();
                self.loops.push(depth);
                let (body, _) = self.branch(body);
                self.loops.pop();
                // moving something declared *outside* the loop happens again on
                // the next iteration, on a value that is already gone. anything
                // declared inside is a fresh binding each time round, and its
                // scope has been popped by now.
                let escaped: Vec<(Binding<'a>, Span)> = self.moved.iter()
                    .filter(|(b, _)| !entry.contains_key(b) && self.slot(**b).is_some())
                    .map(|(b, s)| (*b, s.span()))
                    .collect();
                for (b, at) in escaped {
                    let name = self.binding_name(b);
                    self.error(at, format!(
                        "'{}' is declared outside this loop and its value is moved away inside it, \
                         so the next iteration would move a value that is already gone", name));
                }
                // the body may run zero times, so nothing it did is guaranteed.
                self.moved = entry;
                out.push(Metadata { span, id, value: StmtNode::While { condition, body } });
                false
            }

            StmtNode::Match { scrutinee, arms } => {
                // payload bindings are views into the scrutinee's storage rather
                // than copies, so an arm borrows rather than takes.
                self.visit(&scrutinee);
                let entry = self.moved.clone();
                let mut merged: Option<(Flow<'a>, bool)> = None;
                let mut new_arms = Vec::with_capacity(arms.len());
                for (pat, body) in arms {
                    self.moved = entry.clone();
                    let (body, div) = self.branch(body);
                    let state = std::mem::take(&mut self.moved);
                    merged = Some(match merged {
                        None => (state, div),
                        Some((acc, acc_div)) =>
                            (merge(&entry, acc, acc_div, state, div), acc_div && div),
                    });
                    new_arms.push((pat, body));
                }
                let (state, all_diverge) = merged.unwrap_or((entry, false));
                self.moved = state;
                out.push(Metadata { span, id, value: StmtNode::Match { scrutinee, arms: new_arms } });
                all_diverge
            }

            StmtNode::Return(e) => {
                self.consume(&e);
                let drops = self.unwind(0, span);
                if drops.is_empty() {
                    out.push(Metadata { span, id, value: StmtNode::Return(e) });
                } else if self.ret_ty == Type::Void {
                    // nothing carries a value out, so the owners can go first -
                    // as long as the returned expression does not read them.
                    if let Some(b) = self.first_dropped_read(&e) {
                        let name = self.binding_name(b);
                        self.error(span, format!(
                            "'{}' is destroyed when this scope ends, but the returned expression \
                             reads it", name));
                    }
                    out.extend(drops);
                    out.push(Metadata { span, id, value: StmtNode::Return(e) });
                } else {
                    // the value has to be computed before this scope's owners are
                    // destroyed - it may well be reading them - so it is bound to
                    // a temporary that outlives them.
                    let ty = self.ret_ty.clone();
                    let decl = Metadata::new(StmtNode::Declare {
                        name: RET_TEMP, ty: ty.clone(), value: e,
                    }, span);
                    let var = self.expr(ExprNode::Var(RET_TEMP), ty, span);
                    self.cx.resolved.insert(var.id, Binding::Local(decl.id));
                    out.push(decl);
                    out.extend(drops);
                    out.push(Metadata { span, id, value: StmtNode::Return(var) });
                }
                true
            }

            StmtNode::Break => {
                let from = self.loops.last().copied().unwrap_or(self.scopes.len());
                let drops = self.unwind(from, span);
                out.extend(drops);
                out.push(Metadata { span, id, value: StmtNode::Break });
                true
            }
            StmtNode::Continue => {
                let from = self.loops.last().copied().unwrap_or(self.scopes.len());
                let drops = self.unwind(from, span);
                out.extend(drops);
                out.push(Metadata { span, id, value: StmtNode::Continue });
                true
            }
        }
    }

    /// One branch of an `if`/`while`/`match`: a single statement that may have
    /// to grow into a block once destruction is inserted around it.
    fn branch(&mut self, stmt: Box<Stmt<'a>>) -> (Box<Stmt<'a>>, bool) {
        let span = stmt.span;
        let mut out = Vec::new();
        let diverged = self.one(*stmt, &mut out);
        let stmt = if out.len() == 1 {
            out.pop().unwrap()
        } else {
            Metadata::new(StmtNode::Block(out), span)
        };
        (Box::new(stmt), diverged)
    }

    /// A statement list that owns a scope: whatever it declares is destroyed
    /// when it ends, unless control already left.
    fn scoped(&mut self, stmts: Vec<Stmt<'a>>, span: Span) -> (Vec<Stmt<'a>>, bool) {
        self.scopes.push(Vec::new());
        let (mut out, diverged) = self.sequence(stmts);
        if !diverged {
            let depth = self.scopes.len() - 1;
            let drops = self.unwind(depth, span);
            out.extend(drops);
        }
        self.scopes.pop();
        (out, diverged)
    }

    fn sequence(&mut self, stmts: Vec<Stmt<'a>>) -> (Vec<Stmt<'a>>, bool) {
        let mut out = Vec::with_capacity(stmts.len());
        let mut diverged = false;
        for stmt in stmts {
            if diverged {
                // unreachable after a return/break/continue. the typechecker
                // already accepted it; keep it, but analyse nothing.
                out.push(stmt);
                continue;
            }
            diverged = self.one(stmt, &mut out);
        }
        (out, diverged)
    }
}

/// Combine the flow states of two alternative paths. A binding moved on both is
/// moved; moved on one is only *maybe* moved, which is an error to use and an
/// error to drop.
fn merge<'a>(entry: &Flow<'a>, a: Flow<'a>, a_div: bool, b: Flow<'a>, b_div: bool) -> Flow<'a> {
    // a path that leaves (returns, breaks) contributes nothing to what holds
    // afterwards - it never gets there.
    match (a_div, b_div) {
        (true, true) => return entry.clone(),
        (true, false) => return b,
        (false, true) => return a,
        (false, false) => {}
    }
    let keys: HashSet<Binding<'a>> = a.keys().chain(b.keys()).copied().collect();
    keys.into_iter().map(|k| {
        let state = match (a.get(&k), b.get(&k)) {
            (Some(State::Moved(s)), Some(State::Moved(_))) => State::Moved(*s),
            (Some(x), _) => State::Maybe(x.span()),
            (None, Some(x)) => State::Maybe(x.span()),
            (None, None) => unreachable!("key came from one of the two maps"),
        };
        (k, state)
    }).collect()
}

/// Move-check the program and insert the `delete` calls it needs.
///
/// `impls` is the conformance list from name resolution rather than `cx.impls`:
/// monomorphization drops the trait declarations, so the post-mono typecheck
/// pass has none to record and the context's own copy is empty by this point.
/// It says only *which* types own something; the destructor to call for each
/// comes from the member table, so that a generic owner resolves to its
/// instance's `delete` rather than to the template's.
pub fn ownership_check<'a>(
    program: &mut [TopLevel<'a>],
    cx: &mut Context<'a>,
    delete_trait: Option<DefId>,
    impls: &[ImplDecl<'a>],
) -> Result<(), Vec<Error>> {
    // no `Delete` trait (or nobody implements it) means nothing in this program
    // owns anything, and every value is `Copy` exactly as it was before.
    let Some(delete_trait) = delete_trait else { return Ok(()) };

    // which definitions own something. `extend Vec<T>: Delete` records the
    // *template*, so a concrete `Vec$i32` is an owner by way of the template it
    // instantiates - conformance is a property of the generic type, not of each
    // instance separately. Conformance checking has already rejected a `Delete`
    // impl on anything but a named type, so a non-`Def` head cannot appear here.
    let owning: HashSet<DefId> = impls.iter()
        .filter(|i| i.trait_ == delete_trait)
        .filter_map(|i| match i.head { TyHead::Def(d) => Some(d), _ => None })
        .collect();

    // the destructor to call for each owning type, taken from the member table
    // rather than from `impls` directly: for a generic owner the name wanted is
    // the *instance's* `delete` (`Vec$delete$i32`), which monomorphization minted
    // and registered against the instance when it created it. Reading `impls`
    // alone would find only the template's, which has no code by this point.
    let mut deletes: HashMap<DefId, &'a str> = HashMap::new();
    for ((head, name), m) in cx.members.iter() {
        if *name != DELETE_METHOD { continue; }
        let TyHead::Def(d) = head else { continue };
        let owner = owning.contains(d)
            || cx.instances.get(d).is_some_and(|t| owning.contains(t));
        // a missing `delete` was already reported by conformance checking.
        if owner { deletes.insert(*d, m.name); }
    }
    if deletes.is_empty() { return Ok(()); }

    let model = Model {
        types: cx.types.clone(),
        enums: cx.enums.clone(),
        delete_fns: deletes.values().copied().collect(),
        deletes,
    };

    let mut errors: Vec<Error> = Vec::new();
    for node in program.iter_mut() {
        let span = node.span;
        let TopLevelNode::Function { generics, params, return_type, body, .. } = &mut node.value
            else { continue };
        // a generic template emits no code; its instances are checked instead,
        // and only they have the concrete types this pass needs.
        if !generics.is_empty() { continue; }

        let mut ck = Checker {
            model: &model,
            cx: &mut *cx,
            errors: Vec::new(),
            scopes: vec![Vec::new()],
            loops: Vec::new(),
            moved: Flow::new(),
            ret_ty: return_type.clone(),
        };
        // a by-value parameter of an owning type was moved into this call, so
        // this function destroys it. That is what makes passing one a transfer
        // of ownership rather than a second owner of the same resource.
        for (pname, pty) in params.iter() {
            ck.declare(Binding::Param(pname), pname, pty.clone());
        }
        let (mut stmts, diverged) = ck.sequence(std::mem::take(body));
        if !diverged {
            let drops = ck.unwind(0, span);
            stmts.extend(drops);
        }
        errors.append(&mut ck.errors);
        *body = stmts;
    }

    // one report per distinct problem: a generic instantiated twice would
    // otherwise raise the same error at the same place once per instance.
    let mut seen: HashSet<(FileId, usize, String)> = HashSet::new();
    errors.retain(|e| seen.insert((e.span.file, e.span.start, e.msg.clone())));

    if errors.is_empty() { Ok(()) } else { Err(errors) }
}
