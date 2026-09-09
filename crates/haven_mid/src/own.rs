//! Move checking and automatic `delete` insertion.
//!
//! A type owns a value when it or one of its fields implements the prelude's
//! `Delete` lang item. Passing an owner by value moves it; live owners are
//! destroyed at scope exit, before overwrite, and on return.
//!
//! ## where this runs
//!
//! This pass runs after monomorphization and the second typecheck, when all
//! emitted types are concrete. Inserted calls are recorded in the existing
//! typecheck context for MIL lowering.
//!
//! ## what is deliberately not handled
//!
//! Conditional drops and partial moves are rejected. Supporting them would
//! require runtime drop flags.
//!
//! Array elements and active enum payloads are included in drop paths.

use std::collections::{HashMap, HashSet};

use haven_common::ast::*;
use haven_common::defs::{DefId, TyHead};
use haven_common::layout::TypeTable;

use crate::intrinsics::Intrinsic;

use crate::typecheck::{Context, EnumDef, RecvAdjust};

/// The one method a `Delete` impl provides.
const DELETE_METHOD: &str = "delete";

/// Name given to the temporary that holds a return value while the returning
/// scope's owners are destroyed. Not a legal source identifier, so it cannot
/// collide with a user's local.
const RET_TEMP: &str = "$ret";

/// Name given to a borrowed owning temporary once it is spilled into a `let`
/// whose scope ends with the enclosing statement (see `hoist_stmt`). Like the
/// others, not a legal source identifier; the binding identity is the spill
/// `Declare`'s node id, so any number of them in one statement stay distinct.
const SPILL_TEMP: &str = "$tmp";

/// Locals introduced by the `drop_in_place` expansion: the base pointer, the
/// element count and the loop counter. Like `RET_TEMP` these are not legal
/// source identifiers, and each gets a fresh `Binding::Local` from its own
/// `Declare`, so nesting two expansions is fine despite the shared names.
const DIP_BASE: &str = "$dip_base";
const DIP_COUNT: &str = "$dip_count";
const DIP_INDEX: &str = "$dip_i";
/// Loop index of an array drop. Not shared with `$dip_i`: a `drop_in_place` over
/// an element type that is itself an array nests one inside the other, and
/// keeping the names apart keeps the emitted AST readable.
const DROP_INDEX: &str = "$drop_i";
/// Name bound to the payload field an enum drop path goes through. Like
/// `$drop_i` it may repeat across nested expansions; the `Bind` pattern's node
/// id is the identity, not the name.
const DROP_PAYLOAD: &str = "$drop_p";

/// One `delete` call needed to destroy a value in place: the chain of fields to
/// walk from the owner (empty when the owner implements `Delete` itself), and
/// the function to call on the address of what that chain reaches.
#[derive(Clone, Debug)]
struct DropPath<'a> {
    /// How to walk from the owner to the value being destroyed, outermost first.
    steps: Vec<Step<'a>>,
    /// Emitted name of the `delete` to call.
    target: &'a str,
}

/// One hop from a place towards something that needs destroying.
#[derive(Clone, Debug)]
enum Step<'a> {
    /// `.name`, yielding a value of the given type.
    Field(&'a str, Type<'a>),
    /// `[i]` for every `i` in `0..count`, yielding the element type.
    ///
    /// Unlike a field, this is not one access but a whole loop, which is why a
    /// step cannot simply be a name: the walk stops being a chain of `Access`
    /// nodes and becomes nested `while`s. Everything after this step happens
    /// once per element.
    Elems(Type<'a>, usize),
    /// One payload field of one variant, reached only when the tag says so.
    ///
    /// An enum's own fields are a tag and an opaque byte blob, so a payload is
    /// not addressable by name the way a struct field is - which is why this
    /// emits a `match` rather than an `Access`. The arm binds the field, the
    /// rest of the path continues from that binding, and a `_` arm covers every
    /// other variant (including the ones that own nothing).
    Payload {
        /// The enum being matched, for the arm's variant constant.
        enum_def: DefId,
        /// Which variant this path goes through.
        variant: &'a str,
        /// Index of the payload field within that variant.
        index: usize,
        /// How many payload fields the variant has, so the pattern has one
        /// sub-pattern per field - all `_` except `index`.
        arity: usize,
        /// The bound field's type.
        field_ty: Type<'a>,
    },
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
    fn drop_paths(&self, ty: &Type<'a>, steps: &mut Vec<Step<'a>>, out: &mut Vec<DropPath<'a>>) {
        // an array is inline storage for `count` values, so whatever destroys
        // one of them destroys every one - the same paths, walked under a loop.
        // `is_copy` has always looked through arrays, so a `[Res; 2]` was
        // correctly non-`Copy` and correctly move-checked; only the destruction
        // was missing, which made it a silent leak rather than a diagnostic.
        if let Type::Array(inner, count) = ty {
            steps.push(Step::Elems((**inner).clone(), count.expect_lit()));
            self.drop_paths(inner, steps, out);
            steps.pop();
            return;
        }
        let Type::Named { def, .. } = ty else { return };
        if let Some(target) = self.deletes.get(def) {
            out.push(DropPath { steps: steps.clone(), target });
            return;
        }
        // an enum owns through the payload of whichever variant is live, so
        // there is one path per (variant, owning payload field). Its own
        // `$tag`/`$payload` fields are deliberately not walked: the blob is
        // opaque bytes and would look like it owns nothing.
        if let Some(e) = self.enums.get(def) {
            // sorted so the emitted `match`es are in a stable order rather than
            // whatever the hash map happens to yield
            let mut variants: Vec<_> = e.payloads.iter().collect();
            variants.sort_by_key(|(name, _)| *name);
            for (variant, fields) in variants {
                for (index, (_, fty)) in fields.iter().enumerate() {
                    steps.push(Step::Payload {
                        enum_def: *def,
                        variant,
                        index,
                        arity: fields.len(),
                        field_ty: fty.clone(),
                    });
                    self.drop_paths(fty, steps, out);
                    steps.pop();
                }
            }
            return;
        }
        let Some(info) = self.types.get(def) else { return };
        for (fname, fty) in info.fields.clone() {
            steps.push(Step::Field(fname, fty.clone()));
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
#[derive(Clone, Copy)]
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
    /// The source name of every `Local` binding, for diagnostics. A `Slot` also
    /// carries a name, but only droppable values get a slot: a value moved out of
    /// an enum payload has none, and its move errors would otherwise read
    /// `<local>`. Keyed for every `Declare` and every match payload binding,
    /// whether or not it is droppable.
    names: HashMap<Binding<'a>, &'a str>,
    /// For a match payload binding, where the enum it is a view into lives.
    ///
    /// A payload binding is not storage of its own - it is a window onto the
    /// scrutinee - so moving it out is really a move of *that*, and whether it
    /// is allowed is a question about the scrutinee, not about the window. This
    /// is what stops `match (*p) { Some(v) -> return v }` from moving out of
    /// borrowed storage, and what marks the enum consumed when the move is legal
    /// so its own destructor does not then run over the hole.
    payload_src: HashMap<Binding<'a>, Root<'a>>,
}

impl<'a, 'c> Checker<'a, 'c> {
    /// Record one diagnostic, built by the caller so it can carry its own labels
    /// and note - every ownership error has a second place worth pointing at, or
    /// a rule worth restating, and neither belongs on the header line.
    fn push(&mut self, err: Error) {
        self.errors.push(err);
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

    /// How a binding reads in a diagnostic. `names` covers every source local,
    /// including the slot-less ones (payload move-outs, owning enum locals); the
    /// slot is a fallback and `<local>` a last resort for a synthetic binding.
    fn binding_name(&self, b: Binding<'a>) -> &'a str {
        match b {
            Binding::Param(name) => name,
            Binding::Local(_) => self.names.get(&b).copied()
                .or_else(|| self.slot(b).map(|s| s.name))
                .unwrap_or("<local>"),
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
        // the move site is the other half of the story, so it gets its own
        // underline rather than being described in prose at the use site.
        let err = match state {
            State::Moved(at) => Error::new(span, format!(
                "use of '{}' after its value was moved out of it", name))
                .with_label(span, "used here")
                .with_label(at, "value moved out here"),
            State::Maybe(at) => Error::new(span, format!(
                "use of '{}', whose value is moved away on some paths", name))
                .with_label(span, "used here")
                .with_label(at, "moved here, but not on every path reaching the use"),
        };
        self.push(err);
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
            // `&<temporary>` spills the value into a slot and borrows it - fine for
            // a `Copy` value, but an owning temporary belongs to no scope and its
            // `delete` would never run, the same leak `borrowed_temp` rejects for a
            // `*self` receiver on a temporary.
            ExprNode::Unary { op: UnaryOp::AddrOf, operand } => {
                self.visit(operand);
                self.borrowed_temp(operand);
            }
            ExprNode::Unary { operand, .. } => self.visit(operand),
            ExprNode::Binary { left, right, .. } => { self.visit(left); self.visit(right); }
            // a literal takes ownership of whatever is put into it.
            ExprNode::Struct { fields, .. } => {
                for (_, f) in fields { self.consume(f); }
            }
            ExprNode::Slice(elements) => {
                for x in elements { self.consume(x); }
            }
            // `[value; N]` puts one value into N slots. For a `Copy` element that
            // is just N copies of some bits and means nothing; for an owning one
            // it would hand the same resource to N owners, and the array's drop
            // glue would then release it N times. There is no repair available at
            // the literal - a repeat has one element expression, so there is
            // nowhere to put the `.clone()` calls the other N-1 slots would need -
            // so this is rejected rather than silently duplicated.
            //
            // `N == 1` is exempt: one value, one slot, an ordinary move.
            ExprNode::Repeat { value, count } => {
                let n = count.expect_lit();
                match self.ty_of(value) {
                    Some(ty) if n > 1 && !self.model.is_copy(&ty) => {
                        let shown = self.cx.show(&ty);
                        self.push(Error::new(e.span, format!(
                            "cannot repeat a '{}' into {} elements", shown, n))
                            .with_label(value.span, "it owns a resource")
                            .with_note("every slot would own the same resource and each would \
                                        release it; build the array elementwise, cloning into \
                                        each slot, or repeat a value that owns nothing"));
                    }
                    _ => {}
                }
                self.consume(value);
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
                        self.push(Error::new(e.span, format!(
                            "'{}' is a destructor and cannot be called directly", DELETE_METHOD))
                            .with_label(e.span, "calling it here would release the resource twice")
                            .with_note("it is called for you when the value is destroyed"));
                    }
                } else {
                    self.visit(func);
                }
                for a in args { self.consume(a); }
            }
            _ => {}
        }
    }

    /// Check a borrowed method receiver for an unscoped owning temporary.
    /// Ordinary statements are handled by `hoist_stmt`; `while` conditions
    /// cannot be hoisted and are rejected here.
    fn borrowed_temp(&mut self, base: &Expr<'a>) {
        // a bare name reaches `Temp` only when it is a module-level global.
        // Lowering does copy one of those, but a constant's resource is static
        // and was never acquired by this call, so there is nothing here to leak.
        if matches!(base.value, ExprNode::Var(_)) { return; }
        if !matches!(self.root(base), Root::Temp) { return; }
        let Some(ty) = self.ty_of(base) else { return };
        if self.model.is_copy(&ty) { return; }
        let shown = self.cx.show(&ty);
        self.push(Error::new(base.span, "cannot borrow this temporary")
            .with_label(base.span, format!("`{}` owns a resource", shown))
            .with_note(format!(
                "a temporary belongs to no scope, so its '{}' would never run and the \
                 resource would leak; bind it with a `let` first, then borrow that",
                DELETE_METHOD)));
    }

    // --- temporary lifetime extension

    /// Spill borrowed owning temporaries into locals that live through the
    /// enclosing statement. `while` conditions are excluded because their
    /// temporaries must be recreated on every iteration.
    fn hoist_stmt(&mut self, value: &mut StmtNode<'a>, decls: &mut Vec<Stmt<'a>>) {
        match value {
            StmtNode::Expr(e) => self.hoist_temps(e, decls),
            StmtNode::Return(Some(e)) => self.hoist_temps(e, decls),
            StmtNode::Declare { value, .. } => self.hoist_temps(value, decls),
            StmtNode::Assign { left, value } => {
                self.hoist_temps(left, decls);
                self.hoist_temps(value, decls);
            }
            StmtNode::If { condition, .. } => self.hoist_temps(condition, decls),
            // a match arm's payload bindings are *views* into the scrutinee's
            // storage, so a match borrows its scrutinee exactly as `&` does - and
            // an owning temporary borrowed that way has no binding, so nothing
            // would ever destroy it. Spilling gives it one, and the enum is then
            // dropped at the end of the statement (or moved, if an arm takes its
            // payload, which `payload_src` now attributes to the spill local).
            StmtNode::Match { scrutinee, .. } => {
                self.hoist_temps(scrutinee, decls);
                self.spill(scrutinee, decls);
            }
            StmtNode::While { .. } | StmtNode::Block(_)
            | StmtNode::Return(None) | StmtNode::Break | StmtNode::Continue => {}
        }
    }

    /// Walk `e` for owning temporaries in borrow position - the operand of a `&`
    /// and the receiver of a `&self` method call, exactly where `borrowed_temp`
    /// would object - and spill each. Children are visited first, so a temporary
    /// nested inside another is spilled before the one enclosing it and the
    /// outer `let` refers to the inner's spill local, not to a live temporary.
    fn hoist_temps(&mut self, e: &mut Expr<'a>, decls: &mut Vec<Stmt<'a>>) {
        let id = e.id;
        match &mut e.value {
            ExprNode::Access { base, .. } => self.hoist_temps(base, decls),
            ExprNode::Index { slice, index } => {
                self.hoist_temps(slice, decls);
                self.hoist_temps(index, decls);
            }
            ExprNode::Unary { op: UnaryOp::AddrOf, operand } => {
                self.hoist_temps(operand, decls);
                self.spill(operand, decls);
            }
            ExprNode::Unary { operand, .. } => self.hoist_temps(operand, decls),
            ExprNode::Binary { left, right, .. } => {
                self.hoist_temps(left, decls);
                self.hoist_temps(right, decls);
            }
            ExprNode::Struct { fields, .. } => {
                for (_, f) in fields { self.hoist_temps(f, decls); }
            }
            ExprNode::Slice(elements) => {
                for x in elements { self.hoist_temps(x, decls); }
            }
            ExprNode::Repeat { value, .. } => self.hoist_temps(value, decls),
            ExprNode::Call { func, args, .. } => {
                // a method call borrows its receiver when the adjust is `&recv`;
                // `func` is then the `recv.method` access whose base is that
                // receiver. A plain call has nothing borrowed at this node.
                match self.cx.method_calls.get(&id).map(|mc| matches!(mc.adjust, RecvAdjust::AddrOf)) {
                    Some(borrows_recv) => {
                        let ExprNode::Access { base, .. } = &mut func.value else {
                            unreachable!("method call callee is always a field access")
                        };
                        self.hoist_temps(base, decls);
                        if borrows_recv { self.spill(base, decls); }
                    }
                    None => self.hoist_temps(func, decls),
                }
                for a in args { self.hoist_temps(a, decls); }
            }
            _ => {}
        }
    }

    /// Replace `e` with a `let $tmp = <e>` (appended to `decls`) and a `Var`
    /// naming that local, when `e` is an owning temporary. Mirrors
    /// `borrowed_temp`'s guards: a bare `Var` (a global constant) owns nothing
    /// acquired here, and a `Copy` value needs no `delete`, so neither is
    /// spilled. The new `let` is processed like any other when the wrapping
    /// block runs, so its slot and drop fall out of the ordinary machinery.
    fn spill(&mut self, e: &mut Expr<'a>, decls: &mut Vec<Stmt<'a>>) {
        if matches!(e.value, ExprNode::Var(_)) { return; }
        if !matches!(self.root(e), Root::Temp) { return; }
        let Some(ty) = self.ty_of(e) else { return };
        if self.model.is_copy(&ty) { return; }
        let span = e.span;
        // a typed placeholder to leave behind; it becomes the reference to the
        // spill local once its binding is known.
        let hole = self.expr(ExprNode::Var(SPILL_TEMP), ty.clone(), span);
        let temp = std::mem::replace(e, hole);
        let decl = Metadata::new(
            StmtNode::Declare { name: SPILL_TEMP, ty: Some(ty.clone()), value: temp }, span);
        self.cx.resolved.insert(e.id, Binding::Local(decl.id));
        decls.push(decl);
    }

    /// An expression in a position that takes its value: a `let` initializer, an
    /// argument, a field of a literal, the operand of `return`.
    fn consume(&mut self, e: &Expr<'a>) {
        self.visit(e);
        let Some(ty) = self.ty_of(e) else { return };
        if self.model.is_copy(&ty) { return; }
        // a payload binding stands in for the enum it views, so resolve it to
        // where that enum actually lives before deciding anything.
        let mut root = self.root(e);
        while let Root::Whole(b) = root {
            match self.payload_src.get(&b) {
                // moving the payload consumes the enum too: mark the window
                // moved, then ask the same question of what it looks onto.
                Some(src) => {
                    self.moved.insert(b, State::Moved(e.span));
                    root = *src;
                }
                None => break,
            }
        }
        match root {
            Root::Whole(b) => { self.moved.insert(b, State::Moved(e.span)); }
            Root::Field(b) => {
                let name = self.binding_name(b);
                let shown = self.cx.show(&ty);
                self.push(Error::new(e.span, format!(
                    "cannot move a '{}' out of a field of '{}'", shown, name))
                    .with_label(e.span, "it owns a resource")
                    .with_note("moving one field out of a value would leave the rest without \
                                an owner; borrow it with '&' instead"));
            }
            Root::Borrowed => {
                let shown = self.cx.show(&ty);
                self.push(Error::new(e.span, format!(
                    "cannot move a '{}' out of a pointer or an element", shown))
                    .with_label(e.span, "it owns a resource")
                    .with_note("the place it is moved out of would be left invalid; borrow it \
                                with '&' instead"));
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
            ExprNode::Repeat { value, .. } => self.walk_reads(value, f),
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
        // record the name for diagnostics even when the value needs no slot, so a
        // move out of an owning-but-drop-glue-less local still reads by name.
        if let Binding::Local(_) = binding { self.names.insert(binding, name); }
        let drops = self.model.drops_for(&ty);
        if drops.is_empty() { return; }
        self.scopes.last_mut().unwrap().push(Slot { binding, name, ty, drops });
    }

    /// Record the source names of a match arm's payload bindings, so a move out
    /// of one (`Some(v) -> return v`) names `v` rather than `<local>`. The binding
    /// identity is the sub-pattern's node id, matching what name resolution wired
    /// every use of it to.
    fn note_pattern_binds(&mut self, pat: &Pattern<'a>, src: Root<'a>) {
        match &pat.value {
            PatternNode::Bind(name) => {
                self.names.insert(Binding::Local(pat.id), name);
                self.payload_src.insert(Binding::Local(pat.id), src);
            }
            PatternNode::Variant { fields, .. } => {
                for f in fields { self.note_pattern_binds(f, src); }
            }
            PatternNode::StructVariant { fields, .. } => {
                for (_, f) in fields { self.note_pattern_binds(f, src); }
            }
            _ => {}
        }
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
                self.push(Error::new(at, format!(
                    "'{}' is moved away on some paths but not others", name))
                    .with_label(at, "moved here")
                    .with_note("it owns a resource, so whether it still needs destroying at \
                                the end of its scope is not decidable at compile time; move \
                                it on every path, or on none"));
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

    /// A `Var` reference to a local, wired to its binding so lowering finds the
    /// slot. Every synthesized variable use goes through here.
    fn var(&mut self, name: &'a str, ty: &Type<'a>, binding: Binding<'a>, span: Span) -> Expr<'a> {
        let e = self.expr(ExprNode::Var(name), ty.clone(), span);
        self.cx.resolved.insert(e.id, binding);
        e
    }

    /// `let <name>: <ty> = <value>;`, plus the binding identity that names it.
    /// A local is identified by its `Declare`'s node id, which is what makes the
    /// three locals of one expansion distinct from those of any other.
    fn declare_stmt(&mut self, name: &'a str, ty: Type<'a>, value: Expr<'a>, span: Span)
        -> (Stmt<'a>, Binding<'a>)
    {
        let stmt = Metadata::new(StmtNode::Declare { name, ty: Some(ty), value }, span);
        let binding = Binding::Local(stmt.id);
        (stmt, binding)
    }

    /// An integer literal of exactly `ty`. `None` for a non-integer type, which
    /// typechecking has already ruled out for every caller here.
    fn int_lit(&mut self, ty: &Type<'a>, v: u64, span: Span) -> Option<Expr<'a>> {
        let node = match ty {
            Type::Int8 => ExprNode::Int8(v as i8),
            Type::Int16 => ExprNode::Int16(v as i16),
            Type::Int32 => ExprNode::Int32(v as i32),
            Type::Int64 => ExprNode::Int64(v as i64),
            Type::Uint8 => ExprNode::Uint8(v as u8),
            Type::Uint16 => ExprNode::Uint16(v as u16),
            Type::Uint32 => ExprNode::Uint32(v as u32),
            Type::Uint64 => ExprNode::Uint64(v),
            _ => return None,
        };
        Some(self.expr(node, ty.clone(), span))
    }

    fn binary(&mut self, op: BinaryOp, left: Expr<'a>, right: Expr<'a>, ty: Type<'a>, span: Span)
        -> Expr<'a>
    {
        self.expr(ExprNode::Binary { op, left: Box::new(left), right: Box::new(right) }, ty, span)
    }

    /// `delete(&owner.field...)` as a statement.
    fn delete_call(&mut self, binding: Binding<'a>, name: &'a str, ty: &Type<'a>,
                   path: &DropPath<'a>, span: Span) -> Stmt<'a> {
        let root = self.var(name, ty, binding, span);
        self.delete_at(root, ty.clone(), path, span)
    }

    /// `delete(&place.field...)` for an arbitrary place expression. Split out of
    /// `delete_call` because the `drop_in_place` expansion destroys `base[i]`
    /// rather than a named binding, but walks the same chain to get there.
    fn delete_at(&mut self, place: Expr<'a>, ty: Type<'a>, path: &DropPath<'a>, span: Span)
        -> Stmt<'a>
    {
        self.delete_steps(place, ty, &path.steps, path.target, span)
    }

    /// Walk `steps` from `place` and destroy what is at the end.
    ///
    /// Recursive rather than a loop because a [`Step::Elems`] is not an access
    /// but a `while` that the remaining steps happen *inside*: destroying a
    /// `[[Res; 2]; 3]` is two nested loops, and a `[S; 4]` whose `S` owns two
    /// fields is one loop containing two `delete`s.
    ///
    /// The index local is named the same in every loop; that is fine, and is why
    /// [`Self::declare_stmt`] identifies a local by its `Declare`'s node id
    /// rather than by name - two nested loops get two distinct bindings, and
    /// each `Var` is wired to the one it means.
    fn delete_steps(&mut self, place: Expr<'a>, ty: Type<'a>, steps: &[Step<'a>],
                    target: &'a str, span: Span) -> Stmt<'a>
    {
        match steps.first() {
            Some(Step::Field(field, fty)) => {
                let inner = self.expr(
                    ExprNode::Access { base: Box::new(place), field: *field },
                    fty.clone(), span);
                self.delete_steps(inner, fty.clone(), &steps[1..], target, span)
            }
            Some(Step::Elems(elem_ty, count)) => {
                let idx_ty = Type::Uint64;
                let (Some(zero), Some(one), Some(n)) = (
                    self.int_lit(&idx_ty, 0, span),
                    self.int_lit(&idx_ty, 1, span),
                    self.int_lit(&idx_ty, *count as u64, span),
                ) else {
                    unreachable!("u64 is an integer type, so `int_lit` always yields one")
                };
                let (index_decl, index) = self.declare_stmt(DROP_INDEX, idx_ty.clone(), zero, span);

                let i_read = self.var(DROP_INDEX, &idx_ty, index, span);
                let cond = self.binary(BinaryOp::Lt, i_read, n, Type::Bool, span);

                let i_read = self.var(DROP_INDEX, &idx_ty, index, span);
                let elem = self.expr(
                    ExprNode::Index { slice: Box::new(place), index: Box::new(i_read) },
                    elem_ty.clone(), span);
                let inner = self.delete_steps(
                    elem, elem_ty.clone(), &steps[1..], target, span);

                let i_read = self.var(DROP_INDEX, &idx_ty, index, span);
                let next = self.binary(BinaryOp::Add, i_read, one, idx_ty.clone(), span);
                let i_write = self.var(DROP_INDEX, &idx_ty, index, span);
                let bump = Metadata::new(StmtNode::Assign { left: i_write, value: next }, span);

                let body = Metadata::new(StmtNode::Block(vec![inner, bump]), span);
                let while_ = Metadata::new(
                    StmtNode::While { condition: cond, body: Box::new(body) }, span);
                // one block, so the index is scoped to this loop
                Metadata::new(StmtNode::Block(vec![index_decl, while_]), span)
            }
            Some(Step::Payload { enum_def, variant, index, arity, field_ty }) => {
                // one sub-pattern per payload field: `_` everywhere but the one
                // this path goes through.
                let fields: Vec<Pattern<'a>> = (0..*arity)
                    .map(|i| Metadata::new(
                        if i == *index { PatternNode::Bind(DROP_PAYLOAD) }
                        else { PatternNode::Wildcard },
                        span))
                    .collect();
                // the pattern's binding identity is its node id, exactly as for a
                // hand-written `match` arm - MIL keys the payload view on it.
                let bind = Binding::Local(fields[*index].id);

                // a single segment is enough: everything downstream reads the
                // variant off `path.last()`, and the enum itself comes from the
                // scrutinee's type, never from the pattern's qualifier.
                let name = NameRef { def: *enum_def, path: Path::single(variant) };
                let pat = Metadata::new(
                    PatternNode::Variant { path: name, fields }, span);

                let bound = self.var(DROP_PAYLOAD, field_ty, bind, span);
                let inner = self.delete_steps(
                    bound, field_ty.clone(), &steps[1..], target, span);
                let arm = Box::new(Metadata::new(StmtNode::Block(vec![inner]), span));

                // every other variant, including those that own nothing
                let rest = Metadata::new(PatternNode::Wildcard, span);
                let empty = Box::new(Metadata::new(StmtNode::Block(Vec::new()), span));

                Metadata::new(StmtNode::Match {
                    scrutinee: place,
                    arms: vec![(pat, arm), (rest, empty)],
                }, span)
            }
            None => self.delete_here(place, ty, target, span),
        }
    }

    /// `delete(&place)`, the leaf of every drop path.
    fn delete_here(&mut self, place: Expr<'a>, place_ty: Type<'a>, target: &'a str, span: Span)
        -> Stmt<'a>
    {
        let ptr_ty = Type::Pointer(Box::new(place_ty));
        let addr = self.expr(
            ExprNode::Unary { op: UnaryOp::AddrOf, operand: Box::new(place) },
            ptr_ty.clone(), span);
        // the callee's type is what MIL reads to coerce the argument; a
        // destructor is always `proc(*T) void`.
        let func = self.expr(ExprNode::Var(target), Type::Function {
            params: vec![ptr_ty],
            return_type: Box::new(Type::Void),
        }, span);
        let call = self.expr(ExprNode::Call {
            func: Box::new(func), type_args: Vec::new(), args: vec![addr],
        }, Type::Void, span);
        Metadata::new(StmtNode::Expr(call), span)
    }

    /// The element type of a `drop_in_place::<T>(ptr, count)` call, or `None` if
    /// `expr` is some other call.
    fn drop_in_place_target(&self, expr: &Expr<'a>) -> Option<Type<'a>> {
        let ExprNode::Call { func, type_args, args } = &expr.value else { return None };
        let ExprNode::Var(name) = &func.value else { return None };
        if Intrinsic::lookup(name) != Some(Intrinsic::DropInPlace) { return None }
        // arity and kinds were settled by typechecking; a malformed call never
        // reaches this pass.
        debug_assert_eq!(args.len(), 2);
        match type_args.first() {
            Some(GenericArg::Type(t)) => Some(t.clone()),
            _ => None,
        }
    }

    /// Rewrite `drop_in_place::<T>(ptr, count)` into the loop that destroys
    /// `ptr[0 .. count]`:
    ///
    /// ```text
    /// { let $dip_base = ptr; let $dip_count = count; let $dip_i = 0;
    ///   while ($dip_i < $dip_count) { delete(&$dip_base[$dip_i]...); $dip_i = $dip_i + 1; } }
    /// ```
    ///
    /// The AST loop remains visible to allocation checking. `ptr` and `count`
    /// are bound once before the loop. The caller skips expansion when `drops`
    /// is empty.
    fn expand_drop_in_place(&mut self, call: Expr<'a>, elem_ty: Type<'a>,
                            drops: &[DropPath<'a>], span: Span, out: &mut Vec<Stmt<'a>>) {
        let ExprNode::Call { args, .. } = call.value else { unreachable!("checked by the caller") };
        let mut args = args.into_iter();
        let (ptr_arg, count_arg) = (args.next().unwrap(), args.next().unwrap());

        let ptr_ty = Type::Pointer(Box::new(elem_ty.clone()));
        let Some(count_ty) = self.ty_of(&count_arg) else {
            unreachable!("typechecking recorded the count argument's type")
        };
        let (Some(zero), Some(one)) = (
            self.int_lit(&count_ty, 0, span),
            self.int_lit(&count_ty, 1, span),
        ) else {
            unreachable!("typechecking accepted only integer counts")
        };

        let (base_decl, base) = self.declare_stmt(DIP_BASE, ptr_ty.clone(), ptr_arg, span);
        let (count_decl, count) = self.declare_stmt(DIP_COUNT, count_ty.clone(), count_arg, span);
        let (index_decl, index) = self.declare_stmt(DIP_INDEX, count_ty.clone(), zero, span);

        // while ($dip_i < $dip_count)
        let i_read = self.var(DIP_INDEX, &count_ty, index, span);
        let n_read = self.var(DIP_COUNT, &count_ty, count, span);
        let cond = self.binary(BinaryOp::Lt, i_read, n_read, Type::Bool, span);

        // delete(&$dip_base[$dip_i]...), one per path through the element type
        let mut body: Vec<Stmt<'a>> = Vec::with_capacity(drops.len() + 1);
        for path in drops {
            let base_read = self.var(DIP_BASE, &ptr_ty, base, span);
            let i_read = self.var(DIP_INDEX, &count_ty, index, span);
            let elem = self.expr(
                ExprNode::Index { slice: Box::new(base_read), index: Box::new(i_read) },
                elem_ty.clone(), span);
            let stmt = self.delete_at(elem, elem_ty.clone(), path, span);
            body.push(stmt);
        }

        // $dip_i = $dip_i + 1
        let i_read = self.var(DIP_INDEX, &count_ty, index, span);
        let next = self.binary(BinaryOp::Add, i_read, one, count_ty.clone(), span);
        let i_write = self.var(DIP_INDEX, &count_ty, index, span);
        body.push(Metadata::new(StmtNode::Assign { left: i_write, value: next }, span));

        let body = Metadata::new(StmtNode::Block(body), span);
        let while_ = Metadata::new(
            StmtNode::While { condition: cond, body: Box::new(body) }, span);

        // one block, so the three locals are scoped to the expansion
        out.push(Metadata::new(
            StmtNode::Block(vec![base_decl, count_decl, index_decl, while_]), span));
    }

    // --- statements

    /// Process one statement, appending it (and any destruction around it) to
    /// `out`. Returns whether control definitely leaves here.
    fn one(&mut self, stmt: Stmt<'a>, out: &mut Vec<Stmt<'a>>) -> bool {
        let Metadata { span, id, mut value } = stmt;

        // spill borrowed owning temporaries into `let`s, then wrap the statement
        // in a block that owns them: the block's scope-end unwind destroys each
        // one at the end of the enclosing statement. Re-processing the wrapped
        // statement finds nothing left to hoist, so this recurs exactly once.
        let mut decls = Vec::new();
        self.hoist_stmt(&mut value, &mut decls);
        if !decls.is_empty() {
            decls.push(Metadata { span, id, value });
            return self.one(Metadata::new(StmtNode::Block(decls), span), out);
        }

        match value {
            StmtNode::Expr(e) => {
                self.visit(&e);
                // `drop_in_place` is the one call this pass rewrites rather than
                // just checks: it names work only this pass knows how to emit.
                if let Some(elem_ty) = self.drop_in_place_target(&e) {
                    let drops = self.model.drops_for(&elem_ty);
                    // when `T` owns nothing there is no loop to write: keep the
                    // call, which lowers to nothing but its arguments' effects.
                    if !drops.is_empty() {
                        self.expand_drop_in_place(e, elem_ty, &drops, span, out);
                        return false;
                    }
                    out.push(Metadata { span, id, value: StmtNode::Expr(e) });
                    return false;
                }
                // a produced owner that is never bound has no owner to destroy it.
                if let Some(ty) = self.ty_of(&e) {
                    if !self.model.is_copy(&ty) && matches!(self.root(&e), Root::Temp) {
                        let shown = self.cx.show(&ty);
                        self.push(Error::new(span, format!(
                            "this '{}' owns a resource but is discarded without an owner", shown))
                            .with_label(span, "nothing would ever destroy it")
                            .with_note("bind it with `let`"));
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
                // mono filled in the type of a bare `let`, so it is present here
                // whether or not the source wrote it.
                let declared = ty.clone().expect("monomorphization fills in every `let` type");
                out.push(Metadata { span, id, value: StmtNode::Declare { name, ty, value } });
                // the binding identity is the `Declare`'s node id, matching what
                // name resolution recorded for every use of it.
                self.declare(Binding::Local(id), name, declared);
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
                                self.push(Error::new(span, format!("cannot overwrite '{}'", name))
                                    .with_label(span, "the new value reads what it replaces")
                                    .with_note("it owns a resource that must be destroyed \
                                                first; bind the new value with `let` before \
                                                assigning"));
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
                                self.push(Error::new(span, format!(
                                    "cannot overwrite a '{}' in place", shown))
                                    .with_label(span, "it owns a resource")
                                    .with_note("the value being replaced would never be \
                                                destroyed"));
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
                    self.push(Error::new(at, format!(
                        "'{}' is declared outside this loop but moved away inside it", name))
                        .with_label(at, "moved here")
                        .with_note("the next iteration would move a value that is already gone"));
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
                let scrutinee_root = self.root(&scrutinee);
                let entry = self.moved.clone();
                let mut merged: Option<(Flow<'a>, bool)> = None;
                let mut new_arms = Vec::with_capacity(arms.len());
                for (pat, body) in arms {
                    self.moved = entry.clone();
                    // a payload binding is a view into the scrutinee's storage;
                    // moving it out is a move of *that*, so record both the name
                    // (for diagnostics) and where the storage lives.
                    self.note_pattern_binds(&pat, scrutinee_root);
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

            // `return;` carries no value, so there is nothing to consume and
            // nothing that could read an owner on the way out - the scope's
            // owners are simply dropped ahead of it.
            StmtNode::Return(None) => {
                let drops = self.unwind(0, span);
                out.extend(drops);
                out.push(Metadata { span, id, value: StmtNode::Return(None) });
                true
            }

            StmtNode::Return(Some(e)) => {
                self.consume(&e);
                let drops = self.unwind(0, span);
                if drops.is_empty() {
                    out.push(Metadata { span, id, value: StmtNode::Return(Some(e)) });
                } else if self.ret_ty == Type::Void {
                    // nothing carries a value out, so the owners can go first -
                    // as long as the returned expression does not read them.
                    if let Some(b) = self.first_dropped_read(&e) {
                        let name = self.binding_name(b);
                        self.push(Error::new(span, format!(
                            "'{}' is destroyed when this scope ends", name))
                            .with_label(span, "but the returned expression reads it"));
                    }
                    out.extend(drops);
                    out.push(Metadata { span, id, value: StmtNode::Return(Some(e)) });
                } else {
                    // the value has to be computed before this scope's owners are
                    // destroyed - it may well be reading them - so it is bound to
                    // a temporary that outlives them.
                    let ty = self.ret_ty.clone();
                    let decl = Metadata::new(StmtNode::Declare {
                        name: RET_TEMP, ty: Some(ty.clone()), value: e,
                    }, span);
                    let var = self.expr(ExprNode::Var(RET_TEMP), ty, span);
                    self.cx.resolved.insert(var.id, Binding::Local(decl.id));
                    out.push(decl);
                    out.extend(drops);
                    out.push(Metadata { span, id, value: StmtNode::Return(Some(var)) });
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
    // no `Delete` trait means the program has no prelude at all (`--no-prelude`),
    // so nothing in it owns anything and every value is `Copy` exactly as it was
    // before. This is a real answer, not a failed lookup: a prelude that does not
    // declare `Delete` is rejected at load, precisely so that skipping the whole
    // pass here can never be the silent consequence of not finding the lang item.
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
            || cx.instances.get(d).is_some_and(|i| owning.contains(&i.template));
        // a missing `delete` was already reported by conformance checking.
        if owner { deletes.insert(*d, m.name); }
    }

    // a *concrete* impl on an instantiation of a generic type - `extend Res<i32>:
    // Delete` - is registered against the template like every impl is, but unlike
    // a generic one it desugars to an ordinary non-generic `delete` that survives
    // mono under its own name. So the instance can share the template's entry.
    //
    // The guard is what keeps that from being wrong in the generic case. A
    // generic impl's instances already have their own entry, minted by mono's
    // `instantiate_destructor` and naming the body specialized for them, so they
    // never reach here; but an instance a *conditional* impl does not cover
    // (`Vec<u8>` under `extend Vec<T>: Delete where T: Delete`) has none, and
    // deliberately so - handing it the template's name would call a destructor
    // that was never instantiated. Only a concrete impl has a name to share.
    for (inst, i) in cx.instances.iter() {
        if deletes.contains_key(inst) { continue; }
        let Some(&name) = deletes.get(&i.template) else { continue };
        let concrete_impl = cx.members.get(&(TyHead::Def(i.template), DELETE_METHOD))
            .is_some_and(|m| m.generics.is_empty());
        if concrete_impl { deletes.insert(*inst, name); }
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
            names: HashMap::new(),
            payload_src: HashMap::new(),
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
