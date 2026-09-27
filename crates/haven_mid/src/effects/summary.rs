//! Reusable values for effect inference, independent of source-clause checking.

use haven_common::ast::{Effect, EffectClause, TopLevel, TopLevelNode};
use std::collections::HashMap;
use std::fmt::{self, Display, Formatter};

/// A set in the compiler's currently recognized effect universe.
///
/// Storage is private so no bits outside that universe can be introduced.
/// Display follows `Effect::ALL` order, regardless of construction order.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct EffectSet(u8);

impl EffectSet {
    pub const fn empty() -> Self {
        Self(0)
    }

    pub fn all() -> Self {
        Effect::ALL.into_iter().collect()
    }

    pub const fn singleton(effect: Effect) -> Self {
        Self(Self::bit(effect))
    }

    // Exhaustive matching makes a new Effect require a storage assignment.
    const fn bit(effect: Effect) -> u8 {
        match effect {
            Effect::Alloc => 1,
            Effect::IO => 2,
        }
    }

    pub const fn contains(self, effect: Effect) -> bool {
        self.0 & Self::bit(effect) != 0
    }

    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    pub const fn intersection(self, other: Self) -> Self {
        Self(self.0 & other.0)
    }

    pub const fn difference(self, other: Self) -> Self {
        Self(self.0 & !other.0)
    }
}

impl FromIterator<Effect> for EffectSet {
    fn from_iter<T: IntoIterator<Item = Effect>>(effects: T) -> Self {
        effects.into_iter().fold(Self::empty(), |set, effect| {
            set.union(Self::singleton(effect))
        })
    }
}

impl Display for EffectSet {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{{")?;
        let mut separator = "";
        for effect in Effect::ALL {
            if self.contains(effect) {
                write!(f, "{separator}{effect}")?;
                separator = ", ";
            }
        }
        write!(f, "}}")
    }
}

/// An upper approximation of a callable's effects, not execution guarantees.
///
/// `may_have` records effects admitted by known operations or trusted
/// declarations. `unknown` records recognized effects unresolved code might
/// have. Neither set says an effect occurs on every execution.
///
/// The sets may overlap: a known IO call and an unresolved call that might do
/// IO supply independent evidence. Joins preserve both kinds of evidence;
/// known evidence must not erase uncertainty. `possible()` unions the sets
/// when only the recognized effects that might occur matter.
///
/// `open` records non-exhaustiveness of an effect contract. It is independent
/// of the sets, not shorthand for nonempty `unknown`: an open contract can
/// exclude every currently recognized effect, and a closed summary can retain
/// uncertainty within an exhaustive bound. Thus `possible()` cannot answer
/// whether the whole contract is exhaustive.
///
/// Default is the inference bottom: no evidence and no open contract. Joining
/// summaries only adds evidence, using set union and logical OR. Source bounds
/// must be checked separately; they must not overwrite body inference.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct EffectSummary {
    pub may_have: EffectSet,
    pub unknown: EffectSet,
    pub open: bool,
}

impl EffectSummary {
    /// An indirect or unresolved call has no trusted exhaustive contract.
    pub fn unknown_call() -> Self {
        Self {
            may_have: EffectSet::empty(),
            unknown: EffectSet::all(),
            open: true,
        }
    }

    /// Seed an extern from its trusted declaration, never from a body.
    /// An allowlist is exhaustive; a denylist excludes only its listed effects.
    pub fn for_extern(clause: Option<&EffectClause>) -> Self {
        match clause {
            Some(EffectClause::With(allowed)) => Self {
                may_have: allowed.iter().copied().collect(),
                ..Self::default()
            },
            Some(EffectClause::Without(forbidden)) => Self {
                may_have: EffectSet::empty(),
                unknown: EffectSet::all().difference(forbidden.iter().copied().collect()),
                open: true,
            },
            None => Self::unknown_call(),
        }
    }

    pub const fn possible(self) -> EffectSet {
        self.may_have.union(self.unknown)
    }

    /// Check a source bound against inferred possibilities without changing
    /// the evidence. `with` also requires an exhaustive contract, even if it
    /// allows every recognized effect. `without` ignores openness and excludes
    /// only its listed effects, so an empty denylist accepts unknown calls.
    pub fn satisfies(self, clause: &EffectClause) -> bool {
        match clause {
            EffectClause::With(allowed) => {
                !self.open
                    && self
                        .possible()
                        .difference(allowed.iter().copied().collect())
                        .is_empty()
            }
            EffectClause::Without(forbidden) => self
                .possible()
                .intersection(forbidden.iter().copied().collect())
                .is_empty(),
        }
    }

    pub const fn join(self, other: Self) -> Self {
        Self {
            may_have: self.may_have.union(other.may_have),
            unknown: self.unknown.union(other.unknown),
            open: self.open || other.open,
        }
    }
}

/// Seeds keyed by final post-monomorphization callable names, before propagation.
///
/// Ordinary functions start at bottom even when annotated: their clauses are
/// bounds to check after body inference, not trusted evidence. Non-callable
/// declarations contribute no entries. Indirect and unresolved callees have no
/// declaration here; consumers must use `EffectSummary::unknown_call()` for
/// missing targets. Intrinsics are classified separately by the call collector.
///
/// The checker propagates these seeds through the call graph before checking
/// bounds. Initialization never applies ordinary function bounds to evidence.
pub fn initial_summaries<'a>(program: &[TopLevel<'a>]) -> HashMap<&'a str, EffectSummary> {
    program
        .iter()
        .filter_map(|node| match &node.value {
            TopLevelNode::Extern {
                name,
                effect_clause,
                ..
            } => Some((
                *name,
                EffectSummary::for_extern(effect_clause.as_ref().map(|clause| &clause.value)),
            )),
            TopLevelNode::Function { name, .. } => Some((*name, EffectSummary::default())),
            _ => None,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALLOC: EffectSet = EffectSet::singleton(Effect::Alloc);
    const IO: EffectSet = EffectSet::singleton(Effect::IO);

    #[test]
    fn extern_seeds_respect_exhaustive_and_partial_contracts() {
        let empty = EffectSet::empty();
        let all = EffectSet::all();
        let cases = [
            (EffectClause::With(vec![Effect::Alloc]), ALLOC, empty, false),
            (EffectClause::With(vec![Effect::IO]), IO, empty, false),
            (EffectClause::With(vec![]), empty, empty, false),
            (
                EffectClause::With(vec![Effect::IO, Effect::Alloc]),
                all,
                empty,
                false,
            ),
            (EffectClause::Without(vec![Effect::Alloc]), empty, IO, true),
            (EffectClause::Without(vec![Effect::IO]), empty, ALLOC, true),
            (EffectClause::Without(vec![]), empty, all, true),
            (
                EffectClause::Without(vec![Effect::Alloc, Effect::IO]),
                empty,
                empty,
                true,
            ),
        ];
        for (clause, may_have, unknown, open) in cases {
            assert_eq!(
                EffectSummary::for_extern(Some(&clause)),
                EffectSummary {
                    may_have,
                    unknown,
                    open
                },
                "{clause}"
            );
        }
        let unknown = EffectSummary {
            may_have: empty,
            unknown: all,
            open: true,
        };
        assert_eq!(EffectSummary::for_extern(None), unknown);
        assert_eq!(EffectSummary::unknown_call(), unknown);
    }

    #[test]
    fn program_seeds_trust_only_extern_clauses() {
        use haven_common::ast::{Metadata, Span, Type};
        use haven_common::defs::DefId;

        let mut program = Vec::new();
        let clauses = [
            None,
            Some(EffectClause::With(vec![Effect::IO])),
            Some(EffectClause::Without(vec![Effect::Alloc])),
        ];
        let function_names = ["inferred", "allowed", "excluded"];
        let extern_names = ["native_unknown", "native_io", "native_partial"];
        for (i, clause) in clauses.into_iter().enumerate() {
            let clause = clause.map(|value| Metadata::new(value, Span::unknown()));
            program.push(Metadata::new(
                TopLevelNode::Function {
                    name: function_names[i],
                    def: DefId(i as u32),
                    is_pub: false,
                    attributes: vec![],
                    effect_clause: clause.clone(),
                    generics: vec![],
                    where_bounds: vec![],
                    params: vec![],
                    return_type: Type::Void,
                    body: vec![],
                },
                Span::unknown(),
            ));
            program.push(Metadata::new(
                TopLevelNode::Extern {
                    name: extern_names[i],
                    def: DefId((i + 3) as u32),
                    is_pub: false,
                    attributes: vec![],
                    effect_clause: clause,
                    generics: vec![],
                    params: vec![],
                    return_type: Type::Void,
                },
                Span::unknown(),
            ));
        }
        program.push(Metadata::new(
            TopLevelNode::Struct {
                name: "not_callable",
                def: DefId(6),
                is_pub: false,
                attributes: vec![],
                generics: vec![],
                fields: vec![],
            },
            Span::unknown(),
        ));
        let seeds = initial_summaries(&program);
        assert_eq!(seeds.len(), 6);
        for name in function_names {
            assert_eq!(seeds[name], EffectSummary::default());
        }
        assert_eq!(seeds["native_unknown"], EffectSummary::unknown_call());
        assert_eq!(
            seeds["native_io"],
            EffectSummary {
                may_have: IO,
                ..EffectSummary::default()
            }
        );
        assert_eq!(
            seeds["native_partial"],
            EffectSummary {
                unknown: IO,
                open: true,
                ..EffectSummary::default()
            }
        );
        assert!(!seeds.contains_key("not_callable"));
        assert!(initial_summaries(&[]).is_empty());
    }

    #[test]
    fn set_operations_cover_the_recognized_universe() {
        let empty = EffectSet::empty();
        let all = EffectSet::all();
        assert!(empty.is_empty());
        assert_eq!(EffectSet::default(), empty);
        for effect in Effect::ALL {
            assert!(!empty.contains(effect));
            assert!(all.contains(effect));
            assert!(EffectSet::singleton(effect).contains(effect));
        }
        assert!(!ALLOC.contains(Effect::IO));
        assert_eq!(ALLOC.union(IO), all);
        assert_eq!(ALLOC.union(ALLOC), ALLOC);
        assert_eq!(ALLOC.intersection(IO), empty);
        assert_eq!(all.intersection(IO), IO);
        assert_eq!(all.difference(ALLOC), IO);
        assert_eq!(ALLOC.difference(all), empty);
        assert_eq!(IO.difference(empty), IO);
    }

    #[test]
    fn construction_deduplicates_and_display_is_deterministic() {
        let set: EffectSet = [Effect::IO, Effect::Alloc, Effect::IO]
            .into_iter()
            .collect();
        assert_eq!(set, EffectSet::all());
        assert_eq!(set.to_string(), "{Alloc, IO}");
        assert_eq!(EffectSet::empty().to_string(), "{}");
        assert_eq!(ALLOC.to_string(), "{Alloc}");
        assert_eq!(IO.to_string(), "{IO}");
    }

    #[test]
    fn join_preserves_overlapping_known_and_unknown_evidence() {
        let known = EffectSummary {
            may_have: IO,
            ..EffectSummary::default()
        };
        let unresolved = EffectSummary {
            unknown: EffectSet::all(),
            open: true,
            ..EffectSummary::default()
        };
        let joined = known.join(unresolved);
        assert_eq!(joined.may_have, IO);
        assert_eq!(joined.unknown, EffectSet::all());
        assert!(joined.open);
        assert_eq!(joined.possible(), EffectSet::all());
    }

    #[test]
    fn exhaustiveness_is_independent_of_recognized_possibilities() {
        let closed_io = EffectSummary {
            may_have: IO,
            ..EffectSummary::default()
        };
        let open_io = EffectSummary {
            unknown: IO,
            open: true,
            ..EffectSummary::default()
        };
        assert_eq!(closed_io.possible(), open_io.possible());
        assert_ne!(closed_io, open_io);
        let open_empty = EffectSummary {
            open: true,
            ..EffectSummary::default()
        };
        assert!(open_empty.possible().is_empty());
        assert!(open_empty.join(closed_io).open);
        let closed_unknown = EffectSummary {
            unknown: ALLOC,
            ..EffectSummary::default()
        };
        assert!(!closed_unknown.open);
        assert_eq!(closed_unknown.possible(), ALLOC);
    }

    #[test]
    fn join_has_the_laws_needed_for_fixed_point_propagation() {
        // Exhaust all 32 summaries in the current two-effect universe,
        // including overlap and open summaries with empty sets.
        let sets = [EffectSet::empty(), ALLOC, IO, EffectSet::all()];
        let mut summaries = Vec::new();
        for may_have in sets {
            for unknown in sets {
                for open in [false, true] {
                    summaries.push(EffectSummary {
                        may_have,
                        unknown,
                        open,
                    });
                }
            }
        }
        let bottom = EffectSummary::default();
        for &a in &summaries {
            assert_eq!(a.join(bottom), a);
            assert_eq!(a.join(a), a);
            for &b in &summaries {
                let joined = a.join(b);
                assert_eq!(joined, b.join(a));
                assert!(a.may_have.difference(joined.may_have).is_empty());
                assert!(a.unknown.difference(joined.unknown).is_empty());
                assert!(!a.open || joined.open);
                for &c in &summaries {
                    assert_eq!(a.join(b).join(c), a.join(b.join(c)));
                }
            }
        }
    }
}
