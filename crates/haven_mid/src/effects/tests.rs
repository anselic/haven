use super::*;

#[test]
fn blame_reasons_distinguish_openness_from_possible_effects() {
    let open_empty = EffectSummary::for_extern(Some(&EffectClause::Without(Effect::ALL.to_vec())));
    assert_eq!(
        BlameReason::for_clause(open_empty, &EffectClause::With(vec![])),
        Some(BlameReason::OpenContract)
    );
    assert_eq!(
        BlameReason::for_clause(open_empty, &EffectClause::Without(Effect::ALL.to_vec())),
        None
    );
    let mixed = EffectSummary::unknown_call().join(EffectSummary::for_extern(Some(
        &EffectClause::With(vec![Effect::IO]),
    )));
    assert_eq!(
        BlameReason::for_clause(mixed, &EffectClause::With(vec![])),
        Some(BlameReason::OpenContract)
    );
    assert_eq!(
        BlameReason::for_clause(mixed, &EffectClause::Without(vec![Effect::IO])),
        Some(BlameReason::Possible(Effect::IO))
    );
    // Deterministic effect priority follows Effect::ALL, not clause order.
    assert_eq!(
        BlameReason::for_clause(
            mixed,
            &EffectClause::Without(vec![Effect::IO, Effect::Alloc])
        ),
        Some(BlameReason::Possible(Effect::Alloc))
    );
}

#[test]
fn blame_follows_selected_evidence_through_recursion() {
    let calls = graph(&[
        ("wrapper", &["a_known_io", "b_partial"]),
        ("a_known_io", &["native_io"]),
        ("b_partial", &["wrapper", "native_partial"]),
    ]);
    let mut summaries: SummaryMap = calls
        .keys()
        .map(|&n| (n, EffectSummary::default()))
        .collect();
    summaries.insert(
        "native_io",
        EffectSummary::for_extern(Some(&EffectClause::With(vec![Effect::IO]))),
    );
    summaries.insert(
        "native_partial",
        EffectSummary::for_extern(Some(&EffectClause::Without(vec![Effect::Alloc]))),
    );
    propagate_summaries(&mut summaries, &calls);
    assert_eq!(
        blame_chain(&calls, &summaries, BlameReason::OpenContract, "wrapper"),
        vec!["wrapper", "b_partial", "native_partial"]
    );
    assert_eq!(
        blame_chain(
            &calls,
            &summaries,
            BlameReason::Possible(Effect::IO),
            "wrapper"
        ),
        vec!["wrapper", "a_known_io", "native_io"]
    );
}

#[test]
fn blame_uses_shortest_paths_and_lexical_ties_with_unknown_leaves() {
    let edges: [(&str, &[&str]); 3] = [
        ("entry", &["z_missing", "a_missing", "long"]),
        ("long", &["inner"]),
        ("inner", &[INDIRECT_CALLEE]),
    ];
    for reverse in [false, true] {
        let mut ordered = edges.to_vec();
        if reverse {
            ordered.reverse();
        }
        let calls = graph(&ordered);
        let mut summaries: SummaryMap = calls
            .keys()
            .map(|&n| (n, EffectSummary::default()))
            .collect();
        propagate_summaries(&mut summaries, &calls);
        for reason in [
            BlameReason::OpenContract,
            BlameReason::Possible(Effect::Alloc),
        ] {
            assert_eq!(
                blame_chain(&calls, &summaries, reason, "entry"),
                vec!["entry", "a_missing"]
            );
            assert_eq!(
                blame_chain(&calls, &summaries, reason, "inner"),
                vec!["inner", INDIRECT_CALLEE]
            );
        }
    }
}

#[test]
fn bounds_check_possible_effects_and_exhaustiveness_independently() {
    let alloc = EffectSet::singleton(Effect::Alloc);
    let io = EffectSet::singleton(Effect::IO);
    let sets = [EffectSet::empty(), alloc, io, EffectSet::all()];
    let lists = [
        vec![],
        vec![Effect::Alloc],
        vec![Effect::IO],
        Effect::ALL.to_vec(),
    ];
    for may_have in sets {
        for unknown in sets {
            for open in [false, true] {
                let summary = EffectSummary {
                    may_have,
                    unknown,
                    open,
                };
                for list in &lists {
                    let with = EffectClause::With(list.clone());
                    let without = EffectClause::Without(list.clone());
                    // Independent per-effect oracle also covers overlapping
                    // evidence and closed summaries with unresolved effects.
                    let possible = |effect| may_have.contains(effect) || unknown.contains(effect);
                    assert_eq!(
                        summary.satisfies(&with),
                        !open
                            && Effect::ALL
                                .into_iter()
                                .all(|effect| !possible(effect) || list.contains(&effect))
                    );
                    assert_eq!(
                        summary.satisfies(&without),
                        list.iter().all(|&effect| !possible(effect))
                    );
                }
            }
        }
    }
    let unknown = EffectSummary::unknown_call();
    assert!(unknown.satisfies(&EffectClause::Without(vec![])));
    assert!(!unknown.satisfies(&EffectClause::With(Effect::ALL.to_vec())));
    let partial = EffectSummary::for_extern(Some(&EffectClause::Without(vec![Effect::Alloc])));
    assert!(partial.satisfies(&EffectClause::Without(vec![Effect::Alloc])));
    assert!(!partial.satisfies(&EffectClause::With(vec![Effect::IO])));
}

fn graph<'a>(edges: &[(&'a str, &[&'a str])]) -> CallGraph<'a> {
    edges
        .iter()
        .map(|&(caller, callees)| (caller, callees.iter().copied().collect()))
        .collect()
}

#[test]
fn recursive_components_accumulate_both_kinds_of_evidence() {
    let calls = graph(&[
        ("entry", &["a"]),
        ("a", &["b", "native_alloc"]),
        ("b", &["a", "partial"]),
        ("pure_a", &["pure_b"]),
        ("pure_b", &["pure_a"]),
        ("self", &["self"]),
    ]);
    let alloc = EffectSummary::for_extern(Some(&EffectClause::With(vec![Effect::Alloc])));
    let partial = EffectSummary::for_extern(Some(&EffectClause::Without(vec![Effect::Alloc])));
    let mut summaries: SummaryMap = calls
        .keys()
        .map(|&name| (name, EffectSummary::default()))
        .collect();
    summaries.insert("native_alloc", alloc);
    summaries.insert("partial", partial);
    propagate_summaries(&mut summaries, &calls);
    for name in ["entry", "a", "b"] {
        assert_eq!(summaries[name], alloc.join(partial));
    }
    for name in ["pure_a", "pure_b", "self"] {
        assert_eq!(summaries[name], EffectSummary::default());
    }
    assert_eq!(summaries["native_alloc"], alloc);
    assert_eq!(summaries["partial"], partial);
    let settled = summaries.clone();
    propagate_summaries(&mut summaries, &calls);
    assert_eq!(summaries, settled);
}

#[test]
fn missing_and_indirect_targets_are_conservative_without_erasing_known_effects() {
    let calls = graph(&[
        ("indirect", &[INDIRECT_CALLEE]),
        ("missing", &["unresolved_symbol"]),
        ("mixed", &["native_io", "indirect"]),
        ("wrapper", &["mixed"]),
    ]);
    let io = EffectSummary::for_extern(Some(&EffectClause::With(vec![Effect::IO])));
    let mut summaries: SummaryMap = calls
        .keys()
        .map(|&name| (name, EffectSummary::default()))
        .collect();
    summaries.insert("native_io", io);
    propagate_summaries(&mut summaries, &calls);
    for name in ["indirect", "missing"] {
        assert_eq!(summaries[name], EffectSummary::unknown_call());
    }
    for name in ["mixed", "wrapper"] {
        assert_eq!(summaries[name], io.join(EffectSummary::unknown_call()));
    }
    assert!(!summaries.contains_key(INDIRECT_CALLEE));
    assert!(!summaries.contains_key("unresolved_symbol"));
}

#[test]
fn propagation_is_independent_of_graph_and_seed_insertion_order() {
    let edges: [(&str, &[&str]); 4] = [
        ("first", &["second"]),
        ("second", &["third"]),
        ("third", &["first", "native"]),
        ("empty", &[]),
    ];
    let native = EffectSummary::for_extern(Some(&EffectClause::Without(vec![Effect::IO])));
    let mut expected = None;
    for reverse in [false, true] {
        let mut ordered = edges.to_vec();
        if reverse {
            ordered.reverse();
        }
        let calls = graph(&ordered);
        let mut summaries = SummaryMap::new();
        summaries.insert("native", native);
        for &(name, _) in &ordered {
            summaries.insert(name, EffectSummary::default());
        }
        propagate_summaries(&mut summaries, &calls);
        for name in ["first", "second", "third"] {
            assert_eq!(summaries[name], native);
        }
        assert_eq!(summaries["empty"], EffectSummary::default());
        if let Some(previous) = &expected {
            assert_eq!(&summaries, previous);
        }
        expected = Some(summaries);
    }
}
