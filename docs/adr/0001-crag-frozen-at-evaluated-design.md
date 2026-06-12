# CRAG is frozen at its evaluated design; improvements go to CRAG++

Status: accepted (2026-06-12)

The advanced/CRAG architecture (`src/graph_builder/graph_builder_adv.py`,
`src/nodes/advrag_nodes.py`) produced the l2/g2/d2 benchmark numbers in the final report with a
specific design: sub-questions **reuse the original query's retrieved documents** (no
per-sub-question retrieval), a ≤100-word synthesis cap, and a gemma3:1b document grader. A P1 fix
later added per-sub-question retrieval to these files in place, which silently turned the
benchmark baseline into a different architecture.

We decided to **revert CRAG to the evaluated design** (doc-reuse, word cap) while keeping the
crash-safety fixes that don't change the design (unified TypedDict state, all-docs-rejected
fallback, generation loop guard of 2 attempts, `<think>`-stripping). All behavioral improvements
— per-sub-question retrieval, chunk dedup, removal of the word cap — live in the separate
**CRAG++** architecture (`graph_builder_cragpp.py`), so CRAG-vs-CRAG++ comparisons measure the
design change against the same baseline the report describes.

Considered alternative: keep the fixed-in-place CRAG and differentiate CRAG++ by finer step-level
retrieval only. Rejected because the benchmark could then no longer isolate re-retrieval — the
headline improvement — as the variable under test, and "CRAG" in the code would no longer mean
what "CRAG" means in the report.

Do not "fix" `plan_sub_steps` back to per-sub-question retrieval; the doc-reuse there is
deliberate.
