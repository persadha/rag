"""ILSA-TableQA benchmark scaffolding (proposal-stage).

Modules:
  schema           - the item definition (single source of truth) + validation + JSONL IO
  scorer           - numeric-exact-with-tolerance (real) + table-structure recovery (stub)
  score_benchmark  - CLI harness: validate items, emit template, score a system's answers
"""
