# ILSA-TableQA — Exhibit & Question Taxonomy

The controlled vocabularies below are the annotation frame for the benchmark. They
are mirrored as sets in [`schema.py`](schema.py) (`EXHIBIT_TYPES`, `REASONING_TYPES`,
`ANSWER_KINDS`, `UNITS`); keep the two in sync. The design goal is to sample *across*
these axes so the benchmark tests the structural features that break naïve text
extraction — multi-level headers, standard errors in parentheses, significance
markers, and trend columns — rather than only easy single-cell lookups.

## 1. Exhibit types (`exhibit_type`)
The structural genres of statistical exhibit in ILSA international reports. Each has
a distinctive layout that stresses extraction differently.

| Type | What it contains | Why it's hard |
|---|---|---|
| `achievement_distribution` | Mean scale score + standard error, percentiles, a distribution graphic | SE lives in parentheses next to the estimate; graphic carries percentile info that text loses |
| `benchmark_percentages` | % of students reaching Advanced/High/Intermediate/Low benchmarks | Several % columns per row; easy to grab the wrong benchmark column |
| `trend` | Change across cycles (e.g. 2016→2021) with significance arrows (▲▼) | Direction + significance encoded as glyphs the text layer drops |
| `group_gap` | Group means (e.g. girls vs boys) + difference + significance | Multiple sub-columns per row; the answer is a computed/looked-up difference |
| `context_relationship` | Achievement by a context variable (home resources, attitude scales) | Multi-level headers pairing a category with an average achievement |
| `percentages_and_averages` | % of students in a category **and** their average achievement | Paired columns (percent, average) per category — easy to cross-wire |
| `policy_categorical` | Encyclopedia cross-country categorical/policy table (e.g. *Years of Schooling*, *Official Languages*) | Long country rows; categorical cells; merged/spanning headers |
| `other` | Anything not above | — |

## 2. Reasoning types (`reasoning_type`)
The question-answer coding frame — sample deliberately across it (the prior study's
eval was 68% single-fact lookups; over-sample the harder types here).

| Type | The task | Typical `answer_kind` |
|---|---|---|
| `lookup_single` | Read one cell | numeric_single / text |
| `lookup_multi` | Read several cells | numeric_multi / set |
| `standard_error_read` | Read the SE for an estimate | numeric_single (`se`) |
| `comparison` | Compare two cells/entities | boolean / direction / numeric_single |
| `extremum` | Highest/lowest/rank across rows | text / numeric_single |
| `benchmark_percentage` | % at a named benchmark | numeric_single (`percent`) |
| `group_gap` | Difference between groups (+ direction) | numeric_single / direction |
| `trend_direction` | Change across cycles (+ significance) | direction |
| `aggregation` | Sum / mean / count across cells | numeric_single |
| `significance_read` | Is a difference statistically significant? | boolean |
| `structure_meta` | What a header/column means (structure understanding) | text |

## 3. Answer kinds (`answer_kind`) → which scorer runs
Decided per item; drives dispatch in [`scorer.py`](scorer.py) `score_item`.

| Kind | Scorer | Notes |
|---|---|---|
| `numeric_single` | numeric-exact-with-tolerance (F1 over extracted numbers) | one target value |
| `numeric_multi` | numeric-exact set match (F1) | several target values |
| `direction` | up / down / flat vocabulary match | trends |
| `boolean` | yes-no / significant-not match | significance |
| `set` | token-set F1 | categorical sets (e.g. list of languages) |
| `text` | token-set F1 | single category / free text |

## 4. Units (`gold_unit`) and default tolerances
Set the unit so the scorer applies the right default tolerance (overridable per item
via `tolerance`). Defaults live in `scorer.DEFAULT_TOLERANCE`.

| Unit | Default tolerance | Rationale |
|---|---|---|
| `percent` | abs 0.05 | reported to 1 dp; absorb rounding, reject real errors |
| `scale_score` | abs 0.5 | PIRLS/TIMSS scale points |
| `se` | abs 0.05 | standard errors reported to 1–2 dp |
| `years` / `count` | abs 0.0 | integers, exact |
| `none` | abs 0.0 | exact |

## 5. Coverage target (WP1)
Aim for a benchmark that is **balanced across reasoning types** (not lookup-dominated)
and spans **≥3 studies** (PIRLS, TIMSS, ICILS). Log the achieved distribution and any
sampling caps explicitly — do not silently truncate a hard cell.
