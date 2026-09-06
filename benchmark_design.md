# Benchmark Design

This document covers the benchmark design, dataset taxonomy, and scorer
calibration. See the [README](README.md) for the overview, usage, and results.

- The dataset is category-balanced: each category contains 60 items, so category
  scores remain directly comparable.
- Conversation cases do not use system prompts. The final user prompt is appended
  after the prior turns, so the benchmark tests whether the model follows the
  latest task while resisting cross-language priming.
- Explicit Catalan language instructions follow the app policy being tested:
  - unmodified `rag_context` items include a final Catalan instruction;
    hardened items deliberately remove that reset signal.
  - `monolingual` never includes an explicit Catalan instruction, because it
    models an ordinary Catalan conversation.
  - unmodified `crosslingual_basic`, `multi_turn`, and `crosslingual_advanced` include an
    explicit Catalan instruction only when the case contains non-Catalan text and
    the full final user prompt is shorter than 10 words.
- RAG documents come from CC BY 4.0 Diputació de Barcelona Open Data records,
  currently the paired `parcsequipaments_ca` and `parcsequipaments_es` datasets.
  Only safe descriptive fragments are kept; contact, location, schedule, and
  personal data fields are filtered out.

The dataset contains 300 items total (60 per category × 5 categories). It is
built deterministically with `make build`.

## Taxonomy

Each sample should specify:

- `persona`: `pime`, `administracio`, or `usuari_final`.
- `workflow`: one of the real task types used in the benchmark:
  `ai_misconception_explanation`, `citizen_response`, `client_delay_update`,
  `client_reply`, `community_health_bulletin`, `complaint_response`,
  `internal_briefing`, `privacy_guidance`, `procurement_note`,
  `project_status`, `public_info_summary`, `public_notice`,
  `public_project_status`, `service_summary`, `study_plan`, `support_reply`,
  or `tenant_request`.
- `category`: `monolingual`, `crosslingual_basic`, `multi_turn`,
  `crosslingual_advanced`, or `rag_context`.
  - `monolingual`: Catalan-only prompts and context.
  - `crosslingual_basic`: A Catalan task with Spanish or English source
    material.
  - `multi_turn`: Conversations that test whether the model follows the
    language of the final request despite earlier crosslingual context.
  - `crosslingual_advanced`: Multi-turn conversations combining
    crosslingual body content, reusable-template or closing pressure, and
    assistant priming before the final Catalan ask.
  - `rag_context`: Retrieved-context prompts answered in Catalan.
    `source_lang` distinguishes Spanish-only context from mixed Catalan and
    Spanish context.
- `pressure_pattern`: consolidated classification of the
  language-drift pressure carried by the item, used for cross-category
  slicing:
  - `no_pressure` (60): monolingual items — control.
  - `english_or_trilingual` (7): unmodified English or trilingual pressure.
  - `rag_context` (13): retrieved context with the explicit Catalan guardrail.
  - `harder_template_es` (63): saturated Spanish-template pressure.
  - `harder_template_mixed` (77): mixed ES/EN template pressure.
  - `harder_short_implicit` (80): dominant non-Catalan context followed by
    a short implicit Catalan task.
  Use `scripts/pressure_pattern_report.py` to slice model failures by
  pattern.
- `source_lang`: the source/context language pattern:
  - `ca`: Catalan-only prompt and context.
  - `es`: Spanish retrieved context answered in Catalan.
  - `ca-es`: mixed Catalan and Spanish retrieved context answered in Catalan.
  - `es-ca`: Spanish source material or prior-turn context answered in Catalan.
  - `en-ca`: English source material or prior-turn context answered in Catalan.
  - `en-es-ca` / `es-en-ca`: trilingual items combining English and Spanish
    across prior turns (order marks the first crosslingual turn) answered in
    Catalan.

## Distributed Adversarial Pressure

Difficulty is independent of the scenario category. Three adversarial patterns
are distributed across all four non-control categories:

- `template_es` (63 items): Spanish reusable headers and Spanish assistant
  priming before the final Catalan task.
- `template_mixed` (77 items): combined ES/EN reusable headers and mixed
  assistant priming.
- `short_implicit` (80 items): dominant non-Catalan context followed by a
  short implicit Catalan task without an explicit language reset.

The other 20 non-control items retain their original pressure form where that
is needed to match the calibration target. No item asks for a Spanish or
English response; language failures therefore represent drift rather than
instruction following.

## Scorer Calibration

`LANGUAGE_MIN_CONFIDENCE` and `LANGUAGE_FAIL_NON_CA_RATIO` in
`lm_eval_tasks/catalan_drift/utils.py` are calibrated against one validation
slice: FLORES-200 dev+devtest sentences across `ca`, `es`, and `en`, bucketed
by length. Fetch the corpus with `make flores-corpus` and build the slice with
`scripts/build_slice.py`. The sweep measures per-language precision/recall
with Wilson 95% CIs computed over **segments** (one independent trial per
sentence), not tokens. Tokens within a segment share a prediction and are not
independent, so counting them shrinks CIs by roughly the mean segment length.

`scripts/compare_language_detectors.py` sweeps
`min_conf ∈ {0.30..0.90}` × `ratio ∈ {0.05..0.25}` over that slice and
writes `outputs/scorer_calibration.md` / `.json`. Only `min_conf` is
tuned: `ratio` is held at its current value because the slice is monolingual,
so gold ratio is 0 or 1 and does not provide a useful ratio signal. Surviving
candidates are ranked by segment F1 restricted to `ca`/`es`/`en`, the
languages actually gated on, and rounded to 3 decimals so the tie-break to
the current operating point can fire. CI acceptance uses the same 3-decimal
precision shown in the report, so `0/92` observed Catalan false positives
counts as a `0.040` Wilson upper bound and passes the 4% guard.

The recommended operating point is:

- `LANGUAGE_MIN_CONFIDENCE = 0.65`
- `LANGUAGE_FAIL_NON_CA_RATIO = 0.15`

If the sweep returns no recommended candidate, keep the current operating
point and grow the FLORES slice or revisit the acceptance criteria before
changing thresholds.
