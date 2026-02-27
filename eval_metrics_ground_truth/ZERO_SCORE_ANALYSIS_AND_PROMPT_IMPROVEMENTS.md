# Zero-Score Test Cases: Cross-Check with Eval Prompt and Prompt Improvements

## 1. Ground truth examples that got 0 across all models

From the eval report, these test cases had **overall_score = 0 for every model**:

| test_case_id | Notes |
|--------------|--------|
| **2025.09 Q3 - L Squared Capital Partners IV - Functional Devices Investor Presentation.pdf** | All models scored 0; some (e.g. Claude Haiku) had parse_error "No JSON obj" on this doc. |
| **2026.01 Q1 - L Squared Capital Partners IV - Functional Devices Investor Presentation.pdf** | Same document (different quarter label); all models scored 0. |

In **ground_truth_augmented.json**, the corresponding entry is:

```json
"LSquared Functional Devices": {
  "metrics": {
    "GTPDF": "2026.01 Q1 - L Squared Capital Partners IV - Functional Devices Investor Presentation.pdf",
    "No performance metrics to show": null
  }
}
```

So the ground truth for this document is: **no fund-level performance metrics**. The doc is an **Investor Presentation** (pitch/deck), not a quarterly performance letter. GT expects `net_irr`, `net_moic`, `net_dpi` to be absent (null), not extracted from company-level figures in the deck.

---

## 2. Cross-check: eval prompt vs these edge cases

**What the current EXTRACTION_PROMPT does:**

- Says to extract "fund-level Net IRR, Net MOIC (or TVPI), and Net DPI" and "If not found, use null."
- Does **not** explicitly say:
  - When the document is an **investor presentation**, **pitch deck**, **announcement**, or **calendar** that does **not** report fund-level returns, set `net_irr`, `net_moic`, `net_dpi` to **null**.
  - Do **not** use **company-level** or **portfolio-company** metrics (e.g. revenue, EBITDA multiples, deal-level IRR/MOIC) as **fund-level** Net IRR / Net MOIC / Net DPI.
- So models tend to pull numbers from the deck (e.g. "26% IRR and 3.1x MOIC" for a specific deal or company) and put them in `net_irr` / `net_moic`, which is wrong for this document type and causes mismatches (or parse issues when the model is unsure and output is malformed).

**Why this produces 0 or parse errors:**

- Either the model returns **valid JSON** with **wrong numbers** (deal-level vs fund-level) → strict match gives 0.
- Or the model output is **truncated/malformed** (e.g. "No JSON obj") → parse_error and no credit.
- GT expects **all null** for these metrics; the prompt does not clearly tell the model to **prefer null when the doc type does not contain fund-level returns**.

---

## 3. Proposed prompt improvements to capture these edge cases

Add the following to the extraction prompt (in **eval_metrics_harness.py** and, if used in production, **analysis_agent.py**):

1. **Document-type rule for “no fund-level metrics”**  
   Explicitly instruct: when the document is an **investor presentation**, **pitch deck**, **investment memo**, **announcement** (e.g. trip, close, calendar), or similar and it **does not** contain **fund-level** Net IRR / Net MOIC / Net DPI, set `net_irr`, `net_moic`, and `net_dpi` to **null**. Do not infer fund-level returns from company-level or deal-level metrics.

2. **Fund-level vs company/deal-level**  
   Add a line: **Only** populate `net_irr`, `net_moic`, `net_dpi` when the values are explicitly labeled as **fund-level** (or equivalent). Do **not** use portfolio-company returns, deal-level IRR/MOIC, or company financials (e.g. revenue, EBITDA multiples) as fund-level metrics.

3. **Return only valid JSON**  
   Reinforce: always return a single, well-formed JSON object (no trailing text, no invalid escapes, no embedded snippets). This helps avoid "No JSON obj" and "Invalid \\escape" on edge-case docs.

4. **Optional: explicit “no metrics” phrasing**  
   Add: "If the document clearly has no fund-level performance figures (e.g. 'No performance metrics to show', or presentation-only), set net_irr, net_moic, and net_dpi to null."

These changes align the prompt with the ground truth for the L Squared Functional Devices (and similar) test cases and should reduce zero scores and parse errors on those documents.
