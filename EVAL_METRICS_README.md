# Metrics-Only Eval Harness (IRR, MOIC/TVPI, DPI)

Deterministic evaluation of **performance metric** extraction across multiple LLMs. Models use judgment to determine which metrics to show: ideally **fund-level Net IRR, Net MOIC, Net DPI**; when those aren’t available, **Gross IRR**, **Gross TVPI**, or other headline figures; when the update is about **one specific deal**, deal-level metrics are acceptable. Non-performance documents (marketing notice, holiday card, announcement with no returns) should yield **all nulls**. No LLM-as-judge; results are compared to ground truth and scored 1 (match) or 0 (no match). Output CSV includes predicted values (for debugging), token usage, and **exact cost** using official pricing.

## Models evaluated

| Provider   | Model ID (API)              | Display name     |
|-----------|-----------------------------|------------------|
| OpenAI    | gpt-5.2                     | GPT-5.2          |
| OpenAI    | gpt-5-mini                  | GPT-5-mini        |
| OpenAI    | gpt-5-nano                  | GPT-5-nano        |
| Anthropic | claude-opus-4-20250514      | Claude Opus 4.6  |
| Anthropic | claude-4-opus-20250514      | Claude Opus 4    |
| Anthropic | claude-sonnet-4-20250514    | Claude Sonnet 4.5|
| Anthropic | claude-3-5-haiku-20241022   | Claude Haiku 4.5 |
| Anthropic | claude-3-haiku-20240307     | Claude Haiku 3   |
| Google    | gemini-2.5-flash            | Gemini 2.5 Flash |

Model IDs can be updated in `eval_metrics_harness.py` if the API uses different identifiers. Pricing is set from:

- [OpenAI Pricing](https://developers.openai.com/api/docs/pricing) (Standard tier)
- [Anthropic Pricing](https://platform.claude.com/docs/en/about-claude/pricing)
- [Gemini API Pricing](https://ai.google.dev/gemini-api/docs/pricing)

## Where to put your files

**Option A – PDFs + one Excel file (recommended):**

1. Create a folder for this eval, e.g. `eval_metrics_ground_truth/`.
2. Put all your ground truth PDFs in a subfolder: `eval_metrics_ground_truth/pdfs/`.
3. Put your Excel file with expected answers in the same folder: `eval_metrics_ground_truth/ground_truth.xlsx` (or any path).

Then run:

```bash
python eval_metrics_harness.py --pdf-dir eval_metrics_ground_truth/pdfs --ground-truth-excel eval_metrics_ground_truth/ground_truth.xlsx -o eval_metrics_results.csv
```

The harness matches each PDF to a row in the Excel by **filename**: the Excel must have a column that contains the PDF file name (e.g. `Report Q3 2025.pdf`) or the name without extension (e.g. `Report Q3 2025`). See **Excel format** below.

**Option B – One PDF + one JSON:**  
Put the PDF and a `ground_truth.json` anywhere and use `--pdf` and `--ground-truth`.

**Option C – Directory of test cases:**  
Each subdirectory has its own PDF and `ground_truth.json` (same layout as `eval_test_cases/`). Use `--dir`.

## Excel format (for --pdf-dir + --ground-truth-excel)

Two layouts are supported; the harness auto-detects which you use.

**1) Standard (one row per document)**  
- One column to identify the PDF: header `pdf`, `filename`, `gtpdf`, etc. Values match the PDF file name or stem.
- Metric columns: headers like `net_irr`, `net_moic`, `net_dpi` (or any header containing `irr`, `moic`/`tvpi`, `dpi`).

| pdf                    | net_irr | net_moic | net_dpi |
|------------------------|---------|----------|---------|
| Fund A Q3 2025.pdf      | 15.5    | 2.2      | 0.8     |
| Brand Capital update.pdf| 32      | 2.2      |         |

**2) Alternating label/value rows (one metric per PDF)**  
- **Each PDF has 2 rows:** row 0 = the most relevant metric for that document (e.g. MOIC, Gross IRR, DPI), row 1 = the ground truth value. Different PDFs can show different metrics—no fixed pattern.
- First column = metric label; value in the next row, same column or next column. Optional header row (e.g. "Metric", "Value") is skipped.

Example:

| Metric     | Value |
|------------|-------|
| MOIC       | 2.9   |
| Gross IRR  | 32    |
| DPI        | 0.1   |
| TVPI       | 1.5   |

The 1st PDF gets MOIC 2.9, the 2nd gets Gross IRR 32, etc. Match-by-order pairs the 1st row-pair with the 1st PDF, and so on.

**3) Label–value, one column per document (transposed)**  
- First column = metric names (one per row); next columns = one per document. Row 0 = optional doc names. Used when the first value row has multiple value cells (one per column).

## Ground truth format (JSON)

Ground truth should reflect **whatever the document actually reports** as its primary performance figures—Net or Gross, MOIC or TVPI. For non-performance documents (marketing, holiday cards, etc.), use all nulls. Two formats are supported.

**1) Explicit metrics (recommended when you provide ground truth):**

```json
{
  "net_irr": 32,
  "net_moic": 2.2,
  "net_dpi": null
}
```

Use `null` for any metric not reported (e.g. no DPI when the fund only reports Gross TVPI and Gross IRR). The keys stay `net_irr` / `net_moic` / `net_dpi`; the values are the numbers you expect the model to extract (e.g. 32 for “~32% Gross IRR”, 2.2 for “2.2× Gross TVPI”).

**2) Existing test case format:**  
If the JSON has `investment_performance` (list of strings), the harness parses the first IRR (Net or Gross), first MOIC/TVPI, and first DPI from that list.

## Environment

- `OPENAI_API_KEY` – required for OpenAI models  
- `ANTHROPIC_API_KEY` – required for Anthropic models  
- `GOOGLE_API_KEY` or `GEMINI_API_KEY` – required for Gemini models  

Install deps: `pip install -r requirements.txt` (includes `anthropic`).

## Usage

**Single PDF + ground truth file:**

```bash
python eval_metrics_harness.py --pdf path/to/document.pdf --ground-truth path/to/ground_truth.json --output eval_metrics_results.csv
```

**All test cases in a directory (e.g. existing eval_test_cases):**

Each subdirectory must contain at least one `.pdf` and a `ground_truth.json`. The subdir name is used as `test_case_id`.

```bash
python eval_metrics_harness.py --dir eval_test_cases --output eval_metrics_results.csv
```

**Options:**

- `--output`, `-o` – Output CSV path (default: `eval_metrics_results.csv`)
- `--test-case-id` – Override test case ID (default: PDF filename stem)
- `--tolerance` – Numeric match tolerance (default: 0.01)

## Output CSV

Columns:

- `test_case_id`, `model_id`, `model_display_name`, `provider`
- `irr_gt`, `irr_pred`, `irr_match` (1/0)
- `moic_gt`, `moic_pred`, `moic_match` (1/0)
- `dpi_gt`, `dpi_pred`, `dpi_match` (1/0)
- `input_tokens`, `output_tokens`, `cost_usd`

Use `*_pred` and `*_gt` for troubleshooting; use `*_match` and `cost_usd` for performance and cost comparison.
