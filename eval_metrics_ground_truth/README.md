# Metrics eval ground truth

**Put your files here when using the PDF-dir + ground truth workflow.**

- **`pdfs/`** – Put all your ground truth PDFs in this folder.
- **`ground_truth.json`** – Multi-document format: `{ "DocName": { "metrics": { "GTPDF": "filename.pdf", "Net IRR": 0.1, "Net MOIC": "1.2x", ... } } }`. Use with `--pdf-dir` and `--ground-truth-json`.
- **Excel** – Alternatively use an Excel file (one tab per doc, A1 = document name, or one sheet with GTPDF column). Pass path to `--ground-truth-excel`. See `EVAL_METRICS_README.md` for format details.

Run from project root (JSON):

```bash
python eval_metrics_harness.py --pdf-dir eval_metrics_ground_truth/pdfs --ground-truth-json eval_metrics_ground_truth/ground_truth.json -o eval_metrics_results.csv
```

Or with Excel:

```bash
python eval_metrics_harness.py --pdf-dir eval_metrics_ground_truth/pdfs --ground-truth-excel eval_metrics_ground_truth/GroundTruth_Jan2026Updates.xlsx -o eval_metrics_results.csv
```
