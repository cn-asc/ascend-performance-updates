# Evaluation Test Cases

This directory contains test cases for evaluating the investment updates pipeline.

## Structure

Each test case should be in its own subdirectory with the following structure:

```
eval_test_cases/
  test_case_1/
    document.pdf          # The PDF to process
    ground_truth.json     # The expected output sections
  test_case_2/
    document.pdf
    ground_truth.json
  ...
```

## Ground Truth Format

The `ground_truth.json` file should contain the three sections as follows:

```json
{
  "investment_performance": [
    "Net IRR: 15.5% (vs benchmark 13.1% - Above Median)",
    "Net MOIC: 2.5x (vs benchmark 2.1x - Top Quartile)",
    "Net DPI: 1.2x (vs benchmark 1.0x - Above Median)",
    "Additional performance detail 1",
    "Additional performance detail 2"
  ],
  "key_takeaways": [
    "Key insight 1",
    "Key insight 2",
    "Key insight 3"
  ],
  "business_updates": [
    "Portfolio company X announced acquisition",
    "Market conditions improving in sector Y",
    "New investment in company Z"
  ]
}
```

Alternatively, you can provide the sections as strings (they will be parsed):

```json
{
  "investment_performance": "Investment Performance:\n  • Net IRR: 15.5%\n  • ...",
  "key_takeaways": "Key Takeaways:\n  • Insight 1\n  • ...",
  "business_updates": "Business Updates/Market Commentary:\n  • Update 1\n  • ..."
}
```

## Adding Test Cases

1. Create a new directory for your test case
2. Place the PDF file in that directory
3. Create `ground_truth.json` with the expected output sections
4. Run the evaluation: `python3 eval_harness.py [test_case_name]`
