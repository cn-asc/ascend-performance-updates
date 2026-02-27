# Evaluation Harness

This evaluation harness uses an LLM as a judge to evaluate the investment updates pipeline across three separate dimensions:

1. **Investment Performance** - Accuracy of metrics extraction and benchmark comparisons
2. **Key Takeaways** - Relevance and completeness of insights
3. **Business Updates/Market Commentary** - Quality of business and market updates

## Quick Start

### 1. Create a Test Case

```bash
python3 create_test_case.py <test_case_name> <path_to_pdf> [optional_ground_truth.json]
```

Example:
```bash
python3 create_test_case.py fund_abc_q3_2024 ./updates/fund_abc_q3.pdf
```

This will:
- Create a directory `eval_test_cases/fund_abc_q3_2024/`
- Copy your PDF into that directory
- Create a template `ground_truth.json` file

### 2. Edit Ground Truth

Edit `eval_test_cases/<test_case_name>/ground_truth.json` with the expected output:

```json
{
  "investment_performance": [
    "Net IRR: 15.5% (vs benchmark 13.1% - Above Median)",
    "Net MOIC: 2.5x (vs benchmark 2.1x - Top Quartile)",
    "Additional performance detail..."
  ],
  "key_takeaways": [
    "Key insight 1",
    "Key insight 2"
  ],
  "business_updates": [
    "Portfolio company update...",
    "Market commentary..."
  ]
}
```

### 3. Run Evaluation

Evaluate a single test case:
```bash
python3 eval_harness.py <test_case_name>
```

Evaluate all test cases:
```bash
python3 eval_harness.py
```

### 4. Review Results

Results are saved to `eval_results.json` with:
- Scores for each section (0-100)
- Explanations from the LLM judge
- Missing/incorrect information
- Full ground truth and predicted outputs

## How It Works

1. **PDF Processing**: Extracts text from the PDF using the same pipeline
2. **Pipeline Execution**: Runs the full analysis pipeline (extraction + formatting)
3. **Section Extraction**: Separates the three sections from the output
4. **LLM Judging**: Uses three separate LLM judges (one per section) to compare predicted vs ground truth
5. **Scoring**: Each judge provides a score (0-100) with explanations

## Evaluation Criteria

### Investment Performance Judge
- Accuracy of metrics (IRR, MOIC, DPI)
- Completeness of performance details
- Accuracy of benchmark comparisons
- Clarity of presentation

### Key Takeaways Judge
- Relevance of insights
- Completeness of critical information
- Prioritization (most important first)
- Clarity and actionability

### Business Updates Judge
- Relevance and newsworthiness
- Completeness of developments
- Prioritization
- Exclusion of generic information

## Output Format

The evaluation produces:
- **Section scores**: Average score per section across all test cases
- **Overall average**: Average across all sections
- **Detailed results**: Full evaluation for each test case including:
  - Score and explanation
  - Missing information
  - Incorrect information
  - Ground truth and predicted outputs

## Tips

- Use representative test cases that cover different asset classes and vintages
- Ensure ground truth includes benchmark comparisons if the PDF contains asset class/vintage info
- Review the detailed results to identify patterns in what the pipeline misses
- Use the explanations to improve prompts and extraction logic
