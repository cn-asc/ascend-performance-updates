#!/usr/bin/env python3
"""Evaluation harness for the investment updates pipeline with deterministic metric checking.

This harness evaluates the pipeline by:
- Checking if IRR/MOIC/DPI number strings exist in the Quantitative Performance section
- Tracking tokens and latency across all three sections (Quantitative Performance, Key Takeaways and Business Updates, Market Commentary)
- No LLM judge is used - only deterministic string matching for metrics
"""
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from pdf_processor import extract_text_from_pdf
from analysis_agent import AnalysisAgent


class EvalHarness:
    """Evaluation harness for the investment updates pipeline."""
    
    def __init__(self, test_cases_dir: str = "eval_test_cases"):
        self.test_cases_dir = Path(test_cases_dir)
        self.analysis_agent = AnalysisAgent()
    
    def extract_metric_value(self, text: str, metric_name: str) -> List[str]:
        """
        Extract all metric values (IRR, MOIC, DPI) from text as strings.
        Returns all found number strings since we just need to check if they appear in both texts.
        
        Args:
            text: The text to search
            metric_name: One of 'irr', 'moic', 'dpi'
            
        Returns:
            List of number strings found (e.g., ['13.9', '15.7']), empty list if none found
        """
        # Normalize text to lowercase for matching
        text_lower = text.lower()
        
        # Patterns to match different formats - extract the number part
        patterns = {
            'irr': [
                r'net\s+irr[:\s]+([\d.]+)\s*%',
                r'irr[:\s]+([\d.]+)\s*%',
                r'([\d.]+)\s*%\s*net\s+irr',
                r'([\d.]+)\s*%\s*irr',
            ],
            'moic': [
                r'net\s+moic[:\s]+([\d.]+)\s*x',
                r'net\s+tvpi[:\s]+([\d.]+)\s*x',
                r'moic[:\s]+([\d.]+)\s*x',
                r'tvpi[:\s]+([\d.]+)\s*x',
                r'([\d.]+)\s*x\s*net\s+moic',
                r'([\d.]+)\s*x\s*moic',
            ],
            'dpi': [
                r'net\s+dpi[:\s]+([\d.]+)\s*[x%]',
                r'dpi[:\s]+([\d.]+)\s*[x%]',
                r'([\d.]+)\s*[x%]\s*net\s+dpi',
                r'([\d.]+)\s*[x%]\s*dpi',
            ]
        }
        
        if metric_name.lower() not in patterns:
            return []
        
        values = []
        seen = set()
        
        # Try each pattern and collect all matches as strings
        for pattern in patterns[metric_name.lower()]:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                value_str = match.group(1)
                # Avoid duplicates
                if value_str not in seen:
                    values.append(value_str)
                    seen.add(value_str)
        
        return values
    
    def check_metric_accuracy(self, ground_truth_text: str, predicted_text: str, metric_name: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Check if a metric value matches between ground truth and predicted.
        Simply checks if any number string from ground truth appears in predicted text.
        
        Args:
            ground_truth_text: Ground truth text
            predicted_text: Predicted text
            metric_name: One of 'irr', 'moic', 'dpi'
            
        Returns:
            Tuple of (is_match, gt_value, pred_value)
        """
        gt_value_strings = self.extract_metric_value(ground_truth_text, metric_name)
        pred_value_strings = self.extract_metric_value(predicted_text, metric_name)
        
        # If both are empty, consider it a match (metric not present)
        if not gt_value_strings and not pred_value_strings:
            return True, None, None
        
        # If one is empty and the other isn't, it's a mismatch
        if not gt_value_strings or not pred_value_strings:
            return False, gt_value_strings[0] if gt_value_strings else None, pred_value_strings[0] if pred_value_strings else None
        
        # Check if any ground truth value string appears in predicted text
        # We check both directions: GT value in pred text, and pred value in GT text
        for gt_val_str in gt_value_strings:
            # Check if this number appears in predicted text (as part of the metric)
            # Look for the number followed by % or x (for IRR/MOIC/DPI)
            if metric_name.lower() == 'irr':
                # For IRR, look for number followed by %
                pattern = re.escape(gt_val_str) + r'\s*%'
            else:
                # For MOIC/DPI, look for number followed by x or %
                pattern = re.escape(gt_val_str) + r'\s*[x%]'
            
            if re.search(pattern, predicted_text, re.IGNORECASE):
                return True, gt_val_str, gt_val_str
        
        # Also check reverse: if any predicted value appears in ground truth
        for pred_val_str in pred_value_strings:
            if metric_name.lower() == 'irr':
                pattern = re.escape(pred_val_str) + r'\s*%'
            else:
                pattern = re.escape(pred_val_str) + r'\s*[x%]'
            
            if re.search(pattern, ground_truth_text, re.IGNORECASE):
                return True, pred_val_str, pred_val_str
        
        # No match found
        return False, gt_value_strings[0], pred_value_strings[0]
    
    def load_test_case(self, test_case_name: str) -> Tuple[str, Dict]:
        """
        Load a test case (PDF and ground truth).
        
        Args:
            test_case_name: Name of the test case directory
            
        Returns:
            Tuple of (pdf_path, ground_truth_dict)
        """
        test_case_path = self.test_cases_dir / test_case_name
        
        # Load ground truth first
        ground_truth_file = test_case_path / "ground_truth.json"
        if not ground_truth_file.exists():
            raise FileNotFoundError(f"No ground_truth.json found in {test_case_path}")
        
        with open(ground_truth_file, 'r') as f:
            ground_truth = json.load(f)
        
        # Find PDF file(s)
        pdf_files = list(test_case_path.glob("*.pdf"))
        if not pdf_files:
            raise FileNotFoundError(f"No PDF found in {test_case_path}")
        
        # If multiple PDFs, try to match by name similarity with test case directory name
        if len(pdf_files) > 1:
            print(f"  ⚠️  Warning: Found {len(pdf_files)} PDFs in {test_case_name}, attempting to match...")
            
            # Normalize test case name for matching (remove underscores, dashes, lowercase)
            test_case_normalized = test_case_name.lower().replace('_', ' ').replace('-', ' ')
            # Remove common prefixes/suffixes
            test_case_normalized = test_case_normalized.replace('2025 ', '').replace('q3 ', '').replace('q4 ', '')
            
            best_match = None
            best_score = 0
            
            for pdf_file in pdf_files:
                # Normalize PDF name similarly
                pdf_name_normalized = pdf_file.stem.lower().replace('_', ' ').replace('-', ' ')
                pdf_name_normalized = pdf_name_normalized.replace('2025.', '').replace('q3 ', '').replace('q4 ', '')
                
                # Calculate word overlap score
                test_words = set(test_case_normalized.split())
                pdf_words = set(pdf_name_normalized.split())
                
                # Remove common stop words
                stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'update', 'report', 'quarterly', 'performance'}
                test_words = test_words - stop_words
                pdf_words = pdf_words - stop_words
                
                if test_words and pdf_words:
                    common_words = test_words & pdf_words
                    score = len(common_words) / max(len(test_words), len(pdf_words), 1)
                else:
                    score = 0
                
                if score > best_score:
                    best_score = score
                    best_match = pdf_file
            
            if best_match and best_score > 0.2:  # At least 20% word overlap
                pdf_path = best_match
                print(f"  ✓ Selected PDF: {pdf_path.name} (match score: {best_score:.2f})")
            else:
                # Fall back to first PDF but warn strongly
                pdf_path = pdf_files[0]
                print(f"  ⚠️  WARNING: Could not reliably match PDF!")
                print(f"     Using first PDF: {pdf_path.name}")
                print(f"     Best match score was: {best_score:.2f}")
                print(f"     Please verify this is the correct PDF for ground truth.")
        else:
            pdf_path = pdf_files[0]
        
        return str(pdf_path), ground_truth
    
    def extract_section(self, text: str, section_name: str) -> str:
        """
        Extract a specific section from formatted update text.
        
        Args:
            text: The full formatted update text
            section_name: Name of section to extract (e.g., "Quantitative Performance")
            
        Returns:
            The section text, or empty string if not found
        """
        lines = text.split('\n')
        section_lines = []
        in_section = False
        
        # Section headers that indicate we should stop (new section names)
        stop_headers = ['Quantitative Performance:', 'Key Takeaways and Business Updates:', 
                        'Market Commentary:', 'Key Takeaways:', 'Business Updates/Market Commentary:',
                        'Business Updates:', 'Investment Performance:']
        
        for i, line in enumerate(lines):
            # Remove markdown bold syntax for matching
            line_clean = line.replace('**', '').strip()
            
            # Check if this is the section we're looking for
            if section_name in line_clean and ':' in line_clean:
                in_section = True
                section_lines.append(line)
                continue
            
            if in_section:
                # Stop at next section header
                if any(header in line_clean for header in stop_headers if header != section_name):
                    break
                # Also stop if we hit a blank line followed by what looks like a new section
                if i > 0 and not lines[i-1].strip() and ':' in line_clean and not line_clean.startswith('•'):
                    # Check if it's a section header (not a bullet continuation)
                    if not line_clean.startswith('  •'):
                        break
                section_lines.append(line)
        
        result = '\n'.join(section_lines).strip()
        # Remove trailing blank lines
        while result.endswith('\n'):
            result = result[:-1]
        return result
    
    def evaluate_test_case(self, test_case_name: str) -> Dict:
        """
        Evaluate a single test case.
        
        Args:
            test_case_name: Name of the test case
            
        Returns:
            Dictionary with evaluation results
        """
        print(f"\n{'='*80}")
        print(f"Evaluating: {test_case_name}")
        print(f"{'='*80}")
        
        # Load test case
        pdf_path, ground_truth = self.load_test_case(test_case_name)
        
        # Extract text from PDF
        print(f"Extracting text from PDF...")
        pdf_text = extract_text_from_pdf(pdf_path)
        
        # Run pipeline
        print(f"Running pipeline...")
        predicted_text, metadata, metrics = self.analysis_agent.analyze_update(pdf_text, test_case_name)
        
        # Map old section keys to new section names
        section_mapping = {
            'investment_performance': 'Quantitative Performance',
            'key_takeaways': 'Key Takeaways and Business Updates',
            'business_updates': 'Market Commentary'
        }
        
        results = {
            'test_case': test_case_name,
            'metadata': metadata,
            'metrics': metrics,
            'sections': {}
        }
        
        # Only evaluate Quantitative Performance section for metrics
        print(f"\nEvaluating Quantitative Performance...")
        
        # Extract Quantitative Performance section from ground truth
        ground_truth_section = ground_truth.get('investment_performance', '')
        if isinstance(ground_truth_section, list):
            ground_truth_section = '\n'.join([f"  • {item}" for item in ground_truth_section])
        
        # Extract Quantitative Performance section from predicted (using new section name)
        predicted_section = self.extract_section(predicted_text, 'Quantitative Performance')
        
        # Check IRR, MOIC, DPI accuracy deterministically - just check if number strings exist
        irr_match, irr_gt, irr_pred = self.check_metric_accuracy(ground_truth_section, predicted_section, 'irr')
        moic_match, moic_gt, moic_pred = self.check_metric_accuracy(ground_truth_section, predicted_section, 'moic')
        dpi_match, dpi_gt, dpi_pred = self.check_metric_accuracy(ground_truth_section, predicted_section, 'dpi')
        
        # Calculate scores: 1 point per metric found (total 3 points)
        irr_score = 1.0 if irr_match else 0.0
        if irr_gt is None and irr_pred is None:
            irr_score = 1.0  # Both missing, assume OK
        elif irr_gt is None or irr_pred is None:
            irr_score = 0.0  # One missing, fail
        
        moic_score = 1.0 if moic_match else 0.0
        if moic_gt is None and moic_pred is None:
            moic_score = 1.0  # Both missing, assume OK
        elif moic_gt is None or moic_pred is None:
            moic_score = 0.0  # One missing, fail
        
        dpi_score = 1.0 if dpi_match else 0.0
        if dpi_gt is None and dpi_pred is None:
            dpi_score = 1.0  # Both missing, assume OK
        elif dpi_gt is None or dpi_pred is None:
            dpi_score = 0.0  # One missing, fail
        
        quantitative_performance_score = irr_score + moic_score + dpi_score
        
        print(f"  Metric checks:")
        print(f"    IRR: {'✓' if irr_match else '✗'} (GT: {irr_gt}, Pred: {irr_pred}) - Score: {irr_score:.1f}/1")
        print(f"    MOIC: {'✓' if moic_match else '✗'} (GT: {moic_gt}, Pred: {moic_pred}) - Score: {moic_score:.1f}/1")
        print(f"    DPI: {'✓' if dpi_match else '✗'} (GT: {dpi_gt}, Pred: {dpi_pred}) - Score: {dpi_score:.1f}/1")
        print(f"    Quantitative Performance Total: {quantitative_performance_score:.1f}/3")
        
        results['sections']['quantitative_performance'] = {
            'score': quantitative_performance_score,  # Out of 3 points
            'irr_score': irr_score,
            'moic_score': moic_score,
            'dpi_score': dpi_score,
            'metric_checks': {
                'irr': {'match': irr_match, 'gt_value': irr_gt, 'pred_value': irr_pred, 'score': irr_score},
                'moic': {'match': moic_match, 'gt_value': moic_gt, 'pred_value': moic_pred, 'score': moic_score},
                'dpi': {'match': dpi_match, 'gt_value': dpi_gt, 'pred_value': dpi_pred, 'score': dpi_score}
            },
            'ground_truth': ground_truth_section,
            'predicted': predicted_section
        }
        
        # Extract and store other sections for token/latency tracking (but don't evaluate)
        for section_key, section_name in section_mapping.items():
            if section_key != 'investment_performance':
                ground_truth_section = ground_truth.get(section_key, '')
                if isinstance(ground_truth_section, list):
                    ground_truth_section = '\n'.join([f"  • {item}" for item in ground_truth_section])
                
                predicted_section = self.extract_section(predicted_text, section_name)
                
                results['sections'][section_key] = {
                    'ground_truth': ground_truth_section,
                    'predicted': predicted_section
                }
        
        return results
    
    def run_evaluation(self, test_case_names: List[str] = None) -> Dict:
        """
        Run evaluation on all test cases or specified ones.
        
        Args:
            test_case_names: List of test case names to evaluate, or None for all
            
        Returns:
            Dictionary with overall results
        """
        # Find all test cases
        if not self.test_cases_dir.exists():
            raise FileNotFoundError(f"Test cases directory not found: {self.test_cases_dir}")
        
        if test_case_names is None:
            test_case_names = [d.name for d in self.test_cases_dir.iterdir() if d.is_dir()]
        
        print(f"Found {len(test_case_names)} test case(s) to evaluate")
        
        all_results = []
        quantitative_performance_scores = []
        
        # Track costs
        total_extraction_cost = 0
        total_formatting_cost = 0
        
        for test_case_name in test_case_names:
            try:
                result = self.evaluate_test_case(test_case_name)
                all_results.append(result)
                
                # Collect quantitative performance scores
                if 'quantitative_performance' in result['sections']:
                    quantitative_performance_scores.append(result['sections']['quantitative_performance']['score'])
                
                # Collect costs
                metrics = result.get('metrics', {})
                total_extraction_cost += metrics.get('extraction_cost', 0)
                total_formatting_cost += metrics.get('formatting_cost', 0)
            
            except Exception as e:
                print(f"Error evaluating {test_case_name}: {e}")
                continue
        
        # Calculate averages
        summary = {
            'total_test_cases': len(all_results),
            'quantitative_performance_average': sum(quantitative_performance_scores) / len(quantitative_performance_scores) if quantitative_performance_scores else 0,
            'cost_summary': {
                'total_extraction_cost': total_extraction_cost,
                'total_formatting_cost': total_formatting_cost,
                'total_cost': total_extraction_cost + total_formatting_cost
            },
            'detailed_results': all_results
        }
        
        return summary
    
    def save_results(self, results: Dict, output_file: str = "eval_results.json"):
        """Save evaluation results to JSON file."""
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✅ Results saved to {output_file}")
    
    def save_results_csv(self, results: Dict, output_file: str = "eval_results.csv"):
        """
        Save evaluation results as CSV with token/latency tracking across sections.
        
        Columns:
        - Test Case | Quantitative Performance Score | Quantitative Performance Output | Quantitative Performance Ground Truth |
          Key Takeaways and Business Updates Output | Key Takeaways and Business Updates Ground Truth |
          Market Commentary Output | Market Commentary Ground Truth |
          Extraction Agent Token Usage | Extraction Agent Cost ($) | Extraction Agent P95 Latency (ms) | 
          Formatting Agent Token Usage | Formatting Agent Cost ($) | Formatting Agent P95 Latency (ms) |
          Total Cost ($)
        """
        import csv
        
        # Collect latencies for P95 calculation
        extraction_latencies = []
        formatting_latencies = []
        
        for result in results.get('detailed_results', []):
            metrics = result.get('metrics', {})
            if 'extraction_latency_ms' in metrics:
                extraction_latencies.append(metrics['extraction_latency_ms'])
            if 'formatting_latency_ms' in metrics:
                formatting_latencies.append(metrics['formatting_latency_ms'])
        
        # Calculate P95 latencies (95th percentile)
        def percentile(data, p):
            if not data:
                return 0
            sorted_data = sorted(data)
            k = (len(sorted_data) - 1) * p / 100
            f = int(k)
            c = k - f
            if f + 1 < len(sorted_data):
                return sorted_data[f] * (1 - c) + sorted_data[f + 1] * c
            return sorted_data[f]
        
        extraction_p95 = percentile(extraction_latencies, 95)
        formatting_p95 = percentile(formatting_latencies, 95)
        
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Write header
            writer.writerow([
                'Test Case',
                'Quantitative Performance Score',
                'Quantitative Performance Output',
                'Quantitative Performance Ground Truth',
                'Key Takeaways and Business Updates Output',
                'Key Takeaways and Business Updates Ground Truth',
                'Market Commentary Output',
                'Market Commentary Ground Truth',
                'Extraction Agent Token Usage',
                'Extraction Agent Cost ($)',
                'Extraction Agent P95 Latency (ms)',
                'Formatting Agent Token Usage',
                'Formatting Agent Cost ($)',
                'Formatting Agent P95 Latency (ms)',
                'Total Cost ($)'
            ])
            
            # Write data rows
            for result in results.get('detailed_results', []):
                test_case = result.get('test_case', '')
                sections = result.get('sections', {})
                metrics = result.get('metrics', {})
                
                # Get section data
                qp_section = sections.get('quantitative_performance', {})
                kt_section = sections.get('key_takeaways', {})
                mc_section = sections.get('business_updates', {})
                
                # Get metrics
                extraction_tokens = metrics.get('extraction_tokens', 0)
                extraction_cost = metrics.get('extraction_cost', 0)
                formatting_tokens = metrics.get('formatting_tokens', 0)
                formatting_cost = metrics.get('formatting_cost', 0)
                total_cost = metrics.get('total_cost', 0)
                
                writer.writerow([
                    test_case,
                    f"{qp_section.get('score', 0):.1f}",
                    qp_section.get('predicted', '').replace('\n', ' ').replace('\r', ' '),
                    qp_section.get('ground_truth', '').replace('\n', ' ').replace('\r', ' '),
                    kt_section.get('predicted', '').replace('\n', ' ').replace('\r', ' '),
                    kt_section.get('ground_truth', '').replace('\n', ' ').replace('\r', ' '),
                    mc_section.get('predicted', '').replace('\n', ' ').replace('\r', ' '),
                    mc_section.get('ground_truth', '').replace('\n', ' ').replace('\r', ' '),
                    extraction_tokens,
                    f"{extraction_cost:.4f}",
                    f"{extraction_p95:.2f}",
                    formatting_tokens,
                    f"{formatting_cost:.4f}",
                    f"{formatting_p95:.2f}",
                    f"{total_cost:.4f}"
                ])
        
        print(f"\n✅ CSV results saved to {output_file}")


def main():
    """Main entry point for evaluation harness."""
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='Run evaluation harness on test cases')
    parser.add_argument('test_cases', nargs='*', help='Specific test case names to evaluate (optional)')
    parser.add_argument('--limit', type=int, help='Limit to first N test cases (ignored if specific test cases are provided)')
    
    args = parser.parse_args()
    
    harness = EvalHarness()
    
    # Determine which test cases to evaluate
    if args.test_cases:
        # Use specified test cases
        test_cases = args.test_cases
    elif args.limit:
        # Get all test cases and limit to first N
        test_cases_dir = Path("eval_test_cases")
        if not test_cases_dir.exists():
            raise FileNotFoundError(f"Test cases directory not found: {test_cases_dir}")
        all_test_cases = [d.name for d in test_cases_dir.iterdir() if d.is_dir()]
        test_cases = all_test_cases[:args.limit]
        print(f"Limiting to first {args.limit} of {len(all_test_cases)} test cases")
    else:
        # Use all test cases
        test_cases = None
    
    print("🚀 Starting Evaluation Harness")
    print("=" * 80)
    
    results = harness.run_evaluation(test_case_names=test_cases)
    
    # Print summary
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Total test cases: {results['total_test_cases']}")
    print(f"\nQuantitative Performance Average: {results['quantitative_performance_average']:.1f}/3")
    
    # Print cost summary
    cost_summary = results.get('cost_summary', {})
    if cost_summary:
        print(f"\nCost Summary:")
        print(f"  Extraction Agent Cost: ${cost_summary.get('total_extraction_cost', 0):.4f}")
        print(f"  Formatting Agent Cost: ${cost_summary.get('total_formatting_cost', 0):.4f}")
        print(f"  Total Cost: ${cost_summary.get('total_cost', 0):.4f}")
    
    # Save results
    harness.save_results(results)
    harness.save_results_csv(results)
    
    # Print table preview
    print("\n" + "=" * 80)
    print("RESULTS TABLE PREVIEW")
    print("=" * 80)
    print(f"\nResults saved to CSV with columns:")
    print("  - Test Case")
    print("  - Quantitative Performance Score (out of 3)")
    print("  - Quantitative Performance Output")
    print("  - Quantitative Performance Ground Truth")
    print("  - Key Takeaways and Business Updates Output")
    print("  - Key Takeaways and Business Updates Ground Truth")
    print("  - Market Commentary Output")
    print("  - Market Commentary Ground Truth")
    print("  - Extraction Agent Token Usage")
    print("  - Extraction Agent Cost ($)")
    print("  - Extraction Agent P95 Latency (ms)")
    print("  - Formatting Agent Token Usage")
    print("  - Formatting Agent Cost ($)")
    print("  - Formatting Agent P95 Latency (ms)")
    print("  - Total Cost ($)")
    
    return results


if __name__ == "__main__":
    main()
