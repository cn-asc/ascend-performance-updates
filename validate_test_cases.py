#!/usr/bin/env python3
"""Validate that PDFs and ground truth JSON files are properly matched."""
import json
import re
from pathlib import Path
from pdf_processor import extract_text_from_pdf
from analysis_agent import ExtractionAgent


def extract_fund_name_from_text(text: str) -> str:
    """Extract fund/company name from PDF text."""
    if not text:
        return None
    
    # Look for common patterns
    patterns = [
        r'(?:Fund|Funds?|Partners?|Capital|Investments?|Holdings?|Group|Management)[\s:]+([A-Z][^,\n]{5,50})',
        r'([A-Z][A-Za-z\s&,]+(?:Fund|Partners?|Capital|Investments?|Holdings?|Group|Management))',
    ]
    
    text_upper = text[:2000].upper()  # Check first 2000 chars
    
    for pattern in patterns:
        matches = re.findall(pattern, text[:2000], re.IGNORECASE)
        if matches:
            # Return the longest match (likely most complete)
            return max(matches, key=len).strip()
    
    return None


def extract_key_info_from_text(text: str, pdf_filename: str = None) -> dict:
    """Extract key identifying information from PDF text."""
    if not text:
        return {}
    
    text_lower = text.lower()
    
    info = {
        'fund_name': None,
        'irr_mentions': len(re.findall(r'\birr\b', text_lower[:2000])),
        'moic_mentions': len(re.findall(r'\bmoic\b', text_lower[:2000])),
        'dpi_mentions': len(re.findall(r'\bdpi\b', text_lower[:2000])),
        'date_mentions': len(re.findall(r'2025|q3|q4|september|november|december', text_lower[:1000])),
        'sample_text': text[:300].replace('\n', ' ').strip()
    }
    
    # Try to extract fund name from PDF filename first
    if pdf_filename:
        # Remove common suffixes
        name = pdf_filename.replace('.pdf', '')
        name = re.sub(r'\s*\(1\)\s*$', '', name)
        name = re.sub(r'\s*-\s*Q3\s*2025.*$', '', name, flags=re.IGNORECASE)
        name = re.sub(r'\s*-\s*Q4\s*2025.*$', '', name, flags=re.IGNORECASE)
        name = re.sub(r'\s*-\s*November\s*2025.*$', '', name, flags=re.IGNORECASE)
        name = re.sub(r'\s*-\s*December\s*2025.*$', '', name, flags=re.IGNORECASE)
        name = re.sub(r'^\d{4}\.\d{2}\s*', '', name)  # Remove date prefix like "2025.09"
        info['fund_name'] = name.strip()
    
    # If filename extraction didn't work, try from text
    if not info['fund_name']:
        info['fund_name'] = extract_fund_name_from_text(text)
    
    return info


def extract_key_info_from_json(gt_data: dict, test_case_name: str) -> dict:
    """Extract key identifying information from ground truth JSON."""
    info = {
        'fund_name': None,
        'has_investment_performance': bool(gt_data.get('investment_performance')),
        'has_key_takeaways': bool(gt_data.get('key_takeaways')),
        'has_business_updates': bool(gt_data.get('business_updates')),
        'sample_content': None,
        'irr_mentions': 0,
        'moic_mentions': 0,
        'dpi_mentions': 0
    }
    
    # Combine all sections for analysis
    all_text = []
    for section in ['investment_performance', 'key_takeaways', 'business_updates']:
        section_data = gt_data.get(section, [])
        if isinstance(section_data, list):
            all_text.extend(section_data)
        elif isinstance(section_data, str):
            all_text.append(section_data)
    
    combined_text = ' '.join(all_text).lower()
    
    # Count metric mentions
    info['irr_mentions'] = len(re.findall(r'\birr\b', combined_text, re.IGNORECASE))
    info['moic_mentions'] = len(re.findall(r'\bmoic\b', combined_text, re.IGNORECASE))
    info['dpi_mentions'] = len(re.findall(r'\bdpi\b', combined_text, re.IGNORECASE))
    
    # Try to extract fund name from content
    # Look for patterns like "Fund IV", "Silver Lake VII", etc.
    fund_patterns = [
        r'\b(Fund\s+(?:IV|VII|VI|III|II|I))\b',
        r'\b((?:Silver\s+Lake|MBX|Everside|Blue\s+Sage|LEONID|Peregrine|Star\s+Mountain|Thoma\s+Bravo|CD&R|Coatue|Freedom|Satori|Boram|Merida|Newfront|Figure\s+AI|SpaceX|Riverside|Kaia)[\s\w]+(?:Fund|Partners?|Capital)?)\b',
    ]
    
    for pattern in fund_patterns:
        matches = re.findall(pattern, combined_text, re.IGNORECASE)
        if matches:
            info['fund_name'] = matches[0].strip()
            break
    
    # If no fund name found, try to infer from test case directory name
    if not info['fund_name']:
        # Extract fund name from test case directory name
        # Remove date prefixes and suffixes
        name_parts = test_case_name.replace('2025_09_Q3_-_', '').replace('2025_11_Q4_-_', '').replace('2025_12_Q4_-_', '')
        name_parts = name_parts.replace('_-_Q3_2025_Quarterly_Update', '').replace('_-_Q3_2025_Report', '')
        name_parts = name_parts.replace('_-_November_2025_Investor_Update', '').replace('_-_December_2025_Investor_Update', '')
        name_parts = name_parts.replace('_-_Q3_2025_Quarterly_Performance_Update_(1)', '')
        name_parts = name_parts.replace('_-_Q3_2025_Investor_Update', '').replace('_-_Q3_2025_LP_Report', '')
        name_parts = name_parts.replace('_-_Acquisition', '').replace('_-_Coinvestment_Opportunity__Pepper', '')
        name_parts = name_parts.replace('_-__Message_to_Merida_Investors', '').replace('_-__November_2025_Update_(1)', '')
        name_parts = name_parts.replace('_-_Q3_2025_Quarterly_Letter', '').replace('_-_Q3_2025_Investment_Report_-_SEGIT_(Grojean)_(1)', '')
        name_parts = name_parts.replace('_-_Q3_2025_Quarterly_Update_(1)', '').replace('_-_November_2025_Exposure_Report', '')
        info['fund_name'] = name_parts.replace('_', ' ').strip()
    
    # Get sample content
    if all_text:
        info['sample_content'] = ' '.join(all_text)[:200].replace('\n', ' ').strip()
    
    return info


def normalize_name(name: str) -> str:
    """Normalize fund name for comparison."""
    if not name:
        return ""
    # Remove common words, normalize case, remove punctuation
    normalized = name.lower()
    normalized = re.sub(r'[^\w\s]', '', normalized)
    normalized = re.sub(r'\b(fund|partners|capital|investments|holdings|group|management|lp|llc|inc)\b', '', normalized)
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    return normalized


def check_match(pdf_info: dict, json_info: dict, pdf_name: str, test_case_name: str) -> tuple[bool, str]:
    """Check if PDF and JSON appear to match. Returns (is_match, reason)."""
    issues = []
    
    # Check if both have investment performance metrics
    pdf_has_metrics = (pdf_info.get('irr_mentions', 0) > 0 or 
                       pdf_info.get('moic_mentions', 0) > 0 or 
                       pdf_info.get('dpi_mentions', 0) > 0)
    json_has_metrics = (json_info.get('irr_mentions', 0) > 0 or 
                        json_info.get('moic_mentions', 0) > 0 or 
                        json_info.get('dpi_mentions', 0) > 0)
    
    if not pdf_has_metrics and json_has_metrics:
        issues.append("PDF doesn't appear to contain investment performance metrics (IRR/MOIC/DPI) but JSON does")
    elif pdf_has_metrics and not json_has_metrics:
        issues.append("PDF contains investment performance metrics but JSON doesn't")
    
    # Check if JSON has expected sections
    if not json_info.get('has_investment_performance'):
        issues.append("Ground truth JSON missing investment_performance section")
    
    # Check fund name similarity if both have names
    pdf_fund = pdf_info.get('fund_name')
    json_fund = json_info.get('fund_name')
    
    if pdf_fund and json_fund:
        pdf_norm = normalize_name(pdf_fund)
        json_norm = normalize_name(json_fund)
        
        # Check if normalized names have significant overlap
        pdf_words = set(pdf_norm.split())
        json_words = set(json_norm.split())
        
        if pdf_words and json_words:
            overlap = len(pdf_words & json_words) / max(len(pdf_words), len(json_words))
            if overlap < 0.3:  # Less than 30% word overlap
                issues.append(f"Fund name mismatch: PDF='{pdf_fund[:50]}' vs JSON='{json_fund[:50]}'")
    
    # Check PDF name vs test case directory name
    pdf_name_normalized = pdf_name.lower().replace('_', ' ').replace('-', ' ')
    test_case_normalized = test_case_name.lower().replace('_', ' ').replace('-', ' ')
    
    # Remove dates and common words
    pdf_name_normalized = re.sub(r'2025\.?\s*(09|11|12)?\s*(q3|q4)?', '', pdf_name_normalized)
    test_case_normalized = re.sub(r'2025\s*_\s*(09|11|12)?\s*_?\s*(q3|q4)?', '', test_case_normalized)
    
    pdf_words = set(pdf_name_normalized.split())
    test_words = set(test_case_normalized.split())
    stop_words = {'update', 'report', 'quarterly', 'performance', 'investor', 'letter', 'pdf', 'the', 'a', 'an'}
    pdf_words = pdf_words - stop_words
    test_words = test_words - stop_words
    
    if pdf_words and test_words:
        overlap = len(pdf_words & test_words) / max(len(pdf_words), len(test_words))
        if overlap < 0.3:  # More lenient threshold
            issues.append(f"PDF filename doesn't match test case directory (overlap: {overlap:.2f})")
    
    if issues:
        return False, "; ".join(issues)
    return True, "Match"


def validate_test_case(test_case_dir: Path) -> dict:
    """Validate a single test case."""
    test_case_name = test_case_dir.name
    
    result = {
        'test_case': test_case_name,
        'has_pdf': False,
        'has_json': False,
        'pdf_count': 0,
        'pdf_name': None,
        'is_match': False,
        'issues': [],
        'pdf_info': {},
        'json_info': {}
    }
    
    # Check for PDF(s)
    pdf_files = list(test_case_dir.glob("*.pdf"))
    result['pdf_count'] = len(pdf_files)
    
    if not pdf_files:
        result['issues'].append("No PDF found")
        return result
    
    result['has_pdf'] = True
    
    # Check for ground truth JSON
    json_file = test_case_dir / "ground_truth.json"
    if not json_file.exists():
        result['issues'].append("No ground_truth.json found")
        return result
    
    result['has_json'] = True
    
    # Load ground truth
    try:
        with open(json_file, 'r') as f:
            gt_data = json.load(f)
        result['json_info'] = extract_key_info_from_json(gt_data, test_case_name)
    except Exception as e:
        result['issues'].append(f"Error loading JSON: {e}")
        return result
    
    # If multiple PDFs, try to match
    if len(pdf_files) > 1:
        # Use similar logic to load_test_case to find best match
        test_case_normalized = test_case_name.lower().replace('_', ' ').replace('-', ' ')
        test_case_normalized = test_case_normalized.replace('2025 ', '').replace('q3 ', '').replace('q4 ', '')
        
        best_match = None
        best_score = 0
        
        for pdf_file in pdf_files:
            pdf_name_normalized = pdf_file.stem.lower().replace('_', ' ').replace('-', ' ')
            pdf_name_normalized = pdf_name_normalized.replace('2025.', '').replace('q3 ', '').replace('q4 ', '')
            
            test_words = set(test_case_normalized.split())
            pdf_words = set(pdf_name_normalized.split())
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
        
        if best_match and best_score > 0.2:
            pdf_path = best_match
        else:
            pdf_path = pdf_files[0]
            result['issues'].append(f"Multiple PDFs found ({len(pdf_files)}), using best match (score: {best_score:.2f})")
    else:
        pdf_path = pdf_files[0]
    
    result['pdf_name'] = pdf_path.name
    
    # Extract info from PDF
    try:
        pdf_text = extract_text_from_pdf(str(pdf_path))
        if not pdf_text:
            result['issues'].append("Could not extract text from PDF")
            return result
        
        result['pdf_info'] = extract_key_info_from_text(pdf_text, pdf_path.name)
    except Exception as e:
        result['issues'].append(f"Error extracting PDF text: {e}")
        return result
    
    # Check if they match
    is_match, reason = check_match(result['pdf_info'], result['json_info'], pdf_path.name, test_case_name)
    result['is_match'] = is_match
    if not is_match:
        result['issues'].append(reason)
    
    return result


def main():
    """Validate all test cases."""
    test_cases_dir = Path("eval_test_cases")
    
    if not test_cases_dir.exists():
        print(f"Error: Test cases directory not found: {test_cases_dir}")
        return
    
    test_case_dirs = [d for d in test_cases_dir.iterdir() if d.is_dir()]
    test_case_dirs.sort()
    
    print("=" * 80)
    print("VALIDATING TEST CASES")
    print("=" * 80)
    print(f"Found {len(test_case_dirs)} test case directories\n")
    
    all_results = []
    mismatches = []
    
    for test_case_dir in test_case_dirs:
        result = validate_test_case(test_case_dir)
        all_results.append(result)
        
        status = "✓" if result['is_match'] and not result['issues'] else "✗"
        print(f"{status} {result['test_case']}")
        
        if result['pdf_name']:
            print(f"    PDF: {result['pdf_name']}")
        
        if result['issues']:
            print(f"    Issues:")
            for issue in result['issues']:
                print(f"      - {issue}")
            mismatches.append(result)
        else:
            pdf_fund = result['pdf_info'].get('fund_name', 'N/A')
            json_fund = result['json_info'].get('fund_name', 'N/A')
            print(f"    PDF Fund: {pdf_fund}")
            print(f"    JSON Fund: {json_fund}")
        
        print()
    
    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total test cases: {len(all_results)}")
    print(f"Valid matches: {len(all_results) - len(mismatches)}")
    print(f"Mismatches/Issues: {len(mismatches)}")
    
    if mismatches:
        print("\n⚠️  TEST CASES WITH ISSUES:")
        print("-" * 80)
        for result in mismatches:
            print(f"\n{result['test_case']}:")
            print(f"  PDF: {result['pdf_name']}")
            for issue in result['issues']:
                print(f"  - {issue}")


if __name__ == "__main__":
    main()
