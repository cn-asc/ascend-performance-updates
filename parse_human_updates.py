#!/usr/bin/env python3
"""Parse human-written updates document and convert to JSON test cases."""
import json
import re
from pathlib import Path
import subprocess

def extract_text_from_docx(docx_path):
    """Extract text from docx file using textutil (macOS)."""
    try:
        result = subprocess.run(
            ['textutil', '-convert', 'txt', '-stdout', str(docx_path)],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error extracting text: {e}")
        return None

def parse_updates_document(text):
    """Parse the document text and extract individual fund updates."""
    updates = []
    
    lines = text.split('\n')
    current_update = None
    current_section = None
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        if not line:
            i += 1
            continue
        
        # Check if this is a fund header (contains "Update [" pattern)
        # Examples: "MBX I Update [Outperforming]", "Thoma Bravo XV Update [As Expected]"
        if 'Update [' in line or 'Update [' in line:
            # Save previous update if exists
            if current_update and (current_update.get('investment_performance') or 
                                   current_update.get('key_takeaways') or 
                                   current_update.get('business_updates')):
                updates.append(current_update)
            
            # Extract fund name and performance
            # Pattern: "Fund Name Update [Performance]"
            if 'Update [' in line:
                parts = line.split('Update [')
                fund_name = parts[0].strip()
                performance = parts[1].rstrip(']').strip() if len(parts) > 1 else None
            else:
                fund_name = line.split('Update')[0].strip()
                performance = None
            
            current_update = {
                'fund_name': fund_name,
                'performance_summary': performance or 'Unknown',
                'investment_performance': [],
                'key_takeaways': [],
                'business_updates': []
            }
            current_section = None
            i += 1
            continue
        
        # Check for section headers (various formats)
        # "Fund Performance", "Fund / Company Performance", "Company Performance"
        if 'Fund Performance' in line or 'Company Performance' in line or line == 'Fund /Performance':
            current_section = 'investment_performance'
            i += 1
            continue
        
        # "Takeaways / Action Items" or "Takeaways / Action Items:"
        if 'Takeaways' in line and 'Action Items' in line:
            current_section = 'key_takeaways'
            i += 1
            continue
        
        # "Business Updates / Market Commentary" or "Business Updates / Market Commentary:"
        if 'Business Updates' in line and 'Market Commentary' in line:
            current_section = 'business_updates'
            i += 1
            continue
        
        # Skip "Overview" section
        if line == 'Overview' or line.startswith('Overview'):
            current_section = None
            i += 1
            continue
        
        # Skip template text
        if 'TEMPLATE' in line or 'AB/JD, please put' in line:
            i += 1
            continue
        
        # If we're in a section and have a current update, add bullet points
        if current_update and current_section:
            # Check if line is a bullet point (starts with • or - or is indented with tab)
            if line.startswith('•') or line.startswith('-') or line.startswith('\t'):
                content = line.lstrip('•-\t ').strip()
                if content and len(content) > 5:  # Filter out very short lines
                    current_update[current_section].append(content)
        
        i += 1
    
    # Don't forget the last update
    if current_update and (current_update.get('investment_performance') or 
                           current_update.get('key_takeaways') or 
                           current_update.get('business_updates')):
        updates.append(current_update)
    
    return updates

def match_pdfs_to_updates(updates, pdf_dir):
    """Match PDF files to updates based on fund name."""
    pdf_files = list(Path(pdf_dir).glob('*.pdf'))
    
    matched = []
    unmatched_pdfs = []
    used_fund_names = set()
    
    # Create a mapping of fund names to updates
    fund_name_map = {}
    for update in updates:
        fund_name = update['fund_name'].lower()
        # Normalize fund name variations
        fund_name_map[fund_name] = update
        
        # Also create variations (e.g., "MBX I" -> "MBX Capital I", "MBX I")
        if 'mbx' in fund_name:
            if 'mbx i' in fund_name or fund_name == 'mbx i':
                fund_name_map['mbx capital i'] = update
                fund_name_map['mbx i (frx select)'] = update
            elif 'mbx ii' in fund_name or fund_name == 'mbx ii':
                fund_name_map['mbx capital ii'] = update
            elif 'mbx iii' in fund_name or fund_name == 'mbx iii':
                fund_name_map['mbx capital iii'] = update
        
        # Handle SpaceX variations
        if 'spacex' in fund_name:
            fund_name_map['spacex (zanbato partners fund)'] = update
            fund_name_map['spacex'] = update
        
        # Handle Interplay variations
        if 'interplay' in fund_name:
            fund_name_map['interplay'] = update
            fund_name_map['interplay ventures'] = update
        
        # Handle Newfront
        if 'newfront' in fund_name:
            fund_name_map['newfront'] = update
        
        # Handle Riverside/BWE variations
        if 'riverside' in fund_name or 'vargas' in fund_name:
            fund_name_map['riverside vargas (bwe)'] = update
            fund_name_map['box wilson riverside vargas'] = update
        
        # Handle Everside variations
        if 'everside' in fund_name:
            fund_name_map['everside fund iv'] = update
            fund_name_map['everside capital fund iv'] = update
    
    for pdf_file in pdf_files:
        pdf_name = pdf_file.stem.lower()
        matched_update = None
        matched_fund_name = None
        best_match_score = 0
        
        # Try to match by fund name
        for fund_name, update in fund_name_map.items():
            if fund_name in used_fund_names:
                continue
            
            # Extract key words from fund name
            fund_words = set(re.findall(r'\b\w+\b', fund_name))
            pdf_words = set(re.findall(r'\b\w+\b', pdf_name))
            
            # Check for significant overlap
            common_words = fund_words.intersection(pdf_words)
            # Remove common stop words
            stop_words = {'q3', 'q4', '2025', 'update', 'report', 'letter', 'investor', 'quarterly', 'performance', 'the', 'a', 'an', 'and', 'or', 'of', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'lp', 'fund', 'capital'}
            common_words = common_words - stop_words
            
            score = len(common_words)
            
            # Special handling for specific funds
            if 'mbx' in pdf_name and 'mbx' in fund_name:
                if 'i' in pdf_name and ('i' in fund_name or '1' in fund_name):
                    score += 5
                elif 'ii' in pdf_name and ('ii' in fund_name or '2' in fund_name):
                    score += 5
                elif 'iii' in pdf_name and ('iii' in fund_name or '3' in fund_name):
                    score += 5
            
            # Special handling for SpaceX
            if 'spacex' in pdf_name and 'spacex' in fund_name:
                score += 10
            
            # Special handling for Interplay
            if 'interplay' in pdf_name and 'interplay' in fund_name:
                score += 10
            
            # Special handling for Newfront
            if 'newfront' in pdf_name and 'newfront' in fund_name:
                score += 10
            
            # Lower threshold for single-word matches (like SpaceX, Newfront)
            threshold = 1 if len(fund_words - stop_words) <= 2 else 2
            
            if score > best_match_score and score >= threshold:
                best_match_score = score
                matched_update = update
                matched_fund_name = fund_name
        
        if matched_update:
            matched.append({
                'pdf': pdf_file.name,
                'update': matched_update
            })
            used_fund_names.add(matched_fund_name)
        else:
            unmatched_pdfs.append(pdf_file.name)
    
    return matched, unmatched_pdfs, updates

def main():
    """Main function to parse document and create test cases."""
    human_doc_path = Path('eval_test_cases/test_case_human/2026 Monthly Investment Update [Running].docx')
    pdf_dir = Path('eval_test_cases/test_case_pdf')
    
    print("📄 Extracting text from document...")
    text = extract_text_from_docx(human_doc_path)
    
    if not text:
        print("❌ Failed to extract text from document")
        return
    
    print(f"✅ Extracted {len(text)} characters")
    print("\n📝 Parsing updates...")
    
    updates = parse_updates_document(text)
    print(f"✅ Found {len(updates)} updates")
    
    print("\n🔗 Matching PDFs to updates...")
    matched, unmatched_pdfs, all_updates = match_pdfs_to_updates(updates, pdf_dir)
    
    print(f"✅ Matched {len(matched)} PDFs")
    if unmatched_pdfs:
        print(f"⚠️  {len(unmatched_pdfs)} PDFs could not be matched:")
        for pdf in unmatched_pdfs[:10]:
            print(f"   - {pdf}")
    
    # Create test case structure
    print("\n📦 Creating test case structure...")
    
    for match in matched:
        pdf_name = match['pdf']
        update = match['update']
        
        # Create test case directory
        # Use PDF name as test case name (sanitized)
        test_case_name = Path(pdf_name).stem.replace(' ', '_').replace('.', '_')
        test_case_dir = Path('eval_test_cases') / test_case_name
        test_case_dir.mkdir(exist_ok=True)
        
        # Copy PDF
        import shutil
        pdf_source = pdf_dir / pdf_name
        pdf_dest = test_case_dir / pdf_name
        shutil.copy2(pdf_source, pdf_dest)
        
        # Create ground_truth.json
        ground_truth = {
            'investment_performance': update.get('investment_performance', []),
            'key_takeaways': update.get('key_takeaways', []),
            'business_updates': update.get('business_updates', [])
        }
        
        ground_truth_path = test_case_dir / 'ground_truth.json'
        with open(ground_truth_path, 'w') as f:
            json.dump(ground_truth, f, indent=2)
        
        print(f"✅ Created test case: {test_case_name}")
    
    print(f"\n✅ Created {len(matched)} test cases")
    print(f"\n📋 Summary:")
    print(f"   Total updates found: {len(all_updates)}")
    print(f"   PDFs matched: {len(matched)}")
    print(f"   PDFs unmatched: {len(unmatched_pdfs)}")

if __name__ == '__main__':
    main()
