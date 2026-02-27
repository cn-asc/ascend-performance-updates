#!/usr/bin/env python3
"""Helper script to create a test case from a PDF and ground truth."""
import json
import sys
from pathlib import Path


def create_test_case(test_case_name: str, pdf_path: str, ground_truth_path: str = None):
    """
    Create a test case directory structure.
    
    Args:
        test_case_name: Name for the test case
        pdf_path: Path to the PDF file
        ground_truth_path: Optional path to existing ground truth JSON
    """
    test_cases_dir = Path("eval_test_cases")
    test_case_dir = test_cases_dir / test_case_name
    
    # Create directory
    test_case_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy PDF
    pdf_source = Path(pdf_path)
    if not pdf_source.exists():
        print(f"Error: PDF not found: {pdf_path}")
        return False
    
    pdf_dest = test_case_dir / pdf_source.name
    import shutil
    shutil.copy2(pdf_source, pdf_dest)
    print(f"✅ Copied PDF to {pdf_dest}")
    
    # Create or copy ground truth
    ground_truth_file = test_case_dir / "ground_truth.json"
    
    if ground_truth_path:
        gt_source = Path(ground_truth_path)
        if gt_source.exists():
            shutil.copy2(gt_source, ground_truth_file)
            print(f"✅ Copied ground truth to {ground_truth_file}")
        else:
            print(f"Warning: Ground truth file not found: {ground_truth_path}")
            create_template_ground_truth(ground_truth_file)
    else:
        create_template_ground_truth(ground_truth_file)
    
    print(f"\n✅ Test case created: {test_case_dir}")
    print(f"\nNext steps:")
    print(f"1. Edit {ground_truth_file} with the expected output sections")
    print(f"2. Run evaluation: python3 eval_harness.py {test_case_name}")
    
    return True


def create_template_ground_truth(output_path: Path):
    """Create a template ground truth JSON file."""
    template = {
        "investment_performance": [
            "Net IRR: X.X% (vs benchmark X.X% - Category)",
            "Net MOIC: X.Xx (vs benchmark X.Xx - Category)",
            "Net DPI: X.Xx (vs benchmark X.Xx - Category)",
            "Additional performance detail 1",
            "Additional performance detail 2"
        ],
        "key_takeaways": [
            "Key insight 1",
            "Key insight 2",
            "Key insight 3"
        ],
        "business_updates": [
            "Portfolio company update 1",
            "Market commentary 1",
            "Business development 1"
        ]
    }
    
    with open(output_path, 'w') as f:
        json.dump(template, f, indent=2)
    
    print(f"✅ Created template ground truth at {output_path}")


def main():
    """Main entry point."""
    if len(sys.argv) < 3:
        print("Usage: python3 create_test_case.py <test_case_name> <pdf_path> [ground_truth_path]")
        print("\nExample:")
        print("  python3 create_test_case.py test_case_1 ./document.pdf")
        print("  python3 create_test_case.py test_case_1 ./document.pdf ./ground_truth.json")
        sys.exit(1)
    
    test_case_name = sys.argv[1]
    pdf_path = sys.argv[2]
    ground_truth_path = sys.argv[3] if len(sys.argv) > 3 else None
    
    create_test_case(test_case_name, pdf_path, ground_truth_path)


if __name__ == "__main__":
    main()
