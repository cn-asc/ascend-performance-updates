#!/usr/bin/env python3
"""Test pipeline with just one file."""
import config
from pipeline import UpdatePipeline

print("Testing pipeline with one file...")
try:
    config.validate_config()
    pipeline = UpdatePipeline()
    
    # Get files
    files = pipeline.drive_client.list_files_in_folder(config.PENDING_UPDATES_FOLDER_ID)
    if not files:
        print("No files to process")
        return
    
    # Process first PDF
    pdf_files = [f for f in files if f.get('mimeType') == 'application/pdf']
    if not pdf_files:
        print("No PDF files found")
        return
    
    first_file = pdf_files[0]
    print(f"\nProcessing: {first_file['name']}")
    pipeline._process_file(first_file['id'], first_file['name'])
    print("\n✓ Success!")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
