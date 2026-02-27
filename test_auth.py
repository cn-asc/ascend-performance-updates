#!/usr/bin/env python3
"""Test Google Drive authentication."""
import config
from google_drive_client import GoogleDriveClient

print("Testing Google Drive authentication...")
try:
    config.validate_config()
    print("✓ Config validated")
    
    client = GoogleDriveClient()
    print("✓ Authentication successful")
    
    # Test listing files
    files = client.list_files_in_folder(config.PENDING_UPDATES_FOLDER_ID)
    print(f"✓ Found {len(files)} file(s) in Pending Updates folder")
    
    if files:
        print("\nFiles found:")
        for f in files:
            print(f"  - {f['name']} ({f.get('mimeType', 'unknown type')})")
    else:
        print("\nNo files found in Pending Updates folder.")
        
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
