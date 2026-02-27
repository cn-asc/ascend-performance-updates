"""Main pipeline orchestrator for processing investment updates."""
import os
import time
import tempfile
from datetime import datetime
from google_drive_client import GoogleDriveClient
from pdf_processor import extract_text_from_pdf
from analysis_agent import AnalysisAgent
import config


class UpdatePipeline:
    """Main pipeline for processing investment updates."""
    
    def __init__(self):
        self.drive_client = GoogleDriveClient()
        self.analysis_agent = AnalysisAgent()
        self.processed_files = set()  # Track processed files to avoid duplicates
    
    def process_pending_updates(self):
        """Process all files in the Pending Updates folder."""
        print(f"[{datetime.now()}] Checking for pending updates...")
        
        files = self.drive_client.list_files_in_folder(
            config.PENDING_UPDATES_FOLDER_ID
        )
        
        if not files:
            print("No files found in Pending Updates folder.")
            return
        
        print(f"Found {len(files)} file(s) to process.")
        
        for file_info in files:
            file_id = file_info['id']
            file_name = file_info['name']
            mime_type = file_info.get('mimeType', '')
            
            # Skip if already processed
            if file_id in self.processed_files:
                continue
            
            # Only process PDFs
            if mime_type != 'application/pdf':
                print(f"Skipping {file_name} - not a PDF (type: {mime_type})")
                continue
            
            print(f"\nProcessing: {file_name}")
            
            try:
                self._process_file(file_id, file_name)
                self.processed_files.add(file_id)
            except Exception as e:
                print(f"Error processing {file_name}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    def _process_file(self, file_id, file_name):
        """Process a single file through the pipeline."""
        # Step 1: Download the file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            print(f"  Step 1: Downloading {file_name}...")
            if not self.drive_client.download_file(file_id, file_name, tmp_path):
                raise Exception("Failed to download file")
            
            # Step 2: Extract text from PDF
            print(f"  Step 2: Extracting text from PDF...")
            text = extract_text_from_pdf(tmp_path)
            if not text:
                raise Exception("Failed to extract text from PDF")
            
            print(f"  Extracted {len(text)} characters of text")
            
            # Step 3: Analyze with AI agents (extraction + formatting)
            print(f"  Step 3: Extracting information with AI agent...")
            print(f"  Step 4: Formatting update with AI agent...")
            written_update, metadata, metrics = self.analysis_agent.analyze_update(text, file_name)
            
            # Log metadata for debugging
            print(f"  Extracted metadata:")
            print(f"    Fund Name: {metadata['fund_name']}")
            print(f"    Asset Class: {metadata['asset_class']}")
            print(f"    Vintage: {metadata['vintage'] or 'Not found'}")
            print(f"    Performance Summary: {metadata['performance_summary']}")
            
            # Add timestamp and separator
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            formatted_update = f"\n{'='*80}\n[{timestamp}] {file_name}\n{'='*80}\n\n{written_update}\n"
            
            # Step 5: Append to Written Updates document
            print(f"  Step 5: Appending to Written Updates document...")
            self.drive_client.append_to_document(
                config.WRITTEN_UPDATES_DOCUMENT_ID,
                formatted_update
            )
            
            # Step 6: Move file to Done Updates folder
            print(f"  Step 6: Moving file to Done Updates folder...")
            self.drive_client.move_file(file_id, config.DONE_UPDATES_FOLDER_ID)
            
            print(f"  ✓ Successfully processed {file_name}")
            
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def run_continuous(self):
        """Run the pipeline continuously, checking for updates periodically."""
        print("Starting continuous monitoring...")
        print(f"Checking every {config.POLL_INTERVAL_SECONDS} seconds")
        print("Press Ctrl+C to stop\n")
        
        try:
            while True:
                self.process_pending_updates()
                time.sleep(config.POLL_INTERVAL_SECONDS)
        except KeyboardInterrupt:
            print("\n\nPipeline stopped by user.")
    
    def run_once(self):
        """Run the pipeline once and exit."""
        print("Running pipeline once...\n")
        self.process_pending_updates()
        print("\nPipeline run complete.")


def main():
    """Main entry point."""
    print("Starting pipeline...", flush=True)
    try:
        print("Validating config...", flush=True)
        config.validate_config()
        print("Config validated successfully", flush=True)
    except ValueError as e:
        print(f"Configuration error: {e}", flush=True)
        print("\nPlease check your .env file and ensure all required values are set.", flush=True)
        raise
    
    print("Initializing pipeline...", flush=True)
    pipeline = UpdatePipeline()
    
    # Run once by default, or continuously if POLL_INTERVAL_SECONDS > 0
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--continuous':
        pipeline.run_continuous()
    else:
        pipeline.run_once()


if __name__ == '__main__':
    main()
