"""Configuration management for the Google Drive automation pipeline."""
import os
from dotenv import load_dotenv

# Only load .env file if it exists (for local development)
# In Cloud Run, all config comes from environment variables
if os.path.exists('.env'):
    load_dotenv()

# Google Drive API
GOOGLE_CREDENTIALS_FILE = os.getenv('GOOGLE_CREDENTIALS_FILE', 'credentials.json')
GOOGLE_TOKEN_FILE = os.getenv('GOOGLE_TOKEN_FILE', 'token.json')

# Folder IDs
PENDING_UPDATES_FOLDER_ID = os.getenv('PENDING_UPDATES_FOLDER_ID')
DONE_UPDATES_FOLDER_ID = os.getenv('DONE_UPDATES_FOLDER_ID')
WRITTEN_UPDATES_DOCUMENT_ID = os.getenv('WRITTEN_UPDATES_DOCUMENT_ID')

# OpenAI Configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-4')  # Default/legacy, prefer EXTRACTION_MODEL and FORMATTING_MODEL
OPENAI_TEMPERATURE = float(os.getenv('OPENAI_TEMPERATURE', '0.7'))

# Agent-specific model configuration
EXTRACTION_MODEL = os.getenv('EXTRACTION_MODEL', 'gpt-5.2')
FORMATTING_MODEL = os.getenv('FORMATTING_MODEL', 'gpt-5-mini')

# Pipeline Configuration
POLL_INTERVAL_SECONDS = int(os.getenv('POLL_INTERVAL_SECONDS', '60'))  # Check every minute

def validate_config():
    """Validate that all required configuration is present."""
    missing = []
    
    if not OPENAI_API_KEY:
        missing.append('OPENAI_API_KEY')
    if not PENDING_UPDATES_FOLDER_ID:
        missing.append('PENDING_UPDATES_FOLDER_ID')
    if not DONE_UPDATES_FOLDER_ID:
        missing.append('DONE_UPDATES_FOLDER_ID')
    if not WRITTEN_UPDATES_DOCUMENT_ID:
        missing.append('WRITTEN_UPDATES_DOCUMENT_ID')
    
    # In Cloud Run, we use Application Default Credentials, so credentials.json is not required
    # Only check for it in local development
    if not os.getenv('K_SERVICE') and not os.path.exists(GOOGLE_CREDENTIALS_FILE):
        missing.append(f'Google credentials file: {GOOGLE_CREDENTIALS_FILE} (required for local development)')
    
    if missing:
        raise ValueError(f"Missing required configuration: {', '.join(missing)}")
    
    return True
