"""Google Drive API client for monitoring and managing files."""
import os
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import config

# If modifying these scopes, delete the file token.json.
SCOPES = [
    'https://www.googleapis.com/auth/drive',
    'https://www.googleapis.com/auth/documents',
]


class GoogleDriveClient:
    """Client for interacting with Google Drive API."""
    
    def __init__(self):
        self.service = None
        self.docs_service = None
        self.creds = self._authenticate()
        self.service = build('drive', 'v3', credentials=self.creds)
        self.docs_service = build('docs', 'v1', credentials=self.creds)
    
    def _authenticate(self):
        """
        Authenticate and return credentials.
        Uses OAuth refresh token from environment variables (like InvestmentProcessor),
        or falls back to local OAuth flow for development.
        """
        # Try OAuth refresh token from environment (for Cloud Run)
        CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID")
        CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET")
        REFRESH_TOKEN = os.environ.get("GOOGLE_REFRESH_TOKEN")
        TOKEN_URI = os.environ.get("GOOGLE_TOKEN_URI", "https://oauth2.googleapis.com/token")
        
        if CLIENT_ID and CLIENT_SECRET and REFRESH_TOKEN:
            # Use refresh token from environment (Cloud Run)
            creds = Credentials(
                None,
                refresh_token=REFRESH_TOKEN,
                token_uri=TOKEN_URI,
                client_id=CLIENT_ID,
                client_secret=CLIENT_SECRET,
                scopes=SCOPES,
            )
            creds.refresh(Request())
            return creds
        
        # Fall back to local OAuth flow (for local development)
        creds = None
        
        # The file token.json stores the user's access and refresh tokens.
        if os.path.exists(config.GOOGLE_TOKEN_FILE):
            try:
                creds = Credentials.from_authorized_user_file(
                    config.GOOGLE_TOKEN_FILE, SCOPES
                )
            except Exception as e:
                print(f"Error loading token.json: {e}")
        
        # If there are no (valid) credentials available, let the user log in.
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                try:
                    creds.refresh(Request())
                    print("Refreshed OAuth token")
                except Exception as e:
                    print(f"Error refreshing token: {e}")
                    creds = None
            
            if not creds or not creds.valid:
                if not os.path.exists(config.GOOGLE_CREDENTIALS_FILE):
                    raise FileNotFoundError(
                        f"Credentials file not found: {config.GOOGLE_CREDENTIALS_FILE}\n"
                        "Please download credentials.json from Google Cloud Console."
                    )
                flow = InstalledAppFlow.from_client_secrets_file(
                    config.GOOGLE_CREDENTIALS_FILE, SCOPES
                )
                creds = flow.run_local_server(port=0)
            
            # Save the credentials for the next run
            with open(config.GOOGLE_TOKEN_FILE, 'w') as token:
                token.write(creds.to_json())
        
        return creds
    
    def list_files_in_folder(self, folder_id):
        """List all files in a specific folder."""
        try:
            query = f"'{folder_id}' in parents and trashed=false"
            results = self.service.files().list(
                q=query,
                fields="files(id, name, mimeType, modifiedTime)",
                supportsAllDrives=True,
                includeItemsFromAllDrives=True
            ).execute()
            return results.get('files', [])
        except HttpError as error:
            print(f"An error occurred: {error}")
            return []
    
    def download_file(self, file_id, file_name, destination_path):
        """Download a file from Google Drive."""
        try:
            request = self.service.files().get_media(fileId=file_id)
            file_content = request.execute()
            
            with open(destination_path, 'wb') as f:
                f.write(file_content)
            
            return True
        except HttpError as error:
            print(f"An error occurred downloading file {file_name}: {error}")
            return False
    
    def move_file(self, file_id, new_parent_folder_id):
        """Move a file to a new folder."""
        try:
            # Get the current parents
            file = self.service.files().get(
                fileId=file_id,
                fields='parents',
                supportsAllDrives=True
            ).execute()
            
            previous_parents = ",".join(file.get('parents'))
            
            # Move the file to the new parent
            self.service.files().update(
                fileId=file_id,
                addParents=new_parent_folder_id,
                removeParents=previous_parents,
                fields='id, parents',
                supportsAllDrives=True
            ).execute()
            
            return True
        except HttpError as error:
            print(f"An error occurred moving file: {error}")
            return False
    
    def append_to_document(self, document_id, text):
        """Append text to a Google Docs document with bold formatting for headers and proper bullet lists."""
        try:
            # Get the current document to find the end index
            doc = self.docs_service.documents().get(documentId=document_id).execute()
            start_index = doc['body']['content'][-1]['endIndex'] - 1
            
            # Parse the text into structured sections
            lines = text.split('\n')
            requests = []
            current_index = start_index
            
            i = 0
            while i < len(lines):
                line = lines[i]
                line_length = len(line)
                
                # Check if this is a header line (title or section header)
                is_title = 'Update -' in line and not line.startswith('  ')
                # Remove markdown bold syntax for detection
                line_for_detection = line.replace('**', '').strip()
                is_section_header = (
                    line_for_detection.endswith(':') and 
                    not line.startswith('  ') and
                    (line_for_detection.startswith('Quantitative Performance:') or 
                     line_for_detection.startswith('Key Takeaways and Business Updates:') or 
                     line_for_detection.startswith('Market Commentary:') or
                     line_for_detection.startswith('Investment Performance:') or 
                     line_for_detection.startswith('Key Takeaways:') or 
                     line_for_detection.startswith('Business Updates/Market Commentary:') or
                     line_for_detection.startswith('Performance:'))
                )
                
                if is_title or is_section_header:
                    # Remove markdown bold syntax (**) from the text before inserting
                    clean_line = line.replace('**', '')
                    clean_line_length = len(clean_line)
                    
                    # Insert header text (without markdown syntax)
                    requests.append({
                        'insertText': {
                            'location': {'index': current_index},
                            'text': clean_line + '\n'
                        }
                    })
                    
                    # Apply bold formatting to header
                    requests.append({
                        'updateTextStyle': {
                            'range': {
                                'startIndex': current_index,
                                'endIndex': current_index + clean_line_length
                            },
                            'textStyle': {'bold': True},
                            'fields': 'bold'
                        }
                    })
                    
                    current_index += clean_line_length + 1
                    i += 1
                    
                    # Collect bullet points for this section
                    bullet_points = []
                    
                    while i < len(lines) and (lines[i].startswith('  • ') or lines[i].strip() == ''):
                        if lines[i].startswith('  • '):
                            # Remove the bullet character and leading spaces, keep the text
                            bullet_text = lines[i].replace('  • ', '', 1).strip()
                            if bullet_text:
                                bullet_points.append(bullet_text)
                        elif lines[i].strip() == '':
                            # Empty line - end of bullet list
                            break
                        i += 1
                    
                    # Insert bullet points as proper formatted list
                    if bullet_points:
                        bullet_start_index = current_index
                        
                        # Insert each bullet point as a separate paragraph
                        for bullet_text in bullet_points:
                            requests.append({
                                'insertText': {
                                    'location': {'index': current_index},
                                    'text': bullet_text + '\n'
                                }
                            })
                            current_index += len(bullet_text) + 1
                        
                        # Create bullet list for all bullet points
                        # The end index should be before the final newline
                        bullet_end_index = current_index - 1
                        requests.append({
                            'createParagraphBullets': {
                                'range': {
                                    'startIndex': bullet_start_index,
                                    'endIndex': bullet_end_index
                                },
                                'bulletPreset': 'BULLET_DISC_CIRCLE_SQUARE'
                            }
                        })
                    
                    # Skip empty line after bullets if present
                    if i < len(lines) and lines[i].strip() == '':
                        i += 1
                else:
                    # Regular line - just insert it
                    if line.strip():  # Only insert non-empty lines
                        requests.append({
                            'insertText': {
                                'location': {'index': current_index},
                                'text': line + '\n'
                            }
                        })
                        current_index += line_length + 1
                    i += 1
            
            # Add final spacing
            requests.append({
                'insertText': {
                    'location': {'index': current_index},
                    'text': '\n'
                }
            })
            
            # Execute all requests
            self.docs_service.documents().batchUpdate(
                documentId=document_id,
                body={'requests': requests}
            ).execute()
            
            return True
        except HttpError as error:
            print(f"An error occurred appending to document: {error}")
            import traceback
            traceback.print_exc()
            return False
