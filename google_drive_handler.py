import os
import streamlit as st
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import io

SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

def get_google_drive_service():
    """Initialize and return a Google Drive service instance."""
    creds = None
    
    # The file token.json stores the user's access and refresh tokens
    if 'google_drive_token' in st.session_state:
        creds = Credentials.from_authorized_user_info(
            st.session_state.google_drive_token, 
            SCOPES
        )

    # If there are no (valid) credentials available, let the user log in
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        
        # Save the credentials for the next run
        st.session_state.google_drive_token = creds.to_json()

    # Build and return the Drive API service
    return build('drive', 'v3', credentials=creds)

def list_pdf_files(service):
    """List PDF files from Google Drive."""
    results = service.files().list(
        q="mimeType='application/pdf'",
        pageSize=100,
        fields="nextPageToken, files(id, name)"
    ).execute()
    
    return results.get('files', [])

def download_file(service, file_id):
    """Download a file from Google Drive."""
    request = service.files().get_media(fileId=file_id)
    file = io.BytesIO()
    downloader = MediaIoBaseDownload(file, request)
    done = False
    while done is False:
        status, done = downloader.next_chunk()
    file.seek(0)
    return file
