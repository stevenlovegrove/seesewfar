from __future__ import print_function
import os
import os.path
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
import glob

# If modifying these scopes, delete the file token.json.
SCOPES = ['https://www.googleapis.com/auth/drive']

def main(google_folder_id, to_dir):
    creds = None
    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open('token.json', 'w') as token:
            token.write(creds.to_json())

    service = build('drive', 'v3', credentials=creds)

    page_token = None
    items = []

    while True:
        # Call the Drive v3 API
        results = service.files().list(
            # q="mimeType='application/vnd.google-apps.folder'",
            q="'" + google_folder_id + "' in parents",
            spaces='drive',
            fields='nextPageToken, files(id, name)',
            pageToken=page_token
        ).execute()
        items.extend(results.get('files', []))
        page_token = results.get('nextPageToken', None)
        if page_token is None:
            break

    # Get current files in folder to_dir
    os.chdir(to_dir)
    existing_files = set(glob.glob("*.ARW"))

    # Download all the files in items
    for item in items:
        if item['name'] in existing_files:
            continue

        print(u'{0} ({1})'.format(item['name'], item['id']))
        request = service.files().get_media(fileId=item['id'])
        fh = open(item['name'], "wb")
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()
            print("Download %d%%." % int(status.progress() * 100))



if __name__ == '__main__':
    to_dir = './data/'
    google_folder_id = '1CEyzj75em4uBtMp970Lw0yezgUt7Mart'
    main(google_folder_id, to_dir)