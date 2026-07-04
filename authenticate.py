from google_auth_oauthlib.flow import InstalledAppFlow
import json

# Point this to the JSON file you just downloaded
CLIENT_SECRETS_FILE = 'coldcases_client_secret.json' 
SCOPES = ['https://www.googleapis.com/auth/youtube.upload']

flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRETS_FILE, SCOPES)
# This will open a browser for you to log in to your @ColdCases-pov account
credentials = flow.run_local_server(port=0)

# Save the resulting credentials to a new file
with open('coldcases_auth.json', 'w') as f:
    f.write(credentials.to_json())