import os
import pickle
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

SCOPES = ['https://www.googleapis.com/auth/youtube.upload']

def get_authenticated_service(channel_name):
    credentials = None
    
    # Dynamically name the pickle file based on the target channel
    pickle_file = f"{channel_name.lower().replace(' ', '_')}_token.pickle"
    client_secrets = 'client_secret.json'

    # 1. Check if the specific channel token exists
    if os.path.exists(pickle_file):
        with open(pickle_file, 'rb') as token:
            credentials = pickle.load(token)
            print(f"Loaded existing session for: {channel_name}")

    # 2. Authenticate or refresh if needed
    if not credentials or not credentials.valid:
        if credentials and credentials.expired and credentials.refresh_token:
            print(f"Refreshing expired access token for {channel_name}...")
            credentials.refresh(Request())
        else:
            if not os.path.exists(client_secrets):
                raise FileNotFoundError(f"Missing '{client_secrets}' in this directory.")
            
            # This triggers a new browser window for authorization
            print(f"\n--- AUTHENTICATION REQUIRED FOR: {channel_name} ---")
            print("Select the correct Google Account and choose the matching YouTube Channel profile.")
            
            flow = InstalledAppFlow.from_client_secrets_file(client_secrets, SCOPES)
            credentials = flow.run_local_server(port=0)

        # 3. Save to a distinct pickle file so it doesn't conflict with other channels
        with open(pickle_file, 'wb') as token:
            pickle.dump(credentials, token)
            print(f"Session saved successfully to {pickle_file}.\n")

    return build('youtube', 'v3', credentials=credentials)

def upload_video(youtube, video_path, title, description, tags, category_id="22", privacy="public"):
    # (Keep your existing upload_video logic exactly the same here)
    pass

if __name__ == '__main__':
    
    # ==========================================
    # CHOOSE YOUR TARGET CHANNEL HERE
    # OPTIONS: "Warrior Ethos" or "MindScribble"
    # ==========================================
    TARGET_CHANNEL = "MindScribble" 
    
    # Get the isolated service profile
    youtube_service = get_authenticated_service(TARGET_CHANNEL)

    # # Set up metadata for the video
    # video_to_upload = "mind_scribble_output.mp4" 
    # video_title = "[curious] Ever wonder why your brain completely forgets... #shorts"
    # video_desc = "Exploring the Zeigarnik Effect with sketchy visuals. #psychology #shorts"
    # video_tags = ["psychology", "mind tricks", "facts", "shorts"]

    # # Execute upload
    # upload_video(
    #     youtube=youtube_service,
    #     video_path=video_to_upload,
    #     title=video_title,
    #     description=video_desc,
    #     tags=video_tags,
    #     privacy="public"
    # )