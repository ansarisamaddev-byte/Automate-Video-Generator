import os
import glob
import random
import pickle
import pandas as pd

# ================= CLOUDINARY =================
import cloudinary
import cloudinary.uploader

cloudinary.config(
    cloud_name="dusdbgfey",
    api_key="545263495647551",
    api_secret="KFRuIRsx-LkevEBul4YvfYBWfiY"
)

# ================= YOUTUBE =================
import googleapiclient.discovery
from google.auth.transport.requests import Request
from googleapiclient.http import MediaFileUpload
from google_auth_oauthlib.flow import InstalledAppFlow

# ================= VIDEO GENERATOR =================
from video_generator import generate_reel

SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload"
]

def get_service():
    creds = None
    pickle_file = "mindscribble_token.pickle"
    client_secrets = "client_secret.json"

    # Load existing authentication session for MindScribble
    if os.path.exists(pickle_file):
        with open(pickle_file, "rb") as f:
            creds = pickle.load(f)

    # Refresh the token dynamically if it's expired
    if creds and creds.expired and creds.refresh_token:
        print("🔄 Refreshing MindScribble access token...")
        creds.refresh(Request())

    # First-time setup: Create the distinct pickle token file using client_secret.json
    if not creds or not creds.valid:
        if not os.path.exists(client_secrets):
            raise FileNotFoundError(f"❌ Missing '{client_secrets}' in this folder. Add it to authenticate MindScribble.")
        
        print("\n--- AUTHENTICATION REQUIRED FOR: MindScribble ---")
        print("Opening browser. Select your Google account and explicitly pick the MindScribble channel.")
        flow = InstalledAppFlow.from_client_secrets_file(client_secrets, SCOPES)
        creds = flow.run_local_server(port=0)

        with open(pickle_file, "wb") as f:
            pickle.dump(creds, f)
            print(f"✅ Session saved successfully to {pickle_file}.")

    return googleapiclient.discovery.build(
        "youtube", "v3", credentials=creds
    )

def upload_to_youtube(video_path, title, description, tags):
    try:
        print("📤 Uploading to MindScribble YouTube Channel...")
        youtube = get_service()

        request = youtube.videos().insert(
            part="snippet,status",
            body={
                "snippet": {
                    "title": title,
                    "description": description,
                    "tags": tags,
                    "categoryId": "22" # People & Blogs
                },
                "status": {
                    "privacyStatus": "public"
                }
            },
            media_body=MediaFileUpload(video_path)
        )

        response = request.execute()
        print(f"✅ Uploaded: https://www.youtube.com/watch?v={response['id']}")
        return True

    except Exception as e:
        print(f"❌ YouTube Upload Error: {e}")
        return False

# ================= MAIN AUTOMATION LOGIC =================

def run_automation():
    csv_file = "mind_scribble.csv"

    if not os.path.exists(csv_file):
        print("❌ CSV not found")
        return

    # Load data and standardize the 'posted' column 
    df = pd.read_csv(csv_file)
    df["posted"] = df["posted"].astype(str).str.lower()

    # Find rows that haven't been processed yet
    unposted_df = df[df["posted"] == "false"]

    if unposted_df.empty:
        print("✅ All MindScribble shorts have been posted!")
        return

    # Get the index of the first unposted row
    current_index = unposted_df.index[0]
    row = df.loc[current_index]

    # --- THE IMAGE INDEX FIX ---
    if current_index > 0:
        start_idx = int(df.loc[current_index - 1, "last_image_index"])
    else:
        start_idx = int(row["last_image_index"])

    print(f"🧠 Processing ID {row.get('id', current_index)} | Starting at sketch image index: {start_idx}")

    # --- ASSET SELECTION ---
    bg_music_files = glob.glob("background_music/*.mp3")
    bg_music = random.choice(bg_music_files) if bg_music_files else None

    ending_assets = glob.glob("ending/mindscribble/*.mp4")
    if not ending_assets:
        print("❌ Error: No ending/outro videos found.")
        return
    selected_ending = random.choice(ending_assets)

    output_video = f"mindscribble_output_{current_index}.mp4"

    # --- GENERATION ---
    print("🎬 Generating pencil scribble video content...")
    try:
        result = generate_reel(
            audio_path=row["audio_path"],
            image_folder=row["image_folder"],
            music_path=bg_music,
            credit_video_path=selected_ending,
            output_name=output_video,
            start_at=start_idx
        )
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        return

    # --- METADATA & HASHTAGS ---
    caption = result["caption"]
    
    # Custom psychology conditional tags depending on path signatures
    is_dark_psych = any(x in row["audio_path"].upper() for x in ["/P", "\\P", "P (", "DARK"])
    
    if is_dark_psych:
        hashtags = ["darkpsychology", "manipulation", "mindcontrol", "psychologyfacts", "shorts"]
        emoji = "🧠👁️"
    else:
        hashtags = ["psychologyfacts", "humanbehavior", "mindset", "mentalhealth", "shorts"]
        emoji = "🧠💡"

    title = f"{caption} {emoji} #shorts"
    description = f"{caption}\n\n#{' #'.join(hashtags)}\n\n🧠 Deep Human Insights\n🚀 Subscribe for daily mental loops."

    # --- UPLOAD & UPDATE ---
    if upload_to_youtube(output_video, title, description, hashtags):
        # Update current row as posted
        df.at[current_index, "posted"] = "true"
        # Store where we ended so the NEXT row knows where to start
        df.at[current_index, "last_image_index"] = result["last_index"]

        # Save CSV immediately
        df.to_csv(csv_file, index=False)
        
        if os.path.exists(output_video):
            os.remove(output_video)
        print("✅ Process Complete.")
    else:
        print("❌ Process aborted due to upload failure.")

if __name__ == "__main__":
    run_automation()