import os
import glob
import math
import random
import pickle
import datetime
import pandas as pd

# ================= YOUTUBE API =================
import googleapiclient.discovery
from google.auth.transport.requests import Request
from googleapiclient.http import MediaFileUpload
from google_auth_oauthlib.flow import InstalledAppFlow

# ================= VIDEO GENERATOR ENGINE =================
# Updated module name to match mindscribble_editor.py
from mindscribble_editor import generate_video_from_csv, load_folder_images
from moviepy import AudioFileClip

SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload"
]

def resolve_project_path(path_str):
    """
    Resolves relative paths cleanly based on the script's root folder location.
    Handles cross-platform paths (Windows/Linux/GitHub Actions).
    """
    if not path_str or pd.isna(path_str):
        return ""
    path_str = str(path_str).strip()
    if os.path.isabs(path_str):
        return path_str
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(base_dir, path_str))


def get_service():
    creds = None
    pickle_file = resolve_project_path("mindscribble_token.pickle")
    client_secrets = resolve_project_path("client_secret_mindscribble.json")

    # Load existing authentication session for MindScribble
    if os.path.exists(pickle_file):
        with open(pickle_file, "rb") as f:
            creds = pickle.load(f)

    # Refresh the token dynamically if it's expired
    if creds and creds.expired and creds.refresh_token:
        print("🔄 Refreshing MindScribble access token...")
        try:
            creds.refresh(Request())
        except Exception as e:
            print(f"⚠️ Token refresh failed ({e}). Forcing re-authentication...")
            creds = None

    # First-time setup: Create the token file using client_secret.json
    if not creds or not creds.valid:
        if not os.path.exists(client_secrets):
            raise FileNotFoundError(f"❌ Missing '{client_secrets}' at {client_secrets}. Add it to authenticate MindScribble.")
        
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

        media = MediaFileUpload(
            video_path,
            mimetype="video/mp4",
            chunksize=1024 * 1024 * 5,  # 5MB chunks for stability
            resumable=True
        )

        request = youtube.videos().insert(
            part="snippet,status",
            body={
                "snippet": {
                    "title": title,
                    "description": description,
                    "tags": tags,
                    "categoryId": "22"  # People & Blogs
                },
                "status": {
                    "privacyStatus": "public",
                    "selfDeclaredMadeForKids": False
                }
            },
            media_body=media
        )

        response = None
        while response is None:
            status, response = request.next_chunk()
            if status:
                print(f"⏳ Upload Progress: {int(status.progress() * 100)}%")

        print(f"✅ Uploaded Successfully! Video URL: https://www.youtube.com/watch?v={response['id']}")
        return True

    except Exception as e:
        print(f"❌ YouTube Upload Error: {e}")
        return False


# ================= MAIN AUTOMATION LOGIC =================
def run_automation():
    csv_file = "mind_scribble2.csv"
    csv_abs = resolve_project_path(csv_file)

    if not os.path.exists(csv_abs):
        print(f"❌ CSV not found at: {csv_abs}")
        return

    # Load CSV preserving Integer type for last_image_index
    df = pd.read_csv(csv_abs, dtype={"last_image_index": "Int64"})
    
    # Standardize posted column comparison
    df["posted_clean"] = df["posted"].astype(str).str.lower().str.strip()
    unposted_df = df[df["posted_clean"].isin(["false", "0", "no", "nan"])]

    if unposted_df.empty:
        print("✅ All MindScribble shorts have been posted!")
        return

    current_index = unposted_df.index[0]
    row = df.loc[current_index]

    target_id = int(row["id"]) if "id" in row and pd.notna(row["id"]) else current_index + 1

    print(f"🕒 Time detected: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"🚀 Selected Row Index: {current_index} | ID: {target_id}")

    # Read current index offset (defaults to 0 if NaN)
    current_img_index = int(row["last_image_index"]) if pd.notna(row.get("last_image_index")) else 0

    output_video = resolve_project_path(f"mindscribble_output_{target_id}.mp4")

    # --- GENERATION CALL ---
    print(f"🎬 Generating video using mindscribble_editor script...")
    try:
        generate_video_from_csv(
            csv_path=csv_abs,
            target_id=target_id,
            output_name=output_video
        )
    except Exception as e:
        print(f"❌ Video Generation failed: {e}")
        return

    if not os.path.exists(output_video):
        print(f"❌ Output video file was not generated: {output_video}")
        return

    # --- CALCULATE CONSUMED BACKGROUND IMAGES ---
    audio_path = resolve_project_path(str(row["audio_path"]))
    if os.path.exists(audio_path):
        with AudioFileClip(audio_path) as speech_audio:
            # Editor adds HEADING_DUR = 1.0s and switches backgrounds every 6.0s
            total_dur = speech_audio.duration + 1.0
            bg_images_used = math.ceil(total_dur / 6.0)
    else:
        print(f"⚠️ Audio path not found: {audio_path}, defaulting to 1 image used.")
        bg_images_used = 1

    new_image_index = current_img_index + bg_images_used

    # --- CAPTION & METADATA SELECTION ---
    csv_heading = str(row.get("heading", "")).strip() if pd.notna(row.get("heading")) else ""
    csv_caption = str(row.get("caption", "")).strip() if pd.notna(row.get("caption")) else ""

    # Use the actual title/caption from the CSV as the public-facing caption.
    # Fall back to the heading only if no custom title was provided.
    caption = csv_caption or csv_heading or "Deep Human Insights"
    hashtags = ["darkpsychology", "manipulation", "mindcontrol", "psychologyfacts", "shorts"]
    emoji = "🧠👁️"

    title = f"{caption} {emoji} #shorts"
    if len(title) > 100:
        title = title[:95] + "... #shorts"

    description = f"{caption}\n\n#{' #'.join(hashtags)}\n\n🧠 Deep Human Insights\n🚀 Subscribe for daily mental loops."

    # --- UPLOAD & UPDATE CSV ---
    if upload_to_youtube(output_video, title, description, hashtags):
        # Clean up temporary column before saving
        if "posted_clean" in df.columns:
            df = df.drop(columns=["posted_clean"])

        # Update CSV fields
        df.at[current_index, "posted"] = "true"
        df.at[current_index, "last_image_index"] = new_image_index

        df.to_csv(csv_abs, index=False)
        print(f"✅ CSV Updated -> ID {target_id} | posted: true | last_image_index: {new_image_index}")
        
        # Cleanup rendered video
        if os.path.exists(output_video):
            os.remove(output_video)
            print("🧹 Rendered video cleaned up from disk.")
        print("🎉 Process Complete Successfully.")
    else:
        print("❌ Process aborted due to upload failure.")


if __name__ == "__main__":
    run_automation()