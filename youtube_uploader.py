import pandas as pd
import os
import glob
import random
import pickle

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

# ================= VIDEO GENERATOR =================
from video_generator import generate_reel


SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload"
]

def get_service():
    creds = None
    if os.path.exists("token.pickle"):
        with open("token.pickle", "rb") as f:
            creds = pickle.load(f)

    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())

    return googleapiclient.discovery.build(
        "youtube", "v3", credentials=creds
    )

def upload_to_youtube(video_path, title, description, tags):
    try:
        print("📤 Uploading to YouTube...")
        youtube = get_service()

        request = youtube.videos().insert(
            part="snippet,status",
            body={
                "snippet": {
                    "title": title,
                    "description": description,
                    "tags": tags,
                    "categoryId": "22"
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
    csv_file = "shorts.csv"

    if not os.path.exists(csv_file):
        print("❌ CSV not found")
        return

    # Load data and standardize the 'posted' column
    df = pd.read_csv(csv_file)
    df["posted"] = df["posted"].astype(str).str.lower()

    # Find rows that haven't been processed yet
    unposted_df = df[df["posted"] == "false"]

    if unposted_df.empty:
        print("✅ All reels have been posted!")
        return

    # Get the index of the first unposted row
    current_index = unposted_df.index[0]
    row = df.loc[current_index]

    # --- THE IMAGE INDEX FIX ---
    # We look at the row immediately before this one to see where it stopped.
    # This prevents every video from starting back at image 0.
    if current_index > 0:
        start_idx = int(df.loc[current_index - 1, "last_image_index"])
    else:
        start_idx = int(row["last_image_index"])

    print(f"🚀 Processing ID {row.get('id', current_index)} | Starting at image index: {start_idx}")

    # --- ASSET SELECTION ---
    bg_music_files = glob.glob("background_music/*.mp3")
    bg_music = random.choice(bg_music_files) if bg_music_files else None

    ending_assets = glob.glob("ending/warrior/*.mp4")
    if not ending_assets:
        print("❌ Error: No ending/outro videos found.")
        return
    selected_ending = random.choice(ending_assets)

    output_video = f"temp_output_{current_index}.mp4"

    # --- GENERATION ---
    print("🎬 Generating video content...")
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
    is_warrior_theme = any(x in row["audio_path"].upper() for x in ["/W", "\\W", "W ("])
    
    if is_warrior_theme:
        hashtags = ["warrior", "discipline", "grind", "stoic", "shorts"]
    else:
        hashtags = ["motivation", "mindset", "success", "growth", "shorts"]

    title = f"{caption} 💪 #shorts"
    description = f"{caption}\n\n#{' #'.join(hashtags)}\n\n🔥 Daily Motivation\n🚀 Subscribe For More"

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