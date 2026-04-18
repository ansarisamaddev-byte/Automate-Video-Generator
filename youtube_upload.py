import pandas as pd
import os
import glob
import random

# ================= CLOUDINARY =================
import cloudinary
import cloudinary.uploader

cloudinary.config(
    cloud_name="dusdbgfey",
    api_key="545263495647551",
    api_secret="KFRuIRsx-LkevEBul4YvfYBWfiY"
)

# ================= YOUTUBE =================
import pickle
import googleapiclient.discovery
from google.auth.transport.requests import Request
from googleapiclient.http import MediaFileUpload

# ================= VIDEO GENERATOR =================
from insta_caption_post import generate_reel

SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]


def get_service():
    creds = None

    if os.path.exists("token.pickle"):
        with open("token.pickle", "rb") as f:
            creds = pickle.load(f)

    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())

    return googleapiclient.discovery.build("youtube", "v3", credentials=creds)


# ================= UPLOAD =================
def upload_to_youtube(video_path, title, description, tags):
    try:
        print("☁️ Uploading to Cloudinary...")
        upload_result = cloudinary.uploader.upload(video_path, resource_type="video")
        video_url = upload_result["secure_url"]

        print("📤 Uploading to YouTube...")

        youtube = get_service()

        request = youtube.videos().insert(
            part="snippet,status",
            body={
                "snippet": {
                    "title": title,
                    "description": description + f"\n\nSource: {video_url}",
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
        print(f"❌ Upload Error: {e}")
        return False


# ================= MAIN =================
def run_automation():
    csv_file = "yt_database.csv"

    if not os.path.exists(csv_file):
        print("❌ CSV not found")
        return

    df = pd.read_csv(csv_file)

    unposted_mask = df['posted'].astype(str).str.lower() == 'false'
    if not unposted_mask.any():
        print("✅ All videos uploaded!")
        return

    index = df[unposted_mask].index[0]
    row = df.loc[index]

    start_idx = 0
    if index > 0:
        posted_rows = df[df['posted'].astype(str).str.lower() == 'true']
        if not posted_rows.empty:
            start_idx = int(posted_rows.iloc[-1]['last_image_index'])

    bg_music = random.choice(glob.glob("bg_music/*.mp3")) if glob.glob("bg_music/*.mp3") else None
    selected_ending = random.choice(glob.glob("ending/*.mp4"))

    output_video = "yt_output.mp4"

    print("🎬 Generating video...")

    new_last_index = generate_reel(
        audio_path=row['audio_path'],
        image_folder=row['image_folder'],
        music_path=bg_music,
        credit_video_path=selected_ending,
        output_name=output_video,
        start_at=start_idx
    )

    title = f"{row['caption']} 💪 #shorts"

    description = f"""{row['caption']}

{row['hashtags']}

🔥 Daily Motivation
🚀 Follow for more
"""

    tags = ["motivation", "success", "shorts"]

    if upload_to_youtube(output_video, title, description, tags):
        df.at[index, 'posted'] = True
        df.at[index, 'last_image_index'] = new_last_index
        df.to_csv(csv_file, index=False)

        os.remove(output_video)

        print("✅ DONE")
    else:
        print("❌ Upload failed")


if __name__ == "__main__":
    run_automation()
