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


# ================= AUTH =================

def get_service():

    creds = None

    if os.path.exists("token.pickle"):
        with open("token.pickle", "rb") as f:
            creds = pickle.load(f)

    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())

    return googleapiclient.discovery.build(
        "youtube",
        "v3",
        credentials=creds
    )


# ================= UPLOAD =================

def upload_to_youtube(
    video_path,
    title,
    description,
    tags
):
    try:

        print("☁️ Uploading to Cloudinary...")

        cloudinary.uploader.upload(
            video_path,
            resource_type="video"
        )

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

        print(
            f"✅ Uploaded: "
            f"https://www.youtube.com/watch?v={response['id']}"
        )

        return True

    except Exception as e:
        print(f"❌ Upload Error: {e}")
        return False


# ================= MAIN =================

def run_automation():

    csv_file = "shorts.csv"

    if not os.path.exists(csv_file):
        print("❌ CSV not found")
        return

    df = pd.read_csv(csv_file)

    unposted = (
        df["posted"]
        .astype(str)
        .str.lower() == "false"
    )

    if not unposted.any():
        print("✅ All uploaded")
        return

    index = df[unposted].index[0]
    row = df.loc[index]

    start_idx = int(
        row["last_image_index"]
    )

    bg_music_files = glob.glob(
        "background_music/*.mp3"
    )

    bg_music = (
        random.choice(bg_music_files)
        if bg_music_files else None
    )

    ending_assets = glob.glob(
        "ending/*.mp4"
    )

    if not ending_assets:
        print("❌ No ending videos")
        return

    selected_ending = random.choice(
        ending_assets
    )

    output_video = "yt_output.mp4"

    print("🎬 Generating video...")

    result = generate_reel(
        audio_path=row["audio_path"],
        image_folder=row["image_folder"],
        music_path=bg_music,
        credit_video_path=selected_ending,
        output_name=output_video,
        start_at=start_idx
    )

    caption = result["caption"]
    new_last_index = result["last_index"]

    # Detect W / M
    if (
        "/W" in row["audio_path"]
        or "\\W" in row["audio_path"]
    ):
        hashtags = [
            "warrior",
            "discipline",
            "grind",
            "stoic",
            "shorts"
        ]

    else:
        hashtags = [
            "motivation",
            "mindset",
            "success",
            "growth",
            "shorts"
        ]

    title = f"{caption} 💪 #shorts"

    description = f"""
{caption}

#{' #'.join(hashtags)}

🔥 Daily Motivation
🚀 Subscribe For More
"""

    if upload_to_youtube(
        output_video,
        title,
        description,
        hashtags
    ):

        df.at[index, "posted"] = True
        df.at[
            index,
            "last_image_index"
        ] = new_last_index

        df.to_csv(
            csv_file,
            index=False
        )

        if os.path.exists(output_video):
            os.remove(output_video)

        print("✅ DONE")

    else:
        print("❌ Upload failed")


if __name__ == "__main__":
    run_automation()