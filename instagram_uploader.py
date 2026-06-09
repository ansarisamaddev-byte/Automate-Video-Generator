import pandas as pd
import os
import requests
import time
import glob
import random
import cloudinary
import cloudinary.uploader
from video_generator import generate_reel 

# --- CONFIGURATION ---
ACCESS_TOKEN = "EAAdDD4cKxacBRPCWWL5mYCz0aFWrA3N41ZBBFnXSZBa9sslFdPfHxyyzVXemwUAckiv19zWJYUul9ZAGwLSWZATI9ae5UFRHfCGH43OmOdGySgLOWYV4zZBhaEfNkK6ZCWr9cBxLqvZCVcMSF3j2cKZBPQZCyZAVuX2CP3d1FcvHrKluuyUeRc7tt4PbXhhxl70ZARK2eLqAU73"
IG_USER_ID = "17841480606710089"

cloudinary.config(
    cloud_name="dusdbgfey",
    api_key="545263495647551",
    api_secret="KFRuIRsx-LkevEBul4YvfYBWfiY"
)

# ---------------- UPLOAD ---------------- #

def upload_reel_to_instagram(local_video_path, caption):
    try:
        print(f"Uploading to Cloudinary: {local_video_path}")
        upload_result = cloudinary.uploader.upload(
            local_video_path,
            resource_type="video"
        )
        public_url = upload_result["secure_url"]

        # Using stable/current unversioned endpoint pathing
        post_url = f"https://graph.facebook.com/{IG_USER_ID}/media"

        payload = {
            "media_type": "REELS",
            "video_url": public_url,
            "caption": caption,
            "access_token": ACCESS_TOKEN
        }

        result = requests.post(post_url, data=payload).json()

        if "id" not in result:
            print("Container Error:", result)
            return False

        creation_id = result["id"]
        print("Processing Instagram Reel on Meta servers...")
        status_url = f"https://graph.facebook.com/{creation_id}"

        for _ in range(30): # Extended slightly to account for longer dynamic rendering processing
            status_res = requests.get(
                status_url,
                params={
                    "fields": "status_code",
                    "access_token": ACCESS_TOKEN
                }
            ).json()

            status = status_res.get("status_code")
            print(f"Current Status: {status}")

            if status == "FINISHED":
                break
            if status == "ERROR":
                return False

            time.sleep(10)
        else:
            return False

        publish_res = requests.post(
            f"https://graph.facebook.com/{IG_USER_ID}/media_publish",
            data={
                "creation_id": creation_id,
                "access_token": ACCESS_TOKEN
            }
        ).json()

        return "id" in publish_res

    except Exception as e:
        print(f"❌ Upload Execution Exception: {e}")
        return False


# ---------------- AUTOMATION ---------------- #

def run_automation():
    csv_file = "reels.csv"

    if not os.path.exists(csv_file):
        print("❌ CSV file missing")
        return

    df = pd.read_csv(csv_file)
    
    # Standardize as clean strings matching your working reference sample
    df["posted"] = df["posted"].astype(str).str.lower().str.strip()

    unposted_df = df[df["posted"] == "false"]

    if unposted_df.empty:
        print("✅ All reels posted.")
        return

    index = unposted_df.index[0]
    row = df.loc[index]

    # --- THE IMAGE INDEX FIX (Referencing previous row progression) ---
    current_loc = df.index.get_loc(index)
    if current_loc > 0:
        prev_idx_label = df.index[current_loc - 1]
        start_idx = int(df.loc[prev_idx_label, "last_image_index"])
    else:
        start_idx = int(row["last_image_index"])

    print(f"🚀 Processing Index Row: {index} | Starting at image array pointer: {start_idx}")

    music_files = glob.glob("background_music/*.mp3")
    bg_music = random.choice(music_files) if music_files else None

    endings = glob.glob("ending/warrior/*.mp4")
    if not endings:
        print("❌ No ending assets found. Process aborted.")
        return

    selected_ending = random.choice(endings)
    output_video = f"final_reel_output_{index}.mp4"

    print("🎬 Generating reel video elements...")
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
        print(f"❌ Video Generation Crashed: {e}")
        return

    caption = result["caption"]
    new_last_index = result["last_index"]

    # Theme Hashtag Detection
    if any(x in row["audio_path"].upper() for x in ["/W", "\\W", "W ("]):
        hashtags = "#warrior #grind #discipline #stoic"
    else:
        hashtags = "#motivation #mindset #success #growth"

    full_caption = f"{caption}...\n\n{hashtags}"
    print(f"Caption Draft:\n{full_caption}")

    if upload_reel_to_instagram(output_video, full_caption):
        # Update your dataframe records using uniform string allocations
        df.at[index, "posted"] = "true"
        df.at[index, "last_image_index"] = int(new_last_index)

        df.to_csv(csv_file, index=False)
        print(f"✅ Posted Successfully. Next index logged: {new_last_index}")

        if os.path.exists(output_video):
            os.remove(output_video)
    else:
        print("❌ Upload failed.")

if __name__ == "__main__":
    run_automation()