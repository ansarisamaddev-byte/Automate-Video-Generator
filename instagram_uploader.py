import pandas as pd
import os
import requests
import time
import glob
import random
import cloudinary
import cloudinary.uploader
from video_generator import generate_reel # Ensure your rendering script is named this

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
        print(f"Uploading: {local_video_path}")

        upload_result = cloudinary.uploader.upload(
            local_video_path,
            resource_type="video"
        )

        public_url = upload_result["secure_url"]

        post_url = f"https://graph.facebook.com/v19.0/{IG_USER_ID}/media"

        payload = {
            "media_type": "REELS",
            "video_url": public_url,
            "caption": caption,
            "access_token": ACCESS_TOKEN
        }

        result = requests.post(
            post_url,
            data=payload
        ).json()

        if "id" not in result:
            print("Container Error:", result)
            return False

        creation_id = result["id"]

        print("Processing Instagram Reel...")

        status_url = f"https://graph.facebook.com/v19.0/{creation_id}"

        for _ in range(20):

            status_res = requests.get(
                status_url,
                params={
                    "fields": "status_code",
                    "access_token": ACCESS_TOKEN
                }
            ).json()

            status = status_res.get("status_code")

            print(status)

            if status == "FINISHED":
                break

            if status == "ERROR":
                return False

            time.sleep(10)

        else:
            return False

        publish_res = requests.post(
            f"https://graph.facebook.com/v19.0/{IG_USER_ID}/media_publish",
            data={
                "creation_id": creation_id,
                "access_token": ACCESS_TOKEN
            }
        ).json()

        return "id" in publish_res

    except Exception as e:
        print(e)
        return False


# ---------------- AUTOMATION ---------------- #

def run_automation():

    csv_file = "reels.csv"

    if not os.path.exists(csv_file):
        print("CSV missing")
        return

    df = pd.read_csv(csv_file)

    unposted = (
        df["posted"]
        .astype(str)
        .str.lower() == "false"
    )

    if not unposted.any():
        print("All reels posted.")
        return

    index = df[unposted].index[0]
    row = df.loc[index]

    start_idx = int(row["last_image_index"])

    music_files = glob.glob(
        "background_music/*.mp3"
    )

    bg_music = (
        random.choice(music_files)
        if music_files else None
    )

    endings = glob.glob(
        "ending/warrior/*.mp4"
    )

    if not endings:
        print("No ending assets found")
        return

    selected_ending = random.choice(endings)

    print(f"Ending: {selected_ending}")

    output_video = "final_reel_output.mp4"

    print("Generating reel...")

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

    # Hashtag auto detect
    if (
        "/W" in row["audio_path"]
        or "\\W" in row["audio_path"]
    ):
        hashtags = (
            "#warrior #grind "
            "#discipline #stoic"
        )
    else:
        hashtags = (
            "#motivation #mindset "
            "#success #growth"
        )

    full_caption = (
        f"{caption}...\n\n{hashtags}"
    )

    print(full_caption)

    if upload_reel_to_instagram(
        output_video,
        full_caption
    ):

        df.at[index, "posted"] = True
        df.at[index, "last_image_index"] = new_last_index

        df.to_csv(
            csv_file,
            index=False
        )

        print(
            f"Posted. Next image index: "
            f"{new_last_index}"
        )

    else:
        print("Upload failed")


# ---------------- RUN ---------------- #

if __name__ == "__main__":
    run_automation()