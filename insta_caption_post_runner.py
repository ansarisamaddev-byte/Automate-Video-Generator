import pandas as pd
import os
import requests
import time
import glob
import random
import cloudinary
import cloudinary.uploader
from insta_caption_post import generate_reel # Ensure your rendering script is named this

# --- CONFIGURATION ---
ACCESS_TOKEN = "EAAdDD4cKxacBRPCWWL5mYCz0aFWrA3N41ZBBFnXSZBa9sslFdPfHxyyzVXemwUAckiv19zWJYUul9ZAGwLSWZATI9ae5UFRHfCGH43OmOdGySgLOWYV4zZBhaEfNkK6ZCWr9cBxLqvZCVcMSF3j2cKZBPQZCyZAVuX2CP3d1FcvHrKluuyUeRc7tt4PbXhhxl70ZARK2eLqAU73"
IG_USER_ID = "17841480606710089"

cloudinary.config(
    cloud_name="dusdbgfey",
    api_key="545263495647551",
    api_secret="KFRuIRsx-LkevEBul4YvfYBWfiY"
)

def upload_reel_to_instagram(local_video_path, caption):
    try:
        print(f"--- Step 1: Uploading {local_video_path} to Cloudinary ---")
        # Use resource_type="video" for Reels
        upload_result = cloudinary.uploader.upload(local_video_path, resource_type="video")
        public_url = upload_result["secure_url"]

        print("--- Step 2: Creating Instagram Reel Container ---")
        post_url = f"https://graph.facebook.com/v19.0/{IG_USER_ID}/media"
        payload = {
            "media_type": "REELS",
            "video_url": public_url,
            "caption": caption,
            "access_token": ACCESS_TOKEN
        }
        result = requests.post(post_url, data=payload).json()

        if "id" not in result:
            print("❌ Container Error:", result)
            return False

        creation_id = result["id"]
        
        print("--- Step 3: Processing Video ---")
        status_url = f"https://graph.facebook.com/v19.0/{creation_id}"
        for _ in range(20): # Reels take longer to process
            status_res = requests.get(status_url, params={"fields": "status_code", "access_token": ACCESS_TOKEN}).json()
            status = status_res.get("status_code")
            print(f"Status: {status}")
            if status == "FINISHED": break
            if status == "ERROR": return False
            time.sleep(10)
        else:
            return False

        print("--- Step 4: Publishing ---")
        publish_res = requests.post(f"https://graph.facebook.com/v19.0/{IG_USER_ID}/media_publish", data={
            "creation_id": creation_id,
            "access_token": ACCESS_TOKEN
        }).json()

        return "id" in publish_res
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def run_automation():
    csv_file = 'database.csv'
    if not os.path.exists(csv_file):
        print(f"❌ CSV not found: {csv_file}")
        return

    df = pd.read_csv(csv_file)
    
    # 2. Find first unposted row
    unposted_mask = df['posted'].astype(str).str.lower() == 'false'
    if not unposted_mask.any():
        print("All content posted!")
        return

    index = df[unposted_mask].index[0]
    row = df.loc[index]

    # 3. Determine Starting Image Index
    start_idx = 0
    if index > 0:
        posted_rows = df[df['posted'].astype(str).str.lower() == 'true']
        if not posted_rows.empty:
            start_idx = int(posted_rows.iloc[-1]['last_image_index'])

    # 4. Pick Random Background Music
    music_list = glob.glob("background_music/*.mp3")
    bg_music = random.choice(music_list) if music_list else None
    
    # --- 5. PICK RANDOM ENDING ASSET ---
    ending_assets = glob.glob("ending/*.mp4") # Looks inside 'ending' folder
    if not ending_assets:
        print("❌ Error: No ending assets found in 'ending' folder!")
        return
    
    selected_ending = random.choice(ending_assets)
    print(f"🎬 Selected Ending Asset: {selected_ending}")
    
    output_video = "final_reel_output.mp4"

    # 6. Generate the Reel
    print(f"🎬 Generating Reel starting from Image Index: {start_idx}")
    new_last_index = generate_reel(
        audio_path=row['audio_path'],
        image_folder=row['image_folder'],
        music_path=bg_music,
        credit_video_path=selected_ending, # Now passes the random ending
        output_name=output_video,
        start_at=start_idx
    )

    # 7. Upload and Update
    full_caption = f"{row['caption']}\n\n{row['hashtags']}"
    if upload_reel_to_instagram(output_video, full_caption):
        df.at[index, 'posted'] = True
        df.at[index, 'last_image_index'] = new_last_index
        df.to_csv(csv_file, index=False)
        print(f"✅ Posted. Next start index: {new_last_index}")
    else:
        print("❌ Upload failed.")
        
if __name__ == "__main__":
    run_automation()
