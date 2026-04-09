import pandas as pd
import os
import requests
import time
import cloudinary
import cloudinary.uploader

# --- CONFIGURATION ---
ACCESS_TOKEN = "EAAdDD4cKxacBRPCWWL5mYCz0aFWrA3N41ZBBFnXSZBa9sslFdPfHxyyzVXemwUAckiv19zWJYUul9ZAGwLSWZATI9ae5UFRHfCGH43OmOdGySgLOWYV4zZBhaEfNkK6ZCWr9cBxLqvZCVcMSF3j2cKZBPQZCyZAVuX2CP3d1FcvHrKluuyUeRc7tt4PbXhhxl70ZARK2eLqAU73" #os.getenv("INSTAGRAM_ACCESS_TOKEN")
IG_USER_ID = "17841480606710089"  #os.getenv("INSTAGRAM_ACCOUNT_ID")

cloudinary.config(
    cloud_name="dusdbgfey",
    api_key="545263495647551",
    api_secret="KFRuIRsx-LkevEBul4YvfYBWfiY"
)

def upload_video_to_instagram(local_video_path, caption):
    try:
        # -------------------------------
        # STEP 1: Upload to Cloudinary
        # -------------------------------
        print(f"--- Step 1: Uploading {local_video_path} to Cloudinary ---")
        # Resource_type="video" is mandatory for video files
        upload_result = cloudinary.uploader.upload(local_video_path, resource_type="video")
        public_video_url = upload_result["secure_url"]
        print(f"Public Video URL: {public_video_url}")

        # -------------------------------
        # STEP 2: Create Media Container (VIDEO TYPE)
        # -------------------------------
        print("--- Step 2: Creating Instagram Video Container ---")
        post_url = f"https://graph.facebook.com/v19.0/{IG_USER_ID}/media"
        
        payload = {
            "media_type": "REELS",  # Required for videos
            "video_url": public_video_url,
            "caption": caption,
            "access_token": ACCESS_TOKEN
        }

        response = requests.post(post_url, data=payload)
        result = response.json()

        if "id" not in result:
            print("❌ Error creating container:", result)
            return False

        creation_id = result["id"]
        print(f"Creation ID: {creation_id}")

        # -------------------------------
        # STEP 3: Poll Status (Videos take longer than images)
        # -------------------------------
        print("--- Step 3: Waiting for video processing ---")
        status_url = f"https://graph.facebook.com/v19.0/{creation_id}"
        params = {"fields": "status_code", "access_token": ACCESS_TOKEN}

        for i in range(20):  # Max 100 seconds
            status_res = requests.get(status_url, params=params).json()
            status = status_res.get("status_code")
            print(f"Attempt {i+1}: Status is {status}")

            if status == "FINISHED":
                break
            elif status == "ERROR":
                print("❌ Media processing failed:", status_res)
                return False
            time.sleep(5)
        else:
            print("❌ Timeout waiting for video processing")
            return False

        # -------------------------------
        # STEP 4: Publish
        # -------------------------------
        print("--- Step 4: Publishing Reel/Video to Instagram ---")
        publish_url = f"https://graph.facebook.com/v19.0/{IG_USER_ID}/media_publish"
        publish_res = requests.post(publish_url, data={
            "creation_id": creation_id,
            "access_token": ACCESS_TOKEN
        }).json()

        if "id" in publish_res:
            print("✅ Success! Video/Reel posted to Instagram.")
            return True
        else:
            print("❌ Publish failed:", publish_res)
            return False

    except Exception as e:
        print("❌ Unexpected error:", str(e))
        return False

def run_video_automation():
    # Load CSV
    df = pd.read_csv('quotes.csv')
    
    # Find first unposted quote
    try:
        index = df[df['Posted'] == False].index[0]
    except IndexError:
        print("All quotes have been posted!")
        return

    row = df.loc[index]
    sn = row['SN']
    
    # Path logic
    raw_video = f"videos/video ({sn}).mp4"  # Assuming you have a videos folder
    music_file = "music.mp3"               # Your background music
    output_video = "final_output.mp4"
    
    if not os.path.exists(raw_video):
        print(f"Error: Raw video {raw_video} not found.")
        return

    # Generate the Video (Importing the new MoviePy function)
    from video_post import create_video_post  
    
    create_video_post(
        video_in=raw_video,
        audio_in=music_file if os.path.exists(music_file) else None,
        quote=row['Quote'],
        author=row['Author'],
        logo="profile.png",
        video_out=output_video
    )

    # Prepare Caption
    caption = str(row.get("Caption", "")).strip()
    if not caption:
        caption = "Stay disciplined. ⚔️"

    hashtags = "\n\n#reels #motivation #mindset #discipline #success"
    full_caption = caption + hashtags
        
    # Upload
    if upload_video_to_instagram(output_video, full_caption):
        df.at[index, 'Posted'] = True
        df.to_csv('quotes.csv', index=False)
        print(f"CSV updated for Video #{sn}")
        
        # Cleanup local file to save space
        if os.path.exists(output_video):
            os.remove(output_video)

if __name__ == "__main__":
    run_video_automation()