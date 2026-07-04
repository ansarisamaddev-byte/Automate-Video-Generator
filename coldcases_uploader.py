import os
import glob
import random
import pickle
import pandas as pd

# ================= YOUTUBE =================
import googleapiclient.discovery
from google.auth.transport.requests import Request
from googleapiclient.http import MediaFileUpload
from google_auth_oauthlib.flow import InstalledAppFlow

# ================= VIDEO GENERATOR =================
from video_generator_coldcases import generate_coldcase_video

SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]

def get_service():
    creds = None
    pickle_file = "coldcases_pov_token.pickle"
    client_secrets = "coldcases_auth.json"

    if os.path.exists(pickle_file):
        with open(pickle_file, "rb") as f:
            creds = pickle.load(f)

    if creds and creds.expired and creds.refresh_token:
        print("🔄 Refreshing ColdCases access token...")
        creds.refresh(Request())

    if not creds or not creds.valid:
        if not os.path.exists(client_secrets):
            raise FileNotFoundError(f"❌ Missing '{client_secrets}' in this folder.")
        
        print("\n--- AUTHENTICATION REQUIRED FOR: ColdCases-POV ---")
        flow = InstalledAppFlow.from_client_secrets_file(client_secrets, SCOPES)
        creds = flow.run_local_server(port=0)

        with open(pickle_file, "wb") as f:
            pickle.dump(creds, f)
            print(f"✅ Session saved successfully to {pickle_file}.")

    return googleapiclient.discovery.build("youtube", "v3", credentials=creds)


def upload_to_youtube(video_path, title, description, tags):
    """Safely uploads to YouTube just like the MindScribble script."""
    try:
        print("📤 Uploading to @ColdCases-pov YouTube Channel...")
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
            media_body=MediaFileUpload(video_path, chunksize=1024*1024, resumable=True)
        )

        response = request.execute()
        print(f"✅ YouTube Uploaded: https://www.youtube.com/watch?v={response['id']}")
        return True

    except Exception as e:
        print(f"❌ YouTube Upload Error: {e}")
        return False


def run_automation():
    csv_file = "coldcases.csv"
    if not os.path.exists(csv_file):
        print("❌ CSV not found")
        return

    df = pd.read_csv(csv_file)
    df["posted"] = df["posted"].astype(str).str.lower()
    unposted_df = df[df["posted"] == "false"]

    if unposted_df.empty:
        print("✅ Channel @ColdCases-pov is up to date.")
        return

    idx = unposted_df.index[0]
    row = df.loc[idx]
    
    # Explicitly set the output name BEFORE generation
    output_video = f"temp_coldcase_render_{idx}.mp4"
    
    print(f"🎬 Generating content for Case {idx}...")
    try:
        # We capture the result, but remember: for coldcase, it's just a string!
        generator_result = generate_coldcase_video(
            audio_path=row["audio_path"],
            bg_folder="images/coldcases/background_images",
            evidence_folder=f"images/coldcases/{idx}",
            music_path=random.choice(glob.glob("background_music/*.mp3") or [None]),
            credit_video_path=random.choice(glob.glob("ending/coldcases/*.mp4") or [None]),
            output_name=output_video,
            char_folder="images/coldcases/characters",
            sub_folder="images/coldcases/characters/subscribe"
        )
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        return

    # --- CAPTION LOGIC ---
    # Since generator_result is just a string, we MUST get the caption from the CSV
    if "title" in row and pd.notna(row["title"]) and str(row["title"]).strip() != "":
        caption = str(row["title"]).strip()
        print(f"📖 Using title as caption: {caption}")
    else:
        caption = "Unsolved Mystery Investigation"

    # --- METADATA ---
    title = f"{caption} | @ColdCases-pov #shorts"
    if len(title) > 100:
        title = title[:90] + "... #shorts"

    description = (
        f"{caption}\n\n"
        "Explore the chilling details of this cold case. Join us at @ColdCases-pov for daily investigations into history's most baffling mysteries.\n\n"
        "#truecrime #coldcase #unsolved #mysteries #investigation #shorts"
    )
    tags = ["truecrime", "coldcase", "unsolved", "mysteries", "coldcasespov"]

    # --- YOUTUBE UPLOAD & CSV UPDATE ---
    if upload_to_youtube(output_video, title, description, tags):
        df.at[idx, "posted"] = "true"
        df.to_csv(csv_file, index=False)
        
        if os.path.exists(output_video):
            os.remove(output_video)
        print("✅ Process Complete.")
    else:
        print("❌ Process aborted due to YouTube upload failure. Local video retained.")

if __name__ == "__main__":
    run_automation()