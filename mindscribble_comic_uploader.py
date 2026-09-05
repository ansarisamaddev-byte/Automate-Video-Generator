import os
import sys

# Set up absolute base path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

if BASE_DIR in sys.path:
    sys.path.remove(BASE_DIR)
sys.path.insert(0, BASE_DIR)

import json
import pickle

import googleapiclient.discovery
from google.auth.transport.requests import Request
from googleapiclient.http import MediaFileUpload
from google_auth_oauthlib.flow import InstalledAppFlow

# Import the processing function from your pipeline module
from mindscribble_comic.main import process_script_item

SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]


def get_service(
    pickle_file: str = os.path.join(BASE_DIR, "mindscribble_token.pickle"),
    client_secrets: str = os.path.join(BASE_DIR, "client_secret_mindscribble.json")
):
    """Authenticates and returns the YouTube API service instance."""
    creds = None

    # Restore pickle token from Base64 environment variable if running in CI/GitHub Actions
    if not os.path.exists(pickle_file) and os.environ.get("YOUTUBE_TOKEN_PICKLE_BASE64"):
        import base64
        print("🔑 Restoring pickle token from GitHub Secrets...")
        token_data = base64.b64decode(os.environ["YOUTUBE_TOKEN_PICKLE_BASE64"])
        with open(pickle_file, "wb") as f:
            f.write(token_data)

    if os.path.exists(pickle_file):
        with open(pickle_file, "rb") as f:
            creds = pickle.load(f)

    if creds and creds.expired and creds.refresh_token:
        print("🔄 Refreshing YouTube access token...")
        try:
            creds.refresh(Request())
        except Exception as e:
            print(f"⚠️ Could not refresh token: {e}")
            creds = None

    if not creds or not creds.valid:
        if not os.path.exists(client_secrets):
            raise FileNotFoundError(f"❌ Missing '{client_secrets}'.")

        print("--- AUTHENTICATION REQUIRED ---")
        flow = InstalledAppFlow.from_client_secrets_file(client_secrets, SCOPES)
        creds = flow.run_local_server(port=0)

        with open(pickle_file, "wb") as f:
            pickle.dump(creds, f)

    return googleapiclient.discovery.build("youtube", "v3", credentials=creds)


def upload_to_youtube(video_path: str, title: str, description: str, tags: list) -> bool:
    """Uploads the specified MP4 video to YouTube with provided metadata."""
    try:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        print(f"📤 Uploading '{os.path.basename(video_path)}'...")
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
                    "privacyStatus": "public",
                    "selfDeclaredMadeForKids": False
                }
            },
            media_body=MediaFileUpload(video_path, chunksize=-1, resumable=True, mimetype="video/mp4")
        )

        response = request.execute()
        print(f"✅ Upload successful! URL: https://www.youtube.com/watch?v={response.get('id')}")
        return True

    except Exception as e:
        print(f"❌ YouTube Upload Error: {e}")
        return False


def build_metadata(script_title: str, custom_caption: str = None) -> tuple[str, str, list[str]]:
    """Generates Youtube title, description, and tags based on script topic."""
    base_text = custom_caption if (custom_caption and custom_caption.strip()) else script_title

    is_dark_psych = any(
        k in base_text.lower() or k in script_title.lower()
        for k in ["dark", "psychology", "manipulation", "mind"]
    )

    if is_dark_psych:
        hashtags = ["darkpsychology", "manipulation", "mindcontrol", "psychologyfacts", "shorts"]
        emoji = "🧠👁️"
    else:
        hashtags = ["psychologyfacts", "humanbehavior", "mindset", "mentalhealth", "shorts"]
        emoji = "🧠💡"

    title = f"{base_text} {emoji} #shorts"
    if len(title) > 100:
        title = title[:90].strip() + "... #shorts"

    formatted_hashtags = " ".join([f"#{tag}" for tag in hashtags])
    description = f"{base_text}\n\n{formatted_hashtags}\n\n🧠 Deep Human Insights\n🚀 Subscribe for daily mental loops."

    return title, description, hashtags


def run_automation():
    """Main function to scan JSON queue, generate video, upload, and update JSON."""
    json_file = os.path.join(BASE_DIR, "mindscribble_comic", "mindscribble.json")
    assets_dir = os.path.join(BASE_DIR, "asset_library")
    audio_dir = os.path.join(BASE_DIR, "mindscribble_comic", "voiceovers")
    output_dir = os.path.join(BASE_DIR, "mindscribble_comic", "output")
    bgm_dir = os.path.join(BASE_DIR, "background_music", "mindscribble")

    if not os.path.exists(json_file):
        print(f"❌ JSON file missing: {json_file}")
        return

    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    target_index = None
    target_item = None

    # Locate the first unposted script entry
    for idx, item in enumerate(data):
        is_posted = item.get("posted", False)
        if is_posted is False or str(is_posted).lower() == "false":
            target_index = idx
            target_item = item
            break

    if target_item is None:
        print("✅ All queued scripts have been posted!")
        return

    script_title = target_item.get("script_title", "").strip()
    print(f"\n🚀 Processing [{target_index + 1}/{len(data)}]: '{script_title}'")

    try:
        generated_video_path = process_script_item(
            script_data=target_item,
            assets_dir=assets_dir,
            audio_dir=audio_dir,
            output_dir=output_dir,
            bgm_dir=bgm_dir,
            bgm_volume=0.12,
            transition_type=target_item.get("transition_type", "zoom_dissolve"),
            transition_duration=float(target_item.get("transition_duration", 0.6))
        )
    except Exception as exc:
        print(f"❌ Video Generation Failed: {exc}")
        return

    title, description, tags = build_metadata(script_title, target_item.get("caption"))

    if upload_to_youtube(generated_video_path, title, description, tags):
        data[target_index]["posted"] = True

        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

        print(f"💾 Updated JSON: Marked '{script_title}' as posted.")

        if os.path.exists(generated_video_path):
            os.remove(generated_video_path)
            print("🧹 Cleaned up temporary video.")
    else:
        print("❌ Upload failed. JSON state left unchanged.")


if __name__ == "__main__":
    run_automation()