import os
import glob
from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.video.VideoClip import ImageClip
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip
from moviepy import VideoFileClip, concatenate_videoclips
from caption_generator import generate_caption_overlay, STYLE_CONFIG

def run_test_pipeline(audio_path, image_folder, output_path="test_output.mp4"):
    print("--- STARTING TEST PIPELINE ---")
    
    # 1. Load Audio and get duration
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
    audio_clip = AudioFileClip(audio_path)
    audio_duration = audio_clip.duration
    print(f"Loaded Audio Duration: {audio_duration:.2f} seconds")

    # 2. Gather Images from specified directory
    valid_extensions = ("*.jpg", "*.jpeg", "*.png", "*.webp")
    image_files = []
    for ext in valid_extensions:
        image_files.extend(glob.glob(os.path.join(image_folder, ext)))
        
    image_files = image_files[:10]  # Limit to first 10 images for testing
    
    if not image_files:
        raise FileNotFoundError(f"No image files found in folder: {image_folder}")
        
    print(f"Found {len(image_files)} image(s) for background visuals.")

    # 3. Create Background Visual Track matching audio length
    # Equal time distribution across collected images
    clip_duration = audio_duration / len(image_files)
    image_clips = []

    for img_path in image_files:
        # Load image, scale/crop to vertical 1080x1920 format
        img_clip = (
            ImageClip(img_path)
            .with_duration(clip_duration)
            .resized(height=1920)  # Scale to vertical height
        )
        
        # Crop horizontally if wider than 1080
        if img_clip.w > 1080:
            crop_x = (img_clip.w - 1080) // 2
            img_clip = img_clip.cropped(x1=crop_x, width=1080)
        
        image_clips.append(img_clip)

    background_clip = concatenate_videoclips(image_clips, method="compose")

    # 4. Generate Caption Overlay
    print("Generating Whisper word timestamps and white-card overlay...")
    caption_overlay = generate_caption_overlay(audio_path, config=STYLE_CONFIG)

    # 5. Composite Final Layers
    print("Compositing background + caption overlay...")
    final_video = CompositeVideoClip(
        [background_clip, caption_overlay],
        size=(1080, 1920)
    ).with_audio(audio_clip).with_duration(audio_duration)

    # 6. Render MP4
    print(f"Exporting video to: {output_path}")
    final_video.write_videofile(
        output_path,
        fps=30,
        codec="libx264",
        audio_codec="aac",
        preset="fast"
    )
    
    # Clean up memory
    audio_clip.close()
    final_video.close()
    print("--- TEST PIPELINE COMPLETED SUCCESSFULLY ---")

if __name__ == "__main__":
    # Update these paths to point to a test audio and an asset directory with images
    TEST_AUDIO = "voiceovers/testing.mp3"
    TEST_IMAGE_DIR = "D:\\AI\\Automate-Video-Generator\\images\\mindscribble"
    
    run_test_pipeline(TEST_AUDIO, TEST_IMAGE_DIR)