import os
import glob

from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.video.VideoClip import ImageClip
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip
import moviepy.video.fx as vfx

from modules.caption_generator import CONFIG_STYLE_2, generate_caption_overlay
from modules.transitions import build_transitioned_timeline


def run_test_pipeline(
    audio_path,
    image_folder,
    output_path="test_output.mp4",
    transition_type="blur_dissolve",
    transition_duration=0.6
):
    print("==========================================")
    print(" STARTING TEST PIPELINE")
    print("==========================================")

    # --------------------------------------------------------
    # 1. AUDIO
    # --------------------------------------------------------
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    audio_clip = AudioFileClip(audio_path)
    audio_duration = audio_clip.duration
    print(f"Audio duration: {audio_duration:.2f}s")

    # --------------------------------------------------------
    # 2. FIND IMAGES AND VIDEOS
    # --------------------------------------------------------
    valid_extensions = ("*.jpg", "*.jpeg", "*.mp4", "*.png", "*.webp")
    image_files = []

    for ext in valid_extensions:
        image_files.extend(
            glob.glob(os.path.join(image_folder, ext))
        )

    image_files.sort()
    image_files = image_files[:3]  # Limit to 3 files for testing

    if not image_files:
        raise FileNotFoundError(f"No media found in: {image_folder}")

    print(f"Found {len(image_files)} media files")

    # --------------------------------------------------------
    # 3. CALCULATE CLIP DURATION
    # --------------------------------------------------------
    num_images = len(image_files)
    if num_images == 1:
        clip_duration = audio_duration
    else:
        clip_duration = (
            audio_duration + ((num_images - 1) * transition_duration)
        ) / num_images

    print(f"Each clip duration: {clip_duration:.2f}s")
    print(f"Transition duration: {transition_duration:.2f}s")
    
    # --------------------------------------------------------
    # 4. LOAD AND NORMALIZE CLIPS
    # --------------------------------------------------------
    media_clips = []

    for index, file_path in enumerate(image_files):
        ext = os.path.splitext(file_path)[1].lower()

        if ext == ".mp4":
            vid_clip = VideoFileClip(file_path).without_audio()
            
            # Loop if shorter than required duration
            if vid_clip.duration < clip_duration:
                vid_clip = vid_clip.with_effects([vfx.Loop(duration=clip_duration)])
            
            # HARD TRUNCATE video duration property so transitions math works
            media_clip = vid_clip.subclipped(0, clip_duration).with_duration(clip_duration).resized(height=1920)
        else:
            media_clip = (
                ImageClip(file_path)
                .with_duration(clip_duration)
                .resized(height=1920)
            )

        # Center Crop / Fit to 1080x1920 Vertical Canvas
        if media_clip.w > 1080:
            crop_x = (media_clip.w - 1080) / 2
            media_clip = media_clip.cropped(x1=crop_x, width=1080)
        elif media_clip.w < 1080:
            media_clip = media_clip.resized(width=1080)
            if media_clip.h > 1920:
                crop_y = (media_clip.h - 1920) / 2
                media_clip = media_clip.cropped(y1=crop_y, height=1920)

        media_clips.append(media_clip)

    # --------------------------------------------------------
    # 5. BUILD TRANSITION TIMELINE
    # --------------------------------------------------------
    background = build_transitioned_timeline(
        media_clips,
        transition_type=transition_type,
        duration=transition_duration,
        size=(1080, 1920),
        final_duration=audio_duration
    )

    # --------------------------------------------------------
    # 6. GENERATE CAPTION OVERLAY & COMPOSITE
    # --------------------------------------------------------
    print("Generating Whisper caption overlay...")
    caption_overlay = generate_caption_overlay(audio_path, config=CONFIG_STYLE_2)

    print("Compositing background + caption overlay...")
    final_video = CompositeVideoClip(
        [background, caption_overlay],
        size=(1080, 1920)
    ).with_audio(audio_clip).with_duration(audio_duration)

    # --------------------------------------------------------
    # 7. EXPORT
    # --------------------------------------------------------
    print(f"Final video duration: {final_video.duration:.2f}s")
    print("Rendering...")

    final_video.write_videofile(
        output_path,
        fps=30,
        codec="libx264",
        audio_codec="aac",
        preset="medium"
    )

    print("==========================================")
    print(" VIDEO COMPLETE")
    print(f" Output: {output_path}")
    print("==========================================")

    # Close resources
    audio_clip.close()
    final_video.close()


if __name__ == "__main__":
    TEST_AUDIO = "voiceovers/testing.mp3"
    TEST_IMAGE_DIR = r"D:\AI\Automate-Video-Generator\asset_library\05_internal_conflict"

    run_test_pipeline(
        TEST_AUDIO,
        TEST_IMAGE_DIR,
        output_path="test_transition_output.mp4",
        transition_type="blur_dissolve",
        transition_duration=0.6
    )