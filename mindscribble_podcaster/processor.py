import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import re
import glob

from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.video.VideoClip import ImageClip
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip
import moviepy.video.fx as vfx

# Import modules from D:\AI\Automate-Video-Generator\modules
from modules.caption_generator import (
    CONFIG_STYLE_2,
    generate_caption_overlay
)

from modules.transitions import (
    build_transitioned_timeline
)


def sanitize_filename(name: str) -> str:
    clean_name = re.sub(r'[\\/*?:"<>|]', "", name)
    return re.sub(r'\s+', ' ', clean_name).strip()


def find_audio_file(audio_dir: str, target_title: str) -> str:
    clean_target = sanitize_filename(target_title).lower()
    if not os.path.exists(audio_dir):
        raise FileNotFoundError(f"Audio directory does not exist: {os.path.abspath(audio_dir)}")

    for file in os.listdir(audio_dir):
        if file.lower().endswith(".mp3"):
            file_stem = os.path.splitext(file)[0]
            clean_file_stem = re.sub(r'\s+', ' ', file_stem).strip().lower()
            if clean_file_stem == clean_target:
                return os.path.join(audio_dir, file)

    raise FileNotFoundError(f"Audio file matching '{target_title}' not found in '{audio_dir}'.")


def normalize_media_dimensions(clip, target_w=1080, target_h=1920):
    clip = clip.resized(height=target_h)

    if clip.w > target_w:
        crop_x = (clip.w - target_w) / 2
        clip = clip.cropped(x1=crop_x, width=target_w)
    elif clip.w < target_w:
        clip = clip.resized(width=target_w)

    if clip.h > target_h:
        crop_y = (clip.h - target_h) / 2
        clip = clip.cropped(y1=crop_y, height=target_h)

    return clip


def process_script_item(
    script_data: dict,
    assets_dir: str,
    audio_dir: str,
    output_dir: str = "output",
    transition_type: str = "zoom_dissolve",
    transition_duration: float = 0.6,
    caption_config: dict = CONFIG_STYLE_2
) -> str:
    print("==========================================")
    print(" STARTING PRODUCTION PIPELINE")
    print("==========================================")

    if isinstance(script_data, tuple):
        script_data = script_data[0]

    script_title = script_data.get("script_title")
    timeline = script_data.get("timeline", [])

    if not script_title or not timeline:
        raise ValueError("Script object must contain 'script_title' and 'timeline'.")

    os.makedirs(output_dir, exist_ok=True)

    audio_path = find_audio_file(audio_dir, script_title)
    audio_clip = AudioFileClip(audio_path)
    audio_duration = audio_clip.duration
    print(f"[+] Audio loaded: {os.path.basename(audio_path)} ({audio_duration:.2f}s)")

    media_clips = []

    try:
        for idx, segment in enumerate(timeline):
            folder_name = segment.get("folder")
            start_time = segment.get("start")
            end_time = segment.get("end")
            clip_duration = end_time - start_time

            segment_folder_path = os.path.join(assets_dir, folder_name)
            if not os.path.exists(segment_folder_path):
                raise FileNotFoundError(f"Asset folder missing: {os.path.abspath(segment_folder_path)}")

            valid_extensions = ("*.jpg", "*.jpeg", "*.mp4", "*.png", "*.webp")
            media_files = []
            for ext in valid_extensions:
                media_files.extend(glob.glob(os.path.join(segment_folder_path, ext)))

            if not media_files:
                raise FileNotFoundError(f"No media files found in: {segment_folder_path}")

            file_path = sorted(media_files)[0]
            ext = os.path.splitext(file_path)[1].lower()

            print(f"\nProcessing Segment {idx + 1}: {folder_name} ({clip_duration:.2f}s)")

            if ext == ".mp4":
                try:
                    source_video = VideoFileClip(file_path, audio=False)
                except Exception as exc:
                    print(f"Skipping unreadable video: {os.path.basename(file_path)} ({exc})")
                    continue

                source_video = source_video.with_fps(30)

                if source_video.duration < clip_duration:
                    source_video = source_video.with_effects([vfx.Loop(duration=clip_duration)])

                media_clip = source_video.subclipped(0, clip_duration).with_duration(clip_duration)
            else:
                media_clip = ImageClip(file_path).with_duration(clip_duration)

            media_clip = normalize_media_dimensions(media_clip, target_w=1080, target_h=1920)
            media_clip = media_clip.with_duration(clip_duration)

            media_clips.append(media_clip)

        print("\n[+] Building transition timeline...")
        background = build_transitioned_timeline(
            media_clips,
            transition_type=transition_type,
            duration=transition_duration,
            size=(1080, 1920),
            final_duration=audio_duration
        )

        print("[+] Generating Whisper caption overlay...")
        caption_overlay = generate_caption_overlay(
            audio_path,
            config=caption_config
        )

        print("[+] Compositing background + captions...")
        final_video = (
            CompositeVideoClip(
                [background, caption_overlay],
                size=(1080, 1920)
            )
            .with_audio(audio_clip)
            .with_duration(audio_duration)
        )

        clean_output_name = sanitize_filename(script_title)
        output_filepath = os.path.abspath(os.path.join(output_dir, f"{clean_output_name}.mp4"))

        print(f"\n[+] Rendering final video to: {output_filepath}")
        final_video.write_videofile(
            output_filepath,
            fps=30,
            codec="libx264",
            audio_codec="aac",
            preset="medium"
        )

        print("==========================================")
        print(" VIDEO COMPLETE")
        print("==========================================")
        return output_filepath

    finally:
        if 'audio_clip' in locals():
            audio_clip.close()
        if 'background' in locals():
            background.close()
        if 'caption_overlay' in locals():
            caption_overlay.close()
        for m in media_clips:
            m.close()
if __name__ == "__main__":
    sample_script ={
        "id": 2,
        "script_title": "Why You Procrastinate on Things You Want Copy",
        "total_segments": 11,
        "posted": False,
        "timeline": [
            {
                "segment_id": 1,
                "start": 0.00,
                "end": 3.50,
                "folder": "16_podcaster_hoody",
                "text": "Procrastination is almost never about laziness—it is an emotional regulation problem."
            },
            {
                "segment_id": 2,
                "start": 3.50,
                "end": 7.20,
                "folder": "04_anxiety_overwhelm",
                "text": "When you avoid starting a project, your brain isn't rejecting the work; it’s rejecting the fear of failure attached to it."
            },
            {
                "segment_id": 3,
                "start": 7.20,
                "end": 10.90,
                "folder": "01_brain_anatomy",
                "text": "Your amygdala perceives creative uncertainty as an immediate physical threat..."
            },
            {
                "segment_id": 4,
                "start": 10.90,
                "end": 14.50,
                "folder": "13_addiction_dopamine",
                "text": "...hijacking your logical prefrontal cortex and driving you toward cheap dopamine like social media."
            },
            {
                "segment_id": 5,
                "start": 14.50,
                "end": 18.00,
                "folder": "16_podcaster_hoody",
                "text": "This creates an instant relief loop that tricks your brain into thinking avoidance is safety."
            }
        ]
    }
    process_script_item(
        script_data=sample_script,
        assets_dir=r"D:\AI\Automate-Video-Generator\asset_library",
        audio_dir=r"D:\AI\Automate-Video-Generator\mindscribble_podcaster\voiceover",
        output_dir=r"D:\AI\Automate-Video-Generator\mindscribble_podcaster",
        transition_type="zoom_dissolve",  # Options: cross_dissolve, blur_dissolve, dip_to_black, zoom_dissolve, slide_left, etc.
        transition_duration=0.6,
        caption_config=CONFIG_STYLE_2     # Use CONFIG_STYLE_1 or CONFIG_STYLE_2
    )