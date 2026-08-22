import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import re
import glob
import math

from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.video.VideoClip import ImageClip
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.video import fx as vfx
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip
from moviepy import concatenate_videoclips

# Import modules from D:\AI\Automate-Video-Generator\modules
from modules.caption_generator import (
    CONFIG_STYLE_2,
    generate_caption_overlay,
    align_timeline_with_audio  # Dynamically aligns timestamps
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

def loop_video_to_duration(source_clip, target_duration, crossfade_duration=0.5):
    """
    Safely loops a video clip with a crossfade transition between loops
    to cover target_duration seamlessly.
    """
    if source_clip.duration >= target_duration:
        return source_clip.subclipped(0, target_duration)

    # Determine crossfade length (ensure it doesn't exceed half the clip's length)
    cf_dur = min(crossfade_duration, source_clip.duration / 2.0)
    effective_unit_duration = source_clip.duration - cf_dur

    # Calculate exact repetitions needed including overlap loss
    repeats = math.ceil((target_duration - cf_dur) / effective_unit_duration) + 1

    looped_clips = []
    current_start = 0.0

    for i in range(repeats):
        clip = source_clip.copy()
        if i > 0 and cf_dur > 0:
            # Fade in each looped iteration over the tail of the previous iteration
            clip = clip.with_effects([vfx.CrossFadeIn(cf_dur)])
        
        clip = clip.with_start(current_start)
        looped_clips.append(clip)
        current_start += effective_unit_duration

    # Composite overlapping looped clips together into a single continuous track
    composite = CompositeVideoClip(looped_clips)
    return composite.subclipped(0, target_duration)

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

    # Dynamically align timeline timestamps using Whisper word transcription
    print("[+] Calculating timeline timestamps directly from audio...")
    timeline = align_timeline_with_audio(audio_path, timeline)

    media_clips = []
    background = None
    caption_overlay = None

    try:
        num_segments = len(timeline)
        
        for idx, segment in enumerate(timeline):
            folder_name = segment.get("folder")
            start_time = segment.get("start", 0.0)
            
            # Calculate next segment start to absorb voiceover silence gaps
            if idx < num_segments - 1:
                next_start = timeline[idx + 1].get("start", audio_duration)
                base_duration = next_start - start_time
            else:
                base_duration = audio_duration - start_time

            # Add transition padding to absorb crossfade overlap time loss
            # Every clip except the first loses transition_duration during build_transitioned_timeline
            clip_duration = base_duration + transition_duration

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

            print(f"\nProcessing Segment {idx + 1}/{num_segments}: {folder_name} (Calculated Target Dur: {clip_duration:.2f}s)")

            if ext == ".mp4":
                try:
                    source_video = VideoFileClip(file_path, audio=False)
                except Exception as exc:
                    print(f"Skipping unreadable video: {os.path.basename(file_path)} ({exc})")
                    continue

                source_video = source_video.with_fps(30)
                media_clip = loop_video_to_duration(source_video, clip_duration)
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

        # Ensure background matches exact audio duration cleanly
        background = background.with_duration(audio_duration)

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
        if 'audio_clip' in locals() and audio_clip is not None:
            audio_clip.close()
        if background is not None:
            background.close()
        if caption_overlay is not None:
            caption_overlay.close()
        for m in media_clips:
            if m is not None:
                m.close()


if __name__ == "__main__":
    sample_script = {
        "id": 1,
        "script_title": "Testing File",
        "total_segments": 3,
        "posted": False,
        "timeline": [
            {
                "segment_id": 1,
                "folder": "16_podcaster_hoody",
                "text": "Procrastination is almost never about laziness—it is an emotional regulation problem."
            },
            {
                "segment_id": 2,
                "folder": "04_anxiety_overwhelm",
                "text": "When you avoid starting a project, your brain isn't rejecting the work; it’s rejecting the fear of failure attached to it."
            },
            {
                "segment_id": 3,
                "folder": "01_brain_anatomy",
                "text": "Your amygdala perceives creative uncertainty as an immediate physical threat,"
            }
        ]
    }

    process_script_item(
        script_data=sample_script,
        assets_dir=r"D:\AI\Automate-Video-Generator\asset_library",
        audio_dir=r"D:\AI\Automate-Video-Generator\mindscribble_podcaster\voiceovers",
        output_dir=r"D:\AI\Automate-Video-Generator\mindscribble_podcaster",
        transition_type="zoom_dissolve",
        transition_duration=0.6,
        caption_config=CONFIG_STYLE_2
    )