import os
import sys
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import re
import glob
import math
import random
import cv2
import numpy as np

from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.audio.AudioClip import CompositeAudioClip
from moviepy.video.VideoClip import ImageClip, VideoClip
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip
import moviepy.video.fx as vfx
from moviepy import concatenate_videoclips

from modules.caption_generator import (
    CONFIG_STYLE_2,
    generate_caption_overlay,
    align_timeline_with_audio
)
from modules.transitions import build_transitioned_timeline


def resolve_project_path(*path_parts: str) -> str:
    path_inside = os.path.abspath(os.path.join(BASE_DIR, *path_parts))
    if os.path.exists(path_inside):
        return path_inside

    path_parent = os.path.abspath(os.path.join(BASE_DIR, "..", *path_parts))
    return path_parent


DEFAULT_POP_SFX = resolve_project_path("audio", "sound_effect", "dragon-pop.mp3")
DEFAULT_STICKERS_DIR = resolve_project_path("stickers")
DEFAULT_ASSETS_DIR = resolve_project_path("asset_library")


def sanitize_filename(name: str) -> str:
    clean_name = re.sub(r'[\\/*?:"<>|]', "", name)
    return re.sub(r'\s+', ' ', clean_name).strip()


def find_audio_file(audio_dir: str, target_title: str) -> str:
    clean_target = sanitize_filename(target_title).lower()
    if not os.path.exists(audio_dir):
        raise FileNotFoundError(f"Audio directory does not exist: {os.path.abspath(audio_dir)}")

    for file in os.listdir(audio_dir):
        if file.lower().endswith((".mp3", ".wav", ".m4a")):
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
    if source_clip.duration >= target_duration:
        return source_clip.subclipped(0, target_duration)

    cf_dur = min(crossfade_duration, source_clip.duration / 2.0)
    effective_unit_duration = source_clip.duration - cf_dur

    repeats = math.ceil((target_duration - cf_dur) / effective_unit_duration) + 1

    looped_clips = []
    current_start = 0.0

    for i in range(repeats):
        clip = source_clip.copy()
        if i > 0 and cf_dur > 0:
            clip = clip.with_effects([vfx.CrossFadeIn(cf_dur)])

        clip = clip.with_start(current_start)
        looped_clips.append(clip)
        current_start += effective_unit_duration

    composite = CompositeVideoClip(looped_clips)
    return composite.subclipped(0, target_duration)


def create_feathered_rotated_sticker(
    image_path: str,
    target_width: int = 420,
    rotation_angle: float = 0.0,
    feather_radius: int = 15
) -> np.ndarray:
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not load image at path: {image_path}")

    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    elif img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

    h, w = img.shape[:2]
    aspect_ratio = target_width / float(w)
    target_height = max(1, int(h * aspect_ratio))
    img = cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)

    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    f = max(1, feather_radius)
    if f < min(h, w) // 2:
        cv2.rectangle(mask, (f, f), (w - f, h - f), 255, -1)
        mask = cv2.GaussianBlur(mask, (f * 2 + 1, f * 2 + 1), 0)
    else:
        mask.fill(255)

    img[:, :, 3] = (img[:, :, 3].astype(float) * (mask.astype(float) / 255.0)).astype(np.uint8)

    if rotation_angle != 0:
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)

        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))

        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]

        img = cv2.warpAffine(
            img, M, (new_w, new_h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0)
        )

    return cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
def load_floating_sticker_clip(
    sticker_path: str,
    duration: float,
    start_time: float,
    target_width: int = 420,
    position: tuple = (580, 350),
    rotation_angle: float = 0.0,
    feather_radius: int = 15
):
    # 1. Load feathered, rotated RGBA image matrix
    rgba_matrix = create_feathered_rotated_sticker(
        image_path=sticker_path,
        target_width=target_width,
        rotation_angle=rotation_angle,
        feather_radius=feather_radius
    )

    base_x, base_y = float(position[0]), float(position[1])

    # 2. Dynamic Zoom-In & Zoom-Out on the full RGBA array together
    def make_frame(t):
        # Oscillates smoothly between 1.0 (100%) and 1.08 (108%)
        scale = 1.0 + 0.01 * (1.0 + math.sin(t * 1.5))

        h, w = rgba_matrix.shape[:2]
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))

        # Scale RGB and Alpha simultaneously to prevent boundary clipping/shadows
        return cv2.resize(rgba_matrix, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    # 3. Construct VideoClip directly from animated RGBA frames
    full_rgba_clip = VideoClip(make_frame, is_mask=False, duration=duration)

    rgb_clip = full_rgba_clip.image_transform(lambda frame: frame[:, :, :3])
    alpha_clip = full_rgba_clip.image_transform(
        lambda frame: (frame[:, :, 3] / 255.0).astype(np.float32)
    )

    # 4. Attach dynamically animated mask to dynamically animated RGB clip
    final_sticker_clip = (
        rgb_clip
        .with_mask(alpha_clip)
        .with_position((base_x, base_y))
        .with_start(start_time)
        .with_duration(duration)
    )

    return final_sticker_clip

def generate_segment_sticker_overlays(
    stickers_list: list,
    segment_start_time: float,
    segment_duration: float,
    stickers_dir: str = DEFAULT_STICKERS_DIR,
    sound_effects_list: list = None,
    pop_sound_path: str = DEFAULT_POP_SFX
) -> list:
    """Generates independent sticker overlay clips for a given timeline segment."""
    if not stickers_list:
        return []

    generated_stickers = []
    total_stickers = len(stickers_list)

    STICKER_WIDTH_SINGLE = 650
    STICKER_WIDTH_MULTI = 500
    TOP_Y_POS = 300

    for idx, item in enumerate(stickers_list):
        filename = item.get("file")
        delay = float(item.get("delay", 0.0))
        abs_start_time = segment_start_time + delay

        if total_stickers == 1:
            scale = STICKER_WIDTH_SINGLE
            side = random.choice(["left", "right"])
            pos_x = 50 if side == "left" else (1080 - scale - 50)
            assigned_angle = 15.0 if side == "left" else -15.0
            pos_y = TOP_Y_POS
        elif total_stickers == 2:
            scale = STICKER_WIDTH_MULTI
            margin = 130  
            pos_x = margin if idx == 0 else (1080 - scale - margin)
            assigned_angle = 8.0 if idx == 0 else -8.0
            pos_y = TOP_Y_POS
        else:
            scale = STICKER_WIDTH_MULTI
            spacing = (1080 - (scale * total_stickers)) // (total_stickers + 1)
            pos_x = max(20, spacing + idx * (scale + spacing))
            pos_y = TOP_Y_POS
            assigned_angle = 8.0 if idx == 0 else (-8.0 if idx == 1 else 0.0)

        position = (pos_x, pos_y)
        sticker_path = os.path.join(stickers_dir, filename)

        if not os.path.exists(sticker_path) or delay >= segment_duration:
            print(f"[!] Sticker missing or offset beyond segment duration: {sticker_path}")
            continue

        sticker_duration = max(0.1, segment_duration - delay)

        try:
            sticker_clip = load_floating_sticker_clip(
                sticker_path=sticker_path,
                target_width=scale,
                duration=sticker_duration,
                start_time=abs_start_time,
                position=position,
                rotation_angle=assigned_angle,
                feather_radius=15
            )

            generated_stickers.append(sticker_clip)

            if sound_effects_list is not None and pop_sound_path and os.path.exists(pop_sound_path):
                sfx_clip = AudioFileClip(pop_sound_path).with_start(abs_start_time)
                sound_effects_list.append(sfx_clip)

            print(f"    [+] Top-Level Sticker Prepared: {filename} | Global Start: {abs_start_time:.2f}s | Dur: {sticker_duration:.2f}s")
        except Exception as e:
            print(f"[!] Failed to prepare sticker {filename}: {e}")

    return generated_stickers


def get_random_outro_video() -> str:
    ending_dir = resolve_project_path("ending", "mindscribble")

    if not os.path.exists(ending_dir):
        print(f"[!] Warning: Ending directory not found at: {ending_dir}")
        return None

    valid_extensions = ("*.mp4", "*.mov", "*.mkv", "*.avi", "*.webm")
    outro_files = []
    for ext in valid_extensions:
        outro_files.extend(glob.glob(os.path.join(ending_dir, ext)))

    if not outro_files:
        print(f"[!] Warning: No video files found in: {ending_dir}")
        return None

    selected_outro = random.choice(outro_files)
    print(f"[+] Selected Outro Video: {os.path.basename(selected_outro)}")
    return selected_outro


def process_script_item(
    script_data: dict,
    assets_dir: str = DEFAULT_ASSETS_DIR,
    audio_dir: str = None,
    stickers_dir: str = DEFAULT_STICKERS_DIR,
    pop_sound_path: str = DEFAULT_POP_SFX,
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

    if audio_dir is None:
        audio_dir = resolve_project_path("mindscribble_podcaster", "voiceovers")

    os.makedirs(output_dir, exist_ok=True)

    audio_path = find_audio_file(audio_dir, script_title)
    main_audio = AudioFileClip(audio_path)
    audio_duration = main_audio.duration
    print(f"[+] Audio loaded: {os.path.basename(audio_path)} ({audio_duration:.2f}s)")

    print("[+] Calculating timeline timestamps directly from audio...")
    timeline = align_timeline_with_audio(audio_path, timeline)

    media_clips = []
    all_sticker_clips = []
    sound_effects = [main_audio]
    background = None
    caption_overlay = None
    credit_clip = None

    try:
        num_segments = len(timeline)

        for idx, segment in enumerate(timeline):
            folder_name = segment.get("folder")
            start_time = segment.get("start", 0.0)
            stickers_list = segment.get("stickers", [])

            if idx < num_segments - 1:
                next_start = timeline[idx + 1].get("start", audio_duration)
                base_duration = next_start - start_time
            else:
                base_duration = audio_duration - start_time

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

            print(f"\nProcessing Segment {idx + 1}/{num_segments}: {folder_name} (Base Dur: {base_duration:.2f}s)")

            # 1. Prepare Base Video/Image (NO STICKERS ATTACHED YET)
            if ext == ".mp4":
                try:
                    source_video = VideoFileClip(file_path, audio=False)
                except Exception as exc:
                    print(f"Skipping unreadable video: {os.path.basename(file_path)} ({exc})")
                    continue

                source_video = source_video.with_fps(30)
                media_clip = loop_video_to_duration(source_video, clip_duration, crossfade_duration=0.4)
            else:
                media_clip = ImageClip(file_path).with_duration(clip_duration)

            media_clip = normalize_media_dimensions(media_clip, target_w=1080, target_h=1920)
            media_clip = media_clip.with_duration(clip_duration)
            media_clips.append(media_clip)

            # 2. Extract Top-Level Sticker Overlays
            if stickers_list:
                stickers_for_segment = generate_segment_sticker_overlays(
                    stickers_list=stickers_list,
                    segment_start_time=start_time,
                    segment_duration=base_duration,
                    stickers_dir=stickers_dir,
                    sound_effects_list=sound_effects,
                    pop_sound_path=pop_sound_path
                )
                all_sticker_clips.extend(stickers_for_segment)

        # 3. Render Background Transitions
        print("\n[+] Building background transition timeline...")
        background = build_transitioned_timeline(
            media_clips,
            transition_type=transition_type,
            duration=transition_duration,
            size=(1080, 1920),
            final_duration=audio_duration
        )
        background = background.with_duration(audio_duration)

        # 4. Generate Caption Overlay
        print("[+] Generating Whisper caption overlay...")
        caption_overlay = generate_caption_overlay(
            audio_path,
            config=caption_config
        )

        # 5. Composite Final Layer Stack: Background -> Stickers -> Captions
        print(f"[+] Compositing Final Stack: Background + {len(all_sticker_clips)} Top-Level Sticker(s) + Caption Overlay")
        final_video_layers = [background] + all_sticker_clips + [caption_overlay]

        final_audio = CompositeAudioClip(sound_effects)

        final_video = (
            CompositeVideoClip(
                final_video_layers,
                size=(1080, 1920),
                bg_color=(0, 0, 0)
            )
            .with_audio(final_audio)
            .with_duration(audio_duration)
        )

        # 6. Append Credit / Outro Video
        credit_video_path = get_random_outro_video()
        if credit_video_path and os.path.exists(credit_video_path):
            try:
                print(f"[+] Appending Outro Video: {os.path.basename(credit_video_path)}")
                credit_clip = VideoFileClip(credit_video_path)
                credit_clip = normalize_media_dimensions(credit_clip, target_w=1080, target_h=1920)

                final_video = concatenate_videoclips(
                    [final_video, credit_clip],
                    method="compose"
                )
            except Exception as e:
                print(f"[!] Warning: Could not append credit video ({e})")

        clean_output_name = sanitize_filename(script_title)
        output_filepath = os.path.abspath(os.path.join(output_dir, f"{clean_output_name}_gpt.mp4"))

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
        if 'main_audio' in locals() and main_audio is not None:
            main_audio.close()
        if background is not None:
            background.close()
        if caption_overlay is not None:
            caption_overlay.close()
        if credit_clip is not None:
            credit_clip.close()
        for m in media_clips:
            if m is not None:
                m.close()
        for s in all_sticker_clips:
            if s is not None:
                s.close()


if __name__ == "__main__":
    sample_script = {
        "id": 4,
        "script_title": "Testing Decoy",
        "total_segments": 10,
        "posted": False,
        "timeline": [
            {
                "segment_id": 1,
                "folder": "16_podcaster_hoody",
                "text": "Companies use an asymmetric choice model to manipulate your brain into spending more money.",
                "stickers": [
                    { "file": "money_trap__transparent.png", "delay": 0.2 }
                ]
            },
            {
                "segment_id": 2,
                "folder": "09_traps_and_mazes",
                "text": "It’s called the Decoy Effect, and it triggers an automatic flaw in human value judgment.",
                "stickers": []
            },
            {
                "segment_id": 3,
                "folder": "08_time_mechanisms",
                "text": "When offered a small option for three dollars and a large for seven, most people choose the small.",
                "stickers": [
                    { "file": "popcorn_small.png", "delay": 0.3 },
                    { "file": "popcorn_large.png", "delay": 1.5 }
                ]
            }
        ]
    }

    process_script_item(
        script_data=sample_script,
        assets_dir=r"D:\AI\Automate-Video-Generator\asset_library",
        audio_dir=r"D:\AI\Automate-Video-Generator\mindscribble_podcaster\voiceovers",
        stickers_dir=r"D:\AI\Automate-Video-Generator\stickers",
        output_dir=r"D:\AI\Automate-Video-Generator\mindscribble_podcaster",
        transition_type="zoom_dissolve",
        transition_duration=0.6,
        caption_config=CONFIG_STYLE_2
    )