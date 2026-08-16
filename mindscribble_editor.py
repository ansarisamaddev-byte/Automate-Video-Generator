import pandas as pd
import numpy as np
import random
import os
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps
from moviepy import (
    AudioFileClip,
    ImageClip,
    CompositeVideoClip,
    CompositeAudioClip
)
from faster_whisper import WhisperModel

# ---------------- CORE CONFIG ---------------- #

model = WhisperModel("tiny", device="cpu", compute_type="int8", cpu_threads=4)

SCREEN_W = 1080
SCREEN_H = 1920

COLOR_GREEN_BG = (38, 64, 52, 245)      # Dark green highlighter box
COLOR_WHITE = (245, 245, 245)
COLOR_RED_HEADING = (245, 68, 68)
COLOR_UNDERLINE = (255, 255, 255)       # Active word underline

FONT_HEADING = "fonts/dejavu-sans-bold.ttf"
FONT_CAPTION = "fonts/dejavu-sans-bold.ttf"


# ---------------- AUDIO & IMAGE PROCESSORS ---------------- #

def load_random_bgm(folder_path):
    """Selects a random audio file from the specified folder."""
    if not folder_path or pd.isna(folder_path):
        return None
    
    clean_path = str(folder_path).strip().replace("\\", "/")
    p_folder = Path(clean_path)
    
    if not p_folder.exists():
        return None
        
    valid_exts = {".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac"}
    audio_files = [str(p) for p in p_folder.glob("*") if p.is_file() and p.suffix.lower() in valid_exts]
    
    return random.choice(audio_files) if audio_files else None


def load_folder_images(folder_path):
    """Loads all valid image paths from a given folder relative to current execution context."""
    if not folder_path or pd.isna(folder_path):
        return []
    
    clean_path = str(folder_path).strip().replace("\\", "/")
    p_folder = Path(clean_path)
    
    if not p_folder.exists():
        return []
        
    valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".jfif"}
    return sorted([str(p) for p in p_folder.glob("*") if p.is_file() and p.suffix.lower() in valid_exts])


def process_bg_image(img_path):
    """Processes background image with extra dark overlay and deep radial vignette shadow."""
    try:
        img = Image.open(img_path).convert("RGBA")
    except Exception:
        img = Image.new("RGBA", (SCREEN_W, SCREEN_H), (20, 20, 20, 255))

    ratio = max(SCREEN_W / img.width, SCREEN_H / img.height) * 1.50
    w_new, h_new = int(img.width * ratio), int(img.height * ratio)
    img = img.resize((w_new, h_new), Image.Resampling.LANCZOS)

    dark_overlay = Image.new("RGBA", (w_new, h_new), (0, 0, 0, 200))
    img = Image.alpha_composite(img, dark_overlay)

    vignette_mask = Image.new("L", (w_new, h_new), 0)
    draw_vignette = ImageDraw.Draw(vignette_mask)
    draw_vignette.ellipse([int(w_new * 0.15), int(h_new * 0.15), int(w_new * 0.85), int(h_new * 0.85)], fill=255)
    vignette_mask = vignette_mask.filter(ImageFilter.GaussianBlur(180))
    vignette_mask = ImageOps.invert(vignette_mask)

    shadow_layer = Image.new("RGBA", (w_new, h_new), (0, 0, 0, 220))
    shadow_layer.putalpha(vignette_mask)

    img = Image.alpha_composite(img, shadow_layer)
    return np.array(img.convert("RGB"))


def process_person_image(img_path):
    """Loads and resizes the main portrait cutout."""
    try:
        img = Image.open(img_path).convert("RGBA")
    except Exception:
        img = Image.new("RGBA", (700, 900), (40, 40, 40, 255))

    target_w = 900
    ratio = target_w / img.width
    target_h = int(img.height * ratio)

    img = img.resize((target_w, target_h), Image.Resampling.LANCZOS)
    return np.array(img)


def process_cutout_image(img_path):
    """Processes character cutout image to fit in the bottom corner."""
    try:
        img = Image.open(img_path).convert("RGBA")
    except Exception:
        img = Image.new("RGBA", (300, 500), (0, 0, 0, 0))

    target_h = int(SCREEN_H * 0.38)
    scale_w = target_h / img.height
    target_w = int(img.width * scale_w)

    img = img.resize((target_w, target_h), Image.Resampling.LANCZOS)
    return np.array(img)


# ---------------- TEXT CARD RENDERERS ---------------- #

def create_heading_card(text):
    """Creates a bold YouTube Shorts-style heading card."""
    font_path = FONT_HEADING.replace("\\", "/")

    try:
        font = ImageFont.truetype(font_path, 105) if os.path.exists(font_path) else ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()

    words = text.upper().split()
    lines, current_line = [], ""
    max_chars = 16

    for word in words:
        test_line = f"{current_line} {word}".strip()
        if len(test_line) > max_chars and current_line:
            lines.append(current_line)
            current_line = word
        else:
            current_line = test_line

    if current_line:
        lines.append(current_line)

    lines = lines[:3]

    temp_img = Image.new("RGBA", (SCREEN_W, 700), (0, 0, 0, 0))
    draw = ImageDraw.Draw(temp_img)
    line_spacing = 10

    line_heights = []
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font, stroke_width=10)
        line_heights.append(bbox[3] - bbox[1])

    total_height = sum(line_heights) + line_spacing * (len(lines) - 1)
    y = (700 - total_height) // 2

    for line, line_h in zip(lines, line_heights):
        bbox = draw.textbbox((0, 0), line, font=font, stroke_width=10)
        text_w = bbox[2] - bbox[0]
        x = (SCREEN_W - text_w) // 2

        # Black drop shadow
        draw.text((x + 10, y + 14), line, font=font, fill=(0, 0, 0, 255), stroke_width=10, stroke_fill=(0, 0, 0, 255))
        # Main text
        draw.text((x, y), line, font=font, fill=(245, 65, 65, 255), stroke_width=8, stroke_fill=(0, 0, 0, 255))

        y += line_h + line_spacing

    # Forward slant effect
    shear_factor = -0.025
    new_width = int(temp_img.width + abs(shear_factor) * temp_img.height)

    temp_img = temp_img.transform(
        (new_width, temp_img.height),
        Image.Transform.AFFINE,
        (1, shear_factor, 0, 0, 1, 0),
        resample=Image.Resampling.BICUBIC
    )

    final_img = Image.new("RGBA", (SCREEN_W, 700), (0, 0, 0, 0))
    x_offset = (SCREEN_W - temp_img.width) // 2
    final_img.alpha_composite(temp_img, (x_offset, 0))

    return np.array(final_img)


def render_evergreen_caption_frame(words_group, active_word_index):
    """Renders Evergreen captions with green line box and active word underline."""
    font_path = FONT_CAPTION.replace("\\", "/")
    try:
        font = ImageFont.truetype(font_path, 54) if os.path.exists(font_path) else ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()

    pad_x, pad_y = 20, 12
    max_line_w = 880
    temp_draw = ImageDraw.Draw(Image.new("RGBA", (1, 1)))

    lines, current_line, current_w = [], [], 0
    for i, w_data in enumerate(words_group):
        w_str = w_data["word"]
        bbox = temp_draw.textbbox((0, 0), w_str + " ", font=font)
        w_len = bbox[2] - bbox[0]

        if current_w + w_len > max_line_w and current_line:
            lines.append(current_line)
            current_line = []
            current_w = 0

        current_line.append((i, w_str, w_len))
        current_w += w_len

    if current_line:
        lines.append(current_line)

    lines = lines[:3]
    line_spacing = 76
    total_h = len(lines) * line_spacing + (pad_y * 2)

    img = Image.new("RGBA", (SCREEN_W, total_h + 100), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    curr_y = pad_y
    for line in lines:
        line_text_w = sum(w_len for _, _, w_len in line)
        box_w = line_text_w + (pad_x * 2)
        start_x = (SCREEN_W - box_w) // 2

        draw.rounded_rectangle(
            [start_x, curr_y, start_x + box_w, curr_y + line_spacing - 8],
            radius=10,
            fill=COLOR_GREEN_BG
        )

        x_pos = start_x + pad_x
        for word_idx, word_str, word_w in line:
            draw.text((x_pos, curr_y + 4), word_str, font=font, fill=COLOR_WHITE)

            if word_idx == active_word_index:
                bbox = temp_draw.textbbox((0, 0), word_str, font=font)
                actual_w = bbox[2] - bbox[0]
                underline_y = curr_y + 58
                draw.line([(x_pos, underline_y), (x_pos + actual_w, underline_y)], fill=COLOR_UNDERLINE, width=5)

            x_pos += word_w

        curr_y += line_spacing

    return np.array(img)


# ---------------- ANIMATION UTILS ---------------- #

def animate_person_clip(person_arr, img_path, start_t, dur):
    """Applies smooth pan, zoom, and fade effects to person overlay."""
    base_clip = (
        ImageClip(person_arr)
        .with_start(start_t)
        .with_duration(dur)
    )

    filename = Path(img_path).stem.lower()
    is_left = "left" in filename
    is_right = "right" in filename
    image_w = person_arr.shape[1]

    center_x = (SCREEN_W - image_w) / 2
    left_x = 20
    right_x = SCREEN_W - image_w - 20

    if is_left:
        start_x = -image_w * 0.10
        end_x = left_x
    elif is_right:
        start_x = SCREEN_W - image_w * 0.90
        end_x = right_x
    else:
        start_x = center_x
        end_x = center_x

    def position_func(t):
        progress = min(1.0, t / dur)
        smooth = 1 - (1 - progress) ** 2
        x_pos = start_x + (end_x - start_x) * smooth
        return (x_pos, 100)

    animated = (
        base_clip
        .resized(lambda t: 1.0 + 0.05 * (t / dur))
        .with_position(position_func)
    )

    return animated


# ---------------- MASTER ENGINE ---------------- #
def generate_video_from_csv(csv_path="mind_scribble2.csv", target_id=1, output_name="rendered_output.mp4"):
    clean_csv_path = str(csv_path).strip().replace("\\", "/")
    df = pd.read_csv(clean_csv_path)

    row = df[df["id"] == target_id].iloc[0]

    audio_file = str(row["audio_path"]).strip().replace("\\", "/")
    bg_folder = str(row["image_folder"]).strip().replace("\\", "/")
    person_folder = str(row["person_image_folder"]).strip().replace("\\", "/")
    cutout_folder = str(row.get("character_folder", row.get("characters_folder", "images/cutouts"))).strip().replace("\\", "/")
    heading_text = str(row["heading"]).strip().replace("\\", "/")

    bg_files = load_folder_images(bg_folder)
    person_files = load_folder_images(person_folder)
    cutout_files = load_folder_images(cutout_folder)

    if not bg_files:
        raise ValueError(f"No background images found in folder: {bg_folder}")

    # Transcribe speech for caption sync
    segments_gen, _ = model.transcribe(audio_file, word_timestamps=True)
    all_words = [w for seg in segments_gen for w in seg.words]

    speech_audio = AudioFileClip(audio_file)

    HEADING_DUR = 1.0
    total_duration = speech_audio.duration + HEADING_DUR

    layer_clips = []
    text_clips = []

    # 1. LAYER 1: ANIMATED BACKGROUND (Random selection)
    bg_interval = 6.0
    bg_count = math.ceil(total_duration / bg_interval)
    for i in range(bg_count):
        t_start = i * bg_interval
        dur = min(bg_interval, total_duration - t_start)
        
        # Pick a random background image
        bg_path = random.choice(bg_files)
        bg_arr = process_bg_image(bg_path)

        clip = ImageClip(bg_arr).with_start(t_start).with_duration(dur)
        zoom_delta = 0.024 * (1 if i % 2 == 0 else -1)
        clip = clip.resized(lambda t: 1.0 + zoom_delta * (t / dur))
        clip = clip.with_position(("center", "center")).cropped(y1=0, y2=SCREEN_H, x1=0, x2=SCREEN_W)
        layer_clips.append(clip)

    # 2. LAYER 2: PERSON OVERLAY (Random selection)
    if person_files:
        person_interval = 10.0
        person_count = math.ceil(total_duration / person_interval)
        for i in range(person_count):
            t_start = i * person_interval
            dur = min(person_interval, total_duration - t_start)
            
            # Pick a random person overlay image
            person_path = random.choice(person_files)

            person_arr = process_person_image(person_path)
            person_clip = animate_person_clip(person_arr, person_path, t_start, dur)
            layer_clips.append(person_clip)

    # 3. LAYER 3: CHARACTER CUTOUT (Random selection)
    if cutout_files:
        cutout_interval = 8.0
        cutout_count = math.ceil(total_duration / cutout_interval)
        for i in range(cutout_count):
            t_start = i * cutout_interval
            dur = min(cutout_interval, total_duration - t_start)
            
            # Pick a random character cutout image
            cutout_path = random.choice(cutout_files)
            c_arr = process_cutout_image(cutout_path)

            cutout_clip = (
                ImageClip(c_arr)
                .with_start(t_start)
                .with_duration(dur)
                .with_position((SCREEN_W - c_arr.shape[1] - 20, SCREEN_H - c_arr.shape[0] - 10))
            )
            layer_clips.append(cutout_clip)

    # 4. LAYER 4: HEADING CARD AT BEGINNING
    heading_img = create_heading_card(heading_text)
    heading_clip = (
        ImageClip(heading_img)
        .with_start(0.0)
        .with_duration(HEADING_DUR)
        .with_position(("center", 350))
    )
    text_clips.append(heading_clip)

    # 5. LAYER 5: SYNCHRONIZED CAPTIONS
    chunk_size = 10
    for i in range(0, len(all_words), chunk_size):
        group_words = all_words[i:i + chunk_size]
        if not group_words:
            continue

        group_data = [{"word": w.word.strip()} for w in group_words]

        for idx, w_obj in enumerate(group_words):
            word_start = HEADING_DUR + w_obj.start
            word_end = HEADING_DUR + (w_obj.end if idx == len(group_words) - 1 else group_words[idx + 1].start)
            word_dur = max(0.1, word_end - word_start)

            frame_arr = render_evergreen_caption_frame(group_data, active_word_index=idx)

            caption_clip = (
                ImageClip(frame_arr)
                .with_start(word_start)
                .with_duration(word_dur)
                .with_position(("center", 760))
            )
            text_clips.append(caption_clip)

    # 6. AUDIO COMPOSITION: VOICE OVER + BACKGROUND MUSIC
    audio_tracks = [speech_audio.with_start(HEADING_DUR)]

    bgm_folder = "background_music/mindscribble"
    bgm_file = load_random_bgm(bgm_folder)

    if bgm_file:
        try:
            print(f"[Audio] Selected BGM track: {bgm_file}")
            bgm_clip = AudioFileClip(bgm_file)

            if bgm_clip.duration < total_duration:
                loop_count = math.ceil(total_duration / bgm_clip.duration)
                bgm_clip = CompositeAudioClip([bgm_clip.with_start(i * bgm_clip.duration) for i in range(loop_count)])

            bgm_clip = bgm_clip.with_duration(total_duration)

            if hasattr(bgm_clip, "multiply_volume"):
                bgm_clip = bgm_clip.multiply_volume(0.5)
            elif hasattr(bgm_clip, "volumex"):
                bgm_clip = bgm_clip.volumex(0.5)
            else:
                from moviepy.audio.fx.MultiplyVolume import MultiplyVolume
                bgm_clip = bgm_clip.with_effects([MultiplyVolume(0.5)])

            audio_tracks.append(bgm_clip.with_start(0.0))
        except Exception as e:
            print(f"[Warning] Failed to mix background music: {e}")

    final_audio = CompositeAudioClip(audio_tracks).with_duration(total_duration)

    video = (
        CompositeVideoClip(layer_clips + text_clips, size=(SCREEN_W, SCREEN_H))
        .with_duration(total_duration)
        .with_audio(final_audio)
    )

    clean_output = str(output_name).strip().replace("\\", "/")
    print(f"[Render] Writing video to: {clean_output}")
    video.write_videofile(clean_output, fps=30, codec="libx264", audio_codec="aac", threads=4, preset="ultrafast")

if __name__ == "__main__":
    generate_video_from_csv("mind_scribble2.csv", target_id=1, output_name="rendered_output.mp4")