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


# ---------------- PATH RESOLUTION ---------------- #
def resolve_project_path(raw_path):
    if not raw_path:
        return ""
    
    # Clean slash direction and whitespace from CSV input
    clean_path = str(raw_path).strip().replace("/", "\\").lstrip("\\")
    p = Path(clean_path)
    
    if p.is_absolute():
        return str(p)

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent  # Points directly to D:\Project\Automate-Video-Generator

    # 1. Check directly relative to Project Root (Automate-Video-Generator)
    if (project_root / p).exists():
        return str(project_root / p)

    # 2. Check relative to Script Directory (mind_scribble)
    if (script_dir / p).exists():
        return str(script_dir / p)

    # Fallback to project root path
    return str(project_root / p)

# ---------------- CORE CONFIG ---------------- #

model = WhisperModel("tiny", device="cpu", compute_type="int8", cpu_threads=4)

SCREEN_W = 1080
SCREEN_H = 1920

COLOR_GREEN_BG = (38, 64, 52, 245)    # Dark green highlighter box
COLOR_WHITE = (245, 245, 245)
COLOR_RED_HEADING = (245, 68, 68)
COLOR_UNDERLINE = (255, 255, 255)     # Active word underline

FONT_HEADING = r"fonts/dejavu-sans-bold.ttf"
FONT_CAPTION = r"fonts/dejavu-sans-bold.ttf"


# ---------------- IMAGE PROCESSORS ---------------- #

def load_folder_images(folder_path):
    if not folder_path:
        return []
    p_folder = Path(resolve_project_path(folder_path))
    print(p_folder)
    if not p_folder.exists():
        return []
    valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".jfif"}
    return sorted([str(p) for p in p_folder.glob("*") if p.is_file() and p.suffix.lower() in valid_exts])


def process_bg_image(img_path):
    """Processes background image with dark overlay and radial vignette shadow."""
    try:
        img = Image.open(img_path).convert("RGBA")
    except Exception:
        img = Image.new("RGBA", (SCREEN_W, SCREEN_H), (20, 20, 20, 255))

    ratio = max(SCREEN_W / img.width, SCREEN_H / img.height) * 1.50
    w_new, h_new = int(img.width * ratio), int(img.height * ratio)
    img = img.resize((w_new, h_new), Image.Resampling.LANCZOS)

    dark_overlay = Image.new("RGBA", (w_new, h_new), (0, 0, 0, 90))
    img = Image.alpha_composite(img, dark_overlay)

    vignette_mask = Image.new("L", (w_new, h_new), 0)
    draw_vignette = ImageDraw.Draw(vignette_mask)
    draw_vignette.ellipse([int(w_new * 0.05), int(h_new * 0.05), int(w_new * 0.95), int(h_new * 0.95)], fill=255)
    vignette_mask = vignette_mask.filter(ImageFilter.GaussianBlur(150))
    vignette_mask = ImageOps.invert(vignette_mask)

    shadow_layer = Image.new("RGBA", (w_new, h_new), (0, 0, 0, 180))
    shadow_layer.putalpha(vignette_mask)

    img = Image.alpha_composite(img, shadow_layer)
    return np.array(img.convert("RGB"))


def process_person_image(img_path):
    """Processes person image with heavily feathered alpha mask."""
    try:
        img = Image.open(img_path).convert("RGBA")
    except Exception:
        img = Image.new("RGBA", (700, 900), (40, 40, 40, 255))

    target_w = 750
    ratio = target_w / img.width
    target_h = int(img.height * ratio)
    img = img.resize((target_w, target_h), Image.Resampling.LANCZOS)

    mask = Image.new("L", (target_w, target_h), 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse([60, 60, target_w - 60, target_h - 60], fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(60))

    img.putalpha(mask)
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
    """Renders larger scene heading in red uppercase text with a slight tilt."""
    font_path = resolve_project_path(FONT_HEADING)
    try:
        font = ImageFont.truetype(font_path, 110) if os.path.exists(font_path) else ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()

    words = text.upper().split()
    lines, curr_line = [], ""
    for w in words:
        test_line = f"{curr_line} {w}".strip()
        if len(test_line) > 12:
            lines.append(curr_line)
            curr_line = w
        else:
            curr_line = test_line
    if curr_line:
        lines.append(curr_line)

    temp_img = Image.new("RGBA", (SCREEN_W, 800), (0, 0, 0, 0))
    draw = ImageDraw.Draw(temp_img)

    draw.multiline_text(
        (SCREEN_W // 2, 400),
        "\n".join(lines),
        font=font,
        fill=COLOR_RED_HEADING,
        stroke_width=12,
        stroke_fill=(0, 0, 0),
        align="center",
        anchor="mm",
        spacing=20
    )

    rotated_img = temp_img.rotate(-4, resample=Image.Resampling.BICUBIC, expand=False)
    return np.array(rotated_img)


def render_evergreen_caption_frame(words_group, active_word_index):
    """Renders Evergreen captions with green line box and active word underline."""
    font_path = resolve_project_path(FONT_CAPTION)
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


# ---------------- MASTER ENGINE ---------------- #

def generate_video_from_csv(csv_path, target_id=1, output_name="rendered_output.mp4"):
    csv_abs_path = resolve_project_path(csv_path)
    df = pd.read_csv(csv_abs_path)

    row = df[df["id"] == target_id].iloc[0]

    audio_file = resolve_project_path(str(row["audio_path"]))
    bg_folder = resolve_project_path(str(row["image_folder"]))
    person_folder = resolve_project_path(str(row["person_image_folder"]))
    cutout_folder = resolve_project_path(str(row.get("character_folder", row.get("characters_folder", "images/cutouts"))))
    heading_text = str(row.get("heading", row.get("caption", "HEADER")))

    bg_files = load_folder_images(bg_folder)
    person_files = load_folder_images(person_folder)
    cutout_files = load_folder_images(cutout_folder)

    if not bg_files:
        raise ValueError(f"No background images found in folder: {bg_folder}")

    # Transcribe speech for synchronization
    segments_gen, _ = model.transcribe(audio_file, word_timestamps=True)
    all_words = [w for seg in segments_gen for w in seg.words]

    speech_audio = AudioFileClip(audio_file)

    HEADING_DUR = 1.0  # Heading displays for 1.0s at the start
    total_duration = speech_audio.duration + HEADING_DUR

    layer_clips = []
    text_clips = []

    # 1. LAYER 1: ANIMATED BACKGROUND
    bg_interval = 6.0
    bg_count = math.ceil(total_duration / bg_interval)
    for i in range(bg_count):
        t_start = i * bg_interval
        dur = min(bg_interval, total_duration - t_start)
        bg_arr = process_bg_image(bg_files[i % len(bg_files)])

        clip = ImageClip(bg_arr).with_start(t_start).with_duration(dur)
        zoom_delta = 0.024 * (1 if i % 2 == 0 else -1)
        clip = clip.resized(lambda t: 1.0 + zoom_delta * (t / dur))
        clip = clip.with_position(("center", "center")).cropped(y1=0, y2=SCREEN_H, x1=0, x2=SCREEN_W)
        layer_clips.append(clip)

    # 2. LAYER 2: PERSON IMAGE (Positioned at Top-Center)
    if person_files:
        person_interval = 5.0
        person_count = math.ceil(total_duration / person_interval)
        for i in range(person_count):
            t_start = i * person_interval
            dur = min(person_interval, total_duration - t_start)
            person_arr = process_person_image(person_files[i % len(person_files)])

            person_clip = (
                ImageClip(person_arr)
                .with_start(t_start)
                .with_duration(dur)
                .resized(lambda t: 1.0 + 0.012 * (t / dur))
                .with_position(("center", 180))
            )
            layer_clips.append(person_clip)

    # 3. LAYER 3: CHARACTER CUTOUT (Positioned at Bottom-Right Corner)
    if cutout_files:
        cutout_interval = 3.5
        cutout_count = math.ceil(total_duration / cutout_interval)
        for i in range(cutout_count):
            t_start = i * cutout_interval
            dur = min(cutout_interval, total_duration - t_start)
            c_arr = process_cutout_image(cutout_files[i % len(cutout_files)])

            cutout_clip = (
                ImageClip(c_arr)
                .with_start(t_start)
                .with_duration(dur)
                .with_position((SCREEN_W - c_arr.shape[1] - 20, SCREEN_H - c_arr.shape[0] - 10))
            )
            layer_clips.append(cutout_clip)

    # 4. LAYER 4: LARGER HEADING AT BEGINNING (0.0s to 1.0s)
    heading_img = create_heading_card(heading_text)
    heading_clip = (
        ImageClip(heading_img)
        .with_start(0.0)
        .with_duration(HEADING_DUR)
        .with_position(("center", 350))
    )
    text_clips.append(heading_clip)

    # 5. LAYER 5: SYNCHRONIZED CAPTIONS (Starts at 1.0s)
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

    # Audio track starts at t = 1.0s
    final_audio = CompositeAudioClip([speech_audio.with_start(HEADING_DUR)]).with_duration(total_duration)

    video = (
        CompositeVideoClip(layer_clips + text_clips, size=(SCREEN_W, SCREEN_H))
        .with_duration(total_duration)
        .with_audio(final_audio)
    )

    out_file = resolve_project_path(output_name)
    print(f"[Render] Writing video to: {out_file}")
    video.write_videofile(out_file, fps=30, codec="libx264", audio_codec="aac", threads=4, preset="ultrafast")


if __name__ == "__main__":
    generate_video_from_csv("data.csv", target_id=1, output_name="rendered_output.mp4")