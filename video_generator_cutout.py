import numpy as np
import random
import os
import glob
import math

from PIL import Image, ImageDraw, ImageFont, ImageEnhance

# MoviePy 2.x: the "editor" module is gone, import directly from moviepy
from moviepy import (
    AudioFileClip,
    ImageClip,
    VideoFileClip,
    CompositeVideoClip,
    CompositeAudioClip,
    concatenate_videoclips
)

from faster_whisper import WhisperModel


# ---------------- CORE CONFIG ---------------- #

model = WhisperModel(
    "tiny",
    device="cpu",
    compute_type="int8",
    cpu_threads=4
)

SCREEN_W = 1080
SCREEN_H = 1920
SAFE_MARGIN = 75

# --- Central Banner Configuration ---
BANNER_H = 700
BANNER_Y = (SCREEN_H // 2) - (BANNER_H // 2)  # Perfectly centered vertically

FONTS = [
    r"fonts/dejavu-sans-bold.ttf",
    r"fonts/Blackburn Free.ttf",
    r"fonts/WorldstarRegular.ttf",
    r"fonts/Sugiono-3zqyy.ttf",
    r"fonts/Cintaly-Eazdl.ttf",
    r"fonts/Blankit-8MW2B.ttf",
]


# ---------------- VISUAL EFFECTS & PIL PROCESSING ---------------- #

def process_bg_image(img_path):
    """Resizes, darkens, and desaturates the background for full screen."""
    try:
        img = Image.open(img_path).convert("RGB")
    except:
        # Fallback empty image if path fails
        img = Image.new("RGB", (SCREEN_W, SCREEN_H), (30, 30, 30))

    ratio = max(SCREEN_W / img.width, SCREEN_H / img.height) * 1.35
    img = img.resize((int(img.width * ratio), int(img.height * ratio)), Image.Resampling.LANCZOS)

    # Desaturate slightly
    img = ImageEnhance.Color(img).enhance(0.3)
    # Darken so text/cutouts pop
    img = ImageEnhance.Brightness(img).enhance(0.4)

    return np.array(img)


def process_cutout_image(img_path):
    """Scales cutouts for the screen."""
    img = Image.open(img_path).convert("RGBA")

    # MODIFIED: Fills 35% of screen height instead of 55% to make it smaller
    target_h = int(SCREEN_H * 0.35)

    scale_w = target_h / img.height
    target_w = int(img.width * scale_w)

    img = img.resize((target_w, target_h), Image.Resampling.LANCZOS)
    return np.array(img)


def create_central_banner():
    """Generates the semi-transparent black background for the text."""
    # RGBA: 0,0,0 (Black) with 160/255 opacity
    img = Image.new("RGBA", (SCREEN_W, BANNER_H), (0, 0, 0, 160))
    return np.array(img)


def apply_ken_burns(clip, duration):
    """Subtle zoom for the backgrounds so they aren't completely static."""
    mode = random.choice(["zoom_in", "zoom_out"])
    if mode == "zoom_in":
        # MoviePy 2.x: resize -> resized
        return clip.resized(lambda t: 1.0 + 0.05 * (t / duration))
    else:
        return clip.resized(lambda t: 1.05 - 0.05 * (t / duration))


def apply_slide_and_fade(clip, duration):
    """Slides the cutout in from Bottom, Left, or Right, holds, and fades out."""
    w, h = clip.size

    direction = random.choice(["bottom", "left", "right"])

    # --- POSITIONING LOGIC ---
    # Pushes the image to the right (100 pixels from the right edge)
    parked_x = SCREEN_W - w - 100

    # Pushes the image down to the bottom (20 pixels from the bottom edge)
    parked_y = SCREEN_H - h - 20

    slide_time = 0.8  # Takes 0.8 seconds to enter
    fade_time = 1.0   # Takes 1.0 second to fade out at the end

    # 1. Setup Slide Logic
    if direction == "bottom":
        start_pos = (parked_x, SCREEN_H + 100)
    elif direction == "left":
        start_pos = (-w - 100, parked_y)
    else:  # right
        start_pos = (SCREEN_W + 100, parked_y)

    parked_pos = (parked_x, parked_y)

    def pos_func(t):
        if t < slide_time:
            # Ease out slide
            progress = t / slide_time
            progress = 1 - (1 - progress) ** 3

            curr_x = start_pos[0] + (parked_pos[0] - start_pos[0]) * progress
            curr_y = start_pos[1] + (parked_pos[1] - start_pos[1]) * progress
        else:
            curr_x, curr_y = parked_pos

        # Clamp so the clip always keeps at least 1px of overlap with the canvas frame
        curr_x = max(-w + 1, min(SCREEN_W - 1, curr_x))
        curr_y = max(-h + 1, min(SCREEN_H - 1, curr_y))

        return (int(curr_x), int(curr_y))

    # MoviePy 2.x: set_position -> with_position
    clip = clip.with_position(pos_func)

    # 2. Setup Foolproof Fade-Out Logic
    def fade_mask(get_frame, t):
        mask_frame = get_frame(t)
        if t >= duration - fade_time:
            factor = 1.0 - ((t - (duration - fade_time)) / fade_time)
            return mask_frame * max(0.0, factor)
        return mask_frame

    # MoviePy 2.x: Clip.fl -> Clip.transform
    clip.mask = clip.mask.transform(fade_mask)
    return clip


# ---------------- TEXT RENDER ---------------- #

def create_word_data(text, font_path, max_width):
    target_size = 125
    min_size = 90

    try:
        font = ImageFont.truetype(font_path, target_size)
    except:
        font = ImageFont.load_default()

    temp_draw = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    bbox = temp_draw.textbbox((0, 0), text, font=font)

    w_text = bbox[2] - bbox[0]
    h_text = bbox[3] - bbox[1]

    if w_text > max_width:
        scale = max_width / w_text
        new_size = max(min_size, int(target_size * scale))
        font = ImageFont.truetype(font_path, new_size)
        bbox = temp_draw.textbbox((0, 0), text, font=font)
        w_text = bbox[2] - bbox[0]
        h_text = bbox[3] - bbox[1]

    canvas_w = int(w_text + 50)
    canvas_h = int(h_text + 50)

    img = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    draw.text(
        (canvas_w // 2, canvas_h // 2),
        text,
        font=font,
        fill=(255, 255, 255),
        stroke_width=10,
        stroke_fill=(0, 0, 0),
        anchor="mm"
    )

    return np.array(img), canvas_w, canvas_h


# ---------------- MAIN GENERATOR ---------------- #

def generate_reel(
    audio_path,
    bg_folder,
    cutout_folder,
    music_path=None,
    credit_video_path=None,
    output_name="output.mp4"
):
    # Fetch Backgrounds
    bg_files = []
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        bg_files.extend(glob.glob(os.path.join(bg_folder, ext)))
        bg_files.extend(glob.glob(os.path.join(bg_folder, ext.upper())))
    bg_files = sorted(bg_files)

    # Fetch Cutouts
    raw_cutouts = glob.glob(os.path.join(cutout_folder, "*.png")) + glob.glob(os.path.join(cutout_folder, "*.PNG"))
    cutout_files = sorted(list(set(raw_cutouts)))

    if not bg_files or not cutout_files:
        raise ValueError("Missing background or cutout images in the specified folders!")

    # Transcribe
    segments_gen, _ = model.transcribe(audio_path, word_timestamps=True)
    all_words = [w for seg in segments_gen for w in seg.words]

    speech_audio = AudioFileClip(audio_path)
    total_duration = speech_audio.duration - 0.05

    layer_clips = []
    text_clips = []

    # 1. BUILD BACKGROUND TIMELINE (Random every 3 seconds)
    bg_interval = 3.0
    bg_count = math.ceil(total_duration / bg_interval)

    for i in range(bg_count):
        t_start = i * bg_interval
        dur = min(bg_interval, total_duration - t_start)
        if dur <= 0:
            break

        bg_path = random.choice(bg_files)
        bg_arr = process_bg_image(bg_path)

        clip = ImageClip(bg_arr).with_start(t_start).with_duration(dur)
        clip = apply_ken_burns(clip, dur)
        clip = clip.with_position(("center", "center")).cropped(y1=0, y2=SCREEN_H, x1=0, x2=SCREEN_W)

        layer_clips.append(clip)

    # 2. BUILD CENTRAL BANNER (Transparent Black)
    banner_arr = create_central_banner()
    banner_clip = (ImageClip(banner_arr)
                   .with_position((0, BANNER_Y))
                   .with_duration(total_duration))
    layer_clips.append(banner_clip)

    # 3. BUILD CUTOUT TIMELINE (Every 6 seconds)
    cutout_interval = 6.0
    cutout_count = math.ceil(total_duration / cutout_interval)

    for i in range(cutout_count):
        t_start = i * cutout_interval
        dur = min(cutout_interval, total_duration - t_start)
        
        # --- ADD THESE 3 LINES TO PREVENT OVERLAP ---
        sub_start = max(0.0, total_duration - 4.0)
        if t_start + dur > sub_start:
            dur = max(0.0, sub_start - t_start)


        if dur <= 1.5:
            break

        cut_path = random.choice(cutout_files)
        cut_arr = process_cutout_image(cut_path)

        clip = ImageClip(cut_arr).with_start(t_start).with_duration(dur)
        clip = apply_slide_and_fade(clip, dur)

        layer_clips.append(clip)

    # --- 3b. SPECIAL END SUBSCRIBE CUTOUT (Last 2 seconds) ---
    subscribe_folder = os.path.join(cutout_folder, "subscribe")
    if os.path.exists(subscribe_folder):
        sub_files = glob.glob(os.path.join(subscribe_folder, "*.png")) + glob.glob(os.path.join(subscribe_folder, "*.PNG"))
        if sub_files:
            sub_path = random.choice(sub_files)
            sub_arr = process_cutout_image(sub_path)
            
            # Appears exactly 2 seconds before the voice audio finishes
            sub_start = max(0.0, total_duration - 4.0)
            sub_dur = total_duration - sub_start
            
            if sub_dur > 0:
                sub_clip = ImageClip(sub_arr).with_start(sub_start).with_duration(sub_dur)
                sub_clip = apply_slide_and_fade(sub_clip, sub_dur)
                layer_clips.append(sub_clip)

    # 4. BUILD CAPTIONS (Locked inside the central banner - 5 WORDS MAX)
    curr_x = SAFE_MARGIN
    curr_y = BANNER_Y + 40
    line_h = 0
    max_w = SCREEN_W - (SAFE_MARGIN * 2)
    bottom_limit = BANNER_Y + BANNER_H - 100

    words_in_current_view = []
    word_counter = 0

    def flush_text(flush_time):
        for wd in words_in_current_view:
            clip_dur = flush_time - wd["start"]
            if clip_dur > 0:
                text_clips.append(wd["clip"].with_duration(clip_dur))
        words_in_current_view.clear()

    for w_obj in all_words:
        word = w_obj.word.strip().upper()
        if not word:
            continue

        arr, w, h = create_word_data(word, random.choice(FONTS), max_w)

        if curr_x + w > SCREEN_W - SAFE_MARGIN:
            curr_x = SAFE_MARGIN
            curr_y += line_h + 45
            line_h = 0

        if curr_y + h > bottom_limit or word_counter >= 5:
            flush_text(w_obj.start)
            curr_x = SAFE_MARGIN
            curr_y = BANNER_Y + 40
            line_h = 0
            word_counter = 0

        clip = ImageClip(arr).with_start(w_obj.start).with_position((curr_x, curr_y))

        words_in_current_view.append({
            "clip": clip,
            "start": w_obj.start
        })

        curr_x += w + 30
        line_h = max(line_h, h)
        word_counter += 1

    flush_text(total_duration)

    # -------- AUDIO & VIDEO COMPILE -------- #
    voice = speech_audio.with_duration(total_duration)
    if music_path and os.path.exists(music_path):
        bgm = AudioFileClip(music_path).with_volume_scaled(0.15).with_duration(total_duration)
        final_audio = CompositeAudioClip([voice, bgm])
    else:
        final_audio = voice

    video = (
        CompositeVideoClip(
            layer_clips + text_clips,
            size=(SCREEN_W, SCREEN_H)
        )
        .with_duration(total_duration)
        .with_audio(final_audio)
    )

    if credit_video_path and os.path.exists(credit_video_path):
        try:
            credit = VideoFileClip(credit_video_path).resized(width=SCREEN_W)
            video = concatenate_videoclips([video, credit], method="compose")
        except Exception:
            pass

    video.write_videofile(
        output_name,
        fps=30,
        codec="libx264",
        audio_codec="aac",
        threads=4,
        preset="ultrafast"
    )

    caption_words = [w.word.strip() for w in all_words if w.word.strip()][:5]
    return {
        "caption": " ".join(caption_words).title()
    }


if __name__ == "__main__":
    result = generate_reel(
        audio_path="audio/mindscribble/audio (3).mp3",
        bg_folder="images/backgrounds",
        cutout_folder="images/cutouts",
        music_path="background_music/background_audio (1).mp3",
        credit_video_path="ending/outro.mp4",
        output_name="test_reel_final.mp4"
    )
    print(result)
