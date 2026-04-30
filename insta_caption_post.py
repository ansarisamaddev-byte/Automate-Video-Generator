import numpy as np
import random
import os
import glob
from PIL import Image, ImageDraw, ImageFont
from moviepy import (
    AudioFileClip, ColorClip, ImageClip, VideoFileClip,
    CompositeVideoClip, CompositeAudioClip, concatenate_videoclips
)
from faster_whisper import WhisperModel

# ---------------- GLOBAL INIT ---------------- #
model = WhisperModel("tiny", device="cpu", compute_type="int8")

SCREEN_W, SCREEN_H = 1080, 1920
SAFE_MARGIN = 60
SLIDE_OFFSET = 80
FONTS = [
    r"fonts/dejavu-sans-bold.ttf",
    r"fonts/Blankit-8MW2B.ttf", r"fonts/Cintaly-ax7v9.otf",
    r"fonts/dejavu-sans-bold.ttf", r"fonts/Sabering-gwAG3.otf",
    r"fonts/Sugiono-3zqyy.ttf", r"fonts/SukaCoffee-DYmzE.otf"
     # Ensure these paths are correct
]

# ---------------- UTILS ---------------- #
def make_even(x):
    x = int(round(x))
    return x if x % 2 == 0 else x + 1

def clamp(val, min_val, max_val):
    return max(min_val, min(val, max_val))

def get_sliding_position(t, tx, ty, direction, w, h):
    slide_time = 0.12
    if t < slide_time:
        p = t / slide_time
        if direction == "left": x = (tx - SLIDE_OFFSET) + (SLIDE_OFFSET * p); y = ty
        elif direction == "right": x = (tx + SLIDE_OFFSET) - (SLIDE_OFFSET * p); y = ty
    else:
        x, y = tx, ty
    return clamp(x, SAFE_MARGIN, SCREEN_W - w - SAFE_MARGIN), clamp(y, SAFE_MARGIN, SCREEN_H - h - SAFE_MARGIN)

# ---------------- TEXT RENDER (COLOR & MIN SIZE) ---------------- #
def create_word_data(text, font_path, max_horizontal_available, fill_color=(255, 255, 255)):
    target_size = random.randint(115, 130)
    min_font_size = 100 

    try:
        font = ImageFont.truetype(font_path, target_size)
    except:
        font = ImageFont.load_default()

    temp_draw = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    bbox = temp_draw.textbbox((0, 0), text, font=font)
    w, h = (bbox[2] - bbox[0]) + 60, (bbox[3] - bbox[1]) + 40

    if w > max_horizontal_available:
        scale = max_horizontal_available / w
        new_size = max(min_font_size, int(target_size * scale))
        font = ImageFont.truetype(font_path, new_size)
        bbox = temp_draw.textbbox((0, 0), text, font=font)
        w, h = (bbox[2] - bbox[0]) + 60, (bbox[3] - bbox[1]) + 40

    w, h = make_even(w), make_even(h)
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.text((w // 2, h // 2), text, font=font, fill=fill_color,
              stroke_width=7, stroke_fill=(0, 0, 0), anchor="mm")
    return np.array(img), w, h

# ---------------- PAGINATION ---------------- #
def render_paginated_text(segment, segment_end, clips):
    words_in_current_page = []
    curr_x, curr_y, line_h = SAFE_MARGIN, SCREEN_H // 3, 0
    USABLE_WIDTH = SCREEN_W - (SAFE_MARGIN * 2)
    HIGHLIGHT_COLORS = [(255, 255, 0), (0, 255, 255), (50, 255, 50)]

    for i, w_obj in enumerate(segment.words):
        txt = w_obj.word.strip().upper()
        if not txt: continue

        word_color = random.choice(HIGHLIGHT_COLORS) if random.random() < 0.15 else (255, 255, 255)
        arr, w, h = create_word_data(txt, random.choice(FONTS), USABLE_WIDTH, fill_color=word_color)

        if curr_x + w > (SCREEN_W - SAFE_MARGIN):
            curr_x = SAFE_MARGIN
            curr_y += line_h + 30 
            line_h = 0

        if curr_y + h > (SCREEN_H - SAFE_MARGIN):
            page_end_time = w_obj.start
            for p_word in words_in_current_page:
                p_word['clip'] = p_word['clip'].with_duration(max(0.1, page_end_time - p_word['start']))
                clips.append(p_word['clip'])
            words_in_current_page, curr_x, curr_y, line_h = [], SAFE_MARGIN, SCREEN_H // 3, 0

        direction = "left" if curr_x < SCREEN_W // 2 else "right"
        word_clip = (ImageClip(arr).with_start(w_obj.start).with_duration(max(0.1, segment_end - w_obj.start))
                     .with_position(lambda t, x=curr_x, y=curr_y, d=direction, ww=w, hh=h: 
                                    get_sliding_position(t, x, y, d, ww, hh)))
        
        words_in_current_page.append({'clip': word_clip, 'start': w_obj.start})
        curr_x += w + 20
        line_h = max(line_h, h)

    for p_word in words_in_current_page: clips.append(p_word['clip'])

# ---------------- MAIN ENGINE ---------------- #
def generate_reel(audio_path, image_folder, music_path=None, credit_video_path=None, output_name="output.mp4", start_at=0):
    image_files = sorted(glob.glob(os.path.join(image_folder, "*.jpg")))
    result = model.transcribe(audio_path, word_timestamps=True)
    segments = list(result[0])
    
    speech_audio = AudioFileClip(audio_path)
    video_duration = speech_audio.duration - 0.01 
    clips = [ColorClip((SCREEN_W, SCREEN_H), (0, 0, 0)).with_duration(video_duration)]

    for i, segment in enumerate(segments):
        start, end = segment.start, min(segments[i+1].start if i+1 < len(segments) else video_duration, video_duration)
        if (end - start) <= 0: continue

        if image_files:
            img_idx = (start_at + i) % len(image_files)
            img = Image.open(image_files[img_idx]).convert("RGB")
            ratio = max(SCREEN_W/img.size[0], SCREEN_H/img.size[1])
            img = img.resize((int(img.size[0]*ratio), int(img.size[1]*ratio)), Image.LANCZOS)
            
            l, t = (img.size[0]-SCREEN_W)//2, (img.size[1]-SCREEN_H)//2
            img_arr = np.array(img.crop((l, t, l+SCREEN_W, t+SCREEN_H)))
            
            bg_clip = ImageClip(img_arr).with_start(start).with_duration(end-start)
            overlay = ColorClip((SCREEN_W, SCREEN_H), (0, 0, 0)).with_opacity(0.5).with_start(start).with_duration(end-start)
            clips.extend([bg_clip, overlay])

        render_paginated_text(segment, end, clips)

    # Audio Mixing
    voice = speech_audio.with_duration(video_duration)
    if music_path and os.path.exists(music_path):
        bg_m = AudioFileClip(music_path).with_volume_scaled(0.12).with_duration(video_duration)
        final_audio = CompositeAudioClip([voice, bg_m])
    else:
        final_audio = voice
    
    main_reel = CompositeVideoClip(clips, size=(SCREEN_W, SCREEN_H)).with_duration(video_duration).with_audio(final_audio)

    # Append Credit Video
    if credit_video_path and os.path.exists(credit_video_path):
        credit = VideoFileClip(credit_video_path).resized(width=SCREEN_W)
        final_video = concatenate_videoclips([main_reel, credit], method="compose")
    else:
        final_video = main_reel

    final_video.write_videofile(output_name, fps=30, codec="libx264", audio_codec="aac", threads=4)
    
    # Return index for database update
    return start_at + len(segments)

if __name__ == "__main__":
    # Example of a manual test call
    next_idx = generate_reel(
        audio_path="reel_voice/voice (22).mp3",
        image_folder="gym_images",
        music_path="background_music/workout_beat.mp3", # Optional: add a path if you want music
        credit_video_path="ending/outro.mp4",           # Optional: add a path if you want an ending
        output_name="test_reel.mp4",
        start_at=0                                      # Start from the first image
    )
