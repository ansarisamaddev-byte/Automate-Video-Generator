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
    r"fonts/Blackburn Free.ttf",
    r"fonts/Blankit-8MW2B.ttf",
    r"fonts/Cintaly-Eazdl.ttf",
    r"fonts/dejavu-sans-bold.ttf",
    r"fonts/LemonJellyPersonalUse-dEqR.ttf",
    r"fonts/NotoColorEmoji.ttf",
    r"fonts/PoemLettersSerifRegular-lgggX.ttf",
    r"fonts/Sugiono-3zqyy.ttf",
    r"fonts/WorldstarRegular.ttf"
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
def create_word_data(text, font_path, max_horizontal_available):
    # Fixed target size and a strict minimum floor
    target_size = 120
    MIN_FONT_SIZE = 85 

    try:
        font = ImageFont.truetype(font_path, target_size)
    except:
        font = ImageFont.load_default()

    temp_draw = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    
    def get_dims(f):
        bbox = temp_draw.textbbox((0, 0), text, font=f)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]

    w_text, h_text = get_dims(font)

    # Scale down if too wide, but never below MIN_FONT_SIZE
    if w_text > max_horizontal_available:
        scale = max_horizontal_available / w_text
        final_size = max(MIN_FONT_SIZE, int(target_size * scale))
        font = ImageFont.truetype(font_path, final_size)
        w_text, h_text = get_dims(font)

    padding = 25
    w_canvas, h_canvas = make_even(w_text + padding), make_even(h_text + padding)
    
    img = Image.new("RGBA", (w_canvas, h_canvas), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # fill_color is now strictly white
    draw.text((w_canvas // 2, h_canvas // 2), text, font=font, fill=(255, 255, 255),
              stroke_width=8, stroke_fill=(0, 0, 0), anchor="mm")
    
    return np.array(img), w_canvas, h_canvas

# ---------------- PAGINATION ---------------- #
def render_paginated_text(segment, segment_end, clips):
    words_on_screen = []
    curr_x = SAFE_MARGIN
    curr_y = SCREEN_H // 3 
    line_h = 0
    
    RIGHT_BOUNDARY = SCREEN_W - SAFE_MARGIN
    BOTTOM_BOUNDARY = SCREEN_H - 450 # Safety buffer for Reels/TikTok UI
    USABLE_WIDTH = SCREEN_W - (SAFE_MARGIN * 2)

    # Process words that actually have text
    valid_words = [w for w in segment.words if w.word.strip()]

    for i, w_obj in enumerate(valid_words):
        txt = w_obj.word.strip().upper()
        
        # Call the updated renderer (no color logic inside)
        arr, w, h = create_word_data(txt, random.choice(FONTS), USABLE_WIDTH)

        # Line Wrap Check
        if curr_x + w > RIGHT_BOUNDARY:
            curr_x = SAFE_MARGIN
            curr_y += line_h + 35 
            line_h = 0

        # Page Wrap Check (Clear screen when full)
        if curr_y + h > BOTTOM_BOUNDARY:
            page_clear_time = w_obj.start
            for item in words_on_screen:
                item['clip'] = item['clip'].with_duration(max(0.1, page_clear_time - item['start']))
                clips.append(item['clip'])
            
            # Reset for fresh page
            words_on_screen = []
            curr_x = SAFE_MARGIN
            curr_y = SCREEN_H // 3
            line_h = 0

        # Create Clip
        start_t = w_obj.start
        word_clip = (ImageClip(arr)
                     .with_start(start_t)
                     .with_position(lambda t, x=curr_x, y=curr_y, ww=w, hh=h: 
                                    get_sliding_position(t, x, y, "left", ww, hh)))

        words_on_screen.append({'clip': word_clip, 'start': start_t})

        # Advance spacing
        curr_x += w + 25 
        line_h = max(line_h, h)

    # Ensure the last batch of words stays until the end of the segment
    for item in words_on_screen:
        item['clip'] = item['clip'].with_duration(max(0.1, segment_end - item['start']))
        clips.append(item['clip'])

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
        audio_path="reel_voice/voice (28).mp3",
        image_folder="gym_images",
        music_path="background_music/workout_beat.mp3", # Optional: add a path if you want music
        credit_video_path="ending/outro.mp4",           # Optional: add a path if you want an ending
        output_name="test_reel.mp4",
        start_at=0                                      # Start from the first image
    )
