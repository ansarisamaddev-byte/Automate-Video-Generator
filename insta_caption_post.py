import numpy as np
import random
import os
import glob
from PIL import Image, ImageDraw, ImageFont
from moviepy import (
    AudioFileClip, ImageClip, VideoFileClip,
    CompositeVideoClip, CompositeAudioClip, concatenate_videoclips
)
from moviepy.video.fx import CrossFadeIn
from faster_whisper import WhisperModel

# ---------------- CORE CONFIG ---------------- #
model = WhisperModel("tiny", device="cpu", compute_type="int8", cpu_threads=4)
SCREEN_W, SCREEN_H = 1080, 1920
SAFE_MARGIN = 75
TRANSITION_TIME = 0.6 

# Update these paths to your actual font locations
FONTS = [
    r"fonts/Blackburn Free.ttf",
    r"fonts/WorldstarRegular.ttf",
    r"fonts/LemonJellyPersonalUse-dEqR.ttf",
    r"fonts/Sugiono-3zqyy.ttf",
    r"fonts/Cintaly-Eazdl.ttf",
    r"fonts/Blankit-8MW2B.ttf",
]

# ---------------- VISUAL EFFECTS ---------------- #

def apply_ken_burns(clip, duration):
    mode = random.choice(["zoom_in", "zoom_out", "pan_left", "pan_right"])
    if mode == "zoom_in":
        return clip.resized(lambda t: 1 + 0.12 * (t / duration))
    elif mode == "zoom_out":
        return clip.resized(lambda t: 1.15 - 0.12 * (t / duration))
    elif mode == "pan_left":
        return clip.with_position(lambda t: (int(-60 * (t / duration)), "center"))
    else:
        return clip.with_position(lambda t: (int(-120 + 60 * (t / duration)), "center"))

def apply_cinematic_darken(clip):
    return clip.image_transform(lambda pic: (pic * 0.7).astype('uint8'))

def create_word_data(text, font_path, max_width):
    target_size, MIN_FONT_SIZE = 125, 90
    try:
        font = ImageFont.truetype(font_path, target_size)
    except:
        font = ImageFont.load_default()

    temp_draw = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    bbox = temp_draw.textbbox((0, 0), text, font=font)
    w_text, h_text = bbox[2] - bbox[0], bbox[3] - bbox[1]

    if w_text > max_width:
        scale = max_width / w_text
        font = ImageFont.truetype(font_path, max(MIN_FONT_SIZE, int(target_size * scale)))
        bbox = temp_draw.textbbox((0, 0), text, font=font)
        w_text, h_text = bbox[2] - bbox[0], bbox[3] - bbox[1]

    w_canvas, h_canvas = int(w_text + 50), int(h_text + 50)
    img = Image.new("RGBA", (w_canvas, h_canvas), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.text((w_canvas // 2, h_canvas // 2), text, font=font, fill=(255, 255, 255),
              stroke_width=15, stroke_fill=(0, 0, 0), anchor="mm")
    return np.array(img), w_canvas, h_canvas

# ---------------- MAIN PIPELINE ---------------- #

def generate_reel(audio_path, image_folder, music_path=None, credit_video_path=None, output_name="output.mp4", start_at=0):
    image_files = sorted(glob.glob(os.path.join(image_folder, "*.jpg")))
    if not image_files:
        raise ValueError(f"No images found in {image_folder}")

    segments_gen, _ = model.transcribe(audio_path, word_timestamps=True)
    all_words = []
    for seg in segments_gen:
        all_words.extend(seg.words)
    
    speech_audio = AudioFileClip(audio_path)
    total_duration = speech_audio.duration - 0.05

    bg_clips = []
    text_clips = []
    curr_x, curr_y, line_h = SAFE_MARGIN, SCREEN_H // 3, 0
    max_w = SCREEN_W - (SAFE_MARGIN * 2)
    bottom_limit = SCREEN_H - 600
    
    # Track image usage
    image_index = int(start_at)
    current_image_start = 0
    words_in_current_view = []

    def flush_view(end_time):
        nonlocal image_index, current_image_start, words_in_current_view
        view_duration = end_time - current_image_start
        if view_duration <= 0 or not words_in_current_view: return

        img_path = image_files[image_index % len(image_files)]
        img_pil = Image.open(img_path).convert("RGB")
        
        ratio = max(SCREEN_W / img_pil.width, SCREEN_H / img_pil.height) * 1.35
        img_pil = img_pil.resize((int(img_pil.width * ratio), int(img_pil.height * ratio)), Image.LANCZOS)
        
        bg = (ImageClip(np.array(img_pil))
              .with_start(current_image_start)
              .with_duration(view_duration + TRANSITION_TIME))
        
        bg = apply_cinematic_darken(bg)
        bg = apply_ken_burns(bg, view_duration + TRANSITION_TIME)
        
        if current_image_start > 0:
            bg = bg.with_effects([CrossFadeIn(TRANSITION_TIME)])
        
        bg_clips.append(bg)
        for w_data in words_in_current_view:
            text_clips.append(w_data['clip'].with_duration(end_time - w_data['start']))
        
        image_index += 1
        current_image_start = end_time
        words_in_current_view = []

    for w_obj in all_words:
        word_text = w_obj.word.strip().upper()
        if not word_text: continue
        
        arr, w, h = create_word_data(word_text, random.choice(FONTS), max_w)
        if curr_x + w > SCREEN_W - SAFE_MARGIN:
            curr_x, curr_y, line_h = SAFE_MARGIN, curr_y + line_h + 45, 0
        if curr_y + h > bottom_limit:
            flush_view(w_obj.start)
            curr_x, curr_y, line_h = SAFE_MARGIN, SCREEN_H // 3, 0

        word_clip = (ImageClip(arr).with_start(w_obj.start).with_position((curr_x, curr_y)))
        words_in_current_view.append({'clip': word_clip, 'start': w_obj.start})
        curr_x, line_h = curr_x + w + 30, max(line_h, h)

    flush_view(total_duration)

    # Final Composite
    voice = speech_audio.with_duration(total_duration)
    if music_path and os.path.exists(music_path):
        bgm = AudioFileClip(music_path).with_volume_scaled(0.15).with_duration(total_duration)
        final_audio = CompositeAudioClip([voice, bgm])
    else:
        final_audio = voice

    video = (CompositeVideoClip(bg_clips + text_clips, size=(SCREEN_W, SCREEN_H))
             .with_duration(total_duration).with_audio(final_audio))

    if credit_video_path and os.path.exists(credit_video_path):
        try:
            credit = VideoFileClip(credit_video_path).resized(width=SCREEN_W)
            video = concatenate_videoclips([video, credit], method="compose")
        except: pass

    video.write_videofile(output_name, fps=30, codec="libx264", audio_codec="aac", threads=4, preset="ultrafast")
    return image_index # Returns the last index used


if __name__ == "__main__":
    next_idx = generate_reel(
        audio_path="reel_voice/voice (28).mp3",
        image_folder="gym_images",
        music_path="background_music/workout_beat.mp3", # Optional: add a path if you want music
        credit_video_path="ending/outro.mp4",           # Optional: add a path if you want an ending
        output_name="test_reel.mp4",
        start_at=0     
    )
