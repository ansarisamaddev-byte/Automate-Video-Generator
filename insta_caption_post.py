import numpy as np
import random
import os
import glob
from PIL import Image, ImageDraw, ImageFont
from moviepy import (
    AudioFileClip, ColorClip, ImageClip, VideoFileClip,
    CompositeVideoClip, CompositeAudioClip, concatenate_videoclips
)
from moviepy.video.fx import CrossFadeIn, FadeOut
from faster_whisper import WhisperModel

# ---------------- CORE CONFIG ---------------- #
model = WhisperModel("tiny", device="cpu", compute_type="int8", cpu_threads=4)
SCREEN_W, SCREEN_H = 1080, 1920
SAFE_MARGIN = 75
TRANSITION_TIME = 0.6  # Slightly longer for a smoother "merge"

FONTS = [
    r"D:\Projects\Automation\font\Blankit-BWyOl.otf",
    r"D:\Projects\Automation\font\Cintaly-ax7v9.otf",
    r"D:\Projects\Automation\font\MilkyCoffee-X3mWd.otf",
]

# ---------------- ANIMATION & VISUALS ---------------- #

def apply_ken_burns(clip, duration):
    """Cinematic movement engine."""
    mode = random.choice(["zoom_in", "zoom_out", "pan_left", "pan_right"])
    if mode == "zoom_in":
        return clip.resized(lambda t: 1 + 0.12 * (t / duration))
    elif mode == "zoom_out":
        return clip.resized(lambda t: 1.15 - 0.12 * (t / duration))
    elif mode == "pan_left":
        return clip.with_position(lambda t: (int(-60 * (t / duration)), "center"))
    else: # pan_right
        return clip.with_position(lambda t: (int(-120 + 60 * (t / duration)), "center"))

def apply_cinematic_darken(clip):
    return clip.image_transform(lambda pic: (pic * 0.7).astype('uint8'))

# ---------------- TEXT ENGINE ---------------- #

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

def generate_reel(audio_path, image_folder, music_path=None, credit_video_path=None, output_name="output.mp4"):
    image_files = sorted(glob.glob(os.path.join(image_folder, "*.jpg")))
    
    segments_gen, _ = model.transcribe(audio_path, word_timestamps=True)
    all_words = []
    for seg in segments_gen:
        all_words.extend(seg.words)
    
    speech_audio = AudioFileClip(audio_path)
    total_duration = speech_audio.duration - 0.05

    # Separate lists for Z-index management
    bg_clips = []
    text_clips = []
    
    curr_x, curr_y, line_h = SAFE_MARGIN, SCREEN_H // 3, 0
    max_w = SCREEN_W - (SAFE_MARGIN * 2)
    bottom_limit = SCREEN_H - 600
    
    image_index = 0
    current_image_start = 0
    words_in_current_view = []

    def flush_view(end_time):
        """Creates the 'Merge' transition by overlapping background clips."""
        nonlocal image_index, current_image_start, words_in_current_view
        
        view_duration = end_time - current_image_start
        if view_duration <= 0 or not words_in_current_view: return

        img_path = image_files[image_index % len(image_files)]
        img_pil = Image.open(img_path).convert("RGB")
        
        ratio = max(SCREEN_W / img_pil.width, SCREEN_H / img_pil.height) * 1.35
        img_pil = img_pil.resize((int(img_pil.width * ratio), int(img_pil.height * ratio)), Image.LANCZOS)
        img_arr = np.array(img_pil)

        # The 'Merge' Secret: Extend duration into the next clip's start time
        bg = (ImageClip(img_arr)
              .with_start(current_image_start)
              .with_duration(view_duration + TRANSITION_TIME))
        
        bg = apply_cinematic_darken(bg)
        bg = apply_ken_burns(bg, view_duration + TRANSITION_TIME)
        
        # CrossFadeIn handles the transparent blend over the previous image
        if current_image_start > 0:
            bg = bg.with_effects([CrossFadeIn(TRANSITION_TIME)])
        
        bg_clips.append(bg)
        
        for w_data in words_in_current_view:
            text_clips.append(w_data['clip'].with_duration(end_time - w_data['start']))
        
        image_index += 1
        current_image_start = end_time
        words_in_current_view = []

    # Processing loop
    for w_obj in all_words:
        word_text = w_obj.word.strip().upper()
        if not word_text: continue
        
        arr, w, h = create_word_data(word_text, random.choice(FONTS), max_w)

        if curr_x + w > SCREEN_W - SAFE_MARGIN:
            curr_x, curr_y, line_h = SAFE_MARGIN, curr_y + line_h + 45, 0

        if curr_y + h > bottom_limit:
            flush_view(w_obj.start)
            curr_x, curr_y, line_h = SAFE_MARGIN, SCREEN_H // 3, 0

        safe_x, safe_y = max(0, min(curr_x, SCREEN_W - w)), max(0, min(curr_y, SCREEN_H - h))
        
        word_clip = (ImageClip(arr)
                     .with_start(w_obj.start)
                     .with_position((safe_x, safe_y)))
        
        words_in_current_view.append({'clip': word_clip, 'start': w_obj.start})
        curr_x, line_h = curr_x + w + 30, max(line_h, h)

    flush_view(total_duration)

    # Audio Logic
    voice = speech_audio.with_duration(total_duration)
    if music_path and os.path.exists(music_path):
        bgm = AudioFileClip(music_path).with_volume_scaled(0.16).with_duration(total_duration)
        final_audio = CompositeAudioClip([voice, bgm])
    else:
        final_audio = voice

    # Layering: Backgrounds first, then Text on top
    final_video_clips = bg_clips + text_clips
    
    video = (CompositeVideoClip(final_video_clips, size=(SCREEN_W, SCREEN_H))
             .with_duration(total_duration)
             .with_audio(final_audio))

    if credit_video_path and os.path.exists(credit_video_path):
        try:
            credit = VideoFileClip(credit_video_path).resized(width=SCREEN_W)
            video = concatenate_videoclips([video, credit], method="compose")
        except: pass

    video.write_videofile(output_name, fps=30, codec="libx264", audio_codec="aac", threads=4, preset="ultrafast")

if __name__ == "__main__":
    generate_reel(
        audio_path="audio/voice (36).mp3",
        image_folder="images",
        music_path="background_music/music.mp3",
        credit_video_path="ending/outro.mp4",
        output_name="warrior_seamless_edition.mp4"
    )
