import numpy as np
import random
import os
import glob
from PIL import Image, ImageDraw, ImageFont
from moviepy import (AudioFileClip, ColorClip, ImageClip, VideoFileClip, 
                    CompositeVideoClip, CompositeAudioClip, concatenate_videoclips)
import moviepy.video.fx as vfx
from faster_whisper import WhisperModel

# --- GLOBAL INITIALIZATION ---
model = WhisperModel("tiny", device="cpu", compute_type="int8")

FONTS = [
    r"fonts/Blankit-8MW2B.ttf", r"fonts/Cintaly-ax7v9.otf",
    r"fonts/dejavu-sans-bold.ttf", r"fonts/Sabering-gwAG3.otf",
    r"fonts/Sugiono-3zqyy.ttf", r"fonts/SukaCoffee-DYmzE.otf"
]

def make_even(x):
    x = int(round(x))
    return x if x % 2 == 0 else x + 1

def get_sliding_position(t, duration, target_x, target_y, direction):
    slide_time = 0.15 
    offset = 150 
    if t < slide_time:
        p = t / slide_time
        if direction == "left": return (target_x - offset) + (offset * p), target_y
        if direction == "right": return (target_x + offset) - (offset * p), target_y
        if direction == "top": return target_x, (target_y - offset) + (offset * p)
        if direction == "bottom": return target_x, (target_y + offset) - (offset * p)
    return target_x, target_y

def create_word_data(text, font_path, is_final=False):
    base_size = random.randint(110, 140) if not is_final else random.randint(150, 180)
    try:
        font = ImageFont.truetype(font_path, base_size)
    except:
        font = ImageFont.load_default()
    temp_draw = ImageDraw.Draw(Image.new("RGBA", (1,1)))
    bbox = temp_draw.textbbox((0, 0), text, font=font)
    w = make_even((bbox[2] - bbox[0]) + 100)
    h = make_even((bbox[3] - bbox[1]) + 80)
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    color = (255, 255, 255) if not is_final else (255, 215, 0)
    draw.text((w//2, h//2), text, font=font, fill=color, stroke_width=8, stroke_fill=(0, 0, 0), anchor="mm")
    return np.array(img), w, h

def render_final_style(words, video_duration, clips_list):
    style = random.choice(["cascade", "split"])
    if style == "cascade":
        current_y = 300
        for word in words:
            pixel_array, w, h = create_word_data(word.word.strip().upper(), random.choice(FONTS), True)
            tx, ty = (1080 - w) // 2, current_y
            dur = max(0.1, video_duration - word.start)
            clips_list.append(ImageClip(pixel_array).with_start(word.start).with_duration(dur).with_position(lambda t, tx=tx, ty=ty, d=dur: get_sliding_position(t, d, tx, ty, "top")))
            current_y += (h - 30)
    elif style == "split":
        current_y = 350
        margin = 150 
        for i, word in enumerate(words):
            pixel_array, w, h = create_word_data(word.word.strip().upper(), random.choice(FONTS), True)
            tx = margin if i % 2 == 0 else (1080 - w - margin)
            if current_y + h > 1750: break
            dur = max(0.1, video_duration - word.start)
            clips_list.append(ImageClip(pixel_array).with_start(word.start).with_duration(dur).with_position((tx, current_y)).with_effects([vfx.CrossFadeIn(0.1)]))
            current_y += (h - 20)

def generate_reel(audio_path, image_folder, music_path, credit_video_path, output_name, start_at=0):
    image_files = glob.glob(os.path.join(image_folder, "*.jpg"))
    image_files.sort(key=lambda f: int(''.join(filter(str.isdigit, os.path.basename(f))) or 0))

    segments = list(model.transcribe(audio_path, word_timestamps=True)[0])
    speech_audio = AudioFileClip(audio_path)
    video_duration = speech_audio.duration + 1.0
    
    if music_path and os.path.exists(music_path):
        bg_music = AudioFileClip(music_path).with_volume_scaled(0.19).with_duration(video_duration)
        final_audio = CompositeAudioClip([speech_audio, bg_music])
    else:
        final_audio = speech_audio

    all_clips = [ColorClip(size=(1080, 1920), color=(0, 0, 0)).with_duration(video_duration)]

    for i, segment in enumerate(segments):
        sentence_start = segment.start
        sentence_end = segments[i+1].start if (i+1) < len(segments) else video_duration
        dur = max(0.1, sentence_end - sentence_start)
        is_last = (i == len(segments) - 1)

        if image_files:
            img_idx = (start_at + i) % len(image_files)
            pil_img = Image.open(image_files[img_idx]).convert("RGB")
            ratio = 1920 / pil_img.size[1]
            new_w = make_even(pil_img.size[0] * ratio)
            pil_img = pil_img.resize((new_w, 1920), Image.LANCZOS)
            left = (pil_img.size[0] - 1080) // 2
            pil_img = pil_img.crop((left, 0, left + 1080, 1920))
            
            main_img_clip = ImageClip(np.array(pil_img)).with_start(sentence_start).with_duration(dur)
            all_clips.append(main_img_clip)
            opacity = 0.6 if is_last else 0.4
            all_clips.append(ColorClip(size=(1080, 1920), color=(0, 0, 0)).with_opacity(opacity).with_start(sentence_start).with_duration(dur))

        if is_last:
            render_final_style(segment.words, video_duration, all_clips)
        else:
            current_y, current_x = random.randint(450, 600), 100
            word_in_batch, max_words = 0, 6
            for idx, word in enumerate(segment.words):
                clean_text = word.word.strip().upper()
                if not clean_text: continue
                pixel_array, w, h = create_word_data(clean_text, random.choice(FONTS))
                if word_in_batch >= max_words:
                    current_y, current_x, word_in_batch = random.randint(450, 600), 100, 0
                dis_at = segment.words[idx + (max_words - word_in_batch)].start if idx + (max_words - word_in_batch) < len(segment.words) else sentence_end
                word_dur = max(0.1, dis_at - word.start)
                if current_x + w > 900:
                    current_x, current_y = 100, current_y + h + 20
                tx, ty = current_x, current_y
                move_dir = random.choice(["left", "right", "top", "bottom"])
                txt_clip = (ImageClip(pixel_array).with_start(word.start).with_duration(word_dur).with_effects([vfx.CrossFadeOut(0.2)]).with_position(lambda t, tx=tx, ty=ty, d=move_dir, dur=word_dur: get_sliding_position(t, dur, tx, ty, d)))
                all_clips.append(txt_clip)
                current_x += w - 10
                word_in_batch += 1

    generated_reel = CompositeVideoClip(all_clips, size=(1080, 1920)).with_duration(video_duration).with_audio(final_audio)

    # --- TRANSITION LOGIC ---
    if credit_video_path and os.path.exists(credit_video_path):
        print("DEBUG: Creating crossfade transition...")
        credit_clip = VideoFileClip(credit_video_path).with_effects([vfx.Resize(width=1080)])
        
        # Apply a subtle fade out to the generated reel and fade in to credits
        generated_reel = generated_reel.with_effects([vfx.CrossFadeOut(0.5)])
        credit_clip = credit_clip.with_effects([vfx.CrossFadeIn(0.5)])
        
        # padding=-0.5 creates a 0.5s overlap/transition
        final_video = concatenate_videoclips([generated_reel, credit_clip], method="compose", padding=-0.5)
    else:
        final_video = generated_reel

    final_video.write_videofile(output_name, fps=30, codec="libx264", audio_codec="aac")
    return start_at + len(segments)

if __name__ == "__main__":
    generate_reel(
        audio_path="reel_voice/voice (1).mp3", 
        image_folder="gym_images", 
        music_path=None, 
        credit_video_path="ending/FOLLOW.mp4", 
        output_name="no_flicker_reel.mp4"
    )