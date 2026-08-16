import os
import glob
import random
import datetime
import math
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageOps

# ================= MOVIEPY 2.X & WHISPER =================
from moviepy import (
    VideoFileClip, 
    AudioFileClip, 
    ImageClip, 
    CompositeVideoClip, 
    CompositeAudioClip, 
    concatenate_videoclips,
    afx  # Contains audio effects like afx.audio_loop
)
from faster_whisper import WhisperModel

# ================= CORE ENGINE SETUP =================
SCREEN_W = 1080
SCREEN_H = 1920
SAFE_MARGIN = 75
BANNER_Y = int(SCREEN_H * 0.45) 

FONTS = [r"fonts/dejavu-sans-bold.ttf"]

_whisper_model = None
def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        _whisper_model = WhisperModel("tiny", device="cpu", compute_type="int8", cpu_threads=4)
    return _whisper_model

def apply_ken_burns(clip, duration):
    mode = random.choice(["zoom_in", "zoom_out"])
    return clip.resized(lambda t: 1.0 + 0.04 * (t / duration) if mode == "zoom_in" else 1.04 - 0.04 * (t / duration))

def process_bg_image(img_path):
    try:
        img = Image.open(img_path).convert("RGB")
    except Exception:
        img = Image.new("RGB", (SCREEN_W, SCREEN_H), (20, 20, 20))
    ratio = max(SCREEN_W / img.width, SCREEN_H / img.height) * 1.1
    img = img.resize((int(img.width * ratio), int(img.height * ratio)), Image.Resampling.LANCZOS)
    img = ImageEnhance.Color(img).enhance(0.4)
    img = ImageEnhance.Brightness(img).enhance(0.35)
    return np.array(img)

def process_evidence_image(img_path):
    img = Image.open(img_path).convert("RGBA")
    
    max_allowed_w = int(SCREEN_W * 0.88)  
    max_allowed_h = int(SCREEN_H * 0.38)
    
    ratio_w = max_allowed_w / img.width
    ratio_h = max_allowed_h / img.height
    scale = min(ratio_w, ratio_h)
    
    target_w = int(img.width * scale)
    target_h = int(img.height * scale)
    img = img.resize((target_w, target_h), Image.Resampling.LANCZOS)
    
    img = ImageOps.expand(img, border=12, fill=(255, 255, 255, 255))
    rotation_angle = random.uniform(-5.0, 5.0)
    img = img.rotate(rotation_angle, resample=Image.Resampling.BICUBIC, expand=True)
    
    return np.array(img)

def process_character_image(img_path, max_w, max_h):
    img = Image.open(img_path).convert("RGBA")
    
    ratio_w = max_w / img.width
    ratio_h = max_h / img.height
    scale = min(ratio_w, ratio_h)
    
    target_w = int(img.width * scale)
    target_h = int(img.height * scale)
    img = img.resize((target_w, target_h), Image.Resampling.LANCZOS)
    return np.array(img)

def create_word_data(text, font_path, max_width):
    target_size, min_size = 75, 55
    font = ImageFont.truetype(font_path, target_size) if os.path.exists(font_path) else ImageFont.load_default()
    tdraw = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    bbox = tdraw.textbbox((0, 0), text, font=font)
    w_text, h_text = bbox[2] - bbox[0], bbox[3] - bbox[1]

    if w_text > max_width:
        font = ImageFont.truetype(font_path, max(min_size, int(target_size * (max_width / w_text))))
        bbox = tdraw.textbbox((0, 0), text, font=font)
        w_text, h_text = bbox[2] - bbox[0], bbox[3] - bbox[1]

    canvas_w, canvas_h = int(w_text + 40), int(h_text + 40)
    img = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    text_color = (244, 169, 59) 
    draw.text((canvas_w // 2, canvas_h // 2), text, font=font, fill=text_color, stroke_width=7, stroke_fill=(0, 0, 0), anchor="mm")
    return np.array(img), canvas_w, canvas_h

def get_supported_files(folder_path, exts=None):
    """Helper to fetch supported files case-insensitively"""
    if exts is None:
        exts = {".jpg", ".jpeg", ".png", ".jfif", ".gif", ".webp"}
    files = []
    if not os.path.exists(folder_path):
        return []
        
    for fname in os.listdir(folder_path):
        ext = os.path.splitext(fname)[1].lower()
        if ext in exts:
            files.append(os.path.join(folder_path, fname))
            
    return sorted(files)

def generate_coldcase_video(
    audio_path, 
    bg_folder, 
    evidence_folder, 
    music_path, 
    credit_video_path,
    output_name,
    char_folder, 
    sub_folder,
):
    
    model = get_whisper_model()
    bg_files = get_supported_files(bg_folder)
    evidence_files = get_supported_files(evidence_folder)

    if not bg_files:
        raise ValueError(f"❌ Background pool empty: {bg_folder}")
    if not evidence_files:
        raise ValueError(f"❌ Evidence pool empty: {evidence_folder}")

    segments_gen, _ = model.transcribe(audio_path, word_timestamps=True)
    segments = list(segments_gen)
    all_words = [w for seg in segments for w in seg.words]
    
    speech_audio = AudioFileClip(audio_path)
    total_duration = speech_audio.duration - 0.05
    layer_clips, text_clips = [], []
    clips_to_close = []

    # ================= 1. Background Track (Every 5 sec - Sequential) =================
    bg_interval = 5.0
    num_bg_steps = math.ceil(total_duration / bg_interval)
    
    for i in range(num_bg_steps):
        t_start = i * bg_interval
        dur = min(bg_interval, total_duration - t_start)
        if dur <= 0:
            break
        
        chosen_bg = bg_files[i % len(bg_files)]
        
        if chosen_bg.lower().endswith('.gif'):
            clip = VideoFileClip(chosen_bg).with_start(t_start).with_duration(dur)
            ratio = max(SCREEN_W / clip.w, SCREEN_H / clip.h) * 1.1
            clip = clip.resized((int(clip.w * ratio), int(clip.h * ratio)))
            clips_to_close.append(clip)
        else:
            clip = ImageClip(process_bg_image(chosen_bg)).with_start(t_start).with_duration(dur)
            clip = apply_ken_burns(clip, dur)
            
        layer_clips.append(clip.with_position(("center", "center")).cropped(y1=0, y2=SCREEN_H, x1=0, x2=SCREEN_W))

    # ================= 2. Evidence Frame (Every 8 sec - Sequential) =================
    evidence_interval = 8.0
    num_evidence_steps = math.ceil(total_duration / evidence_interval)
    
    for i in range(num_evidence_steps):
        t_start = i * evidence_interval
        dur = min(evidence_interval, total_duration - t_start)
        if dur <= 0.5:
            break
        
        chosen_evidence = evidence_files[i % len(evidence_files)]
        
        if chosen_evidence.lower().endswith('.gif'):
            clip = VideoFileClip(chosen_evidence).with_start(t_start).with_duration(dur)
            max_allowed_w = int(SCREEN_W * 0.88)
            max_allowed_h = int(SCREEN_H * 0.38)
            scale = min(max_allowed_w / clip.w, max_allowed_h / clip.h)
            clip = clip.resized((int(clip.w * scale), int(clip.h * scale)))
            clips_to_close.append(clip)
        else:
            raw_img = process_evidence_image(chosen_evidence)
            clip = ImageClip(raw_img)
            
        clip = clip.with_start(t_start).with_duration(dur).with_position(("center", int(SCREEN_H * 0.12)))
        clip = clip.transform(lambda gf, t: gf(t) * min(1.0, t / 0.5))
        layer_clips.append(clip)

    # ================= 3. Character Poses =================
    detective_poses = get_supported_files(char_folder)
    subscribe_poses = get_supported_files(sub_folder)

    outro_threshold = max(0.0, total_duration - 5.0)

    CHAR_W, CHAR_H = 605, 871  
    SUB_W, SUB_H = 726, 992

    if detective_poses and outro_threshold > 0:
        char_interval = 4.0
        num_char_loops = math.ceil(outro_threshold / char_interval)
        
        for i in range(num_char_loops):
            c_start = i * char_interval
            c_dur = min(char_interval, outro_threshold - c_start)
            if c_dur <= 0:
                break
            
            chosen_pose = detective_poses[i % len(detective_poses)]
            
            det_arr = process_character_image(chosen_pose, CHAR_W, CHAR_H)
            char_clip = ImageClip(det_arr).with_start(c_start).with_duration(c_dur).with_position(("left", "bottom"))
            char_clip = char_clip.transform(lambda gf, t: gf(t) * min(1.0, t / 0.4))
            layer_clips.append(char_clip)

    if subscribe_poses:
        sub_arr = process_character_image(subscribe_poses[0], SUB_W, SUB_H)
        sub_clip = ImageClip(sub_arr).with_start(outro_threshold).with_duration(total_duration - outro_threshold).with_position(("center", "bottom"))
        sub_clip = sub_clip.transform(lambda gf, t: gf(t) * min(1.0, t / 0.4))
        layer_clips.append(sub_clip)
    elif detective_poses and outro_threshold == 0:
        det_arr = process_character_image(detective_poses[0], CHAR_W, CHAR_H)
        char_clip = ImageClip(det_arr).with_start(0).with_duration(total_duration).with_position(("left", "bottom"))
        layer_clips.append(char_clip)

    # ================= 4. Subtitle Engine =================
    active_font = FONTS[0] if os.path.exists(FONTS[0]) else "default"
    max_w = SCREEN_W - (SAFE_MARGIN * 2)
    
    current_sentence_words = []
    
    def process_and_flush_sentence(word_group):
        if not word_group:
            return
        sentence_end_time = word_group[-1]["end"]
        curr_x, curr_y = SAFE_MARGIN, BANNER_Y + 30
        line_h = 0
        
        for item in word_group:
            word_str = item["word"]
            arr, w, h = create_word_data(word_str, active_font, max_w)
            
            if curr_x + w > SCREEN_W - SAFE_MARGIN:
                curr_x = SAFE_MARGIN
                curr_y += line_h + 20
                line_h = 0
            
            line_h = max(line_h, h)
            duration = sentence_end_time - item["start"]
            if duration > 0:
                word_clip = (ImageClip(arr)
                             .with_start(item["start"])
                             .with_duration(duration)
                             .with_position((curr_x, curr_y)))
                text_clips.append(word_clip)
                
            curr_x += w + 20

    for w_obj in all_words:
        word_text = w_obj.word.strip().upper()
        if not word_text:
            continue
            
        current_sentence_words.append({
            "word": word_text,
            "start": w_obj.start,
            "end": w_obj.end
        })
        
        is_terminal = any(char in word_text for char in [".", "?", "!"])
        if is_terminal or len(current_sentence_words) >= 6:
            process_and_flush_sentence(current_sentence_words)
            current_sentence_words = []
            
    if current_sentence_words:
        process_and_flush_sentence(current_sentence_words)

    # ================= 5. Audio Compiling (Voice + Background Music) =================
    voice = speech_audio.with_duration(total_duration)
    
    bg_music_clip = None
    if music_path and os.path.exists(music_path):
        print(f"🎵 Adding background music: {os.path.basename(music_path)}")
        raw_music = AudioFileClip(music_path)
        
        if raw_music.duration < total_duration:
            bg_music_clip = raw_music.with_effects([
                afx.AudioLoop(duration=total_duration)
            ])
        else:
            bg_music_clip = raw_music.with_duration(total_duration)

        bg_music_clip = bg_music_clip.with_volume_scaled(0.30)

        # Combine voice (100% volume) and background music (30% volume)
        final_audio = CompositeAudioClip([
            voice.with_volume_scaled(1.0),
            bg_music_clip
        ])
    else:
        print("⚠️ No valid music file found, generating video with voiceover only.")
        final_audio = voice.with_volume_scaled(1.0)
    
    video = CompositeVideoClip(
        layer_clips + text_clips,
        size=(SCREEN_W, SCREEN_H)
    ).with_duration(total_duration).with_audio(final_audio)

    # Append credits if provided
    if credit_video_path and os.path.exists(credit_video_path):
        print("🎬 Appending credit video...")
        credit_clip = VideoFileClip(credit_video_path)

        if credit_clip.size != (SCREEN_W, SCREEN_H):
            credit_clip = credit_clip.resized((SCREEN_W, SCREEN_H))

        final_video = concatenate_videoclips(
            [video, credit_clip],
            method="compose"
        )
    else:
        final_video = video

    print("🎥 Writing final video...")
    final_video.write_videofile(
        output_name,
        fps=30,
        codec="libx264",
        audio_codec="aac",
        threads=4,
        preset="ultrafast"
    )

    # Cleanup memory
    final_video.close()

    if final_video is not video:
        credit_clip.close()

    video.close()

    if bg_music_clip:
        bg_music_clip.close()
    
    voice.close()
    speech_audio.close()
    
    if final_audio is not voice:
        final_audio.close()

    for c in clips_to_close:
        try:
            c.close()
        except Exception:
            pass

    print(f"✅ Video generation complete: {output_name}")
    return output_name


if __name__ == "__main__":
    test_audio = "audio/coldcases/audio (9).mp3"  
    bg_folder = "images/coldcases/background_images"
    evidence_folder = "images/coldcases/9" 
    char_folder = "images/coldcases/characters"
    sub_folder = "images/coldcases/characters/subscribe"
    music_folder = "background_music/coldcases"
    output = "final_test_video.mp4"
    
    selected_music = None
    music_files = get_supported_files(music_folder, exts={".mp3", ".wav", ".aac", ".m4a", ".ogg"})
    
    if music_files:
        selected_music = random.choice(music_files)
        print(f"🎧 Selected background track: {selected_music}")
    else:
        print(f"⚠️ Warning: No music files found in folder '{music_folder}'.")

    if not os.path.exists(bg_folder):
        print(f"❌ Error: Background folder not found: {bg_folder}")
    elif not os.path.exists(test_audio):
        print(f"❌ Error: Audio file '{test_audio}' not found.")
    else:
        print("🚀 Starting test generation...")
        try:
            generate_coldcase_video(
                audio_path=test_audio,
                bg_folder=bg_folder,
                evidence_folder=evidence_folder,
                music_path=selected_music,
                credit_video_path=None,
                output_name=output,
                char_folder=char_folder,
                sub_folder=sub_folder
            )
            print(f"✅ Success! Saved to {output}")
        except Exception as e:
            print(f"❌ Generation failed: {e}")
