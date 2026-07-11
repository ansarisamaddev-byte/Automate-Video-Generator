import os
import glob
import random
import datetime
import math
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageOps
from moviepy import VideoFileClip, concatenate_videoclips

# ================= MOVIEPY 2.X & WHISPER =================
from moviepy import AudioFileClip, ImageClip, CompositeVideoClip, CompositeAudioClip
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
    except:
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

def get_supported_files(folder_path):
    """Helper to fetch images including GIFs case-insensitively"""
    extensions = ["*.jpg", "*.jpeg", "*.png", "*.jfif", "*.gif", "*.GIF"]
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(folder_path, ext)))
    return sorted(list(set(files)))

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
    clips_to_close = []  # Keep track of dynamically opened video clips (like animated GIFs)

    # 1. Background Track
    bg_interval = 5.0
    for i in range(math.ceil(total_duration / bg_interval)):
        t_start = i * bg_interval
        dur = min(bg_interval, total_duration - t_start)
        if dur <= 0: break
        
        chosen_bg = random.choice(bg_files)
        
        # Check if background is an animated GIF
        if chosen_bg.lower().endswith('.gif'):
            clip = VideoFileClip(chosen_bg).with_start(t_start).with_duration(dur)
            # Resize to fill screen
            ratio = max(SCREEN_W / clip.w, SCREEN_H / clip.h) * 1.1
            clip = clip.resized(newsize=(int(clip.w * ratio), int(clip.h * ratio)))
            clips_to_close.append(clip)
        else:
            clip = ImageClip(process_bg_image(chosen_bg)).with_start(t_start).with_duration(dur)
            clip = apply_ken_burns(clip, dur)
            
        layer_clips.append(clip.with_position(("center", "center")).cropped(y1=0, y2=SCREEN_H, x1=0, x2=SCREEN_W))

    # 2. Evidence Frame
    evidence_interval = 12.0
    for i in range(math.ceil(total_duration / evidence_interval)):
        t_start = i * evidence_interval
        dur = min(evidence_interval, total_duration - t_start)
        if dur <= 1.0: break
        
        chosen_evidence = random.choice(evidence_files)
        
        if chosen_evidence.lower().endswith('.gif'):
            clip = VideoFileClip(chosen_evidence).with_start(t_start).with_duration(dur)
            # Scale down to fit evidence bounds
            max_allowed_w = int(SCREEN_W * 0.88)
            max_allowed_h = int(SCREEN_H * 0.38)
            scale = min(max_allowed_w / clip.w, max_allowed_h / clip.h)
            clip = clip.resized(newsize=(int(clip.w * scale), int(clip.h * scale)))
            clips_to_close.append(clip)
        else:
            raw_img = process_evidence_image(chosen_evidence)
            clip = ImageClip(raw_img)
            
        clip = clip.with_start(t_start).with_duration(dur).with_position(("center", int(SCREEN_H * 0.12)))
        clip = clip.transform(lambda gf, t: gf(t) * min(1.0, t / 0.5))
        layer_clips.append(clip)

    detective_poses = get_supported_files(char_folder)
    subscribe_poses = get_supported_files(sub_folder)

    outro_threshold = max(0.0, total_duration - 5.0)

    CHAR_W, CHAR_H = 605, 871  
    SUB_W, SUB_H = 726, 992

    if detective_poses and outro_threshold > 0:
        char_interval = 4.0
        num_char_loops = math.ceil(outro_threshold / char_interval)
        
        shuffled_detective_poses = detective_poses.copy()
        random.shuffle(shuffled_detective_poses)
        
        for i in range(num_char_loops):
            c_start = i * char_interval
            c_dur = min(char_interval, outro_threshold - c_start)
            if c_dur <= 0: break
            
            if not shuffled_detective_poses:
                shuffled_detective_poses = detective_poses.copy()
                random.shuffle(shuffled_detective_poses)
                
            chosen_pose = shuffled_detective_poses.pop(0)
            
            # Character images are processed as static frames via PIL
            det_arr = process_character_image(chosen_pose, CHAR_W, CHAR_H)
            char_clip = ImageClip(det_arr).with_start(c_start).with_duration(c_dur).with_position(("left", "bottom"))
            char_clip = char_clip.transform(lambda gf, t: gf(t) * min(1.0, t / 0.4))
            layer_clips.append(char_clip)

    if subscribe_poses:
        sub_arr = process_character_image(random.choice(subscribe_poses), SUB_W, SUB_H)
        sub_clip = ImageClip(sub_arr).with_start(outro_threshold).with_duration(total_duration - outro_threshold).with_position(("center", "bottom"))
        sub_clip = sub_clip.transform(lambda gf, t: gf(t) * min(1.0, t / 0.4))
        layer_clips.append(sub_clip)
    elif detective_poses and outro_threshold == 0:
        det_arr = process_character_image(random.choice(detective_poses), CHAR_W, CHAR_H)
        char_clip = ImageClip(det_arr).with_start(0).with_duration(total_duration).with_position(("left", "bottom"))
        layer_clips.append(char_clip)

    # 4. Accumulative Subtitle Engine
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

    # Audio Compiling & Windows Safe Closing
    voice = speech_audio.with_duration(total_duration)
    
    bg_music_clip = None
    if music_path:
        bg_music_clip = AudioFileClip(music_path).with_volume_scaled(0.08).with_duration(total_duration)
        final_audio = CompositeAudioClip([voice, bg_music_clip])
    else:
        final_audio = voice
    
    video = CompositeVideoClip(
    layer_clips + text_clips,
    size=(SCREEN_W, SCREEN_H)
    ).with_duration(total_duration).with_audio(final_audio)

    # Append credits if provided
    if credit_video_path and os.path.exists(credit_video_path):
        print("🎬 Appending credit video...")
        credit_clip = VideoFileClip(credit_video_path)

        # Resize if necessary
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

    # 1. Close Final Video
    final_video.close()

    # 2. Safely close credit clip using 'is not' (Fixes the TypeError)
    if final_video is not video:
        credit_clip.close()

    video.close()

    # 3. Safely close audio clips using 'is not' (Fixes the TypeError)
    if music_path and bg_music_clip:
        bg_music_clip.close()
    
    voice.close()
    speech_audio.close()
    
    if final_audio is not voice:
        final_audio.close()

    # Close any GIF video clips opened dynamically
    for c in clips_to_close:
        try:
            c.close()
        except:
            pass

    print(f"✅ Video generation complete: {output_name}")
    return output_name


if __name__ == "__main__":
    # Define test parameters
    test_audio = "audio/coldcases/audio (9).mp3"  
    bg_folder = "images/coldcases/background_images"
    evidence_folder = "images/coldcases/9" 
    char_folder = "images/coldcases/characters"
    sub_folder = "images/coldcases/characters/subscribe"
    output = "final_test_video.mp4"
    
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
                music_path=None, 
                credit_video_path=None,
                output_name=output,
                char_folder=char_folder,
                sub_folder=sub_folder
            )
            print(f"✅ Success! Saved to {output}")
        except Exception as e:
            print(f"❌ Generation failed: {e}")