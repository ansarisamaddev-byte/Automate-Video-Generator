import os
import re
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from moviepy.video.VideoClip import ImageClip
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip
from faster_whisper import WhisperModel


# -------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------
STYLE_CONFIG = {
    # Fonts
    "font_path": "D:\\AI\\Automate-Video-Generator\\fonts\\dejavu-sans-bold.ttf", 
    "highlight_font_path": "D:\\AI\\Automate-Video-Generator\\fonts\\MilkyCoffee-X3mWd.otf", 

    # Sizing & Spacing
    "font_size": 52,                           
    "padding_x": 24,                           # Clear side margin
    "padding_y": 12,                           # Top/bottom padding
    "line_gap": 8,                             # Vertical spacing between line boxes
    "max_words_per_batch": 5,                  # Total words on screen before clearing
    "max_line_width": 850,                     # Max pixel width before wrapping to next line
    "video_width": 1080,
    "video_height": 1920,
    "vertical_pos": 0.60,                      # Position (60% down screen)
    
    # Colors
    "bg_color": (255, 255, 255, 255),          # Solid white background box
    "text_color": (26, 26, 26, 255),           # Deep jet black for inactive words
    "highlight_color": (217, 112, 118, 255),   # Coral red for currently active words
    "special_color": (138, 43, 226, 255)       # Purple/Accent color exclusively for special words
}

# Special words: Custom Font + Custom Special Color
SPECIAL_WORDS = {"PROCRASTINATION", "HURTS", "MIND", "PAIN", "PROBLEM"}

def clean_word(word):
    return re.sub(r'[^\w\s]', '', word).upper()

# -------------------------------------------------------------------
# 1. FASTER-WHISPER TIMESTAMP EXTRACTION
# -------------------------------------------------------------------
def get_word_timestamps(audio_path, model_size="base"):
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments, _ = model.transcribe(audio_path, word_timestamps=True)
    
    words_data = []
    for segment in segments:
        for word_info in segment.words:
            words_data.append({
                "word": word_info.word.strip().upper(),
                "start": word_info.start,
                "end": word_info.end
            })
    return words_data

# -------------------------------------------------------------------
# 2. INDIVIDUAL LINE CARD RENDERING
# -------------------------------------------------------------------
def render_caption_card(word_batch, active_index, config):
    font_path = config.get("font_path")
    highlight_font_path = config.get("highlight_font_path", font_path)
    
    FONT_SIZE = config.get("font_size", 52)
    PADDING_X = config.get("padding_x", 24)
    PADDING_Y = config.get("padding_y", 12)
    LINE_GAP = config.get("line_gap", 8)
    MAX_LINE_WIDTH = config.get("max_line_width", 850)

    try:
        base_font = ImageFont.truetype(font_path, FONT_SIZE)
    except Exception:
        base_font = ImageFont.load_default()

    try:
        special_font = ImageFont.truetype(highlight_font_path, FONT_SIZE)
    except Exception:
        special_font = base_font

    space_w = base_font.getlength(" ") if hasattr(base_font, "getlength") else (base_font.getbbox("A A")[2] - base_font.getbbox("AA")[2])

    # -------------------------------------------------------------------
    # DYNAMIC LINE WRAPPING
    # -------------------------------------------------------------------
    lines = []
    current_line = []
    current_line_width = 0

    for item in word_batch:
        w = item["word"]
        is_special = clean_word(w) in SPECIAL_WORDS
        
        target_font = special_font if is_special else base_font
        w_w = target_font.getlength(w) if hasattr(target_font, "getlength") else (target_font.getbbox(w)[2] - target_font.getbbox(w)[0])
        
        test_width = current_line_width + w_w + (space_w if current_line else 0)
        
        if current_line and test_width > MAX_LINE_WIDTH:
            lines.append(current_line)
            current_line = [item]
            current_line_width = w_w
        else:
            current_line.append(item)
            current_line_width = test_width

    if current_line:
        lines.append(current_line)

    # -------------------------------------------------------------------
    # CALCULATE METRICS & ASSIGN STYLES
    # -------------------------------------------------------------------
    line_data = []

    for line_items in lines:
        words_in_line = []
        total_line_w = 0
        max_ascent = 0
        max_height = 0

        for idx, item in enumerate(line_items):
            w = item["word"]
            is_active = (item["global_idx"] == active_index)
            is_special = clean_word(w) in SPECIAL_WORDS
            
            # 1. Font Family Logic
            chosen_font = special_font if is_special else base_font
            
            # 2. Text Color Logic
            if is_special:
                chosen_color = config.get("special_color", (138, 43, 226, 255))
            elif is_active:
                chosen_color = config.get("highlight_color", (217, 112, 118, 255))
            else:
                chosen_color = config.get("text_color", (26, 26, 26, 255))

            w_w = chosen_font.getlength(w) if hasattr(chosen_font, "getlength") else (chosen_font.getbbox(w)[2] - chosen_font.getbbox(w)[0])
            
            bbox = chosen_font.getbbox(w)
            w_ascent = bbox[1]
            w_h = bbox[3] - bbox[1]

            if w_ascent < max_ascent:
                max_ascent = w_ascent
            if w_h > max_height:
                max_height = w_h
            
            words_in_line.append((w, w_w, chosen_color, chosen_font, w_ascent))
            
            total_line_w += w_w
            if idx < len(line_items) - 1:
                total_line_w += space_w

        line_data.append({
            "words": words_in_line,
            "width": total_line_w,
            "height": max_height,
            "top_ascent": max_ascent
        })

    total_card_h = sum(ld["height"] + (PADDING_Y * 2) for ld in line_data) + ((len(line_data) - 1) * LINE_GAP)
    
    img = Image.new("RGBA", (config["video_width"], config["video_height"]), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    start_y = int(config["video_height"] * config.get("vertical_pos", 0.60)) - (total_card_h // 2)
    curr_y = start_y

    for ld in line_data:
        box_w = ld["width"] + (PADDING_X * 2)
        box_h = ld["height"] + (PADDING_Y * 2)
        box_x = (config["video_width"] - box_w) // 2

        # Background box
        draw.rectangle(
            [box_x, curr_y, box_x + box_w, curr_y + box_h],
            fill=config.get("bg_color", (255, 255, 255, 255))
        )

        text_x = box_x + PADDING_X

        for word, w_w, color, word_font, w_ascent in ld["words"]:
            text_y = curr_y + PADDING_Y - w_ascent
            draw.text((text_x, text_y), word, font=word_font, fill=color)
            text_x += w_w + space_w

        curr_y += box_h + LINE_GAP

    return np.array(img)

# -------------------------------------------------------------------
# 3. MOVIEPY TIMELINE COMPOSITOR
# -------------------------------------------------------------------
def generate_caption_overlay(audio_path, config=STYLE_CONFIG):
    words_data = get_word_timestamps(audio_path)
    if not words_data:
        return None

    for idx, w in enumerate(words_data):
        w["global_idx"] = idx

    batches = [words_data[i : i + config["max_words_per_batch"]] 
               for i in range(0, len(words_data), config["max_words_per_batch"])]

    clip_list = []

    for batch in batches:
        for idx, word_info in enumerate(batch):
            start_t = word_info["start"]
            end_t = word_info["end"]
            duration = max(0.05, end_t - start_t)

            img_np = render_caption_card(batch, active_index=word_info["global_idx"], config=config)

            ic = ImageClip(img_np).with_start(start_t).with_duration(duration)
            clip_list.append(ic)

    return CompositeVideoClip(clip_list, size=(config["video_width"], config["video_height"]))