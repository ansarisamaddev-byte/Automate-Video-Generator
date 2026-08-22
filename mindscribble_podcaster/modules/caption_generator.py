import os
import re
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from moviepy.video.VideoClip import ImageClip
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip
from faster_whisper import WhisperModel

# -------------------------------------------------------------------
# CONFIGURATIONS
# -------------------------------------------------------------------
CONFIG_STYLE_1 = {
    "style": "card_box",                        # Full line white background box
    "font_path": "D:\\AI\\Automate-Video-Generator\\fonts\\dejavu-sans-bold.ttf", 
    "highlight_font_path": "D:\\AI\\Automate-Video-Generator\\fonts\\MilkyCoffee-X3mWd.otf", 
    "font_size": 52,                           
    "text_transform": "uppercase",             # 'uppercase', 'lowercase', 'none'
    "padding_x": 24,                           
    "padding_y": 12,                           
    "line_gap": 8,                             
    "max_words_per_batch": 5,                  
    "max_line_width": 850,                     
    "video_width": 1080,
    "video_height": 1920,
    "vertical_pos": 0.50,                      
    
    "bg_color": (255, 255, 255, 255),          
    "text_color": (26, 26, 26, 255),           
    "highlight_color": (217, 112, 118, 255),   
    "special_color": (138, 43, 226, 255)       
}

CONFIG_STYLE_2 = {
    "style": "active_word_box",                # Floating text + active word orange box
    "font_path": "D:\\AI\\Automate-Video-Generator\\fonts\\dejavu-sans-bold.ttf", 
    "highlight_font_path": "D:\\AI\\Automate-Video-Generator\\fonts\\MilkyCoffee-X3mWd.otf", 
    "font_size": 56,                           
    "text_transform": "lowercase",             
    "padding_x": 16,                           
    "padding_y": 8,                            
    "box_corner_radius": 10,                   
    "line_gap": 16,                            
    "max_words_per_batch": 5,                  
    "max_line_width": 800,                     
    "video_width": 1080,
    "video_height": 1920,
    "vertical_pos": 0.70,                      
    
    "text_color": (255, 255, 255, 255),        
    "active_text_color": (255, 255, 255, 255), 
    "active_box_color": (255, 87, 34, 255),    
    "special_color": (255, 215, 0, 255)        
}

SPECIAL_WORDS = {"PROCRASTINATION", "HURTS", "MIND", "PAIN", "PROBLEM"}

def clean_word(word):
    return re.sub(r'[^\w\s]', '', word).strip().upper()

def format_text(word, transform_type):
    if transform_type == "uppercase":
        return word.upper()
    elif transform_type == "lowercase":
        return word.lower()
    return word

# -------------------------------------------------------------------
# 1. RENDERER ENGINE 1: CARD BOX (Full Line Box)
# -------------------------------------------------------------------
def render_style_card_box(lines, config, base_font, special_font, space_w, active_index):
    PAD_X = config.get("padding_x", 24)
    PAD_Y = config.get("padding_y", 12)
    LINE_GAP = config.get("line_gap", 8)
    TRANSFORM = config.get("text_transform", "uppercase")

    line_data = []
    for line_items in lines:
        words_in_line = []
        total_line_w = 0
        max_height = 0

        for idx, item in enumerate(line_items):
            raw_w = item["word"]
            formatted_w = format_text(raw_w, TRANSFORM)
            is_active = (item["global_idx"] == active_index)
            is_special = clean_word(raw_w) in SPECIAL_WORDS
            
            chosen_font = special_font if is_special else base_font
            
            if is_special:
                color = config.get("special_color", (138, 43, 226, 255))
            elif is_active:
                color = config.get("highlight_color", (217, 112, 118, 255))
            else:
                color = config.get("text_color", (26, 26, 26, 255))

            w_w = chosen_font.getlength(formatted_w) if hasattr(chosen_font, "getlength") else (chosen_font.getbbox(formatted_w)[2] - chosen_font.getbbox(formatted_w)[0])
            bbox = chosen_font.getbbox(formatted_w)
            w_h = bbox[3] - bbox[1]

            if w_h > max_height:
                max_height = w_h
            
            words_in_line.append((formatted_w, w_w, color, chosen_font, bbox[1]))
            total_line_w += w_w
            if idx < len(line_items) - 1:
                total_line_w += space_w

        line_data.append({"words": words_in_line, "width": total_line_w, "height": max_height})

    total_card_h = sum(ld["height"] + (PAD_Y * 2) for ld in line_data) + ((len(line_data) - 1) * LINE_GAP)
    img = Image.new("RGBA", (config["video_width"], config["video_height"]), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    curr_y = int(config["video_height"] * config.get("vertical_pos", 0.50)) - (total_card_h // 2)

    for ld in line_data:
        box_w = ld["width"] + (PAD_X * 2)
        box_h = ld["height"] + (PAD_Y * 2)
        box_x = (config["video_width"] - box_w) // 2

        draw.rectangle([box_x, curr_y, box_x + box_w, curr_y + box_h], fill=config.get("bg_color", (255, 255, 255, 255)))

        text_x = box_x + PAD_X
        for word, w_w, color, word_font, w_ascent in ld["words"]:
            draw.text((text_x, curr_y + PAD_Y - w_ascent), word, font=word_font, fill=color)
            text_x += w_w + space_w

        curr_y += box_h + LINE_GAP

    return np.array(img)

# -------------------------------------------------------------------
# 2. RENDERER ENGINE 2: ACTIVE WORD BOX (Floating text + orange box)
# -------------------------------------------------------------------
def render_style_active_word_box(lines, config, base_font, special_font, space_w, active_index):
    PAD_X = config.get("padding_x", 16)
    PAD_Y = config.get("padding_y", 8)
    RADIUS = config.get("box_corner_radius", 10)
    LINE_GAP = config.get("line_gap", 16)
    TRANSFORM = config.get("text_transform", "lowercase")

    line_data = []
    for line_items in lines:
        words_in_line = []
        total_line_w = 0
        max_height = 0

        for idx, item in enumerate(line_items):
            raw_w = item["word"]
            formatted_w = format_text(raw_w, TRANSFORM)
            is_active = (item["global_idx"] == active_index)
            is_special = clean_word(raw_w) in SPECIAL_WORDS
            
            chosen_font = special_font if is_special else base_font
            w_w = chosen_font.getlength(formatted_w) if hasattr(chosen_font, "getlength") else (chosen_font.getbbox(formatted_w)[2] - chosen_font.getbbox(formatted_w)[0])
            bbox = chosen_font.getbbox(formatted_w)
            w_h = bbox[3] - bbox[1]

            if w_h > max_height:
                max_height = w_h
            
            if is_active:
                text_color = config.get("active_text_color", (255, 255, 255, 255))
            elif is_special:
                text_color = config.get("special_color", (255, 215, 0, 255))
            else:
                text_color = config.get("text_color", (255, 255, 255, 255))

            words_in_line.append({
                "word": formatted_w,
                "width": w_w,
                "is_active": is_active,
                "text_color": text_color,
                "font": chosen_font,
                "ascent": bbox[1]
            })
            
            total_line_w += w_w
            if idx < len(line_items) - 1:
                total_line_w += space_w

        line_data.append({"words": words_in_line, "width": total_line_w, "height": max_height})

    total_card_h = sum(ld["height"] + (PAD_Y * 2) for ld in line_data) + ((len(line_data) - 1) * LINE_GAP)
    img = Image.new("RGBA", (config["video_width"], config["video_height"]), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    curr_y = int(config["video_height"] * config.get("vertical_pos", 0.70)) - (total_card_h // 2)

    for ld in line_data:
        line_w = ld["width"]
        line_h = ld["height"]
        text_x = (config["video_width"] - line_w) // 2

        # Pass 1: Draw Active Box Background (so line layout never twitches)
        temp_x = text_x
        for w_info in ld["words"]:
            w_w = w_info["width"]
            if w_info["is_active"]:
                draw.rounded_rectangle(
                    [temp_x - PAD_X, curr_y, temp_x + w_w + PAD_X, curr_y + line_h + (PAD_Y * 2)], 
                    radius=RADIUS, 
                    fill=config.get("active_box_color", (255, 87, 34, 255))
                )
            temp_x += w_w + space_w

        # Pass 2: Draw Floating Text
        for w_info in ld["words"]:
            word = w_info["word"]
            w_w = w_info["width"]
            draw.text((text_x, curr_y + PAD_Y - w_info["ascent"]), word, font=w_info["font"], fill=w_info["text_color"])
            text_x += w_w + space_w

        curr_y += line_h + (PAD_Y * 2) + LINE_GAP

    return np.array(img)

# -------------------------------------------------------------------
# 3. REGISTRY ROUTER (ADD FUTURE RENDERERS HERE)
# -------------------------------------------------------------------
RENDER_REGISTRY = {
    "card_box": render_style_card_box,
    "active_word_box": render_style_active_word_box,
    # Example for future styles:
    # "neon_glow": render_style_neon_glow,
}

def render_caption_card(word_batch, active_index, config):
    font_path = config.get("font_path")
    highlight_font_path = config.get("highlight_font_path", font_path)
    FONT_SIZE = config.get("font_size", 52)
    MAX_LINE_WIDTH = config.get("max_line_width", 850)

    try:
        base_font = ImageFont.truetype(font_path, FONT_SIZE)
    except Exception:
        base_font = ImageFont.load_default()

    try:
        special_font = ImageFont.truetype(highlight_font_path, FONT_SIZE)
    except Exception:
        special_font = base_font

    space_w = base_font.getlength(" ") if hasattr(base_font, "getlength") else (base_font.getbbox("a a")[2] - base_font.getbbox("aa")[2])

    # Dynamic line calculation
    lines, current_line, current_line_width = [], [], 0
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

    # Dispatch strategy based on "style" key
    style_key = config.get("style", "card_box")
    render_fn = RENDER_REGISTRY.get(style_key, render_style_card_box)
    
    return render_fn(lines, config, base_font, special_font, space_w, active_index)

# -------------------------------------------------------------------
# 4. MOVIEPY PIPELINE COMPOSITOR
# -------------------------------------------------------------------
def get_word_timestamps(audio_path, model_size="base"):
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments, _ = model.transcribe(audio_path, word_timestamps=True)
    
    words_data = []
    for segment in segments:
        for word_info in segment.words:
            words_data.append({"word": word_info.word.strip(), "start": word_info.start, "end": word_info.end})
    return words_data

def generate_caption_overlay(audio_path, config=CONFIG_STYLE_1):
    words_data = get_word_timestamps(audio_path)
    if not words_data:
        return None

    for idx, w in enumerate(words_data):
        w["global_idx"] = idx

    batches = [words_data[i : i + config["max_words_per_batch"]] 
               for i in range(0, len(words_data), config["max_words_per_batch"])]

    clip_list = []
    for batch in batches:
        for word_info in batch:
            start_t = word_info["start"]
            end_t = word_info["end"]
            duration = max(0.05, end_t - start_t)

            img_np = render_caption_card(batch, active_index=word_info["global_idx"], config=config)
            clip_list.append(ImageClip(img_np).with_start(start_t).with_duration(duration))

    return CompositeVideoClip(clip_list, size=(config["video_width"], config["video_height"]))