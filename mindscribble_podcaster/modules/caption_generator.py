import os
import re
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from moviepy.video.VideoClip import ImageClip
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip
from faster_whisper import WhisperModel

# Dynamically set BASE_DIR relative to repository root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_DIR = os.path.dirname(BASE_DIR) 
FONTS_DIR = os.path.join(ROOT_DIR, "fonts")

# -------------------------------------------------------------------
# FONT PATHS (Assigned to distinct files)
# -------------------------------------------------------------------
DEFAULT_FONT = os.path.join(FONTS_DIR, "dejavu-sans-bold.ttf")         # Normal body font
SPECIAL_FONT = os.path.join(FONTS_DIR, "MilkyCoffee-X3mWd.otf")   
HIGHLIGHT_FONT = os.path.join(FONTS_DIR, "dejavu-sans-bold.ttf")      # Highlighted word font

# Fallback to system fonts if local font files are not found in repo
if not os.path.exists(DEFAULT_FONT):
    DEFAULT_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

if not os.path.exists(HIGHLIGHT_FONT):
    HIGHLIGHT_FONT = DEFAULT_FONT

if not os.path.exists(SPECIAL_FONT):
    SPECIAL_FONT = DEFAULT_FONT

# -------------------------------------------------------------------
# CONFIGURATIONS
# -------------------------------------------------------------------
CONFIG_STYLE_1 = {
    "style": "card_box",                       # Full line white background box
    "font_path": DEFAULT_FONT, 
    "highlight_font_path": HIGHLIGHT_FONT, 
    "special_font_path": SPECIAL_FONT,
    "font_size": 52,                           
    "special_font_size": 64,
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
    "font_path": DEFAULT_FONT, 
    "highlight_font_path": HIGHLIGHT_FONT, 
    "special_font_path": SPECIAL_FONT,
    "font_size": 56,                           
    "special_font_size": 68,
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

CONFIG_STYLE_PUNCHY = {
    "style": "active_word_box",                
    "font_path": DEFAULT_FONT, 
    "highlight_font_path": HIGHLIGHT_FONT, 
    "special_font_path": SPECIAL_FONT,
    "font_size": 62,                           # Larger font size for the blocky look
    "special_font_size": 74,
    "text_transform": "uppercase",             # Uppercase makes block fonts hit much harder
    "padding_x": 18,                           
    "padding_y": 10,                            
    "box_corner_radius": 8,                    
    "line_gap": 16,                            
    "max_words_per_batch": 3,                  # 3 words max so it flips fast and keeps attention
    "max_line_width": 850,                     
    "video_width": 1080,
    "video_height": 1920,
    "vertical_pos": 0.65,                      
    
    "text_color": (255, 255, 255, 255),        
    "active_text_color": (0, 0, 0, 255),       # Dark text inside the active highlight box
    "active_box_color": (255, 230, 0, 255),    # Vibrant yellow box backdrop for the spoken word
    "special_color": (255, 69, 0, 255)         # Orange-red pop for special keywords
}

CONFIG_SAMPLE_STYLE = {
    "style": "sample_highlight",
    "font_path": DEFAULT_FONT,
    "highlight_font_path": HIGHLIGHT_FONT,
    "special_font_path": SPECIAL_FONT,           # Dedicated path for special words
    "font_size": 60,                           # Normal text size
    "special_font_size": 90,                   # Larger size for special words
    "text_transform": "uppercase",
    "max_words_per_batch": 3,
    "max_line_width": 620,
    "video_width": 1080,
    "video_height": 1920,
    "vertical_pos": 0.68,
    
    "text_color": (255, 255, 255, 255),
    "highlight_color": (255, 215, 0, 255),     # Color when active
    "special_color": (255, 69, 0, 255),        # Color for special words
    "stroke_width": 5,
    "stroke_color": (0, 0, 0, 255)
}

CONFIG_STYLE_STACKED_GRADIENT = {
    "style": "stacked_gradient",                       # Stacked gradient shadow style
    "font_path": DEFAULT_FONT,
    "highlight_font_path": HIGHLIGHT_FONT,
    "special_font_path": SPECIAL_FONT,
    "font_size": 72,                           # Large, bold display size
    "special_font_size": 84,
    "text_transform": "uppercase",             
    "max_words_per_batch": 4,                  # Keeps a couple of words active to form the stack
    "max_line_width": 700,                     # Narrower width forces natural stacking
    "video_width": 1080,
    "video_height": 1920,
    "vertical_pos": 0.68,                      
    
    "text_color": (255, 255, 255, 255),        
    "highlight_color": (255, 216, 102, 255),   # Warm gold/yellow highlight
    "special_color": (255, 69, 0, 255),
    "stroke_width": 7,                         # Heavy outline for the 3D pop effect
    "stroke_color": (15, 15, 15, 240)          # Deep dark shadow
}

SPECIAL_WORDS = {"COMPANIES", "MONEY", "DOLLAR","PROCRASTINATION", "HURTS", "MIND", "PAIN", "PROBLEM" "MIND",
    "BRAIN",
    "MEMORY",
    "PROCRASTINATION",
    "DOPAMINE",
    "ADDICTION",
    "PAIN",
    "HURT",
    "HURTS",
    "FEAR",
    "ANXIETY",
    "STRESS",
    "SHAME",
    "GUILT",
    "REGRET",
    "DECEPTION",
    "DECOY",
    "DISTRACTION",
    "ATTENTION",
    "FOCUS",
    "LONELINESS",
    "DESIRE",
    "REWARD",
    "POWER",
    "DANGER",
    "SECRET",
    "HIDDEN",
    "TRUTH",
    "DESTROY"}

def clean_word(word):
    return re.sub(r'[^\w\s]', '', word).strip().upper()

def format_text(word, transform_type):
    if transform_type == "uppercase":
        return word.upper()
    elif transform_type == "lowercase":
        return word.lower()
    return word

# -------------------------------------------------------------------
# RENDERER ENGINES (Harmonized signatures accepting all 3 fonts)
# -------------------------------------------------------------------
def render_style_sample_highlight(lines, config, base_font, highlight_font, special_font, space_w, active_index):
    TRANSFORM = config.get("text_transform", "uppercase")
    LINE_GAP = config.get("line_gap", 12)

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
            
            if is_special:
                chosen_font = special_font
                color = config.get("special_color", (255, 69, 0, 255))
            elif is_active:
                chosen_font = highlight_font
                color = config.get("highlight_color", (255, 215, 0, 255))
            else:
                chosen_font = base_font
                color = config.get("text_color", (255, 255, 255, 255))

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

    total_card_h = sum(ld["height"] for ld in line_data) + ((len(line_data) - 1) * LINE_GAP)
    img = Image.new("RGBA", (config["video_width"], config["video_height"]), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    curr_y = int(config["video_height"] * config.get("vertical_pos", 0.70)) - (total_card_h // 2)
    
    # GUARD: Safely cap stroke width to prevent PIL array allocation crashes on custom fonts
    raw_stroke = config.get("stroke_width", 0)
    current_stroke_width = min(raw_stroke, 2) if raw_stroke > 0 else 0
    stroke_color = config.get("stroke_color", (0, 0, 0, 255))

    for ld in line_data:
        text_x = (config["video_width"] - ld["width"]) // 2
        for word, w_w, color, word_font, w_ascent in ld["words"]:
            safe_draw_text(
                    draw,
                    (text_x, curr_y - w_ascent),
                    word,
                    font=word_font,
                    fill=color,
                    fallback_font=base_font,
                    stroke_width=current_stroke_width,
                    stroke_fill=stroke_color,
                )
            text_x += w_w + space_w

        curr_y += ld["height"] + LINE_GAP

    return np.array(img)

def render_style_stacked_gradient(lines, config, base_font, highlight_font, special_font, space_w, active_index):
    TRANSFORM = config.get("text_transform", "uppercase")
    LINE_GAP = config.get("line_gap", 4)

    stacked_lines = []
    for line_items in lines:
        chunk_size = 2
        for i in range(0, len(line_items), chunk_size):
            stacked_lines.append(line_items[i:i + chunk_size])

    line_data = []
    for line_items in stacked_lines:
        words_in_line = []
        total_line_w = 0
        max_height = 0

        for idx, item in enumerate(line_items):
            raw_w = item["word"]
            formatted_w = format_text(raw_w, TRANSFORM)
            is_active = (item["global_idx"] == active_index)
            is_special = clean_word(raw_w) in SPECIAL_WORDS
            
            if is_special:
                chosen_font = special_font
                color = config.get("special_color", (255, 69, 0, 255))
            elif is_active:
                chosen_font = highlight_font
                color = config.get("highlight_color", (255, 215, 0, 255))
            else:
                chosen_font = base_font
                color = config.get("text_color", (255, 255, 255, 255))

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

    total_card_h = sum(ld["height"] for ld in line_data) + ((len(line_data) - 1) * LINE_GAP)
    img = Image.new("RGBA", (config["video_width"], config["video_height"]), (0, 0, 0, 0))
    
    overlay_draw = ImageDraw.Draw(img)
    curr_y = int(config["video_height"] * config.get("vertical_pos", 0.70)) - (total_card_h // 2) - 20
    box_h = total_card_h + 40
    
    overlay_draw.rectangle(
        [100, curr_y, config["video_width"] - 100, curr_y + box_h],
        fill=(0, 0, 0, 90)
    )

    draw = ImageDraw.Draw(img)
    stroke_width = config.get("stroke_width", 6)
    stroke_color = config.get("stroke_color", (0, 0, 0, 255))

    for ld in line_data:
        text_x = (config["video_width"] - ld["width"]) // 2
        for word, w_w, color, word_font, w_ascent in ld["words"]:
            draw.text(
                (text_x, curr_y - w_ascent), 
                word, 
                font=word_font, 
                fill=color,
                stroke_width=stroke_width,
                stroke_fill=stroke_color
            )
            text_x += w_w + space_w

        curr_y += ld["height"] + LINE_GAP

    return np.array(img)


def render_style_card_box(lines, config, base_font, highlight_font, special_font, space_w, active_index):
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
            
            if is_special:
                chosen_font = special_font
                color = config.get("special_color", (138, 43, 226, 255))
            elif is_active:
                chosen_font = highlight_font
                color = config.get("highlight_color", (217, 112, 118, 255))
            else:
                chosen_font = base_font
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


def render_style_active_word_box(lines, config, base_font, highlight_font, special_font, space_w, active_index):
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
            
            if is_special:
                chosen_font = special_font
                text_color = config.get("special_color", (255, 215, 0, 255))
            elif is_active:
                chosen_font = highlight_font
                text_color = config.get("active_text_color", (255, 255, 255, 255))
            else:
                chosen_font = base_font
                text_color = config.get("text_color", (255, 255, 255, 255))

            w_w = chosen_font.getlength(formatted_w) if hasattr(chosen_font, "getlength") else (chosen_font.getbbox(formatted_w)[2] - chosen_font.getbbox(formatted_w)[0])
            bbox = chosen_font.getbbox(formatted_w)
            w_h = bbox[3] - bbox[1]

            if w_h > max_height:
                max_height = w_h

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

        for w_info in ld["words"]:
            word = w_info["word"]
            w_w = w_info["width"]
            draw.text((text_x, curr_y + PAD_Y - w_info["ascent"]), word, font=w_info["font"], fill=w_info["text_color"])
            text_x += w_w + space_w

        curr_y += line_h + (PAD_Y * 2) + LINE_GAP

    return np.array(img)

def safe_draw_text(
    draw,
    position,
    text,
    font,
    fill,
    fallback_font=None,
    stroke_width=0,
    stroke_fill=None,
):
    """
    Safely render text with custom fonts.

    Some custom OTF/TTF fonts contain problematic glyph metrics that
    can cause Pillow/FreeType to attempt enormous memory allocations.
    """

    try:
        draw.text(
            position,
            text,
            font=font,
            fill=fill,
            stroke_width=stroke_width,
            stroke_fill=stroke_fill,
        )
        return font

    except (OSError, ValueError) as exc:
        print(
            f"[!] Font render failed for word={text!r} "
            f"font={getattr(font, 'path', 'unknown')}: {exc}"
        )

        # First fallback: base font
        if fallback_font is not None and fallback_font is not font:
            try:
                draw.text(
                    position,
                    text,
                    font=fallback_font,
                    fill=fill,
                    stroke_width=stroke_width,
                    stroke_fill=stroke_fill,
                )

                print(
                    f"    [+] Fallback font used for {text!r}: "
                    f"{getattr(fallback_font, 'path', 'unknown')}"
                )

                return fallback_font

            except (OSError, ValueError):
                pass

        # Final fallback
        try:
            default_font = ImageFont.load_default()

            draw.text(
                position,
                text,
                font=default_font,
                fill=fill,
                stroke_width=0,
            )

            print(f"    [+] PIL default font used for {text!r}")
            return default_font

        except Exception as final_exc:
            print(
                f"    [X] Could not render word {text!r}: "
                f"{final_exc}"
            )
            return None
# -------------------------------------------------------------------
# REGISTRY ROUTER
# -------------------------------------------------------------------
RENDER_REGISTRY = {
    "card_box": render_style_card_box,
    "active_word_box": render_style_active_word_box,
    "sample_highlight": render_style_sample_highlight,
    "stacked_gradient": render_style_stacked_gradient
}

def render_caption_card(word_batch, active_index, config):
    font_path = config.get("font_path")
    highlight_font_path = config.get("highlight_font_path", font_path)
    special_font_path = config.get("special_font_path", font_path)
    print(f"Using fonts: base={font_path}, highlight={highlight_font_path}, special={special_font_path}")
    FONT_SIZE = config.get("font_size", 52)
    HIGHLIGHT_FONT_SIZE = config.get("highlight_font_size", FONT_SIZE)
    SPECIAL_FONT_SIZE = config.get("special_font_size", FONT_SIZE)
    MAX_LINE_WIDTH = config.get("max_line_width", 850)

    try:
        base_font = ImageFont.truetype(font_path, FONT_SIZE)
    except Exception:
        base_font = ImageFont.load_default()

    try:
        highlight_font = ImageFont.truetype(highlight_font_path, HIGHLIGHT_FONT_SIZE)
    except Exception:
        highlight_font = base_font

    try:
        special_font = ImageFont.truetype(special_font_path, SPECIAL_FONT_SIZE)
    except Exception:
        special_font = base_font

    space_w = base_font.getlength(" ") if hasattr(base_font, "getlength") else (base_font.getbbox("a a")[2] - base_font.getbbox("aa")[2])

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

    style_key = config.get("style", "card_box")
    render_fn = RENDER_REGISTRY.get(style_key, render_style_card_box)
    
    return render_fn(lines, config, base_font, highlight_font, special_font, space_w, active_index)

# -------------------------------------------------------------------
# TIMING & MOVIEPY PIPELINE COMPOSITOR
# -------------------------------------------------------------------
def get_word_timestamps(audio_path, model_size="base"):
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments, _ = model.transcribe(audio_path, word_timestamps=True)
    
    words_data = []
    for segment in segments:
        if segment.words:
            for word_info in segment.words:
                cleaned = clean_word(word_info.word)
                if cleaned:
                    words_data.append({
                        "word": word_info.word.strip(),
                        "clean_word": cleaned,
                        "start": word_info.start,
                        "end": word_info.end
                    })
    return words_data

def align_timeline_with_audio(audio_path: str, timeline: list, model_size: str = "base") -> list:
    words_data = get_word_timestamps(audio_path, model_size=model_size)
    if not words_data:
        return timeline

    search_cursor = 0
    total_words = len(words_data)

    for seg in timeline:
        raw_words = seg.get("text", "").split()
        if not raw_words:
            continue

        first_word_clean = clean_word(raw_words[0])
        last_word_clean = clean_word(raw_words[-1])

        start_time = None
        end_time = None

        match_first_idx = None
        for i in range(search_cursor, total_words):
            if words_data[i]["clean_word"] == first_word_clean:
                match_first_idx = i
                start_time = words_data[i]["start"]
                break

        if start_time is None and search_cursor < total_words:
            start_time = words_data[search_cursor]["start"]
            match_first_idx = search_cursor

        scan_from = match_first_idx if match_first_idx is not None else search_cursor
        for j in range(scan_from, total_words):
            if words_data[j]["clean_word"] == last_word_clean:
                end_time = words_data[j]["end"]
                search_cursor = j + 1
                break

        if end_time is None:
            expected_word_count = len(raw_words)
            fallback_idx = min(scan_from + expected_word_count - 1, total_words - 1)
            end_time = words_data[fallback_idx]["end"]
            search_cursor = fallback_idx + 1

        seg["start"] = round(start_time, 2)
        seg["end"] = round(end_time, 2)

    return timeline

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