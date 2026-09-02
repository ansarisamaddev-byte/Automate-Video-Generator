import os
import re
import numpy as np
import math
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from moviepy.video.VideoClip import ImageClip, VideoClip
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip
from faster_whisper import WhisperModel
from moviepy.video.fx import FadeIn

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_DIR = os.path.dirname(BASE_DIR) 
FONTS_DIR = os.path.join(ROOT_DIR, "fonts")

DEFAULT_FONT = os.path.join(FONTS_DIR, "dejavu-sans-bold.ttf")
SPECIAL_FONT = os.path.join(FONTS_DIR, "MilkyCoffee-X3mWd.otf")   
HIGHLIGHT_FONT = os.path.join(FONTS_DIR, "dejavu-sans-bold.ttf")

if not os.path.exists(DEFAULT_FONT):
    DEFAULT_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

if not os.path.exists(HIGHLIGHT_FONT):
    HIGHLIGHT_FONT = DEFAULT_FONT

if not os.path.exists(SPECIAL_FONT):
    SPECIAL_FONT = DEFAULT_FONT

CONFIG_STYLE_1 = {
    "style": "card_box",
    "font_path": DEFAULT_FONT, 
    "highlight_font_path": HIGHLIGHT_FONT, 
    "special_font_path": SPECIAL_FONT,
    "font_size": 52,                           
    "special_font_size": 64,
    "text_transform": "uppercase",
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
    "style": "active_word_box",
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
    "font_size": 62,                           
    "special_font_size": 74,
    "text_transform": "uppercase",             
    "padding_x": 18,                           
    "padding_y": 10,                            
    "box_corner_radius": 8,                    
    "line_gap": 16,                            
    "max_words_per_batch": 3,                  
    "max_line_width": 850,                     
    "video_width": 1080,
    "video_height": 1920,
    "vertical_pos": 0.65,                      
    "text_color": (255, 255, 255, 255),        
    "active_text_color": (0, 0, 0, 255),       
    "active_box_color": (255, 230, 0, 255),    
    "special_color": (255, 69, 0, 255)         
}

CONFIG_SAMPLE_STYLE = {
    "style": "sample_highlight",
    "font_path": DEFAULT_FONT,
    "highlight_font_path": HIGHLIGHT_FONT,
    "special_font_path": SPECIAL_FONT,
    "font_size": 60,
    "special_font_size": 90,
    "text_transform": "uppercase",
    "max_words_per_batch": 3,
    "max_line_width": 620,
    "video_width": 1080,
    "video_height": 1920,
    "vertical_pos": 0.68,
    "heading_font_path": DEFAULT_FONT,
    "heading_font_size": 92,
    "heading_text_transform": "uppercase",
    "heading_max_width": 900,
    "heading_line_gap": 8,
    "heading_fill_color": (255, 159, 28, 255),
    "heading_outline_color": (58, 128, 222, 255),
    "heading_outline_width": 11,
    "heading_inner_stroke_color": (30, 20, 60, 255),
    "heading_inner_stroke_width": 3,
    "heading_shadow_color": (0, 0, 0, 150),
    "heading_shadow_offset": (0, 10),
    "heading_shadow_blur": 3,
    "heading_bg_color": None,
    "heading_padding_x": 30,
    "heading_padding_y": 30,
    "heading_y": 170,
    "heading_slide_distance": 288,
    "heading_animation_duration": 1.3,
    "heading_lead_time": 0.0,
    "text_color": (255, 255, 255, 255),
    "highlight_color": (255, 215, 0, 255),
    "special_color": (255, 69, 0, 255),
    "stroke_width": 5,
    "stroke_color": (0, 0, 0, 255)
}

CONFIG_STYLE_STACKED_GRADIENT = {
    "style": "stacked_gradient",                       
    "font_path": DEFAULT_FONT,
    "highlight_font_path": HIGHLIGHT_FONT,
    "special_font_path": SPECIAL_FONT,
    "font_size": 72,                           
    "special_font_size": 84,
    "text_transform": "uppercase",             
    "max_words_per_batch": 4,                  
    "max_line_width": 700,                     
    "video_width": 1080,
    "video_height": 1920,
    "vertical_pos": 0.68,                      
    "text_color": (255, 255, 255, 255),        
    "highlight_color": (255, 216, 102, 255),   
    "special_color": (255, 69, 0, 255),
    "stroke_width": 7,                         
    "stroke_color": (15, 15, 15, 240)          
}

SPECIAL_WORDS = {
    "COMPANIES", "MONEY", "DOLLAR", "PROCRASTINATION", "HURTS", "MIND", "PAIN", "PROBLEM",
    "BRAIN", "MEMORY", "DOPAMINE", "ADDICTION", "HURT", "FEAR", "ANXIETY", "STRESS", 
    "SHAME", "GUILT", "REGRET", "DECEPTION", "DECOY", "DISTRACTION", "ATTENTION", "FOCUS", 
    "LONELINESS", "DESIRE", "REWARD", "POWER", "DANGER", "SECRET", "HIDDEN", "TRUTH", "DESTROY"
}

def clean_word(word):
    return re.sub(r'[^\w\s]', '', str(word)).strip().upper()

def format_text(word, transform_type):
    if transform_type == "uppercase":
        return word.upper()
    elif transform_type == "lowercase":
        return word.lower()
    return word

def normalize_match_word(word):
    if not word:
        return ""
    return re.sub(r"[^a-z0-9]", "", str(word).lower())

def words_contain_match(heading_word, whisper_word):
    if not heading_word or not whisper_word:
        return False
    return heading_word in whisper_word or whisper_word in heading_word

def find_heading_start_time(heading_text, word_timestamps):
    if not heading_text or not word_timestamps:
        return None

    if isinstance(heading_text, list):
        heading_text = " ".join(str(x) for x in heading_text)

    raw_heading_words = str(heading_text).split()
    heading_words = [normalize_match_word(w) for w in raw_heading_words if normalize_match_word(w)]

    if not heading_words:
        return None

    transcript_words = []
    for item in word_timestamps:
        norm_w = normalize_match_word(item.get("word", ""))
        if norm_w:
            transcript_words.append({
                "raw": item.get("word", ""),
                "word": norm_w,
                "start": float(item["start"]),
                "end": float(item["end"])
            })

    if not transcript_words:
        return None

    req_count = len(heading_words)
    for i in range(len(transcript_words) - req_count + 1):
        candidate = transcript_words[i:i + req_count]
        matched = True
        for hw, tw in zip(heading_words, candidate):
            if not words_contain_match(hw, tw["word"]):
                matched = False
                break
        if matched:
            return candidate[0]["start"]

    return None

def safe_draw_text(draw, position, text, font, fill, fallback_font=None, stroke_width=0, stroke_fill=None):
    try:
        draw.text(position, text, font=font, fill=fill, stroke_width=stroke_width, stroke_fill=stroke_fill)
        return font
    except Exception:
        if fallback_font is not None:
            try:
                draw.text(position, text, font=fallback_font, fill=fill, stroke_width=stroke_width, stroke_fill=stroke_fill)
                return fallback_font
            except Exception:
                pass
        default_font = ImageFont.load_default()
        draw.text(position, text, font=default_font, fill=fill, stroke_width=0)
        return default_font

def render_heading_image(text: str, config: dict) -> np.ndarray:
    font_path = config.get("heading_font_path", config.get("font_path"))
    font_size = config.get("heading_font_size", 92)

    try:
        font = ImageFont.truetype(font_path, font_size)
    except Exception:
        font = ImageFont.load_default()

    display_text = format_text(text, config.get("heading_text_transform", "uppercase")).strip()
    if not display_text:
        return np.zeros((1, 1, 4), dtype=np.uint8)

    max_width = config.get("heading_max_width", 900)
    line_gap = config.get("heading_line_gap", 8)
    outline_width = config.get("heading_outline_width", 11)
    padding_x = config.get("heading_padding_x", 30)
    padding_y = config.get("heading_padding_y", 30)

    dummy_draw = ImageDraw.Draw(Image.new("RGBA", (1, 1)))

    def get_text_width(line):
        bbox = dummy_draw.textbbox((0, 0), line, font=font, stroke_width=outline_width)
        return bbox[2] - bbox[0]

    words = display_text.split()
    if len(words) <= 1 or get_text_width(display_text) <= max_width:
        lines = [display_text]
    else:
        best_split = None
        best_score = float("inf")
        for idx in range(1, len(words)):
            l1, l2 = " ".join(words[:idx]), " ".join(words[idx:])
            w1, w2 = get_text_width(l1), get_text_width(l2)
            if w1 <= max_width and w2 <= max_width:
                score = abs(w1 - w2) + (max_width - max(w1, w2)) * 0.15
                if score < best_score:
                    best_score = score
                    best_split = (l1, l2)
        lines = [best_split[0], best_split[1]] if best_split else [" ".join(words[:math.ceil(len(words)/2)]), " ".join(words[math.ceil(len(words)/2):])]

    line_data = []
    max_line_width = 0
    total_text_height = 0

    for line in lines:
        bbox = dummy_draw.textbbox((0, 0), line, font=font, stroke_width=outline_width)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        max_line_width = max(max_line_width, w)
        total_text_height += h
        line_data.append({"text": line, "bbox": bbox, "width": w, "height": h})

    if len(line_data) > 1:
        total_text_height += (len(line_data) - 1) * line_gap

    shadow_offset = config.get("heading_shadow_offset", (0, 10))
    shadow_blur = config.get("heading_shadow_blur", 3)
    extra = abs(shadow_offset[0]) + abs(shadow_offset[1]) + (shadow_blur * 2) + (outline_width * 2)

    canvas_w = int(max_line_width + padding_x * 2 + extra)
    canvas_h = int(total_text_height + padding_y * 2 + extra)

    base_layer = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    
    def draw_lines(draw, y_start, fill, stroke_w, stroke_c):
        curr_y = y_start
        for ld in line_data:
            bbox = ld["bbox"]
            x = (canvas_w - ld["width"]) / 2 - bbox[0]
            y_pos = curr_y - bbox[1]
            draw.text((int(round(x)), int(round(y_pos))), ld["text"], font=font, fill=fill, stroke_width=stroke_w, stroke_fill=stroke_c)
            curr_y += ld["height"] + line_gap

    start_y = padding_y + outline_width
    shadow_layer = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    draw_lines(ImageDraw.Draw(shadow_layer), start_y + shadow_offset[1], config.get("heading_shadow_color", (0, 0, 0, 150)), outline_width, config.get("heading_shadow_color", (0, 0, 0, 150)))
    if shadow_blur > 0:
        shadow_layer = shadow_layer.filter(ImageFilter.GaussianBlur(shadow_blur))

    outline_layer = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    draw_lines(ImageDraw.Draw(outline_layer), start_y, config.get("heading_outline_color", (58, 128, 222, 255)), outline_width, config.get("heading_outline_color", (58, 128, 222, 255)))

    fill_layer = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    draw_lines(ImageDraw.Draw(fill_layer), start_y, config.get("heading_fill_color", (255, 159, 28, 255)), config.get("heading_inner_stroke_width", 3), config.get("heading_inner_stroke_color", (30, 20, 60, 255)))

    img = Image.alpha_composite(base_layer, shadow_layer)
    img = Image.alpha_composite(img, outline_layer)
    return np.array(Image.alpha_composite(img, fill_layer))

def create_top_layer_heading(config: dict, audio_path: str, total_duration: float):
    heading_text = config.get("heading_text")
    if not heading_text:
        return None

    words_data = get_word_timestamps(audio_path)
    start_time = find_heading_start_time(heading_text, words_data or [])
    if start_time is None:
        return None

    start_time = max(0.0, start_time - config.get("heading_lead_time", 0.0))
    clip_duration = max(0.1, total_duration - start_time)
    video_h = config.get("video_height", 1920)

    start_y = int(round(video_h * 0.30))
    target_y = int(round(video_h * 0.15))
    slide_distance = start_y - target_y

    heading_rgba = render_heading_image(heading_text, config)
    base_clip = ImageClip(heading_rgba, transparent=True).with_duration(clip_duration)
    anim_duration = min(config.get("heading_animation_duration", 1.3), clip_duration)

    if base_clip.mask is not None:
        def animated_mask(get_frame, t):
            opacity = min(1.0, (t / anim_duration) ** 2) if t < anim_duration else 1.0
            return (get_frame(t).astype(np.float32) * opacity)
        base_clip = base_clip.with_mask(base_clip.mask.transform(animated_mask))

    def dynamic_position(t):
        if t <= 0:
            cy = start_y
        elif t < anim_duration:
            prog = t / anim_duration
            cy = start_y - slide_distance * (1 - (1 - prog) ** 3)
        else:
            cy = target_y
        return ("center", int(round(cy)))

    return base_clip.with_position(dynamic_position).with_start(start_time).with_duration(clip_duration)

def render_style_sample_highlight(lines, config, base_font, highlight_font, special_font, space_w, active_index):
    TRANSFORM = config.get("text_transform", "uppercase")
    LINE_GAP = config.get("line_gap", 12)

    line_data = []
    for line_items in lines:
        words_in_line = []
        total_line_w, max_h = 0, 0
        for idx, item in enumerate(line_items):
            raw_w = item["word"]
            formatted_w = format_text(raw_w, TRANSFORM)
            is_active = (item["global_idx"] == active_index)
            is_special = clean_word(raw_w) in SPECIAL_WORDS

            if is_special:
                font, color = special_font, config.get("special_color", (255, 69, 0, 255))
            elif is_active:
                font, color = highlight_font, config.get("highlight_color", (255, 215, 0, 255))
            else:
                font, color = base_font, config.get("text_color", (255, 255, 255, 255))

            bbox = font.getbbox(formatted_w)
            w_w, w_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            max_h = max(max_h, w_h)
            words_in_line.append((formatted_w, w_w, color, font, bbox[1]))
            total_line_w += w_w + (space_w if idx < len(line_items) - 1 else 0)

        line_data.append({"words": words_in_line, "width": total_line_w, "height": max_h})

    total_h = sum(ld["height"] for ld in line_data) + ((len(line_data) - 1) * LINE_GAP)
    img = Image.new("RGBA", (config["video_width"], config["video_height"]), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    curr_y = int(config["video_height"] * config.get("vertical_pos", 0.68)) - (total_h // 2)
    stroke_w = min(config.get("stroke_width", 0), 2)
    stroke_c = config.get("stroke_color", (0, 0, 0, 255))

    for ld in line_data:
        text_x = (config["video_width"] - ld["width"]) // 2
        for word, w_w, color, font, w_ascent in ld["words"]:
            safe_draw_text(draw, (text_x, curr_y - w_ascent), word, font=font, fill=color, fallback_font=base_font, stroke_width=stroke_w, stroke_fill=stroke_c)
            text_x += w_w + space_w
        curr_y += ld["height"] + LINE_GAP

    return np.array(img)

def render_style_active_word_box(lines, config, base_font, highlight_font, special_font, space_w, active_index):
    PAD_X, PAD_Y = config.get("padding_x", 16), config.get("padding_y", 8)
    RADIUS, LINE_GAP = config.get("box_corner_radius", 10), config.get("line_gap", 16)
    TRANSFORM = config.get("text_transform", "lowercase")

    line_data = []
    for line_items in lines:
        words_in_line, total_line_w, max_h = [], 0, 0
        for idx, item in enumerate(line_items):
            raw_w = item["word"]
            formatted_w = format_text(raw_w, TRANSFORM)
            is_active = (item["global_idx"] == active_index)
            is_special = clean_word(raw_w) in SPECIAL_WORDS

            font = special_font if is_special else (highlight_font if is_active else base_font)
            color = config.get("special_color") if is_special else (config.get("active_text_color") if is_active else config.get("text_color"))

            bbox = font.getbbox(formatted_w)
            w_w, w_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            max_h = max(max_h, w_h)
            words_in_line.append({"word": formatted_w, "width": w_w, "is_active": is_active, "text_color": color, "font": font, "ascent": bbox[1]})
            total_line_w += w_w + (space_w if idx < len(line_items) - 1 else 0)

        line_data.append({"words": words_in_line, "width": total_line_w, "height": max_h})

    total_h = sum(ld["height"] + (PAD_Y * 2) for ld in line_data) + ((len(line_data) - 1) * LINE_GAP)
    img = Image.new("RGBA", (config["video_width"], config["video_height"]), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    curr_y = int(config["video_height"] * config.get("vertical_pos", 0.70)) - (total_h // 2)

    for ld in line_data:
        text_x = (config["video_width"] - ld["width"]) // 2
        temp_x = text_x
        for w in ld["words"]:
            if w["is_active"]:
                draw.rounded_rectangle([temp_x - PAD_X, curr_y, temp_x + w["width"] + PAD_X, curr_y + ld["height"] + (PAD_Y * 2)], radius=RADIUS, fill=config.get("active_box_color", (255, 87, 34, 255)))
            temp_x += w["width"] + space_w

        for w in ld["words"]:
            draw.text((text_x, curr_y + PAD_Y - w["ascent"]), w["word"], font=w["font"], fill=w["text_color"])
            text_x += w["width"] + space_w

        curr_y += ld["height"] + (PAD_Y * 2) + LINE_GAP

    return np.array(img)

RENDER_REGISTRY = {
    "card_box": render_style_sample_highlight,
    "active_word_box": render_style_active_word_box,
    "sample_highlight": render_style_sample_highlight,
    "stacked_gradient": render_style_sample_highlight
}

def render_caption_card(word_batch, active_index, config):
    font_path = config.get("font_path")
    try:
        base_font = ImageFont.truetype(font_path, config.get("font_size", 52))
    except Exception:
        base_font = ImageFont.load_default()

    highlight_font = ImageFont.truetype(config.get("highlight_font_path", font_path), config.get("highlight_font_size", config.get("font_size", 52)))
    special_font = ImageFont.truetype(config.get("special_font_path", font_path), config.get("special_font_size", config.get("font_size", 52)))

    space_w = base_font.getbbox(" ")[2] - base_font.getbbox(" ")[0]
    lines, current_line, current_line_width = [], [], 0

    for item in word_batch:
        w = item["word"]
        target_font = special_font if clean_word(w) in SPECIAL_WORDS else base_font
        w_w = target_font.getbbox(w)[2] - target_font.getbbox(w)[0]
        test_width = current_line_width + w_w + (space_w if current_line else 0)

        if current_line and test_width > config.get("max_line_width", 850):
            lines.append(current_line)
            current_line = [item]
            current_line_width = w_w
        else:
            current_line.append(item)
            current_line_width = test_width

    if current_line:
        lines.append(current_line)

    render_fn = RENDER_REGISTRY.get(config.get("style", "sample_highlight"), render_style_sample_highlight)
    return render_fn(lines, config, base_font, highlight_font, special_font, space_w, active_index)

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

        first_w = clean_word(raw_words[0])
        last_w = clean_word(raw_words[-1])
        start_time, end_time = None, None

        for i in range(search_cursor, total_words):
            if words_data[i]["clean_word"] == first_w:
                start_time = words_data[i]["start"]
                search_cursor = i
                break

        if start_time is None and search_cursor < total_words:
            start_time = words_data[search_cursor]["start"]

        for j in range(search_cursor, total_words):
            if words_data[j]["clean_word"] == last_w:
                end_time = words_data[j]["end"]
                search_cursor = j + 1
                break

        if end_time is None:
            fallback_idx = min(search_cursor + len(raw_words) - 1, total_words - 1)
            end_time = words_data[fallback_idx]["end"]
            search_cursor = fallback_idx + 1

        seg["start"] = round(start_time, 2)
        seg["end"] = round(end_time, 2)

    return timeline

def generate_caption_overlay(audio_path, config=CONFIG_SAMPLE_STYLE):
    words_data = get_word_timestamps(audio_path)
    if not words_data:
        return None

    for idx, w in enumerate(words_data):
        w["global_idx"] = idx

    batch_size = max(1, config.get("max_words_per_batch", 3))
    batches = [words_data[i:i + batch_size] for i in range(0, len(words_data), batch_size)]

    clip_list = []
    for batch in batches:
        for word_info in batch:
            start_t = word_info["start"]
            duration = max(0.05, word_info["end"] - start_t)
            img_np = render_caption_card(batch, active_index=word_info["global_idx"], config=config)
            clip_list.append(ImageClip(img_np).with_start(start_t).with_duration(duration))

    return CompositeVideoClip(clip_list, size=(config["video_width"], config["video_height"]))