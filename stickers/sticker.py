import numpy as np
from PIL import Image
from moviepy.video.VideoClip import ImageClip, ColorClip
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip

pil_sticker = Image.open("money_trap.png").convert("RGBA")
arr = np.array(pil_sticker)
rgb = arr[:, :, :3]
alpha = arr[:, :, 3] / 255.0

rgb_clip = ImageClip(rgb).with_duration(2)
mask_clip = ImageClip(alpha, is_mask=True).with_duration(2)
sticker_clip = rgb_clip.with_mask(mask_clip)

print("mask attached:", sticker_clip.mask is not None)

bg = ColorClip(size=(1080,1920), color=(255,0,0), duration=2)
test = CompositeVideoClip([bg, sticker_clip.with_position("center")], size=(1080,1920))
test.save_frame("test_frame.png", t=1)