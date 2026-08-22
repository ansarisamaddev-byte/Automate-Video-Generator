import numpy as np
from PIL import Image, ImageFilter
from moviepy.video.VideoClip import VideoClip
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip

# ============================================================
# HELPER: ANIMATED FADE MASK (uniform scalar alpha)
# ============================================================
def create_fade_mask(clip, duration, invert=False):
    """
    IMPORTANT: the mask's own .duration must match the CLIP's full duration,
    not the transition length. Several moviepy versions treat a clip's mask
    duration as gating how long that clip stays visible in a composite -
    capping it at the transition length made every clip disappear right
    after its fade-in finished, instead of staying visible for its full
    screen time. The fade itself still only takes `duration` seconds
    because progress is clamped to 1.0 (or 0.0) beyond that point below.
    """
    width = int(clip.w)
    height = int(clip.h)

    def frame_function(t):
        progress = max(0.0, min(1.0, t / duration))
        if invert:
            progress = 1.0 - progress
        return np.full((height, width), progress, dtype=np.float32)

    mask = VideoClip(frame_function=frame_function, is_mask=True)
    return mask.with_duration(clip.duration)


# ============================================================
# HELPER: SYMMETRIC "HAT" CURVE (0 -> 1 -> 0, peak at midpoint)
# ============================================================
def _hat(progress):
    """
    Used by blur_dissolve, additive_dissolve, and smooth_cut so both
    clips intensify and resolve on the SAME clock, peaking together
    exactly at the cut point. This is the shape that hides the seam.
    """
    p = max(0.0, min(1.0, progress))
    return 4.0 * p * (1.0 - p)


# ============================================================
# HELPER: FAST GAUSSIAN BLUR (downsample -> blur -> upsample)
# ============================================================
def _fast_gaussian_blur(frame, radius, downsample=0.5):
    """
    Blurring at full resolution every frame is the main render-time cost.
    Real-time NLE blurs (including Resolve's GPU pipeline) typically work
    at reduced resolution for exactly this reason - visually near-identical
    for the radii used in transitions, much cheaper to compute.
    """
    if radius <= 0.3:
        return frame
    img = Image.fromarray(frame)
    if 0 < downsample < 1.0:
        w, h = img.size
        sw, sh = max(1, int(w * downsample)), max(1, int(h * downsample))
        small = img.resize((sw, sh), Image.Resampling.BILINEAR)
        small = small.filter(ImageFilter.GaussianBlur(radius=radius * downsample))
        img = small.resize((w, h), Image.Resampling.BILINEAR)
    else:
        img = img.filter(ImageFilter.GaussianBlur(radius=radius))
    return np.array(img)


# ============================================================
# 1. CROSS DISSOLVE
# ============================================================
def transition_cross_dissolve(clip_a, clip_b, duration):
    """Standard linear alpha blend - this is how Resolve's default Cross
    Dissolve actually works, no change needed."""
    duration = min(duration, clip_a.duration, clip_b.duration)
    mask = create_fade_mask(clip_b, duration)
    incoming = clip_b.with_mask(mask)
    return clip_a, incoming


# ============================================================
# 2. ADDITIVE DISSOLVE - FIXED
# ============================================================
def transition_additive_dissolve(clip_a, clip_b, duration, boost_amount=0.5):
    """
    Real Additive Dissolve sums luminance from BOTH clips through the
    overlap, producing a brief "glow" at the crossover. Previously only
    the incoming clip was boosted - fixed so both sides brighten on the
    same synced hat curve.
    """
    duration = min(duration, clip_a.duration, clip_b.duration)
    blur_start_a = clip_a.duration - duration

    def boost(frame, progress):
        peak = 1.0 + boost_amount * _hat(progress)
        return np.clip(frame.astype(np.float32) * peak, 0, 255).astype(np.uint8)

    def outgoing_frame(get_frame, t):
        frame = get_frame(t)
        if t >= blur_start_a:
            frame = boost(frame, (t - blur_start_a) / duration)
        return frame

    def incoming_frame(get_frame, t):
        frame = get_frame(t)
        if t <= duration:
            frame = boost(frame, t / duration)
        return frame

    boosted_outgoing = clip_a.transform(outgoing_frame, keep_duration=True)
    boosted_incoming = clip_b.transform(incoming_frame, keep_duration=True)

    mask = create_fade_mask(boosted_incoming, duration)
    return boosted_outgoing, boosted_incoming.with_mask(mask)


# ============================================================
# 3. NON-ADDITIVE DISSOLVE - NEW (was wrongly aliased to additive)
# ============================================================
def transition_non_additive_dissolve(clip_a, clip_b, duration, softness=0.35):
    """
    Real Non-Additive Dissolve is luminance-driven, not a flat alpha fade:
    the incoming clip's brightest pixels reveal first, darkest pixels
    resolve last, giving a textured "developing" look. This is genuinely
    different from Additive Dissolve and needed its own implementation
    rather than reusing that function.
    """
    duration = min(duration, clip_a.duration, clip_b.duration)

    def mask_frame_function(t):
        progress = max(0.0, min(1.0, t / duration))
        frame_b = clip_b.get_frame(t).astype(np.float32) / 255.0
        luminance = 0.299 * frame_b[..., 0] + 0.587 * frame_b[..., 1] + 0.114 * frame_b[..., 2]
        threshold = (1.0 + softness) * (1.0 - progress) - softness * progress
        alpha = np.clip((luminance - threshold) / softness + 0.5, 0.0, 1.0)
        return alpha.astype(np.float32)

    # Same fix as create_fade_mask: mask must last the full clip, not just
    # the transition window, or the clip vanishes right after the reveal.
    mask = VideoClip(frame_function=mask_frame_function, is_mask=True).with_duration(clip_b.duration)
    incoming = clip_b.with_mask(mask)
    return clip_a, incoming


# ============================================================
# 4. BLUR DISSOLVE - FIXED (synced hat curve, fast blur)
# ============================================================
def transition_blur_dissolve(clip_a, clip_b, duration, max_blur=25, downsample=0.5):
    duration = min(duration, clip_a.duration, clip_b.duration)
    blur_start_a = clip_a.duration - duration

    def blur_outgoing_frame(get_frame, t):
        frame = get_frame(t)
        if t >= blur_start_a:
            progress = (t - blur_start_a) / duration
            radius = _hat(progress) * max_blur
            frame = _fast_gaussian_blur(frame, radius, downsample)
        return frame

    def blur_incoming_frame(get_frame, t):
        frame = get_frame(t)
        if t <= duration:
            progress = t / duration
            radius = _hat(progress) * max_blur
            frame = _fast_gaussian_blur(frame, radius, downsample)
        return frame

    blurred_outgoing = clip_a.transform(blur_outgoing_frame, keep_duration=True)
    blurred_incoming = clip_b.transform(blur_incoming_frame, keep_duration=True)

    mask = create_fade_mask(blurred_incoming, duration)
    return blurred_outgoing, blurred_incoming.with_mask(mask)


# ============================================================
# 5. DIP TO COLOR - FIXED (hard step at midpoint, not a crossfade)
# ============================================================
def transition_dip_to_color(clip_a, clip_b, duration, color=(0, 0, 0)):
    """
    Real Dip to Color Dissolve is sequential, not a crossfade of two
    color-blended clips: A dissolves to solid color for the first half,
    then color dissolves to B for the second half. Previously both were
    composited across the FULL duration with a linear mask, making B
    partially visible too early. Fixed with a hard step exactly at the
    midpoint - seamless because both sides equal the same color there.
    """
    duration = min(duration, clip_a.duration, clip_b.duration)
    half_d = duration / 2.0
    color_arr = np.array(color, dtype=np.float32)

    def dip_out_frame(get_frame, t):
        frame = get_frame(t)
        start = clip_a.duration - half_d
        if t >= start:
            progress = max(0.0, min(1.0, (t - start) / half_d))
            blended = frame.astype(np.float32) * (1.0 - progress) + color_arr * progress
            return np.clip(blended, 0, 255).astype(np.uint8)
        return frame

    def dip_in_frame(get_frame, t):
        frame = get_frame(t)
        if t <= half_d:
            progress = max(0.0, min(1.0, t / half_d))
            blended = color_arr * (1.0 - progress) + frame.astype(np.float32) * progress
            return np.clip(blended, 0, 255).astype(np.uint8)
        return frame

    outgoing = clip_a.transform(dip_out_frame, keep_duration=True)
    incoming = clip_b.transform(dip_in_frame, keep_duration=True)

    width, height = int(clip_b.w), int(clip_b.h)

    def step_mask_frame(t):
        val = 0.0 if t < half_d else 1.0
        return np.full((height, width), val, dtype=np.float32)

    # Same fix as create_fade_mask: mask must last the full clip, not just
    # the transition window.
    mask = VideoClip(frame_function=step_mask_frame, is_mask=True).with_duration(clip_b.duration)
    return outgoing, incoming.with_mask(mask)


# ============================================================
# 6. SMOOTH CUT - FIXED (synced hat curve, fast blur)
# ============================================================
def transition_smooth_cut(clip_a, clip_b, duration, punch=0.08, blur_strength=10, downsample=0.5):
    """
    Approximation of Resolve's optical-flow Smooth Cut (see note in prior
    message re: true optical flow being a separate, heavier implementation).
    Keep duration SHORT (0.12-0.25s) - this is meant to be a snap, not a
    slow dissolve.
    """
    duration = min(duration, clip_a.duration, clip_b.duration)
    blur_start_a = clip_a.duration - duration

    def zoom_blur_frame(frame, progress):
        intensity = _hat(progress)
        if intensity <= 0:
            return frame
        scale = 1.0 + intensity * punch
        h, w = frame.shape[:2]
        img = Image.fromarray(frame)
        new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
        img = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
        crop_x, crop_y = (new_w - w) // 2, (new_h - h) // 2
        img = img.crop((crop_x, crop_y, crop_x + w, crop_y + h))
        frame = np.array(img)
        radius = intensity * blur_strength
        return _fast_gaussian_blur(frame, radius, downsample)

    def outgoing_frame(get_frame, t):
        frame = get_frame(t)
        if t >= blur_start_a:
            frame = zoom_blur_frame(frame, (t - blur_start_a) / duration)
        return frame

    def incoming_frame(get_frame, t):
        frame = get_frame(t)
        if t <= duration:
            frame = zoom_blur_frame(frame, t / duration)
        return frame

    outgoing = clip_a.transform(outgoing_frame, keep_duration=True)
    incoming = clip_b.transform(incoming_frame, keep_duration=True)

    mask = create_fade_mask(incoming, duration)
    return outgoing, incoming.with_mask(mask)


# ============================================================
# TRANSITION REGISTRY
# ============================================================
TRANSITION_REGISTRY = {
    "cross_dissolve": transition_cross_dissolve,
    "blur_dissolve": transition_blur_dissolve,
    "additive_dissolve": transition_additive_dissolve,
    "non_additive_dissolve": transition_non_additive_dissolve,
    "smooth_cut": transition_smooth_cut,
    "dip_to_black": lambda a, b, d: transition_dip_to_color(a, b, d, (0, 0, 0)),
    "dip_to_white": lambda a, b, d: transition_dip_to_color(a, b, d, (255, 255, 255)),
}

# ============================================================
# MASTER TIMELINE (unchanged)
# ============================================================
def build_transitioned_timeline(
    clip_list,
    transition_type="cross_dissolve",
    duration=0.6,
    size=(1080, 1920),
    final_duration=None
):
    if not clip_list:
        return None

    if len(clip_list) == 1:
        clip = clip_list[0]
        if final_duration is not None:
            clip = clip.with_duration(final_duration)
        return clip

    transition_fn = TRANSITION_REGISTRY.get(
        transition_type,
        transition_cross_dissolve
    )

    duration = min(
        duration,
        min(clip.duration for clip in clip_list)
    )

    timeline = []
    current_start = 0.0

    first_clip = clip_list[0].with_start(0)
    timeline.append(first_clip)

    for i in range(1, len(clip_list)):
        previous_clip = clip_list[i - 1]
        current_clip = clip_list[i]

        processed_outgoing, incoming_clip = transition_fn(
            previous_clip,
            current_clip,
            duration
        )

        timeline[i - 1] = processed_outgoing.with_start(
            timeline[i - 1].start
        )

        current_start += (previous_clip.duration - duration)
        incoming_clip = incoming_clip.with_start(current_start)
        timeline.append(incoming_clip)

    if final_duration is None:
        final_duration = max(clip.start + clip.duration for clip in timeline)

    result = CompositeVideoClip(timeline, size=size)
    return result.with_duration(final_duration)