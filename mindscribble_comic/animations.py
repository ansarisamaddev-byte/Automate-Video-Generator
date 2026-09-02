import numpy as np
from PIL import Image, ImageFilter

from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip
import moviepy.video.fx as vfx


DEFAULT_SIZE = (1080, 1920)


# ============================================================
# HELPERS
# ============================================================

def _progress(t, duration):
    """0 -> 1 linear progress through a transition, clamped."""
    if duration <= 0:
        return 1.0
    return min(max(t / duration, 0.0), 1.0)


def _blur_frame(frame, radius):
    if radius <= 0.05:
        return frame
    original_dtype = frame.dtype
    img = Image.fromarray(frame.astype(np.uint8))
    img = img.filter(ImageFilter.GaussianBlur(radius=radius))
    return np.array(img).astype(original_dtype)


def _fade_frame_towards_color(frame, color, alpha):
    """Blend `frame` towards `color` by `alpha` (0 = untouched, 1 = solid color)."""
    if alpha <= 0.0:
        return frame
    original_dtype = frame.dtype
    color_arr = np.array(color, dtype=np.float64)
    blended = frame.astype(np.float64) * (1.0 - alpha) + color_arr * alpha
    return np.clip(blended, 0, 255).astype(original_dtype)


def _fade_out_to_color(clip, duration, color):
    """
    Fade the tail of `clip` towards `color` over its last
    `duration` seconds. Works for any color, unlike moviepy's
    built-in FadeOut which only ever fades through black.
    """
    total = clip.duration

    def _make_frame(get_frame, t):
        frame = get_frame(t)
        alpha = _progress(t - (total - duration), duration)
        return _fade_frame_towards_color(frame, color, alpha)

    return clip.transform(_make_frame)


def _fade_in_from_color(clip, duration, color):
    """
    Fade the head of `clip` in from `color` over its first
    `duration` seconds. Works for any color, unlike moviepy's
    built-in FadeIn which only ever fades through black.
    """

    def _make_frame(get_frame, t):
        frame = get_frame(t)
        alpha = 1.0 - _progress(t, duration)
        return _fade_frame_towards_color(frame, color, alpha)

    return clip.transform(_make_frame)


# ============================================================
# CROSS DISSOLVE
# ============================================================

def transition_cross_dissolve(clip_a, clip_b, duration, size=DEFAULT_SIZE):
    """
    Plain crossfade: clip_b fades in on top of clip_a.
    """
    return clip_b.with_effects([vfx.CrossFadeIn(duration)])


# ============================================================
# BLUR DISSOLVE
# ============================================================

def transition_blur_dissolve(clip_a, clip_b, duration, size=DEFAULT_SIZE, max_blur=18):
    """
    clip_b starts blurred and sharpens while it crossfades in.

    Note: this runs a PIL GaussianBlur per-frame during the
    transition window, so it is noticeably slower to render
    than the other transitions. Only the first `duration`
    seconds of clip_b are blurred; after that it's rendered
    at full sharpness.
    """

    def _make_frame(get_frame, t):
        frame = get_frame(t)
        progress = _progress(t, duration)
        radius = max_blur * (1.0 - progress)
        return _blur_frame(frame, radius)

    incoming = clip_b.transform(_make_frame)
    incoming = incoming.with_effects([vfx.CrossFadeIn(duration)])
    return incoming


# ============================================================
# DIP TO BLACK
# ============================================================

def transition_dip_to_black(clip_a, clip_b, duration, size=DEFAULT_SIZE):
    """
    Standalone "incoming half" of a dip-to-black: clip_b fades
    in from black. The matching "outgoing half" (clip_a fading
    OUT to black) is applied to the previous clip directly by
    build_transitioned_timeline, since a true dip needs both
    sides to hit black at the same instant with no crossfade
    overlap. Call this function on its own only if you're
    stitching things together manually.
    """
    half = duration / 2.0
    return _fade_in_from_color(clip_b, half, (0, 0, 0))


# ============================================================
# DIP TO WHITE
# ============================================================

def transition_dip_to_white(clip_a, clip_b, duration, size=DEFAULT_SIZE):
    """
    Same idea as transition_dip_to_black but through white.
    See that function's docstring for the important caveat
    about how the outgoing half is handled. Note this fades
    the actual pixel values towards white (not moviepy's
    built-in FadeIn, which only ever fades through black).
    """
    half = duration / 2.0
    return _fade_in_from_color(clip_b, half, (255, 255, 255))


# ============================================================
# ZOOM DISSOLVE
# ============================================================
def transition_zoom_dissolve(clip_a, clip_b, duration, size=DEFAULT_SIZE, zoom_start=1.15):
    """
    clip_b zooms from `zoom_start`x down to 1.0x while it
    crossfades in, staying perfectly centered on canvas.
    """
    canvas_w, canvas_h = size

    def _zoom_and_center(get_frame, t):
        frame = get_frame(t)
        
        # Only scale during the transition duration
        if t < duration:
            progress = _progress(t, duration)
            scale = zoom_start - (zoom_start - 1.0) * progress
            
            # 1. Resize frame using PIL
            h, w = frame.shape[:2]
            new_w, new_h = int(w * scale), int(h * scale)
            
            img = Image.fromarray(frame)
            img_resized = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
            resized_frame = np.array(img_resized)

            # 2. Crop centered portion back to canvas size (1080x1920)
            crop_x = (new_w - canvas_w) // 2
            crop_y = (new_h - canvas_h) // 2
            
            return resized_frame[crop_y : crop_y + canvas_h, crop_x : crop_x + canvas_w]

        return frame

    incoming = clip_b.transform(_zoom_and_center)
    incoming = incoming.with_effects([vfx.CrossFadeIn(duration)])
    return incoming


# ============================================================
# SLIDE LEFT
# ============================================================

def transition_slide_left(clip_a, clip_b, duration, size=DEFAULT_SIZE):
    """
    clip_b slides in from the right edge and settles at (0, 0),
    covering clip_a as it travels across.
    """
    w = size[0]

    def _pos(t):
        progress = _progress(t, duration)
        x = w * (1.0 - progress)
        return (x, 0)

    return clip_b.with_position(_pos)


# ============================================================
# SLIDE RIGHT
# ============================================================

def transition_slide_right(clip_a, clip_b, duration, size=DEFAULT_SIZE):
    """
    clip_b slides in from the left edge and settles at (0, 0).
    """
    w = size[0]

    def _pos(t):
        progress = _progress(t, duration)
        x = -w * (1.0 - progress)
        return (x, 0)

    return clip_b.with_position(_pos)


# ============================================================
# CUT SMOOTH
# ============================================================

def transition_cut_smooth(clip_a, clip_b, duration, size=DEFAULT_SIZE):
    """
    Approximation of DaVinci Resolve's "Smooth Cut": a very
    short, snappy crossfade (clamped well below the requested
    duration) so it reads as a quick cut rather than a slow
    dissolve.

    Caveat: Resolve's real Smooth Cut uses optical-flow motion
    estimation to morph matching frames together, which needs
    something like OpenCV's calcOpticalFlowFarneback and isn't
    implemented here. This gives a similar "feel" for a fraction
    of the complexity, but it will not intelligently match
    motion between the two clips. If you want the real thing,
    add an OpenCV optical-flow warp here instead of the plain
    crossfade.
    """
    snap_duration = min(duration, 0.15)
    return clip_b.with_effects([vfx.CrossFadeIn(snap_duration)])


# ============================================================
# REGISTRY
# ============================================================

TRANSITION_REGISTRY = {
    "cross_dissolve": transition_cross_dissolve,
    "blur_dissolve": transition_blur_dissolve,
    "dip_to_black": transition_dip_to_black,
    "dip_to_white": transition_dip_to_white,
    "zoom_dissolve": transition_zoom_dissolve,
    "slide_left": transition_slide_left,
    "slide_right": transition_slide_right,
    "cut_smooth": transition_cut_smooth,
}

# Transitions that need a black/white flash handled sequentially
# (no overlap) rather than as a simple overlapping crossfade.
DIP_TRANSITIONS = {"dip_to_black", "dip_to_white"}


def build_transitioned_timeline(
    clip_list,
    transition_type="cross_dissolve",
    duration=0.6,
    size=DEFAULT_SIZE,
    final_duration=None
):

    if not clip_list:
        return None

    if len(clip_list) == 1:
        return clip_list[0].with_duration(
            final_duration
            if final_duration is not None
            else clip_list[0].duration
        )

    if transition_type not in TRANSITION_REGISTRY:
        print(
            f"Unknown transition_type '{transition_type}', "
            f"falling back to 'cross_dissolve'. Valid options: "
            f"{sorted(TRANSITION_REGISTRY.keys())}"
        )
        transition_type = "cross_dissolve"

    duration = min(
        duration,
        min(c.duration for c in clip_list) / 2
    )

    # Smooth cut is meant to be snappy no matter what duration
    # was requested for the other transitions.
    if transition_type == "cut_smooth":
        duration = min(duration, 0.15)

    timeline = []
    current_start = 0.0

    # --------------------------------------------------------
    # FIRST CLIP
    # --------------------------------------------------------

    first = clip_list[0].with_start(0)
    timeline.append(first)

    print(
        f"Clip 1: start=0.00 duration={first.duration:.2f}"
    )

    # --------------------------------------------------------
    # FOLLOWING CLIPS
    # --------------------------------------------------------

    for i in range(1, len(clip_list)):

        previous = clip_list[i - 1]
        incoming = clip_list[i]

        if transition_type in DIP_TRANSITIONS:

            # --------------------------------------------
            # Sequential dip-through-color: no overlap.
            # Previous clip fades OUT to the dip color at
            # its own tail end; incoming clip fades IN from
            # that same color at its own head. Because both
            # sides hit the same solid color at the seam,
            # this reads as a clean dip even without
            # overlapping the two clips in time. Both fades
            # go through the real target color (black OR
            # white) via _fade_out_to_color / _fade_in_from_color,
            # not moviepy's built-in FadeOut/FadeIn (which only
            # ever fade through black).
            # --------------------------------------------

            half = duration / 2.0
            color = (0, 0, 0) if transition_type == "dip_to_black" else (255, 255, 255)

            timeline[-1] = _fade_out_to_color(timeline[-1], half, color)

            start = current_start + previous.duration
            incoming_clip = _fade_in_from_color(incoming, half, color)
            incoming_clip = incoming_clip.with_start(start)

            timeline.append(incoming_clip)

            print(
                f"Clip {i + 1}: start={start:.2f} "
                f"duration={incoming.duration:.2f} (via {transition_type})"
            )

            current_start = start

        else:

            # --------------------------------------------
            # Overlapping transitions: incoming begins
            # `duration` seconds before previous ends, and
            # whichever effect is registered for this
            # transition_type is applied to the incoming clip.
            # --------------------------------------------

            start = (
                current_start
                + previous.duration
                - duration
            )

            effect_fn = TRANSITION_REGISTRY[transition_type]
            incoming_clip = effect_fn(previous, incoming, duration, size=size)
            incoming_clip = incoming_clip.with_start(start)

            timeline.append(incoming_clip)

            print(
                f"Clip {i + 1}: start={start:.2f} "
                f"duration={incoming_clip.duration:.2f} (via {transition_type})"
            )

            current_start = start

    # --------------------------------------------------------
    # FINAL DURATION
    # --------------------------------------------------------

    if final_duration is None:
        final_duration = (
            current_start
            + clip_list[-1].duration
        )

    # --------------------------------------------------------
    # COMPOSITE
    # --------------------------------------------------------

    result = CompositeVideoClip(
        timeline,
        size=size,
        bg_color=(0, 0, 0)
    )

    return result.with_duration(final_duration)