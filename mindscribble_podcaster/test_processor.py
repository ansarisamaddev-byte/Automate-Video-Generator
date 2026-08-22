import os
import glob

from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.video.VideoClip import ImageClip
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip
from moviepy.video.compositing.CompositeVideoClip import concatenate_videoclips
import moviepy.video.fx as vfx

from modules.caption_generator import (
    CONFIG_STYLE_2,
    generate_caption_overlay
)

from modules.transitions import (
    build_transitioned_timeline
)


# ============================================================
# RUN TEST PIPELINE
# ============================================================

def run_test_pipeline(
    audio_path,
    image_folder,
    output_path="test_output.mp4",
    transition_type="zoom_dissolve",
    transition_duration=0.6
):

    print("==========================================")
    print(" STARTING TEST PIPELINE")
    print("==========================================")

    # ========================================================
    # 1. AUDIO
    # ========================================================

    if not os.path.exists(audio_path):
        raise FileNotFoundError(
            f"Audio file not found: {audio_path}"
        )

    audio_clip = AudioFileClip(audio_path)

    audio_duration = audio_clip.duration

    print(
        f"Audio duration: "
        f"{audio_duration:.2f}s"
    )

    # ========================================================
    # 2. FIND MEDIA
    # ========================================================

    valid_extensions = (
        "*.jpg",
        "*.jpeg",
        "*.mp4",
        "*.png",
        "*.webp"
    )

    media_files = []

    for ext in valid_extensions:

        media_files.extend(
            glob.glob(
                os.path.join(
                    image_folder,
                    ext
                )
            )
        )

    media_files.sort()

    # TEST ONLY
    media_files = media_files[:3]

    if not media_files:

        raise FileNotFoundError(
            f"No media found in: {image_folder}"
        )

    print(
        f"Found {len(media_files)} media files"
    )

    for i, path in enumerate(media_files):

        print(
            f"  {i + 1}: "
            f"{os.path.basename(path)}"
        )

    # ========================================================
    # 3. CALCULATE CLIP DURATION
    # ========================================================

    num_media = len(media_files)

    if num_media == 1:

        clip_duration = audio_duration

    else:

        clip_duration = (
            audio_duration
            +
            (
                (num_media - 1)
                * transition_duration
            )
        ) / num_media

    print(
        f"Each clip duration: "
        f"{clip_duration:.2f}s"
    )

    print(
        f"Transition duration: "
        f"{transition_duration:.2f}s"
    )

    # ========================================================
    # 4. LOAD MEDIA
    # ========================================================

    media_clips = []

    for index, file_path in enumerate(
        media_files
    ):

        print(
            f"\nLoading media "
            f"{index + 1}: "
            f"{os.path.basename(file_path)}"
        )

        ext = os.path.splitext(
            file_path
        )[1].lower()

        # ====================================================
        # VIDEO
        # ====================================================

        if ext == ".mp4":

            print("  Type: VIDEO")

            try:
                source_video = (
                    VideoFileClip(
                        file_path,
                        audio=False
                    )
                )
            except Exception as exc:
                print(
                    f"  Skipping unreadable/corrupt video: "
                    f"{os.path.basename(file_path)} ({exc})"
                )
                continue

            print(
                f"  Source duration: "
                f"{source_video.duration:.2f}s"
            )

            print(
                f"  Source size: "
                f"{source_video.w}x"
                f"{source_video.h}"
            )

            # ------------------------------------------------
            # Normalize fps so every video clip lines up the
            # same way once they're composited together for
            # transitions. Mixed source fps (24/25/30/60) is a
            # common cause of mp4 -> mp4 transitions looking
            # stuttery or misaligned.
            # ------------------------------------------------

            source_video = source_video.with_fps(30)

            # ------------------------------------------------
            # Make sure video is long enough
            # ------------------------------------------------

            if source_video.duration < clip_duration:

                print(
                    "  Video shorter than "
                    "required duration."
                )

                # Loop the video
                source_video = source_video.with_effects(
                    [
                        vfx.Loop(
                            duration=clip_duration
                        )
                    ]
                )

            # ------------------------------------------------
            # Cut to exact duration
            # ------------------------------------------------

            media_clip = (
                source_video
                .subclipped(
                    0,
                    clip_duration
                )
                .with_duration(
                    clip_duration
                )
            )

            # ------------------------------------------------
            # Resize
            # ------------------------------------------------

            media_clip = media_clip.resized(
                height=1920
            )

        # ====================================================
        # IMAGE
        # ====================================================

        else:

            print("  Type: IMAGE")

            media_clip = (
                ImageClip(
                    file_path
                )
                .with_duration(
                    clip_duration
                )
                .resized(
                    height=1920
                )
            )

        # ====================================================
        # NORMALIZE WIDTH / HEIGHT
        # ====================================================

        if media_clip.w > 1080:

            crop_x = (
                media_clip.w - 1080
            ) / 2

            media_clip = media_clip.cropped(
                x1=crop_x,
                width=1080
            )

        elif media_clip.w < 1080:

            media_clip = media_clip.resized(
                width=1080
            )

        # ----------------------------------------------------
        # Height correction
        # ----------------------------------------------------

        if media_clip.h > 1920:

            crop_y = (
                media_clip.h - 1920
            ) / 2

            media_clip = media_clip.cropped(
                y1=crop_y,
                height=1920
            )

        # ====================================================
        # FINAL NORMALIZATION
        # ====================================================

        media_clip = (
            media_clip
            .with_duration(
                clip_duration
            )
        )

        print(
            f"  Final size: "
            f"{media_clip.w}x"
            f"{media_clip.h}"
        )

        print(
            f"  Final duration: "
            f"{media_clip.duration:.2f}s"
        )

        media_clips.append(
            media_clip
        )

    # ========================================================
    # 5. BUILD TRANSITION TIMELINE
    # ========================================================

    print("\n==========================================")
    print(" BUILDING TRANSITION TIMELINE")
    print("==========================================")

    print(
        f"Transition type: "
        f"{transition_type}"
    )

    print(
        f"Transition duration: "
        f"{transition_duration:.2f}s"
    )

    background = build_transitioned_timeline(
        media_clips,
        transition_type=transition_type,
        duration=transition_duration,
        size=(1080, 1920),
        final_duration=audio_duration
    )

    print(
        f"Background duration: "
        f"{background.duration:.2f}s"
    )

    # ========================================================
    # 6. CAPTIONS
    # ========================================================

    print(
        "\nGenerating Whisper "
        "caption overlay..."
    )

    caption_overlay = (
        generate_caption_overlay(
            audio_path,
            config=CONFIG_STYLE_2
        )
    )

    # ========================================================
    # 7. FINAL COMPOSITION
    # ========================================================

    print(
        "Compositing "
        "background + captions..."
    )

    final_video = (
        CompositeVideoClip(
            [
                background,
                caption_overlay
            ],
            size=(1080, 1920)
        )
        .with_audio(
            audio_clip
        )
        .with_duration(
            audio_duration
        )
    )

    print(
        f"Final video duration: "
        f"{final_video.duration:.2f}s"
    )

    # ========================================================
    # 8. EXPORT
    # ========================================================

    print("\nRendering...")

    final_video.write_videofile(
        output_path,
        fps=30,
        codec="libx264",
        audio_codec="aac",
        preset="medium"
    )

    print("==========================================")
    print(" VIDEO COMPLETE")
    print(
        f" Output: {output_path}"
    )
    print("==========================================")

    # ========================================================
    # 9. CLOSE
    # ========================================================

    final_video.close()
    background.close()
    caption_overlay.close()
    audio_clip.close()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    TEST_AUDIO = (
        "voiceovers/testing.mp3"
    )

    TEST_IMAGE_DIR = (
        r"D:\AI\Automate-Video-Generator"
        r"\asset_library"
        r"\05_internal_conflict"
    )

    run_test_pipeline(
        TEST_AUDIO,
        TEST_IMAGE_DIR,
        output_path=(
            "test_transition_output.mp4"
        ),
        transition_type=(
            "cut_smooth"
        ),
        transition_duration=0.6
    )