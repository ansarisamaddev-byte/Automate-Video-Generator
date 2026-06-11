import os
import cv2
import yt_dlp
import imageio_ffmpeg
from moviepy import ImageSequenceClip, CompositeVideoClip, TextClip

# 1. ImageMagick configuration for Windows
# Ensure this path points to your actual installation
MAGICK_PATH = r"C:\Program Files\ImageMagick-7.1.1-Q16\magick.exe"
TextClip.search_magick = lambda: MAGICK_PATH

def download_video(url, output_filename="temp_video.mp4"):
    ydl_opts = {'outtmpl': output_filename, 'format': 'best'}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    return output_filename

def edit_video(video_path, output_path, captions, resize_coords):
    # 1. Use OpenCV to get frames (Bypassing MoviePy's metadata parser)
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        # Convert BGR (OpenCV) to RGB (MoviePy)
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    
    # 2. Create the clip
    clip = ImageSequenceClip(frames, fps=25)

    import inspect

    print("MoviePy Version:", moviepy.__version__)
    print("resized signature:", inspect.signature(clip.resized))

    if resize_coords:
        clip = clip.resized(resize_coords)

    # 4. Add Captions
    layers = [clip]
    for cap in captions:
        txt = TextClip(
            text=cap['text'], 
            font_size=70, 
            color='white', 
            font='Arial'
        )
        # MoviePy 2.x method names
        txt = txt.with_start(cap['start']).with_duration(cap['end'] - cap['start'])
        txt = txt.with_position(('center', 'bottom'))
        layers.append(txt)

    # 5. Render
    final_video = CompositeVideoClip(layers)
    final_video.write_videofile(output_path, codec="libx264", audio_codec="aac")
    
    # Cleanup
    clip.close()
    final_video.close()

if __name__ == "__main__":
    # Test
    video_file = download_video("https://www.w3schools.com/tags/mov_bbb.mp4")
    my_caps = [{'text': 'Hello World!', 'start': 0, 'end': 3}]
    edit_video(video_file, "final_output.mp4", my_caps, (640, 360))