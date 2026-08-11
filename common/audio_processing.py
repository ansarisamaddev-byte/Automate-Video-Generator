import os
import numpy as np
import librosa
import soundfile as sf
from pydub import AudioSegment, effects
from pydub.silence import detect_leading_silence, split_on_silence

class AudioProcessor:
    """
    Handles audio preparation for short-form video automation:
    - Trims leading/trailing silence and compresses long internal pauses
    - High-quality speed/tempo adjustment without pitch shifting
    - Peak volume normalization
    - Standardizes audio sample rate to 48kHz for high-quality video encoding
    """

    def __init__(
        self, 
        silence_thresh_db: int = -42, 
        chunk_size_ms: int = 10,
        target_sample_rate: int = 48000
    ):
        self.silence_thresh_db = silence_thresh_db
        self.chunk_size_ms = chunk_size_ms
        self.target_sample_rate = target_sample_rate

    def trim_start_end_silence(self, audio: AudioSegment) -> AudioSegment:
        """Removes dead air from the absolute beginning and end of an AudioSegment."""
        start_trim = detect_leading_silence(
            audio, 
            silence_threshold=self.silence_thresh_db, 
            chunk_size=self.chunk_size_ms
        )
        end_trim = detect_leading_silence(
            audio.reverse(), 
            silence_threshold=self.silence_thresh_db, 
            chunk_size=self.chunk_size_ms
        )
        
        duration = len(audio)
        return audio[start_trim : duration - end_trim]

    def compress_internal_pauses(
        self, 
        audio: AudioSegment, 
        min_silence_len: int = 400, 
        keep_silence_ms: int = 250
    ) -> AudioSegment:
        """
        Splits audio on internal pauses and rejoins them with a capped maximum 
        silence duration (keeps voice natural yet fast-paced).
        """
        chunks = split_on_silence(
            audio,
            min_silence_len=min_silence_len,
            silence_thresh=self.silence_thresh_db,
            keep_silence=keep_silence_ms
        )
        if not chunks:
            return audio
            
        combined = chunks[0]
        for chunk in chunks[1:]:
            combined += chunk
        return combined

    def apply_time_stretch(self, audio: AudioSegment, speed_factor: float = 1.2) -> np.ndarray:
        """
        Converts PyDub AudioSegment to numpy float array and uses Librosa 
        time-stretching in-memory without saving intermediate files.
        """
        # Convert PyDub AudioSegment to float32 numpy array
        samples = np.array(audio.get_array_of_samples()).astype(np.float32)
        
        # Handle stereo channels if necessary
        if audio.channels == 2:
            samples = samples.reshape((-1, 2)).T
            # Librosa expects shape (channels, samples) or 1D mono array
            samples = librosa.to_mono(samples)

        # Normalize sample values to [-1.0, 1.0] for librosa
        max_possible_val = float(1 << (8 * audio.sample_width - 1))
        samples /= max_possible_val

        # Apply time stretch if speed factor differs from 1.0
        if speed_factor != 1.0:
            samples = librosa.effects.time_stretch(samples, rate=speed_factor)

        return samples

    def process_voiceover(
        self,
        input_file: str,
        output_file: str,
        speed_factor: float = 1.2,
        normalize: bool = True,
        tighten_pauses: bool = True
    ) -> str:
        """
        Complete audio processing pipeline:
        1. Load raw audio file
        2. Trim start/end dead air
        3. Compress long internal pauses (optional)
        4. Apply peak normalization
        5. Change speed without pitch distortion (In-memory)
        6. Export 48kHz high-quality master WAV
        """
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file not found: {input_file}")

        print(f"[AudioProcessor] Processing: {input_file}")

        # Step A: Load audio and convert to 48kHz standard
        audio = AudioSegment.from_file(input_file)
        audio = audio.set_frame_rate(self.target_sample_rate)

        # Step B: Trim head/tail dead air
        audio = self.trim_start_end_silence(audio)

        # Step C: Tighten internal speech pauses for YouTube Shorts / Reels pacing
        if tighten_pauses:
            audio = self.compress_internal_pauses(audio)

        # Step D: Peak volume normalization
        if normalize:
            audio = effects.normalize(audio, headroom=0.1)

        # Step E: In-memory time stretching (No temp disk files)
        processed_samples = self.apply_time_stretch(audio, speed_factor=speed_factor)

        # Step F: Export master WAV at 48kHz 16-bit PCM (Optimal for MoviePy)
        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
        sf.write(
            output_file, 
            processed_samples, 
            self.target_sample_rate, 
            subtype="PCM_16"
        )

        print(f"[AudioProcessor] Successfully exported master audio: {output_file} ({speed_factor}x)")
        return output_file


# --- Example Execution ---
if __name__ == "__main__":
    processor = AudioProcessor(silence_thresh_db=-42, target_sample_rate=48000)
    
    input_path = "trash/ms_1.mp3"
    output_path = "trash/ms_1_new.wav"
    
    if os.path.exists(input_path):
        processed_audio_path = processor.process_voiceover(
            input_file=input_path,
            output_file=output_path,
            speed_factor=1.1,
            normalize=True,
            tighten_pauses=True
        )
    else:
        print(f"Place a test file at '{input_path}' to run.")