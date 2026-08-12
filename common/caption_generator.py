import os
from typing import List, Dict, Any
from faster_whisper import WhisperModel

class CaptionGenerator:
    def __init__(self, model_size: str = "base", device: str = "cpu", compute_type: str = "int8"):
        """
        Initializes the Faster-Whisper model.
        - model_size: 'tiny', 'base', 'small', 'medium', 'large-v3'
        - device: 'cpu' or 'cuda' (use 'cpu' for standard GitHub Actions / PC setup)
        - compute_type: 'int8' or 'float32' for CPU; 'float16' for CUDA
        """
        print(f"Loading Faster-Whisper model ({model_size}) on {device}...")
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def generate_word_timestamps(self, audio_path: str) -> List[Dict[str, Any]]:
        """
        Transcribes audio and extracts precise word-level timestamps.
        Returns a list of dicts: [{'word': str, 'start': float, 'end': float}, ...]
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        print(f"Transcribing audio: {audio_path}")
        segments, _ = self.model.transcribe(
            audio_path,
            word_timestamps=True,
            language="en"  # Explicitly set or remove for auto-detection
        )

        word_level_data = []
        for segment in segments:
            for word in segment.words:
                # Clean up word text and store exact timing
                clean_word = word.word.strip()
                if clean_word:
                    word_level_data.append({
                        "word": clean_word,
                        "start": round(word.start, 2),
                        "end": round(word.end, 2)
                    })

        return word_level_data

    def group_words_into_lines(
        self, 
        word_data: List[Dict[str, Any]], 
        max_words_per_line: int = 4, 
        max_duration_sec: float = 2.5
    ) -> List[Dict[str, Any]]:
        """
        Groups single-word timestamps into short subtitle chunks/lines 
        ideal for short-form media (e.g., Shorts/Reels).
        """
        lines = []
        current_chunk = []
        
        for item in word_data:
            current_chunk.append(item)
            
            chunk_duration = current_chunk[-1]["end"] - current_chunk[0]["start"]
            
            # Create a line when word limit or duration limit is reached
            if len(current_chunk) >= max_words_per_line or chunk_duration >= max_duration_sec:
                lines.append({
                    "text": " ".join([w["word"] for w in current_chunk]),
                    "start": current_chunk[0]["start"],
                    "end": current_chunk[-1]["end"],
                    "words": current_chunk
                })
                current_chunk = []

        # Catch any remaining words
        if current_chunk:
            lines.append({
                "text": " ".join([w["word"] for w in current_chunk]),
                "start": current_chunk[0]["start"],
                "end": current_chunk[-1]["end"],
                "words": current_chunk
            })

        return lines


if __name__ == "__main__":
    # Quick Test Execution
    TEST_AUDIO = "trash/ms_1_new.wav"  # Replace with path to your audio file
    
    if os.path.exists(TEST_AUDIO):
        generator = CaptionGenerator(model_size="base")
        
        # 1. Get raw word timings
        raw_words = generator.generate_word_timestamps(TEST_AUDIO)
        print(f"\nTotal words detected: {len(raw_words)}")
        print("First 3 words:", raw_words[:3])

        # 2. Group into short subtitle lines
        caption_lines = generator.group_words_into_lines(raw_words, max_words_per_line=3)
        print(f"\nGenerated {len(caption_lines)} caption lines.")
        print("Sample Caption Line:", caption_lines[0] if caption_lines else "None")
    else:
        print(f"To test, place an audio file named '{TEST_AUDIO}' in this directory.")