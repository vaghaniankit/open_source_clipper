from pathlib import Path
from typing import List, Tuple, Dict, Union
import os
import math
import numpy as np
import librosa
import soundfile as sf
from faster_whisper import WhisperModel


ChunkEntry = Tuple[Path, float]


def chunk_audio(input_wav: Path, out_dir: Path, chunk_seconds: int = 30, overlap_seconds: int = 2) -> List[ChunkEntry]:
    out_dir.mkdir(parents=True, exist_ok=True)
    y, sr = librosa.load(str(input_wav), sr=None, mono=True)
    total_samples = y.shape[0]
    chunk_len = int(chunk_seconds * sr)
    overlap = int(overlap_seconds * sr)
    step = max(1, chunk_len - overlap)
    chunks: List[ChunkEntry] = []
    start = 0
    idx = 0
    while start < total_samples:
        end = min(total_samples, start + chunk_len)
        part = y[start:end]
        out_path = out_dir / f"chunk_{idx:05d}.wav"
        sf.write(str(out_path), part, sr)
        chunk_start_time = start / sr
        chunks.append((out_path, float(chunk_start_time)))
        if end == total_samples:
            break
        start += step
        idx += 1
    return chunks


def _resolve_chunk_entry(entry, default_start: float = 0.0) -> Tuple[Path, float]:
    if isinstance(entry, tuple) or isinstance(entry, list):
        if len(entry) >= 2:
            return Path(entry[0]), float(entry[1])
        if len(entry) == 1:
            return Path(entry[0]), float(default_start)
    if isinstance(entry, dict):
        path_val = entry.get("path")
        start_time = float(entry.get("start_time", default_start))
        return Path(path_val), start_time
    return Path(entry), float(default_start)


def transcribe_chunks(chunks: List[Union[ChunkEntry, Path, str]], model_size: str = "base") -> Dict:
    """Transcribe audio chunks with faster-whisper.

    Returns a dict with:
    - segments: list of {chunk_index, start, end, text, words?}
    - transcript: concatenated text.

    When available, each segment will include a "words" field containing
    [{"start": float, "end": float, "word": str}, ...] which can be used
    later for more precise karaoke-style subtitles.
    """
    # Allow overriding the default model via environment so we can run a
    # stronger model (e.g. "medium" or "large") in production without
    # changing code.
    env_model = os.getenv("TRANSCRIBE_MODEL")
    effective_model = env_model or model_size or "medium"

    model = WhisperModel(effective_model)
    results = []
    full_text_parts: List[str] = []
    for i, raw_entry in enumerate(chunks):
        wav_path, chunk_start = _resolve_chunk_entry(raw_entry, default_start=0.0)

        # Use accuracy‑oriented decoding: beam search and a low temperature so
        # subtitles (especially music/lyrics) are as faithful as possible.
        segments, _ = model.transcribe(
            str(wav_path),
            word_timestamps=True,
            beam_size=5,
            best_of=5,
            temperature=0.2,
        )
        try:
            import soundfile as _sf
            f = _sf.SoundFile(str(wav_path))
            sr = f.samplerate
            frames = len(f)
            dur = frames / float(sr)
            f.close()
        except Exception:
            dur = 0.0
        for seg in segments:
            seg_start = chunk_start + float(seg.start)
            seg_end = chunk_start + float(seg.end)
            seg_dict: Dict = {
                "chunk_index": i,
                "start": seg_start,
                "end": seg_end,
                "text": seg.text,
            }
            # Attach word-level timestamps if present
            words = getattr(seg, "words", None)
            if words:
                seg_dict["words"] = [
                    {
                        "start": chunk_start + float(w.start),
                        "end": chunk_start + float(w.end),
                        "word": w.word,
                        "confidence": getattr(w, "probability", None),
                    }
                    for w in words
                ]
            results.append(seg_dict)
            full_text_parts.append(seg.text)
    transcript = " ".join([t.strip() for t in full_text_parts if t and t.strip()])
    return {"segments": results, "transcript": transcript}


def compute_segment_energy(audio_path: str, segments: List[Dict]) -> List[Dict]:
    """Compute a simple RMS energy per transcript segment and attach as 'energy'.

    This is a lightweight feature used to give the highlight LLM some sense of
    which parts of the audio are more intense. It loads the full audio once and
    computes per-segment RMS over the corresponding sample range.
    """
    if not segments:
        return segments

    y, sr = librosa.load(audio_path, sr=None, mono=True)
    n = y.shape[0]

    def seg_rms(start_s: float, end_s: float) -> float:
        if end_s <= start_s:
            return 0.0
        s = max(0, int(start_s * sr))
        e = min(n, int(end_s * sr))
        if e <= s:
            return 0.0
        window = y[s:e]
        # avoid NaNs if silence
        return float(np.sqrt(np.mean(window**2))) if window.size else 0.0

    for seg in segments:
        try:
            start = float(seg.get("start", 0.0))
            end = float(seg.get("end", 0.0))
        except Exception:
            seg["energy"] = 0.0
            continue
        seg["energy"] = round(seg_rms(start, end), 5)
    return segments
