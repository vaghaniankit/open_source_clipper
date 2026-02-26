# analyze_audio_parallel.py — Parallelized Pipeline (No SER)
# ------------------------------------------------
# Includes: YAMNet TFLite + Energy + Laughter + Whisper Karaoke + Scene Detection
# Parallelized across video chunks for speed

import os, json, subprocess, csv, math
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import librosa
from faster_whisper import WhisperModel

from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector

# ------------------------------
# Config
# ------------------------------
DEFAULT_CONFIG = {
    "pipeline": {"export_mode": "both", "model_size": "small.en", "min_confidence": 0.3},
    "audio_detection": {"energy_percentile": 90, "laughter_backtrack": 6.0},
    "yamnet": {"score_threshold": 0.25,
               "target_labels": ["Laughter", "Applause", "Cheering", "Scream", "Crying, sobbing"]},
    "karaoke": {"font": "Arial", "font_size": 48},
    "scene": {"snap_to_scenes": True, "threshold": 30.0},
    "parallel": {"chunk_sec": 300}  # 5-minute chunks
}
CONFIG = DEFAULT_CONFIG

@dataclass
class Segment:
    start: float
    end: float
    label: str
    confidence: float

# ------------------------------
# Utilities
# ------------------------------
def extract_audio(video_path: str, audio_path: str, ar: int = 16000) -> str:
    subprocess.run(["ffmpeg", "-y", "-i", video_path, "-vn", "-ac", "1", "-ar", str(ar), audio_path],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return audio_path

def transcribe_words(audio_path: str, model_size: str) -> List[Dict[str, float]]:
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments, _ = model.transcribe(audio_path, word_timestamps=True)
    words = []
    for seg in segments:
        for w in seg.words:
            words.append({"word": w.word.strip(), "start": w.start, "end": w.end})
    return words

def words_for_segment(words: List[Dict[str, float]], start: float, end: float) -> List[Dict[str, float]]:
    return [w for w in words if start <= w.get("start", 0) <= end]

# ------------------------------
# Audio detectors
# ------------------------------
def detect_audio_energy(audio_path: str, percentile: int = 90) -> List[Segment]:
    y, sr = librosa.load(audio_path, sr=16000, mono=True)
    energy = librosa.feature.rms(y=y)[0]
    thr = np.percentile(energy, percentile)
    segs = []
    hop = 512 / sr
    active = False; start = 0.0
    for i, e in enumerate(energy):
        t = i * hop
        if e >= thr and not active:
            start = t; active = True
        elif e < thr and active:
            segs.append(Segment(start, t, "audio_energy", 1.0)); active = False
    if active:
        segs.append(Segment(start, len(y)/sr, "audio_energy", 1.0))
    return segs

def detect_laughter(audio_path: str, backtrack: float = 6.0) -> List[Segment]:
    y, sr = librosa.load(audio_path, sr=16000, mono=True)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    times = librosa.times_like(onset_env, sr=sr, hop_length=512)
    segs = []
    for t in times[::200]:
        segs.append(Segment(max(0, t-backtrack), t, "funny/laughter", 0.8))
    return segs

# ------------------------------
# YAMNet TFLite (chunked)
# ------------------------------
_YAMNET_TFLITE = None
_YAMNET_INPUT = None
_YAMNET_OUTPUT = None
_YAMNET_LABELS = []

def _load_yamnet_tflite(model_path="yamnet.tflite", label_path="yamnet_class_map.csv"):
    global _YAMNET_TFLITE, _YAMNET_INPUT, _YAMNET_OUTPUT, _YAMNET_LABELS
    if _YAMNET_TFLITE is None:
        import tensorflow as tf
        _YAMNET_TFLITE = tf.lite.Interpreter(model_path=model_path)
        _YAMNET_TFLITE.allocate_tensors()
        _YAMNET_INPUT = _YAMNET_TFLITE.get_input_details()[0]
        _YAMNET_OUTPUT = _YAMNET_TFLITE.get_output_details()[0]
        with open(label_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            _YAMNET_LABELS = [row["display_name"] for row in reader]

def detect_audio_events_yamnet_tflite(audio_path: str,
                                      target_labels: Optional[List[str]] = None,
                                      score_thr: float = 0.25) -> List[Segment]:
    _load_yamnet_tflite()
    y, sr = librosa.load(audio_path, sr=16000, mono=True)
    if len(y) == 0: return []
    segs: List[Segment] = []
    frame_hop = 0.48
    waveform = np.expand_dims(y.astype(np.float32), axis=0)
    _YAMNET_TFLITE.set_tensor(_YAMNET_INPUT['index'], waveform)
    _YAMNET_TFLITE.invoke()
    scores = _YAMNET_TFLITE.get_tensor(_YAMNET_OUTPUT['index'])
    for i, row in enumerate(scores):
        top_idx = int(np.argmax(row)); top_score = float(row[top_idx])
        label = _YAMNET_LABELS[top_idx]
        if target_labels is None or label in target_labels:
            if top_score >= score_thr:
                segs.append(Segment(i*frame_hop, (i+1)*frame_hop, f"audio/{label}", top_score))
    return segs

# ------------------------------
# Scene detection
# ------------------------------
def detect_scenes(video_path: str, threshold: float = 30.0) -> List[Segment]:
    vm = VideoManager([video_path])
    sm = SceneManager()
    sm.add_detector(ContentDetector(threshold=threshold))
    vm.set_downscale_factor()
    vm.start()
    sm.detect_scenes(frame_source=vm)
    scene_list = sm.get_scene_list()
    vm.release()
    return [Segment(start=s[0].get_seconds(), end=s[1].get_seconds(), label="scene", confidence=1.0)
            for s in scene_list]

# ------------------------------
# Export highlight with karaoke
# ------------------------------
from app.utils.subtitles import make_karaoke_ass_from_words, escape_path_for_ffmpeg

def export_highlight_with_karaoke(video_path: str, seg: Segment, words: List[Dict[str, float]],
                                  output_dir: str = "highlights", export_mode: str = "both", cfg=CONFIG):
    os.makedirs(output_dir, exist_ok=True)
    start, end = seg.start, seg.end
    duration = max(0.1, end-start)
    base = f"clip_{int(start)}_{seg.label.replace('/', '_')}"
    out_clip = os.path.join(output_dir, base+".mp4")
    ass_file = os.path.join(output_dir, base+".ass")
    burned_out = os.path.join(output_dir, base+"_burned.mp4")
    subprocess.run(["ffmpeg", "-y", "-i", video_path, "-ss", f"{start:.2f}", "-t", f"{duration:.2f}", "-c", "copy", out_clip],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    seg_words = words_for_segment(words, start, end)
    if seg_words:
        make_karaoke_ass_from_words(seg_words, ass_file, cfg["karaoke"])
        if export_mode in ("burned", "both"):
            escaped_ass_path = escape_path_for_ffmpeg(ass_file)
            subprocess.run(["ffmpeg", "-y", "-i", out_clip, "-vf", f"subtitles={escaped_ass_path}", "-c:a", "copy", burned_out],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return {"clip": out_clip, "ass": ass_file}

# ------------------------------
# Chunk processor (worker)
# ------------------------------
def process_chunk(video_path, start, end, cfg):
    # Extract audio for chunk
    chunk_audio = f"chunk_{int(start)}_{int(end)}.wav"
    subprocess.run(["ffmpeg", "-y", "-i", video_path, "-ss", str(start), "-t", str(end-start),
                    "-vn", "-ac", "1", "-ar", "16000", chunk_audio],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    # Run detectors on chunk
    words = transcribe_words(chunk_audio, cfg["pipeline"]["model_size"])
    yamnet = detect_audio_events_yamnet_tflite(chunk_audio, cfg["yamnet"]["target_labels"], cfg["yamnet"]["score_threshold"])
    energy = detect_audio_energy(chunk_audio, percentile=cfg["audio_detection"]["energy_percentile"])
    laughter = detect_laughter(chunk_audio, backtrack=cfg["audio_detection"]["laughter_backtrack"])

    # Adjust offsets
    for seg in yamnet+energy+laughter:
        seg.start += start
        seg.end += start
    for w in words:
        w["start"] += start; w["end"] += start

    return {"segments": yamnet+energy+laughter, "words": words}

# ------------------------------
# Main orchestrator (parallel)
# ------------------------------
def run_pipeline_parallel(video_path: str, export_mode: str = "both") -> Dict[str, Any]:
    cfg = CONFIG

    # get duration
    import ffmpeg
    probe = ffmpeg.probe(video_path)
    duration = float(probe['format']['duration'])

    # create chunks
    chunk_len = cfg["parallel"]["chunk_sec"]
    chunks = [(i, min(i+chunk_len, duration)) for i in range(0, math.ceil(duration), chunk_len)]
    print(f"Splitting into {len(chunks)} chunks of {chunk_len}s")

    results = []
    with ProcessPoolExecutor() as ex:
        futures = [ex.submit(process_chunk, video_path, start, end, cfg) for start, end in chunks]
        for f in futures:
            results.append(f.result())

    # merge
    all_segments = []; all_words = []
    for r in results:
        all_segments.extend(r["segments"])
        all_words.extend(r["words"])

    # scene detection
    scenes = detect_scenes(video_path, cfg["scene"]["threshold"])
    print(f"Detected {len(scenes)} scenes")

    exported = []
    for seg in all_segments:
        if seg.confidence < cfg["pipeline"]["min_confidence"]: continue
        print(f"Exporting {seg.label} {seg.start:.1f}-{seg.end:.1f} conf={seg.confidence:.2f}")
        out = export_highlight_with_karaoke(video_path, seg, all_words, export_mode=export_mode, cfg=cfg)
        exported.append({**seg.__dict__, "exported": out})

    print(json.dumps(exported, indent=2))
    return {"segments": exported}

# ------------------------------
# CLI
# ------------------------------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Parallel Highlight Pipeline (no SER)")
    ap.add_argument("video", help="Input video file")
    args = ap.parse_args()
    run_pipeline_parallel(args.video)
