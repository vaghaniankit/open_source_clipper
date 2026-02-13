from typing import List, Dict, Tuple

import numpy as np
import librosa
import tensorflow as tf
import tensorflow_hub as hub

from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector


def detect_scene_cuts(video_path: str, threshold: float = 27.0) -> List[float]:
    """Detect scene cuts using PySceneDetect and return cut timestamps in seconds.

    Returns a sorted list of times (in seconds) where a new scene starts.
    The first scene is assumed to start at t=0 and is not included in the cut list.
    """
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))

    try:
        video_manager.start()
        scene_manager.detect_scenes(video_manager)
        scene_list = scene_manager.get_scene_list()
    finally:
        video_manager.release()

    cuts: List[float] = []
    for idx, (start_time, _end_time) in enumerate(scene_list):
        if idx == 0:
            # first scene starts at 0, skip as a cut point
            continue
        cuts.append(float(start_time.get_seconds()))
    return sorted(cuts)


def attach_scene_metadata(segments: List[Dict], cut_times: List[float], near_cut_window: float = 0.7) -> List[Dict]:
    """Attach scene_id and near_cut to each transcript segment.

    - scene_id: index of the scene interval the segment belongs to (0-based).
    - near_cut: True if the segment start or end is within near_cut_window of a cut.
    """
    if not segments:
        return segments

    cut_times = sorted(cut_times or [])

    # Build scene boundaries: [0, cuts..., +inf)
    first_start = float(segments[0].get("start", 0.0))
    last_end = float(segments[-1].get("end", first_start))
    boundaries = [first_start] + [c for c in cut_times if first_start < c < last_end] + [last_end + 1.0]

    def scene_for_time(t: float) -> int:
        for i in range(len(boundaries) - 1):
            if boundaries[i] <= t < boundaries[i + 1]:
                return i
        return max(0, len(boundaries) - 2)

    for seg in segments:
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", start))
        mid = (start + end) / 2.0 if end >= start else start

        seg["scene_id"] = scene_for_time(mid)

        seg_near_cut = False
        for ct in cut_times:
            if abs(start - ct) <= near_cut_window or abs(end - ct) <= near_cut_window:
                seg_near_cut = True
                break
        seg["near_cut"] = seg_near_cut

    return segments


def _frame_events_from_labels(frame_times: np.ndarray, frame_labels: List[str]) -> List[Dict]:
    """Merge frame-level labels into contiguous events.

    frame_times: array of frame center times in seconds.
    frame_labels: list of labels ("music", "laughter", "none", etc.) per frame.
    """
    events: List[Dict] = []
    if len(frame_times) == 0:
        return events

    current_label = frame_labels[0]
    start_time = float(frame_times[0])
    for i in range(1, len(frame_times)):
        t = float(frame_times[i])
        label = frame_labels[i]
        if label != current_label:
            if current_label != "none":
                events.append({"start": start_time, "end": t, "labels": [current_label]})
            current_label = label
            start_time = t
    # flush last
    if current_label != "none":
        events.append({"start": start_time, "end": float(frame_times[-1]), "labels": [current_label]})
    return events


_YAMNET_MODEL_HANDLE = "https://tfhub.dev/google/yamnet/1"
_yamnet_model = None
_yamnet_class_names: List[str] = []


def _load_yamnet():
    global _yamnet_model, _yamnet_class_names
    if _yamnet_model is not None and _yamnet_class_names:
        return _yamnet_model, _yamnet_class_names

    model = hub.load(_YAMNET_MODEL_HANDLE)
    class_map_path = model.class_map_path().numpy()
    names: List[str] = []
    with tf.io.gfile.GFile(class_map_path, "r") as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            parts = line.strip().split(",")
            if not parts:
                continue
            names.append(parts[-1])
    _yamnet_model = model
    _yamnet_class_names = names
    return _yamnet_model, _yamnet_class_names


def _analyze_audio_events_yamnet(audio_path: str) -> List[Dict]:
    y, sr = librosa.load(audio_path, sr=16000, mono=True)
    if y.size == 0:
        return []

    model, class_names = _load_yamnet()
    waveform = y.astype(np.float32)
    scores, _embeddings, _spectrogram = model(waveform)
    scores_np = scores.numpy()

    frame_hop_seconds = 0.48
    times = np.arange(scores_np.shape[0]) * frame_hop_seconds

    labels: List[str] = []
    for i in range(scores_np.shape[0]):
        row = scores_np[i]
        if row.size == 0:
            labels.append("none")
            continue
        idx = int(np.argmax(row))
        name = class_names[idx] if 0 <= idx < len(class_names) else ""
        name_lower = name.lower()

        # Map YAMNet's rich label space into a smaller set of tags we care
        # about for highlight selection.
        label = "none"
        if "laugh" in name_lower or "laughter" in name_lower or "giggle" in name_lower or "chuckle" in name_lower:
            label = "laughter"
        elif "music" in name_lower or "singing" in name_lower or "choir" in name_lower:
            label = "music"
        elif "gunshot" in name_lower or "gun fire" in name_lower or "machine gun" in name_lower:
            label = "gunshot"
        elif "explosion" in name_lower or "burst" in name_lower or "fireworks" in name_lower:
            label = "explosion"
        elif "fart" in name_lower or "flatulence" in name_lower:
            label = "fart"
        elif "doorbell" in name_lower or "knock" in name_lower:
            label = "doorbell"
        elif "rain" in name_lower or "thunderstorm" in name_lower:
            label = "rain"
        elif "scream" in name_lower or "shout" in name_lower or "yell" in name_lower:
            label = "scream"
        elif "breath" in name_lower or "breathing" in name_lower:
            label = "breathing"
        elif "cheer" in name_lower or "applause" in name_lower or "crowd" in name_lower:
            label = "cheer"
        labels.append(label)

    return _frame_events_from_labels(times, labels)


def _analyze_audio_events_heuristic(audio_path: str, frame_length: float = 0.5, hop_length: float = 0.25) -> List[Dict]:
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    if y.size == 0:
        return []

    frame_len = int(frame_length * sr)
    hop_len = int(hop_length * sr)

    rms = librosa.feature.rms(y=y, frame_length=frame_len, hop_length=hop_len)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=frame_len, hop_length=hop_len)[0]
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=frame_len, hop_length=hop_len)[0]

    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_len)

    rms_med = float(np.median(rms)) if rms.size else 0.0
    rms_high = rms_med * 1.8 if rms_med > 0 else 0.0
    centroid_med = float(np.median(centroid)) if centroid.size else 0.0
    zcr_med = float(np.median(zcr)) if zcr.size else 0.0

    labels: List[str] = []
    for i in range(len(times)):
        r = rms[i]
        c = centroid[i]
        z = zcr[i]
        label = "none"
        if r > rms_high and c > centroid_med * 1.2:
            label = "music"
        elif r > rms_med * 1.3 and z > zcr_med * 1.3:
            label = "laughter"
        labels.append(label)

    return _frame_events_from_labels(times, labels)


def analyze_audio_events(audio_path: str, frame_length: float = 0.5, hop_length: float = 0.25) -> List[Dict]:
    try:
        return _analyze_audio_events_yamnet(audio_path)
    except Exception:
        return _analyze_audio_events_heuristic(audio_path, frame_length=frame_length, hop_length=hop_length)


def attach_audio_tags(segments: List[Dict], events: List[Dict]) -> List[Dict]:
    """Attach tags to segments based on overlapping audio events.

    Each segment gets a list of unique tags, e.g. ["laughter", "music"].
    """
    if not segments or not events:
        for seg in segments:
            seg.setdefault("tags", [])
        return segments

    for seg in segments:
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", start))
        seg_labels = set()
        for ev in events:
            ev_start = float(ev.get("start", 0.0))
            ev_end = float(ev.get("end", ev_start))
            if ev_end <= ev_start:
                continue
            # overlap check
            if max(start, ev_start) < min(end, ev_end):
                for lab in ev.get("labels", []):
                    if lab and lab != "none":
                        seg_labels.add(lab)
        seg["tags"] = sorted(seg_labels)
    return segments


def compute_excitement_score(segments: List[Dict]) -> List[Dict]:
    """Compute a simple excitement score per segment.

    excitement is a weighted combination of normalised energy, near cuts,
    and tags like laughter/music.
    """
    if not segments:
        return segments

    energies = [float(seg.get("energy", 0.0)) for seg in segments]
    max_energy = max(energies) if energies else 0.0

    for seg in segments:
        energy = float(seg.get("energy", 0.0))
        energy_norm = energy / max_energy if max_energy > 0 else 0.0

        near_cut = bool(seg.get("near_cut", False))
        tags = seg.get("tags", []) or []

        # Base excitement from normalised energy
        base = 0.5 * energy_norm

        # Small bonus if the segment is aligned with a scene cut
        cut_bonus = 0.2 if near_cut else 0.0

        # Tag-driven bonus: any non-"none" tag gives a small bump, and
        # certain high-salience tags give additional weight.
        tag_bonus = 0.0
        exciting_tags = {"laughter", "scream", "gunshot", "explosion", "cheer"}
        neutral_tags = {"music", "fart", "doorbell", "rain", "breathing"}

        for t in tags:
            if not t or t == "none":
                continue
            if t in exciting_tags:
                tag_bonus += 0.2
            elif t in neutral_tags:
                tag_bonus += 0.1
            else:
                tag_bonus += 0.05

        excitement = base + cut_bonus + tag_bonus
        if excitement > 1.0:
            excitement = 1.0
        seg["excitement"] = round(float(excitement), 4)

    return segments
