
import json
from app.utils.subtitles import build_clip_srt_and_ass

# Mock segment data mimicking real input with speech and overlap events
segments = [
    {
        "start": 0.0,
        "end": 5.0,
        "words": [
            {"word": "Hello", "start": 0.5, "end": 1.0},
            {"word": "world", "start": 1.2, "end": 1.8},
            {"word": "this", "start": 2.0, "end": 2.5},
            {"word": "is", "start": 2.6, "end": 2.8},
            {"word": "a", "start": 2.9, "end": 3.0},
            {"word": "test.", "start": 3.1, "end": 4.0}
        ],
        "audio_events": [
            {"label": "laughter", "start": 2.0, "end": 4.5}
        ]
    },
    {
        "start": 6.0,
        "end": 10.0,
        "words": [],
        "audio_events": [
            {"label": "applause", "start": 6.5, "end": 9.5}
        ]
    }
]

# Clip window covers both
clip_start = 0.0
clip_end = 10.0

srt_out, ass_out = build_clip_srt_and_ass(segments, clip_start, clip_end)

print("SRT OUTPUT:\n", srt_out)
print("\nASS OUTPUT:\n", ass_out)
