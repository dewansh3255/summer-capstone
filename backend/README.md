# Backend Pipeline

Python backend for the DepthVR gesture recognition system.
Receives hand joint data from the Meta Quest via WebSocket and runs the gesture classification pipeline.

## Architecture

```
Meta Quest (Unity DepthVR)
    └── WebSocket (binary frames) ──► server.py
                                          └── protocol/frame_deserializer.py   [Step 1 ✅]
                                          └── pipeline/point_cloud.py          [Step 2 ✅]
                                          └── pipeline/recorder.py             [Step 3 ✅]
                                          └── pipeline/chamfer.py              [Step 3 ✅]
                                          └── pipeline/classifier.py           [Step 4]
```

## Setup

```bash
pip install -r requirements.txt
```

## Running the server

```bash
# Live mode (no saving):
python3 server.py --port 9002 --verbose

# Recording mode — saves frames for offline development:
python3 server.py --record --gesture ThumbsUp --recordings-dir recordings/
```

Then in Unity, set `WebSocketSender.serverUrl` to `ws://<YOUR_LAN_IP>:9002`.

Once a session is recorded, all downstream pipeline work runs offline:

```python
from pipeline.recorder import load_session
from pipeline.chamfer import extract_pairs, find_optimal_threshold

clouds, meta = load_session("recordings/2026-06-18_120000_ThumbsUp")
```

## Running tests

```bash
python3 -m pytest tests/ -v
```

## Protocol

See [PROTOCOL.md](PROTOCOL.md) for the full binary frame specification.
