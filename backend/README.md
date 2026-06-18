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
                                          └── pipeline/dataset.py              [Step 4 ✅]
                                          └── pipeline/classifier.py           [Step 4 ✅]
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
python3 server.py --record --gesture Thumbs_Up --recordings-dir recordings/

# Live inference mode — classifies gestures in real time:
python3 server.py --model models/gesture_clf.npz --verbose
```

Then in Unity, set `WebSocketSender.serverUrl` to `ws://<YOUR_LAN_IP>:9002`.

## Offline workflow (no Quest needed after first recording)

```python
from pipeline.recorder import load_session, list_sessions
from pipeline.chamfer import extract_pairs, find_optimal_threshold
from pipeline.dataset import load_dataset, tsts_split, loso_splits
from pipeline.classifier import GestureClassifier

# 1. Load all recorded sessions into a labelled dataset
dataset = load_dataset("recordings/")

# 2. Train/test split (temporal — no data leakage)
train, test = tsts_split(dataset, train_ratio=0.8)

# 3. Train the MLP classifier
clf = GestureClassifier()
clf.fit(train, epochs=100, lr=1e-3)
print(f"Test accuracy: {clf.score(test):.3f}")

# 4. Save trained model for live inference
clf.save("models/gesture_clf.npz")

# 5. (Optional) LOSO-CV for OOD evaluation (EMGBench methodology)
for train_fold, test_fold, held_out in loso_splits(dataset):
    clf_fold = GestureClassifier()
    clf_fold.fit(train_fold, epochs=100, verbose=False)
    print(f"LOSO held-out={held_out}  acc={clf_fold.score(test_fold):.3f}")
```

## Gesture vocabulary

The 10 gestures in DepthVR (matching `Assets/Gesture Videos/` filenames):

| Label | Description |
|---|---|
| ASL_L | ASL letter L |
| ASL_Y | ASL letter Y |
| Five | Open hand, all fingers extended |
| Four | Four fingers extended |
| One | Index finger pointing |
| Spiderman | Web-shooting pose |
| Spock | Vulcan salute |
| Three | Three fingers extended |
| Thumbs_Up | Thumbs up |
| Two | Two fingers extended |

## Running tests

```bash
python3 -m pytest tests/ -v
```

## Protocol

See [PROTOCOL.md](PROTOCOL.md) for the full binary frame specification.
