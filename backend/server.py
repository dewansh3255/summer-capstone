"""
server.py
---------
WebSocket server that receives binary frames from the Unity DepthVR frontend
(Meta Quest), runs the gesture pipeline, and optionally records sessions to disk.

Usage
-----
    # Live mode (pipeline only, no saving):
    python3 server.py --port 9002 --verbose

    # Recording mode (saves frames for offline use):
    python3 server.py --record --gesture ThumbsUp --recordings-dir recordings/

    Default WebSocket: ws://0.0.0.0:9002  (matches WebSocketSender.cs port 9002)

On the Quest side
-----------------
    In Unity Inspector → WebSocketSender → serverUrl:
        ws://<YOUR_LAN_IP>:9002
    Both devices must be on the same WiFi network.

Recording workflow
------------------
    1. Run server.py with --record --gesture <label>
    2. Put on the Quest, perform the gesture for ~5–10 seconds
    3. Ctrl-C to stop — session is saved to recordings/<timestamp>_<label>/
    4. All downstream pipeline work (Chamfer Distance, classification) can
       then be run offline using load_session() — no Quest needed.
"""

import asyncio
import argparse
import logging
import time

import websockets
from websockets.server import WebSocketServerProtocol

from protocol import deserialize_frame, Frame
from pipeline import build_point_cloud, Recorder, GestureClassifier

# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("server")


# ──────────────────────────────────────────────────────────────────────────────
# Stats tracker
# ──────────────────────────────────────────────────────────────────────────────

class ConnectionStats:
    def __init__(self):
        self.frames_received = 0
        self.frames_saved    = 0
        self.parse_errors    = 0
        self.start_time      = time.monotonic()
        self._last_report    = time.monotonic()
        self._report_every   = 5.0

    def record_frame(self, frame: Frame, saved: bool = False, verbose: bool = False):
        self.frames_received += 1
        if saved:
            self.frames_saved += 1

        if verbose:
            left_valid  = frame.left_hand.valid_count()  if frame.left_hand  else 0
            right_valid = frame.right_hand.valid_count() if frame.right_hand else 0
            log.debug(
                "Frame %5d | L=%2d/25 joints | R=%2d/25 joints | saved=%s",
                frame.frame_index, left_valid, right_valid, saved,
            )

        elapsed = time.monotonic() - self._last_report
        if elapsed >= self._report_every:
            fps = self.frames_received / (time.monotonic() - self.start_time)
            log.info(
                "Stats → received=%d  saved=%d  errors=%d  avg_fps=%.1f",
                self.frames_received, self.frames_saved, self.parse_errors, fps,
            )
            self._last_report = time.monotonic()

    def record_error(self):
        self.parse_errors += 1


# ──────────────────────────────────────────────────────────────────────────────
# Connection handler
# ──────────────────────────────────────────────────────────────────────────────

async def handle_connection(
    websocket: WebSocketServerProtocol,
    verbose: bool = False,
    recorder: Recorder = None,
    classifier: GestureClassifier = None,
):
    """
    Handles one connected Quest client.
    Runs the full pipeline per frame; optionally records point clouds to disk.
    """
    client_addr = websocket.remote_address
    log.info("Client connected: %s:%s", *client_addr)

    if recorder is not None:
        recorder.start()
        log.info("Recording started — gesture: '%s'", recorder.gesture_label)

    stats = ConnectionStats()

    try:
        async for message in websocket:
            if not isinstance(message, bytes):
                log.warning("Received non-binary message — skipping.")
                continue

            try:
                frame = deserialize_frame(message)
            except ValueError as e:
                stats.record_error()
                log.warning("Parse error (%d bytes): %s", len(message), e)
                continue

            # ── PIPELINE ─────────────────────────────────────────────────────
            # Step 1 ✅  raw bytes → Frame
            # Step 2 ✅  HandData → normalised PointCloud
            # Step 3 ✅  save to disk (if recording mode) / Chamfer Distance
            # Step 4 ✅  MLP classifier → gesture label
            # ─────────────────────────────────────────────────────────────────

            saved = False
            if frame.right_hand is not None:
                pc = build_point_cloud(
                    frame.right_hand, frame_index=frame.frame_index
                )
                if pc is not None:
                    if recorder is not None:
                        recorder.add(pc)
                        saved = True

                    if classifier is not None:
                        label = classifier.predict_label(pc.points)
                        if verbose:
                            log.debug(
                                "Frame %5d → predicted: %s",
                                frame.frame_index, label,
                            )

            stats.record_frame(frame, saved=saved, verbose=verbose)

    except websockets.exceptions.ConnectionClosedOK:
        log.info("Client disconnected cleanly: %s:%s", *client_addr)
    except websockets.exceptions.ConnectionClosedError as e:
        log.warning("Client disconnected with error: %s", e)
    finally:
        if recorder is not None:
            meta = recorder.stop()
            log.info(
                "Recording saved → %d frames  %.1fs  %.1f fps  dir=%s",
                meta.get("frame_count", 0),
                meta.get("duration_s", 0),
                meta.get("avg_fps", 0),
                meta.get("session_dir", "?"),
            )
        total_time = time.monotonic() - stats.start_time
        log.info(
            "Session summary — received=%d  saved=%d  errors=%d  duration=%.1fs",
            stats.frames_received, stats.frames_saved,
            stats.parse_errors, total_time,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

async def main(host: str, port: int, verbose: bool,
               record: bool, gesture: str, recordings_dir: str,
               model_path: str):
    log.info("Starting WebSocket server on ws://%s:%d", host, port)

    recorder   = None
    classifier = None

    if record:
        if not gesture:
            raise ValueError("--gesture is required when --record is set.")
        recorder = Recorder(recordings_dir, gesture_label=gesture)
        log.info("Record mode ON — gesture='%s'  dir='%s'", gesture, recordings_dir)
    else:
        log.info("Live mode (no recording). Use --record --gesture <label> to save.")

    if model_path:
        try:
            classifier = GestureClassifier.load(model_path)
            log.info("Classifier loaded from '%s'", model_path)
        except FileNotFoundError:
            log.warning("Model file '%s' not found — running without classifier.", model_path)

    log.info("Waiting for Quest connection…")

    async with websockets.serve(
        lambda ws: handle_connection(
            ws, verbose=verbose, recorder=recorder, classifier=classifier
        ),
        host,
        port,
        max_size=10 * 1024 * 1024,
        ping_interval=20,
        ping_timeout=60,
    ):
        await asyncio.Future()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DepthVR backend WebSocket server")
    parser.add_argument("--host",  default="0.0.0.0",    help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port",  type=int, default=9002, help="Bind port (default: 9002)")
    parser.add_argument("--verbose", action="store_true",  help="Log every frame")

    # Recording flags
    parser.add_argument("--record",  action="store_true",
                        help="Save incoming frames to disk for offline use")
    parser.add_argument("--gesture", default="",
                        help="Gesture label for this recording session (e.g. ThumbsUp)")
    parser.add_argument("--recordings-dir", default="recordings",
                        help="Root directory for saved sessions (default: recordings/)")
    parser.add_argument("--model", default="",
                        dest="model_path",
                        help="Path to a saved GestureClassifier .npz for live inference")

    args = parser.parse_args()

    try:
        asyncio.run(main(
            args.host, args.port, args.verbose,
            args.record, args.gesture, args.recordings_dir,
            args.model_path,
        ))
    except KeyboardInterrupt:
        log.info("Server stopped.")
