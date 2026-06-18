"""
server.py
---------
WebSocket server that receives binary frames from the Unity DepthVR frontend
(Meta Quest) and deserializes them into structured Frame objects.

This is Step 1 of the pipeline — data ingestion only.
Later steps (point cloud generation, Chamfer Distance, classification)
will be added as separate pipeline modules.

Usage
-----
    python server.py [--host HOST] [--port PORT] [--verbose]

    Default: ws://0.0.0.0:9002  (matches WebSocketSender.cs default IP on port 9002)

On the Quest side
-----------------
    In Unity Inspector → WebSocketSender → serverUrl:
        ws://<YOUR_PC_LAN_IP>:9002
    Both devices must be on the same WiFi network.
"""

import asyncio
import argparse
import logging
import time
from datetime import datetime

import websockets
from websockets.server import WebSocketServerProtocol

from protocol import deserialize_frame, Frame

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
# Stats tracker (printed every N frames)
# ──────────────────────────────────────────────────────────────────────────────

class ConnectionStats:
    def __init__(self):
        self.frames_received = 0
        self.parse_errors    = 0
        self.start_time      = time.monotonic()
        self._last_report    = time.monotonic()
        self._report_every   = 5.0   # seconds

    def record_frame(self, frame: Frame, verbose: bool = False):
        self.frames_received += 1
        elapsed = time.monotonic() - self._last_report

        if verbose:
            left_valid  = frame.left_hand.valid_count()  if frame.left_hand  else 0
            right_valid = frame.right_hand.valid_count() if frame.right_hand else 0
            log.debug(
                "Frame %5d | L=%2d/25 joints | R=%2d/25 joints",
                frame.frame_index, left_valid, right_valid,
            )

        if elapsed >= self._report_every:
            fps = self.frames_received / (time.monotonic() - self.start_time)
            log.info(
                "Stats → received=%d  errors=%d  avg_fps=%.1f",
                self.frames_received, self.parse_errors, fps,
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
):
    """
    Handles one connected Quest client.
    Receives binary frames, deserializes them, and logs basic statistics.
    This is the entry point for all downstream pipeline stages.
    """
    client_addr = websocket.remote_address
    log.info("Client connected: %s:%s", *client_addr)
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
                log.warning("Parse error (frame %d bytes): %s", len(message), e)
                continue

            stats.record_frame(frame, verbose=verbose)

            # ── PIPELINE HOOK ─────────────────────────────────────────────────
            # Step 1: data is deserialized into `frame` (Frame object).
            # Future steps will be called here, e.g.:
            #   point_cloud = build_point_cloud(frame.right_hand)
            #   distance    = chamfer_distance(point_cloud, reference)
            #   label       = classifier.predict(point_cloud)
            # ─────────────────────────────────────────────────────────────────

    except websockets.exceptions.ConnectionClosedOK:
        log.info("Client disconnected cleanly: %s:%s", *client_addr)
    except websockets.exceptions.ConnectionClosedError as e:
        log.warning("Client disconnected with error: %s", e)
    finally:
        total_time = time.monotonic() - stats.start_time
        log.info(
            "Session summary — frames=%d  errors=%d  duration=%.1fs",
            stats.frames_received, stats.parse_errors, total_time,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

async def main(host: str, port: int, verbose: bool):
    log.info("Starting WebSocket server on ws://%s:%d", host, port)
    log.info("Waiting for Quest connection…")

    async with websockets.serve(
        lambda ws: handle_connection(ws, verbose=verbose),
        host,
        port,
        max_size=10 * 1024 * 1024,   # 10 MB max message — well above any frame
        ping_interval=20,
        ping_timeout=60,
    ):
        await asyncio.Future()   # run until Ctrl-C


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DepthVR backend WebSocket server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=9002, help="Bind port (default: 9002)")
    parser.add_argument("--verbose", action="store_true", help="Log every frame's joint counts")
    args = parser.parse_args()

    try:
        asyncio.run(main(args.host, args.port, args.verbose))
    except KeyboardInterrupt:
        log.info("Server stopped.")
