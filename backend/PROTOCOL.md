# WebSocket Binary Frame Protocol

This document specifies the exact binary layout of each frame
sent from the Unity DepthVR frontend (Meta Quest) to the Python backend.

## Source Files (Unity)
- `FrameSerializer.cs`  — assembles the packet
- `HandLandmarkLogger.cs` — writes the hand joint binary blob
- `FrameSyncManager.cs`  — provides FrameIndex and Timestamp
- `XRFrameRecorder.cs`  — calls Serialize() and sends via WebSocket each LateUpdate()

---

## Packet Layout (little-endian, binary)

```
Offset   Type      Size   Field
------   ----      ----   -----
0        int32     4      FrameIndex         (increments each LateUpdate tick)
4        int64     8      Timestamp          (Stopwatch.GetTimestamp(), high-res ticks)
12       uint16    2      ColorWidth
14       uint16    2      ColorHeight
16       uint16    2      DepthWidth
18       uint16    2      DepthHeight
20       int32     4      colorLen           (byte count of color blob)
24       int32     4      depthLen           (byte count of depth blob)
28       int32     4      handsLen           (byte count of hand blob)
32       bytes     N      hands blob         (N = handsLen)
32+N     bytes     M      color blob         (M = colorLen, often 0)
32+N+M   bytes     K      depth blob         (K = depthLen, often 0)
```

Total header size: 32 bytes

---

## Hand Blob Layout

Written by `HandLandmarkLogger.WriteHandData()`.

**Two hands are always written, in order: LEFT then RIGHT.**

For each hand, 25 joints are written in this fixed order:
```
 0  Wrist
 1  ThumbMetacarpal
 2  ThumbProximal
 3  ThumbDistal
 4  ThumbTip
 5  IndexMetacarpal
 6  IndexProximal
 7  IndexIntermediate
 8  IndexDistal
 9  IndexTip
10  MiddleMetacarpal
11  MiddleProximal
12  MiddleIntermediate
13  MiddleDistal
14  MiddleTip
15  RingMetacarpal
16  RingProximal
17  RingIntermediate
18  RingDistal
19  RingTip
20  LittleMetacarpal
21  LittleProximal
22  LittleIntermediate
23  LittleDistal
24  LittleTip
```

For each joint:
```
Type     Size   Field
----     ----   -----
bool     1      valid     (1 = tracked, 0 = not tracked)
float32  4      x         (camera-local space, metres)
float32  4      y         (camera-local space, metres)
float32  4      z         (camera-local space, metres)
float32  4      depth     (Euclidean distance from camera origin, metres)
```

Per-joint size: 1 + 4 + 4 + 4 + 4 = **17 bytes**
Per-hand size:  25 × 17 = **425 bytes**
Total hands blob (both hands): 425 × 2 = **850 bytes**

If a joint is NOT valid, x/y/z/depth are written as 0.0f.

---

## Coordinate System

Positions are in **camera-local space**:
- Origin: XR camera (headset) position
- Computed via `camT.InverseTransformPoint(pose.position)`
- Units: metres
- depth = `Vector3.Distance(camT.position, pose.position)` (world space distance)

---

## Notes

- Color and depth blobs are currently often empty (length = 0).
  The pipeline processes hand joint data only for now.
- WebSocket URL default: `ws://<PC_IP>:9002`
- One packet is sent per Unity LateUpdate() frame (~72 fps on Quest).
