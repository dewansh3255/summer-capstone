using UnityEngine;

public class XRFrameRecorder : MonoBehaviour
{
    FrameBuffer buffer;
    FrameSerializer serializer;
    FrameSyncManager sync;
    WebSocketSender socket;

    void Awake()
    {
        Debug.Log("[REC] XRFrameRecorder Awake");
        buffer = GetComponent<FrameBuffer>();
        serializer = GetComponent<FrameSerializer>();
        sync = GetComponent<FrameSyncManager>();
        socket = GetComponent<WebSocketSender>();

        if (buffer == null || serializer == null || sync == null || socket == null)
        {
            Debug.LogError(
                "XRFrameRecorder missing required component(s).\n" +
                "Ensure FrameBuffer, FrameSerializer, FrameSyncManager, and WebSocketSender\n" +
                "are on the SAME GameObject."
            );
            enabled = false; // ?? prevents crash loop
        }
        else
        {
            Debug.Log("[REC] All required components found");
        }
    }

    void LateUpdate()
    {
        if (!enabled) return;

        // DO NOT RETURN if color is null. 
        // Just send empty bytes so the Python script doesn't freeze.
        if (buffer.LatestColor == null) buffer.LatestColor = System.Array.Empty<byte>();
        if (buffer.LatestDepth == null) buffer.LatestDepth = System.Array.Empty<byte>();
        if (buffer.LatestHands == null) buffer.LatestHands = System.Array.Empty<byte>();

        // Only stop if literally NOTHING is happening
        if (buffer.LatestHands.Length == 0 && buffer.LatestColor.Length == 0)
            return;

        sync.Tick();

        byte[] packet = serializer.Serialize(buffer);
        socket.Send(packet);

        // Reset for next frame
        buffer.LatestColor = null;
        buffer.LatestHands = null;
        buffer.LatestDepth = null;
    }
}
