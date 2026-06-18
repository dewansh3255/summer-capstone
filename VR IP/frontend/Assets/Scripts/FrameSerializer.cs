using System.IO;
using UnityEngine;

public class FrameSerializer : MonoBehaviour
{
    FrameSyncManager sync;

    void Awake()
    {
        sync = GetComponent<FrameSyncManager>();
    }

    public byte[] Serialize(FrameBuffer b)
    {
        using MemoryStream ms = new();
        using BinaryWriter bw = new(ms);

        bw.Write(sync.FrameIndex);
        bw.Write(sync.Timestamp);

        bw.Write((ushort)b.ColorWidth);
        bw.Write((ushort)b.ColorHeight);
        bw.Write((ushort)b.DepthWidth);
        bw.Write((ushort)b.DepthHeight);

        bw.Write(b.LatestColor.Length);
        bw.Write(b.LatestDepth.Length);
        bw.Write(b.LatestHands.Length);

        bw.Write(b.LatestHands);
        bw.Write(b.LatestColor);
        bw.Write(b.LatestDepth);

        return ms.ToArray();
    }
}
