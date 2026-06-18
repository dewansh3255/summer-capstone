using UnityEngine;
using System.Diagnostics;

public class FrameSyncManager : MonoBehaviour
{
    public bool recording = true;

    public int FrameIndex { get; private set; }
    public long Timestamp { get; private set; }

    public void Tick()
    {
        if (!recording) return;

        FrameIndex++;
        Timestamp = Stopwatch.GetTimestamp();
    }
}
