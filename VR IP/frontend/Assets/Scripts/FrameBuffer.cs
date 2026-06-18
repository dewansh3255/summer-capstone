using UnityEngine;

public class FrameBuffer : MonoBehaviour
{
    public byte[] LatestColor;
    public byte[] LatestDepth;
    public byte[] LatestHands;

    public int ColorWidth;
    public int ColorHeight;
    public int DepthWidth;
    public int DepthHeight;
}
