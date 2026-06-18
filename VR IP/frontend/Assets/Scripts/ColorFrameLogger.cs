using UnityEngine;

public class ColorFrameLogger : MonoBehaviour
{
    public FrameBuffer buffer;
    public RenderTexture xrRenderTexture;

    // PERFORMANCE SETTINGS
    public float captureRate = 0.1f; // Capture every 0.1s (10 FPS)
    private float timer = 0;

    private Texture2D tempTex;
    private Rect rect;

    void LateUpdate()
    {
        // 1. Throttling: Only run if enough time has passed
        timer += Time.deltaTime;
        if (timer < captureRate) return;
        timer = 0; // Reset timer

        if (buffer == null || buffer.LatestColor != null) return;
        if (xrRenderTexture == null) return;

        // 2. Efficient Setup
        if (tempTex == null || tempTex.width != xrRenderTexture.width)
        {
            tempTex = new Texture2D(xrRenderTexture.width, xrRenderTexture.height, TextureFormat.RGBA32, false);
            rect = new Rect(0, 0, xrRenderTexture.width, xrRenderTexture.height);
        }

        // 3. The "Heavy" Operation (Now runs less often)
        RenderTexture currentActive = RenderTexture.active;
        RenderTexture.active = xrRenderTexture;

        try
        {
            tempTex.ReadPixels(rect, 0, 0);
            tempTex.Apply();

            buffer.ColorWidth = xrRenderTexture.width;
            buffer.ColorHeight = xrRenderTexture.height;
            buffer.LatestColor = tempTex.GetRawTextureData();
        }
        finally
        {
            RenderTexture.active = currentActive;
        }
    }
}