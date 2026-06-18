using UnityEngine;

public class DepthFrameLogger : MonoBehaviour
{
    public FrameBuffer buffer;
    public Shader depthShader;
    public RenderTexture depthRenderTexture;

    // PERFORMANCE SETTINGS
    public float captureRate = 0.1f; // 10 FPS
    private float timer = 0;

    private Camera depthCam;
    private Texture2D tempTex;
    private Rect rect;

    void Start()
    {
        // Keep your existing Start() code here to create the camera...
        GameObject go = new GameObject("DepthCam_Hidden");
        depthCam = go.AddComponent<Camera>();
        depthCam.enabled = false;

        Camera mainCam = Camera.main;
        if (mainCam != null)
        {
            go.transform.SetParent(mainCam.transform);
            go.transform.localPosition = Vector3.zero;
            go.transform.localRotation = Quaternion.identity;
            depthCam.fieldOfView = mainCam.fieldOfView;
            depthCam.nearClipPlane = mainCam.nearClipPlane;
            depthCam.farClipPlane = mainCam.farClipPlane;
        }
    }

    void LateUpdate()
    {
        // 1. Throttling
        timer += Time.deltaTime;
        if (timer < captureRate) return;
        timer = 0;

        if (buffer == null || buffer.LatestDepth != null) return;
        if (depthRenderTexture == null || depthShader == null) return;

        // 2. Render & Read
        depthCam.targetTexture = depthRenderTexture;
        depthCam.RenderWithShader(depthShader, "");

        if (tempTex == null || tempTex.width != depthRenderTexture.width)
        {
            tempTex = new Texture2D(depthRenderTexture.width, depthRenderTexture.height, TextureFormat.RGBA32, false);
            rect = new Rect(0, 0, depthRenderTexture.width, depthRenderTexture.height);
        }

        RenderTexture currentActive = RenderTexture.active;
        RenderTexture.active = depthRenderTexture;

        tempTex.ReadPixels(rect, 0, 0);
        tempTex.Apply();

        RenderTexture.active = currentActive;

        buffer.DepthWidth = depthRenderTexture.width;
        buffer.DepthHeight = depthRenderTexture.height;
        buffer.LatestDepth = tempTex.GetRawTextureData();
    }
}