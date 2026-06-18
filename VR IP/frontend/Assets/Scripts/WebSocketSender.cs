using NativeWebSocket;
using UnityEngine;

public class WebSocketSender : MonoBehaviour
{
    // CHECK THIS: Did your IP change? Run 'ipconfig' on PC to verify.
    public string serverUrl = "ws://192.168.52.110:9002";

    WebSocket ws;

    void Awake()
    {
        // This runs instantly when the object loads.
        // If you don't see this, the object is NOT in the build.
        Debug.LogError("[WS-DEBUG] !!! SCRIPT IS ALIVE !!!");
    }

    async void Start()
    {
        Debug.Log($"[WS-DEBUG] initializing... Target URL: {serverUrl}");

        ws = new WebSocket(serverUrl);

        // --- NEW DEBUG LISTENERS ---
        ws.OnOpen += () =>
        {
            Debug.Log("[WS-DEBUG] >>> CONNECTION SUCCESSFUL! <<<");
        };

        ws.OnError += (e) =>
        {
            Debug.LogError($"[WS-DEBUG] >>> CONNECTION ERROR: {e}");
        };

        ws.OnClose += (e) =>
        {
            Debug.LogWarning($"[WS-DEBUG] >>> CONNECTION CLOSED. Code: {e}");
        };
        // ---------------------------

        try
        {
            Debug.Log("[WS-DEBUG] Connecting...");
            await ws.Connect();
        }
        catch (System.Exception ex)
        {
            Debug.LogError($"[WS-DEBUG] CRITICAL EXCEPTION: {ex.Message}");
        }
    }

    public async void Send(byte[] data)
    {
        if (ws == null) return;

        if (ws.State == WebSocketState.Open)
        {
            // Debug.Log($"[WS-DEBUG] Sending {data.Length} bytes..."); // Uncomment if needed
            await ws.Send(data);
        }
        else
        {
            // This tells us if it's trying to send while broken
            if (Time.frameCount % 60 == 0) // Only log once per second to avoid spam
                Debug.LogWarning($"[WS-DEBUG] Cannot send. State is: {ws.State}");
        }
    }

    void Update()
    {
#if !UNITY_WEBGL || UNITY_EDITOR
        if (ws != null) ws.DispatchMessageQueue();
#endif
    }

    async void OnApplicationQuit()
    {
        if (ws != null) await ws.Close();
    }
}