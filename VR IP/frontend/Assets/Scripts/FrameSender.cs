using UnityEngine;
using System;
using System.Net.WebSockets;
using System.Threading;
using System.Threading.Tasks;

public class FrameSender : MonoBehaviour
{
    // ?? Singleton
    public static FrameSender Instance { get; private set; }

    ClientWebSocket socket;
    CancellationTokenSource cts;

    public string serverUrl = "ws://192.168.1.100:8765"; // CHANGE to PC IP
    public bool autoConnect = true;

    void Awake()
    {
        // ? Singleton enforcement
        if (Instance != null && Instance != this)
        {
            Destroy(gameObject);
            return;
        }

        Instance = this;
        DontDestroyOnLoad(gameObject);
    }

    async void Start()
    {
        if (autoConnect)
            await Connect();
    }

    async Task Connect()
    {
        socket = new ClientWebSocket();
        cts = new CancellationTokenSource();

        try
        {
            await socket.ConnectAsync(new Uri(serverUrl), cts.Token);
            Debug.Log("WebSocket OPEN");
        }
        catch (Exception e)
        {
            Debug.LogError("WebSocket failed: " + e.Message);
        }
    }

    // ===============================
    // COLOR
    // ===============================
    public async void SendColor(byte[] rgb, int width, int height)
    {
        if (socket == null || socket.State != WebSocketState.Open) return;

        await socket.SendAsync(
            new ArraySegment<byte>(rgb),
            WebSocketMessageType.Binary,
            true,
            cts.Token
        );
    }

    // ===============================
    // DEPTH
    // ===============================
    public async void SendDepth(float[] depth)
    {
        if (socket == null || socket.State != WebSocketState.Open) return;

        byte[] bytes = new byte[depth.Length * 4];
        Buffer.BlockCopy(depth, 0, bytes, 0, bytes.Length);

        await socket.SendAsync(
            new ArraySegment<byte>(bytes),
            WebSocketMessageType.Binary,
            true,
            cts.Token
        );
    }

    void OnDestroy()
    {
        cts?.Cancel();
        socket?.Dispose();
    }
}
