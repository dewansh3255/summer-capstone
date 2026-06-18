////code 6 - hand landmark detection
////using System;
////using System.Collections;
////using System.Collections.Generic;
////using System.IO;
////using System.Text;
////using UnityEngine;
////using UnityEngine.XR;
////using UnityEngine.XR.Hands;

////public class HandLandmarkLogger : MonoBehaviour
////{
////    private XRHandSubsystem _handSubsystem;
////    private string _filePath;

////    void Start()
////    {
////        // Verify MainCamera tag
////        if (Camera.main == null)
////        {
////            Debug.LogError("HandLandmarkLogger: No MainCamera found. Please tag your XR camera as \"MainCamera\".");
////        }
////        else
////        {
////            Debug.Log($"HandLandmarkLogger: Found MainCamera at {Camera.main.transform.position}");
////        }

////        // Setup file path for logging
////        _filePath = Path.Combine(Application.persistentDataPath, "HandLandmarkData.txt");
////        try
////        {
////            File.WriteAllText(_filePath, "Time,Hand,Joint,X,Y,Z\n"); // CSV format
////            Debug.Log($"HandLandmarkLogger: Logging to {_filePath}");
////        }
////        catch (Exception e)
////        {
////            Debug.LogError($"HandLandmarkLogger: Unable to write initial file: {e.Message}");
////        }

////        // Begin coroutine to initialize XRHandSubsystem (delayed start)
////        StartCoroutine(InitializeHandSubsystem());
////    }

////    private IEnumerator InitializeHandSubsystem()
////    {
////        // Wait a bit for XR to initialize (OpenXR often needs 0.5–1s)
////        yield return new WaitForSeconds(1.0f);

////        var subsystems = new List<XRHandSubsystem>();
////        SubsystemManager.GetSubsystems(subsystems);

////        // Debug all detected subsystems for verification
////        foreach (var s in subsystems)
////        {
////            Debug.Log($"HandLandmarkLogger: Subsystem found: {s.GetType().FullName}");
////        }

////        if (subsystems.Count > 0)
////        {
////            _handSubsystem = subsystems[0];
////            _handSubsystem.updatedHands += OnHandsUpdated;
////            _handSubsystem.trackingAcquired += OnHandTrackingAcquired;
////            _handSubsystem.trackingLost += OnHandTrackingLost;
////            Debug.Log("HandLandmarkLogger: Hand subsystem found and callbacks registered.");
////        }
////        else
////        {
////            Debug.LogError("HandLandmarkLogger: No XRHandSubsystem found. Hand tracking won’t work.");
////        }
////    }

////    private void OnHandTrackingAcquired(XRHand hand)
////    {
////        string label = HandLabelFromHandedness(hand.handedness);
////        Debug.Log($"HandLandmarkLogger: Hand tracking acquired for {label} hand.");
////    }

////    private void OnHandTrackingLost(XRHand hand)
////    {
////        string label = HandLabelFromHandedness(hand.handedness);
////        Debug.Log($"HandLandmarkLogger: Hand tracking lost for {label} hand.");
////    }

////    private void OnHandsUpdated(XRHandSubsystem subsystem,
////        XRHandSubsystem.UpdateSuccessFlags successFlags,
////        XRHandSubsystem.UpdateType updateType)
////    {
////        if (_handSubsystem == null) return;

////        if (successFlags.HasFlag(XRHandSubsystem.UpdateSuccessFlags.LeftHandJoints))
////        {
////            LogHandLandmarks(_handSubsystem.leftHand, "Left");
////        }

////        if (successFlags.HasFlag(XRHandSubsystem.UpdateSuccessFlags.RightHandJoints))
////        {
////            LogHandLandmarks(_handSubsystem.rightHand, "Right");
////        }
////    }

////    private void LogHandLandmarks(XRHand hand, string label)
////    {
////        if (Camera.main == null) return;

////        // Define the list of valid joint IDs for a hand
////        List<XRHandJointID> validJointIds = new List<XRHandJointID>
////        {
////            XRHandJointID.Wrist,
////            XRHandJointID.ThumbMetacarpal,
////            XRHandJointID.ThumbProximal,
////            XRHandJointID.ThumbDistal,
////            XRHandJointID.ThumbTip,
////            XRHandJointID.IndexMetacarpal,
////            XRHandJointID.IndexProximal,
////            XRHandJointID.IndexIntermediate,
////            XRHandJointID.IndexDistal,
////            XRHandJointID.IndexTip,
////            XRHandJointID.MiddleMetacarpal,
////            XRHandJointID.MiddleProximal,
////            XRHandJointID.MiddleIntermediate,
////            XRHandJointID.MiddleDistal,
////            XRHandJointID.MiddleTip,
////            XRHandJointID.RingMetacarpal,
////            XRHandJointID.RingProximal,
////            XRHandJointID.RingIntermediate,
////            XRHandJointID.RingDistal,
////            XRHandJointID.RingTip,
////            XRHandJointID.LittleMetacarpal,  // Updated to "Little" from "Pinky"
////            XRHandJointID.LittleProximal,
////            XRHandJointID.LittleIntermediate,
////            XRHandJointID.LittleDistal,
////            XRHandJointID.LittleTip  // Updated to "Little" from "Pinky"
////        };

////        // List of joints that we can track and log (filter out unavailable joints)
////        List<XRHandJointID> availableJointIds = new List<XRHandJointID>();

////        // Iterate over the valid joint IDs to find which ones are available in the hand
////        foreach (var jointId in validJointIds)
////        {
////            var joint = hand.GetJoint(jointId);
////            if (joint != null && joint.TryGetPose(out _)) // Ensure the joint is valid and has pose data
////            {
////                availableJointIds.Add(jointId);
////            }
////        }

////        // Log the positions of each available joint
////        foreach (var jointId in availableJointIds)
////        {
////            var joint = hand.GetJoint(jointId);
////            if (joint.TryGetPose(out Pose jointPose))
////            {
////                string logLine = $"{DateTime.Now:HH:mm:ss.fff},{label},{jointId},{jointPose.position.x:F4},{jointPose.position.y:F4},{jointPose.position.z:F4}";
////                try
////                {
////                    File.AppendAllText(_filePath, logLine + "\n");
////                }
////                catch (Exception e)
////                {
////                    Debug.LogError($"HandLandmarkLogger: Failed to append to log file: {e.Message}");
////                }
////            }
////        }
////    }

////    private string HandLabelFromHandedness(Handedness handedness)
////    {
////        switch (handedness)
////        {
////            case Handedness.Left: return "Left";
////            case Handedness.Right: return "Right";
////            default: return "Unknown";
////        }
////    }

////    void OnDestroy()
////    {
////        if (_handSubsystem != null)
////        {
////            _handSubsystem.updatedHands -= OnHandsUpdated;
////            _handSubsystem.trackingAcquired -= OnHandTrackingAcquired;
////            _handSubsystem.trackingLost -= OnHandTrackingLost;
////            Debug.Log("HandLandmarkLogger: Unregistered hand subsystem callbacks.");
////        }
////    }
////}


////getting names of JointIds
////using UnityEngine;
////using UnityEngine.XR.Hands;
////using System;

////public class JointIDVerifier : MonoBehaviour
////{
////    void Start()
////    {
////        // Log all XRHandJointID enum values
////        foreach (XRHandJointID jointId in Enum.GetValues(typeof(XRHandJointID)))
////        {
////            Debug.Log($"Joint ID: {jointId}");
////        }
////    }
////}

////using System;
////using System.IO;
////using UnityEngine;
////using UnityEngine.XR.Hands;
////using UnityEngine.XR;

////public class HandDepthLogger : MonoBehaviour
////{
////    private XRHandSubsystem _handSubsystem;
////    private string _filePath;

////    void Start()
////    {
////        // Verify MainCamera tag
////        if (Camera.main == null)
////        {
////            Debug.LogError("HandDepthLogger: No MainCamera found. Please tag your XR camera as \"MainCamera\".");
////        }
////        else
////        {
////            Debug.Log($"HandDepthLogger: Found MainCamera at {Camera.main.transform.position}");
////        }

////        // Setup file path
////        _filePath = Path.Combine(Application.persistentDataPath, "HandDepthData.txt");
////        try
////        {
////            File.WriteAllText(_filePath, "Hand Depth Data Log\n");
////            Debug.Log($"HandDepthLogger: Logging to {_filePath}");
////        }
////        catch (Exception e)
////        {
////            Debug.LogError($"HandDepthLogger: Unable to write initial file: {e.Message}");
////        }

////        // Start Coroutine to initialize XRHandSubsystem
////        StartCoroutine(InitializeHandSubsystem());
////    }

////    private System.Collections.IEnumerator InitializeHandSubsystem()
////    {
////        // Wait a few frames for XR to initialize
////        yield return new WaitForSeconds(1.0f);

////        var subsystems = new System.Collections.Generic.List<XRHandSubsystem>();
////        SubsystemManager.GetSubsystems(subsystems);

////        if (subsystems.Count > 0)
////        {
////            _handSubsystem = subsystems[0];
////            _handSubsystem.updatedHands += OnHandsUpdated;
////            _handSubsystem.trackingAcquired += OnHandTrackingAcquired;
////            _handSubsystem.trackingLost += OnHandTrackingLost;
////            Debug.Log("HandDepthLogger: Hand subsystem found and callbacks registered.");
////        }
////        else
////        {
////            Debug.LogError("HandDepthLogger: No XRHandSubsystem found. Hand tracking won’t work.");
////        }
////    }

////    private void OnHandTrackingAcquired(XRHand hand)
////    {
////        string label = HandLabelFromHandedness(hand.handedness);
////        Debug.Log($"HandDepthLogger: Hand tracking acquired for {label} hand.");
////    }

////    private void OnHandTrackingLost(XRHand hand)
////    {
////        string label = HandLabelFromHandedness(hand.handedness);
////        Debug.Log($"HandDepthLogger: Hand tracking lost for {label} hand.");
////    }

////    private void OnHandsUpdated(XRHandSubsystem subsystem, XRHandSubsystem.UpdateSuccessFlags successFlags, XRHandSubsystem.UpdateType updateType)
////    {
////        if (_handSubsystem == null) return;

////        if (successFlags.HasFlag(XRHandSubsystem.UpdateSuccessFlags.LeftHandJoints))
////        {
////            // Check if the left hand joints are available
////            if (_handSubsystem.leftHand != null)
////            {
////                Debug.Log("HandDepthLogger: Left hand joints are available.");
////                LogHandJoints(_handSubsystem.leftHand, "Left");
////            }
////            else
////            {
////                Debug.LogWarning("HandDepthLogger: Left hand data is not available.");
////            }
////        }

////        if (successFlags.HasFlag(XRHandSubsystem.UpdateSuccessFlags.RightHandJoints))
////        {
////            // Check if the right hand joints are available
////            if (_handSubsystem.rightHand != null)
////            {
////                Debug.Log("HandDepthLogger: Right hand joints are available.");
////                LogHandJoints(_handSubsystem.rightHand, "Right");
////            }
////            else
////            {
////                Debug.LogWarning("HandDepthLogger: Right hand data is not available.");
////            }
////        }
////    }

////    private void LogHandJoints(XRHand hand, string label)
////    {
////        if (Camera.main == null) return;

////        // Get all valid joint IDs (26 joints per hand, from the XRHandJointID enum)
////        var jointIds = Enum.GetValues(typeof(XRHandJointID)) as XRHandJointID[];

////        foreach (var jointId in jointIds)
////        {
////            // Ensure the jointId is valid before accessing it
////            var joint = hand.GetJoint(jointId);

////            if (joint.TryGetPose(out Pose jointPose))
////            {
////                // Log joint depth and position
////                float depth = Vector3.Distance(Camera.main.transform.position, jointPose.position);
////                string logLine = $"{DateTime.Now:HH:mm:ss.fff},{label},{jointId},{jointPose.position.x:F4},{jointPose.position.y:F4},{jointPose.position.z:F4},{depth:F4}";

////                try
////                {
////                    File.AppendAllText(_filePath, logLine + "\n");
////                    Debug.Log($"Logged joint {jointId} for {label} hand: {logLine}");
////                }
////                catch (Exception e)
////                {
////                    Debug.LogError($"HandDepthLogger: Failed to append to log file: {e.Message}");
////                }
////            }
////            else
////            {
////                // Explicitly check if joint is available and log the joint ID and label
////                Debug.LogWarning($"HandDepthLogger: Could not get pose for {label} hand joint {jointId} (joint is not available or tracked).");
////            }
////        }
////    }

////    private string HandLabelFromHandedness(Handedness handedness)
////    {
////        switch (handedness)
////        {
////            case Handedness.Left:
////                return "Left";
////            case Handedness.Right:
////                return "Right";
////            default:
////                return "Unknown";
////        }
////    }

////    void OnDestroy()
////    {
////        if (_handSubsystem != null)
////        {
////            _handSubsystem.updatedHands -= OnHandsUpdated;
////            _handSubsystem.trackingAcquired -= OnHandTrackingAcquired;
////            _handSubsystem.trackingLost -= OnHandTrackingLost;
////            Debug.Log("HandDepthLogger: Unregistered hand subsystem callbacks.");
////        }
////    }
////}


//////Code 5 - logging 26 joint positions in addition to depth data (depth  is the distance between the quest camera and palm position)
////using System;
////using System.IO;
////using UnityEngine;
////using UnityEngine.XR.Hands;
////using UnityEngine.XR;

////public class HandDepthLogger : MonoBehaviour
////{
////    private XRHandSubsystem _handSubsystem;
////    private string _filePath;

////    void Start()
////    {
////        // Verify MainCamera tag
////        if (Camera.main == null)
////        {
////            Debug.LogError("HandDepthLogger: No MainCamera found. Please tag your XR camera as \"MainCamera\".");
////        }
////        else
////        {
////            Debug.Log($"HandDepthLogger: Found MainCamera at {Camera.main.transform.position}");
////        }

////        // Setup file path
////        _filePath = Path.Combine(Application.persistentDataPath, "HandDepthData.txt");
////        try
////        {
////            File.WriteAllText(_filePath, "Hand Depth Data Log\n");
////            Debug.Log($"HandDepthLogger: Logging to {_filePath}");
////        }
////        catch (Exception e)
////        {
////            Debug.LogError($"HandDepthLogger: Unable to write initial file: {e.Message}");
////        }

////        // Start Coroutine to initialize XRHandSubsystem
////        StartCoroutine(InitializeHandSubsystem());
////    }

////    private System.Collections.IEnumerator InitializeHandSubsystem()
////    {
////        // Wait a few frames for XR to initialize
////        yield return new WaitForSeconds(1.0f);

////        var subsystems = new System.Collections.Generic.List<XRHandSubsystem>();
////        SubsystemManager.GetSubsystems(subsystems);

////        if (subsystems.Count > 0)
////        {
////            _handSubsystem = subsystems[0];
////            _handSubsystem.updatedHands += OnHandsUpdated;
////            _handSubsystem.trackingAcquired += OnHandTrackingAcquired;
////            _handSubsystem.trackingLost += OnHandTrackingLost;
////            Debug.Log("HandDepthLogger: Hand subsystem found and callbacks registered.");
////        }
////        else
////        {
////            Debug.LogError("HandDepthLogger: No XRHandSubsystem found. Hand tracking won’t work.");
////        }
////    }

////    private void OnHandTrackingAcquired(XRHand hand)
////    {
////        string label = HandLabelFromHandedness(hand.handedness);
////        Debug.Log($"HandDepthLogger: Hand tracking acquired for {label} hand.");
////    }

////    private void OnHandTrackingLost(XRHand hand)
////    {
////        string label = HandLabelFromHandedness(hand.handedness);
////        Debug.Log($"HandDepthLogger: Hand tracking lost for {label} hand.");
////    }

////    private void OnHandsUpdated(XRHandSubsystem subsystem, XRHandSubsystem.UpdateSuccessFlags successFlags, XRHandSubsystem.UpdateType updateType)
////    {
////        if (_handSubsystem == null) return;

////        if (successFlags.HasFlag(XRHandSubsystem.UpdateSuccessFlags.LeftHandJoints))
////        {
////            LogHandJoints(_handSubsystem.leftHand, "Left");
////        }

////        if (successFlags.HasFlag(XRHandSubsystem.UpdateSuccessFlags.RightHandJoints))
////        {
////            LogHandJoints(_handSubsystem.rightHand, "Right");
////        }
////    }

////    private void LogHandJoints(XRHand hand, string label)
////    {
////        if (Camera.main == null) return;

////        // Iterate through all hand joints (26 joints per hand)
////        foreach (XRHandJointID jointId in Enum.GetValues(typeof(XRHandJointID)))
////        {
////            var joint = hand.GetJoint(jointId);
////            if (joint.TryGetPose(out Pose jointPose))
////            {
////                float depth = Vector3.Distance(Camera.main.transform.position, jointPose.position);
////                string logLine = $"{DateTime.Now:HH:mm:ss.fff},{label},{jointId},{jointPose.position.x:F4},{jointPose.position.y:F4},{jointPose.position.z:F4},{depth:F4}";
////                try
////                {
////                    File.AppendAllText(_filePath, logLine + "\n");
////                }
////                catch (Exception e)
////                {
////                    Debug.LogError($"HandDepthLogger: Failed to append to log file: {e.Message}");
////                }
////            }
////            else
////            {
////                Debug.Log($"HandDepthLogger: Could not get pose for {label} hand joint {jointId}");
////            }
////        }
////    }

////    private string HandLabelFromHandedness(Handedness handedness)
////    {
////        switch (handedness)
////        {
////            case Handedness.Left:
////                return "Left";
////            case Handedness.Right:
////                return "Right";
////            default:
////                return "Unknown";
////        }
////    }

////    void OnDestroy()
////    {
////        if (_handSubsystem != null)
////        {
////            _handSubsystem.updatedHands -= OnHandsUpdated;
////            _handSubsystem.trackingAcquired -= OnHandTrackingAcquired;
////            _handSubsystem.trackingLost -= OnHandTrackingLost;
////            Debug.Log("HandDepthLogger: Unregistered hand subsystem callbacks.");
////        }
////    }
////}


//////Code 4 -  logging depth data (depth  is the distance between the quest camera and palm position)- working
////using System;
////using System.IO;
////using System.Collections;
////using System.Collections.Generic;
////using UnityEngine;
////using UnityEngine.XR.Hands;
////using UnityEngine.XR;

////public class HandDepthLogger : MonoBehaviour
////{
////    private XRHandSubsystem _handSubsystem;
////    private string _filePath;

////    void Start()
////    {
////        // Verify MainCamera tag
////        if (Camera.main == null)
////        {
////            Debug.LogError("HandDepthLogger: No MainCamera found. Please tag your XR camera as \"MainCamera\".");
////        }
////        else
////        {
////            Debug.Log($"HandDepthLogger: Found MainCamera at {Camera.main.transform.position}");
////        }

////        // Setup file path for logging
////        _filePath = Path.Combine(Application.persistentDataPath, "HandDepthData.txt");
////        try
////        {
////            File.WriteAllText(_filePath, "Hand Depth Data Log\n");
////            Debug.Log($"HandDepthLogger: Logging to {_filePath}");
////        }
////        catch (Exception e)
////        {
////            Debug.LogError($"HandDepthLogger: Unable to write initial file: {e.Message}");
////        }

////        // Begin coroutine to initialize XRHandSubsystem (delayed start)
////        StartCoroutine(InitializeHandSubsystem());
////    }

////    private IEnumerator InitializeHandSubsystem()
////    {
////        // Wait a bit for XR to initialize (OpenXR often needs 0.5–1s)
////        yield return new WaitForSeconds(1.0f);

////        var subsystems = new List<XRHandSubsystem>();
////        SubsystemManager.GetSubsystems(subsystems);

////        // Debug all detected subsystems for verification (Step 5)
////        foreach (var s in subsystems)
////        {
////            Debug.Log($"HandDepthLogger: Subsystem found: {s.GetType().FullName}");
////        }

////        if (subsystems.Count > 0)
////        {
////            _handSubsystem = subsystems[0];
////            _handSubsystem.updatedHands += OnHandsUpdated;
////            _handSubsystem.trackingAcquired += OnHandTrackingAcquired;
////            _handSubsystem.trackingLost += OnHandTrackingLost;
////            Debug.Log("HandDepthLogger: Hand subsystem found and callbacks registered.");
////        }
////        else
////        {
////            Debug.LogError("HandDepthLogger: No XRHandSubsystem found. Hand tracking won’t work.");
////        }
////    }

////    private void OnHandTrackingAcquired(XRHand hand)
////    {
////        string label = HandLabelFromHandedness(hand.handedness);
////        Debug.Log($"HandDepthLogger: Hand tracking acquired for {label} hand.");
////    }

////    private void OnHandTrackingLost(XRHand hand)
////    {
////        string label = HandLabelFromHandedness(hand.handedness);
////        Debug.Log($"HandDepthLogger: Hand tracking lost for {label} hand.");
////    }

////    private void OnHandsUpdated(XRHandSubsystem subsystem,
////        XRHandSubsystem.UpdateSuccessFlags successFlags,
////        XRHandSubsystem.UpdateType updateType)
////    {
////        if (_handSubsystem == null) return;

////        if (successFlags.HasFlag(XRHandSubsystem.UpdateSuccessFlags.LeftHandJoints))
////        {
////            LogHandDepth(_handSubsystem.leftHand, "Left");
////        }

////        if (successFlags.HasFlag(XRHandSubsystem.UpdateSuccessFlags.RightHandJoints))
////        {
////            LogHandDepth(_handSubsystem.rightHand, "Right");
////        }
////    }

////    private void LogHandDepth(XRHand hand, string label)
////    {
////        if (Camera.main == null) return;

////        var palmJoint = hand.GetJoint(XRHandJointID.Palm);
////        if (palmJoint.TryGetPose(out Pose palmPose))
////        {
////            float depth = Vector3.Distance(Camera.main.transform.position, palmPose.position);
////            string logLine = $"{DateTime.Now:HH:mm:ss.fff},{label},{depth:F4}";

////            try
////            {
////                File.AppendAllText(_filePath, logLine + "\n");
////            }
////            catch (Exception e)
////            {
////                Debug.LogError($"HandDepthLogger: Failed to append to log file: {e.Message}");
////            }
////        }
////        else
////        {
////            Debug.Log($"HandDepthLogger: Could not get pose for {label} hand palm joint.");
////        }
////    }

////    private string HandLabelFromHandedness(Handedness handedness)
////    {
////        switch (handedness)
////        {
////            case Handedness.Left: return "Left";
////            case Handedness.Right: return "Right";
////            default: return "Unknown";
////        }
////    }

////    void OnDestroy()
////    {
////        if (_handSubsystem != null)
////        {
////            _handSubsystem.updatedHands -= OnHandsUpdated;
////            _handSubsystem.trackingAcquired -= OnHandTrackingAcquired;
////            _handSubsystem.trackingLost -= OnHandTrackingLost;
////            Debug.Log("HandDepthLogger: Unregistered hand subsystem callbacks.");
////        }
////    }
////}


//////Code 3 - PC side listener (C# console app example)
////using System;
////using System.IO;
////using System.Net;
////using System.Net.Sockets;
////using System.Text;

////class UDPReceiver
////{
////    static void Main(string[] args)
////    {
////        int port = 5005;
////        UdpClient listener = new UdpClient(port);
////        IPEndPoint groupEP = new IPEndPoint(IPAddress.Any, port);

////        string filePath = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.Desktop), "HandDepthData_PC.txt");
////        File.WriteAllText(filePath, "Hand Depth Data Log (PC)\n");

////        Console.WriteLine($"Listening for depth data on UDP port {port}");
////        try
////        {
////            while (true)
////            {
////                byte[] bytes = listener.Receive(ref groupEP);
////                string message = Encoding.UTF8.GetString(bytes);
////                Console.WriteLine($"Received: {message}");
////                File.AppendAllText(filePath, message + "\n");
////            }
////        }
////        catch (Exception e)
////        {
////            Console.WriteLine(e.ToString());
////        }
////        finally
////        {
////            listener.Close();
////        }
////    }
////}


//////Code 2
////using System;
////using System.IO;
////using UnityEngine;
////using UnityEngine.XR.Hands;
////using UnityEngine.XR;

////public class HandDepthLogger : MonoBehaviour
////{
////    private XRHandSubsystem _handSubsystem;
////    private string _filePath;

////    void Start()
////    {
////        // Verify MainCamera tag
////        if (Camera.main == null)
////        {
////            Debug.LogError("HandDepthLogger: No MainCamera found. Please tag your XR camera as \"MainCamera\".");
////        }
////        else
////        {
////            Debug.Log($"HandDepthLogger: Found MainCamera at {Camera.main.transform.position}");
////        }

////        // Setup file path
////        _filePath = Path.Combine(Application.persistentDataPath, "HandDepthData.txt");
////        try
////        {
////            File.WriteAllText(_filePath, "Hand Depth Data Log\n");
////            Debug.Log($"HandDepthLogger: Logging to {_filePath}");
////        }
////        catch (Exception e)
////        {
////            Debug.LogError($"HandDepthLogger: Unable to write initial file: {e.Message}");
////        }

////        // Get XRHandSubsystem
////        var subsystems = new System.Collections.Generic.List<XRHandSubsystem>();
////        SubsystemManager.GetSubsystems(subsystems);
////        if (subsystems.Count > 0)
////        {
////            _handSubsystem = subsystems[0];
////            _handSubsystem.updatedHands += OnHandsUpdated;
////            _handSubsystem.trackingAcquired += OnHandTrackingAcquired;
////            _handSubsystem.trackingLost += OnHandTrackingLost;
////            Debug.Log("HandDepthLogger: Hand subsystem found and callbacks registered.");
////        }
////        else
////        {
////            Debug.LogError("HandDepthLogger: No XRHandSubsystem found. Hand tracking won’t work.");
////        }
////    }

////    private void OnHandTrackingAcquired(XRHand hand)
////    {
////        string label = HandLabelFromHandedness(hand.handedness);
////        Debug.Log($"HandDepthLogger: Hand tracking acquired for {label} hand.");
////    }

////    private void OnHandTrackingLost(XRHand hand)
////    {
////        string label = HandLabelFromHandedness(hand.handedness);
////        Debug.Log($"HandDepthLogger: Hand tracking lost for {label} hand.");
////    }

////    private void OnHandsUpdated(XRHandSubsystem subsystem, XRHandSubsystem.UpdateSuccessFlags successFlags, XRHandSubsystem.UpdateType updateType)
////    {
////        if (_handSubsystem == null) return;

////        if (successFlags.HasFlag(XRHandSubsystem.UpdateSuccessFlags.LeftHandJoints))
////        {
////            LogHandDepth(_handSubsystem.leftHand, "Left");
////        }

////        if (successFlags.HasFlag(XRHandSubsystem.UpdateSuccessFlags.RightHandJoints))
////        {
////            LogHandDepth(_handSubsystem.rightHand, "Right");
////        }
////    }

////    private void LogHandDepth(XRHand hand, string label)
////    {
////        if (Camera.main == null) return;

////        // Get the palm joint pose
////        var palmJoint = hand.GetJoint(XRHandJointID.Palm);
////        if (palmJoint.TryGetPose(out Pose palmPose))
////        {
////            float depth = Vector3.Distance(Camera.main.transform.position, palmPose.position);
////            string logLine = $"{DateTime.Now:HH:mm:ss.fff},{label},{depth:F4}";
////            try
////            {
////                File.AppendAllText(_filePath, logLine + "\n");
////            }
////            catch (Exception e)
////            {
////                Debug.LogError($"HandDepthLogger: Failed to append to log file: {e.Message}");
////            }
////        }
////        else
////        {
////            Debug.Log($"HandDepthLogger: Could not get pose for {label} hand palm joint.");
////        }
////    }

////    private string HandLabelFromHandedness(Handedness handedness)
////    {
////        switch (handedness)
////        {
////            case Handedness.Left:
////                return "Left";
////            case Handedness.Right:
////                return "Right";
////            default:
////                return "Unknown";
////        }
////    }

////    void OnDestroy()
////    {
////        if (_handSubsystem != null)
////        {
////            _handSubsystem.updatedHands -= OnHandsUpdated;
////            _handSubsystem.trackingAcquired -= OnHandTrackingAcquired;
////            _handSubsystem.trackingLost -= OnHandTrackingLost;
////            Debug.Log("HandDepthLogger: Unregistered hand subsystem callbacks.");
////        }
////    }
////}



////Code 1
////using Oculus.Interaction;
////using System.IO;
////using UnityEngine;
////using UnityEngine.XR;
////using UnityEngine.XR.Hands;

////public class HandDepthLogger : MonoBehaviour
////{
////    private XRHandSubsystem handSubsystem;
////    private string filePath;

////    void Start()
////    {
////        // Create file path in persistent data path (works on Quest 3)
////        filePath = Path.Combine(Application.persistentDataPath, "HandDepthData.txt");
////        File.WriteAllText(filePath, "Hand Depth Data Log\n");

////        // Get XRHandSubsystem for hand tracking
////        var subsystems = new System.Collections.Generic.List<XRHandSubsystem>();
////        SubsystemManager.GetSubsystems(subsystems);
////        if (subsystems.Count > 0)
////        {
////            handSubsystem = subsystems[0];
////            handSubsystem.updatedHands += OnHandsUpdated;
////            Debug.Log("Hand subsystem found and tracking started.");
////        }
////        else
////        {
////            Debug.LogError("No XRHandSubsystem found.");
////        }
////    }

////    private void OnHandsUpdated(XRHandSubsystem subsystem, XRHandSubsystem.UpdateSuccessFlags successFlags, XRHandSubsystem.UpdateType updateType)
////    {
////        // Log left hand depth data if available
////        if (successFlags.HasFlag(XRHandSubsystem.UpdateSuccessFlags.LeftHandJoints))
////        {
////            LogHandDepth(subsystem.leftHand, "Left");
////        }

////        // Log right hand depth data if available
////        if (successFlags.HasFlag(XRHandSubsystem.UpdateSuccessFlags.RightHandJoints))
////        {
////            LogHandDepth(subsystem.rightHand, "Right");
////        }
////    }

////    private void LogHandDepth(XRHand hand, string label)
////    {
////        // For simplicity, measure depth as distance from headset to palm joint
////        XRHandJoint palmJoint = hand.GetJoint(XRHandJointID.Palm);

////        if (palmJoint.TryGetPose(out Pose palmPose))
////        {
////            float depth = Vector3.Distance(Camera.main.transform.position, palmPose.position);
////            string logLine = $"{System.DateTime.Now:HH:mm:ss.fff},{label},{depth:F4}";
////            File.AppendAllText(filePath, logLine + "\n");
////        }
////    }

////    void OnDestroy()
////    {
////        if (handSubsystem != null)
////        {
////            handSubsystem.updatedHands -= OnHandsUpdated;
////        }
////    }
////}

//// --- Sanskar Goyal ----

//using System;
//using System.IO;
//using System.Text;
//using System.Collections;
//using System.Collections.Generic;
//using UnityEngine;
//using UnityEngine.XR;
//using UnityEngine.XR.Hands;

//public class HandLandmarkLogger : MonoBehaviour
//{
//    private XRHandSubsystem handSubsystem;
//    private Camera mainCamera;

//    private ulong frameIndex = 0;
//    private StringBuilder logBuffer;
//    private string filePath;

//    // ---- 26 JOINTS (STATIC, NO ALLOCATIONS) ----
//    private static readonly XRHandJointID[] JointIds =
//    {
//        XRHandJointID.Wrist,

//        XRHandJointID.ThumbMetacarpal,
//        XRHandJointID.ThumbProximal,
//        XRHandJointID.ThumbDistal,
//        XRHandJointID.ThumbTip,

//        XRHandJointID.IndexMetacarpal,
//        XRHandJointID.IndexProximal,
//        XRHandJointID.IndexIntermediate,
//        XRHandJointID.IndexDistal,
//        XRHandJointID.IndexTip,

//        XRHandJointID.MiddleMetacarpal,
//        XRHandJointID.MiddleProximal,
//        XRHandJointID.MiddleIntermediate,
//        XRHandJointID.MiddleDistal,
//        XRHandJointID.MiddleTip,

//        XRHandJointID.RingMetacarpal,
//        XRHandJointID.RingProximal,
//        XRHandJointID.RingIntermediate,
//        XRHandJointID.RingDistal,
//        XRHandJointID.RingTip,

//        XRHandJointID.LittleMetacarpal,
//        XRHandJointID.LittleProximal,
//        XRHandJointID.LittleIntermediate,
//        XRHandJointID.LittleDistal,
//        XRHandJointID.LittleTip
//    };

//    // ---------------- UNITY LIFECYCLE ----------------

//    void Start()
//    {
//        mainCamera = Camera.main;
//        if (mainCamera == null)
//        {
//            Debug.LogError("HandLandmarkLogger: MainCamera not found. Ensure XR Origin camera is tagged MainCamera.");
//            enabled = false;
//            return;
//        }

//        logBuffer = new StringBuilder(1024 * 64);

//        filePath = Path.Combine(Application.persistentDataPath, "HandLandmarks_HeadLocal.csv");

//        // CSV HEADER
//        logBuffer.AppendLine(
//            "frameIndex,timestampTicks,hand,joint,x,y,z"
//        );

//        StartCoroutine(InitializeHandSubsystem());
//    }

//    private IEnumerator InitializeHandSubsystem()
//    {
//        // OpenXR needs time to boot
//        yield return new WaitForSeconds(1.0f);

//        List<XRHandSubsystem> subsystems = new List<XRHandSubsystem>();
//        SubsystemManager.GetSubsystems(subsystems);

//        if (subsystems.Count == 0)
//        {
//            Debug.LogError("HandLandmarkLogger: No XRHandSubsystem found.");
//            enabled = false;
//            yield break;
//        }

//        handSubsystem = subsystems[0];
//        handSubsystem.updatedHands += OnHandsUpdated;

//        Debug.Log("HandLandmarkLogger: XRHandSubsystem initialized.");
//    }

//    // ---------------- HAND UPDATE CALLBACK ----------------

//    private void OnHandsUpdated(
//        XRHandSubsystem subsystem,
//        XRHandSubsystem.UpdateSuccessFlags successFlags,
//        XRHandSubsystem.UpdateType updateType)
//    {
//        frameIndex++;
//        long timestampTicks = System.Diagnostics.Stopwatch.GetTimestamp();

//        if (successFlags.HasFlag(XRHandSubsystem.UpdateSuccessFlags.LeftHandJoints))
//        {
//            LogHand(subsystem.leftHand, "Left", timestampTicks);
//        }

//        if (successFlags.HasFlag(XRHandSubsystem.UpdateSuccessFlags.RightHandJoints))
//        {
//            LogHand(subsystem.rightHand, "Right", timestampTicks);
//        }
//    }

//    private void LogHand(XRHand hand, string handLabel, long timestampTicks)
//    {
//        Transform camTransform = mainCamera.transform;

//        for (int i = 0; i < JointIds.Length; i++)
//        {
//            XRHandJoint joint = hand.GetJoint(JointIds[i]);

//            if (!joint.TryGetPose(out Pose jointPose))
//                continue;

//            // ---- HEAD-LOCAL SPACE (CRITICAL DECISION) ----
//            Vector3 headLocalPos =
//                camTransform.InverseTransformPoint(jointPose.position);

//            logBuffer.Append(frameIndex).Append(',');
//            logBuffer.Append(timestampTicks).Append(',');
//            logBuffer.Append(handLabel).Append(',');
//            logBuffer.Append(JointIds[i]).Append(',');
//            logBuffer.Append(headLocalPos.x.ToString("F6")).Append(',');
//            logBuffer.Append(headLocalPos.y.ToString("F6")).Append(',');
//            logBuffer.Append(headLocalPos.z.ToString("F6")).Append('\n');
//        }
//    }

//    // ---------------- CLEANUP & FLUSH ----------------

//    void OnDestroy()
//    {
//        if (handSubsystem != null)
//        {
//            handSubsystem.updatedHands -= OnHandsUpdated;
//        }

//        try
//        {
//            File.WriteAllText(filePath, logBuffer.ToString());
//            Debug.Log($"HandLandmarkLogger: Data written to {filePath}");
//        }
//        catch (Exception e)
//        {
//            Debug.LogError($"HandLandmarkLogger: Failed to write file: {e.Message}");
//        }
//    }
//}

