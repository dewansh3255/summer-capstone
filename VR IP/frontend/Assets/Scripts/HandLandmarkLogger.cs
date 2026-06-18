using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using UnityEngine.XR.Hands;

public class HandLandmarkLogger : MonoBehaviour
{
    private XRHandSubsystem _handSubsystem;
    private FrameBuffer _buffer;
    private Camera _mainCamera;

    // Your original joint list
    private readonly List<XRHandJointID> _validJointIds = new List<XRHandJointID>
    {
        XRHandJointID.Wrist,
        XRHandJointID.ThumbMetacarpal, XRHandJointID.ThumbProximal, XRHandJointID.ThumbDistal, XRHandJointID.ThumbTip,
        XRHandJointID.IndexMetacarpal, XRHandJointID.IndexProximal, XRHandJointID.IndexIntermediate, XRHandJointID.IndexDistal, XRHandJointID.IndexTip,
        XRHandJointID.MiddleMetacarpal, XRHandJointID.MiddleProximal, XRHandJointID.MiddleIntermediate, XRHandJointID.MiddleDistal, XRHandJointID.MiddleTip,
        XRHandJointID.RingMetacarpal, XRHandJointID.RingProximal, XRHandJointID.RingIntermediate, XRHandJointID.RingDistal, XRHandJointID.RingTip,
        XRHandJointID.LittleMetacarpal, XRHandJointID.LittleProximal, XRHandJointID.LittleIntermediate, XRHandJointID.LittleDistal, XRHandJointID.LittleTip
    };

    void Start()
    {
        _mainCamera = Camera.main;

        // --- FIX: Use FindFirstObjectByType instead of FindObjectOfType ---
        _buffer = FindFirstObjectByType<FrameBuffer>();
        // -----------------------------------------------------------------

        if (_mainCamera == null) Debug.LogError("HandLandmarkLogger: MainCamera not found!");
        if (_buffer == null) Debug.LogError("HandLandmarkLogger: FrameBuffer not found! Is WebSocketSender in the scene?");

        StartCoroutine(InitializeHandSubsystem());
    }

    private IEnumerator InitializeHandSubsystem()
    {
        yield return new WaitForSeconds(1.0f);

        var subsystems = new List<XRHandSubsystem>();
        SubsystemManager.GetSubsystems(subsystems);

        if (subsystems.Count > 0)
        {
            _handSubsystem = subsystems[0];
            _handSubsystem.updatedHands += OnHandsUpdated;
            Debug.Log("HandLandmarkLogger: Hand subsystem found and callbacks registered.");
        }
        else
        {
            Debug.LogError("HandLandmarkLogger: No XRHandSubsystem found.");
        }
    }

    private void OnHandsUpdated(XRHandSubsystem subsystem, XRHandSubsystem.UpdateSuccessFlags successFlags, XRHandSubsystem.UpdateType updateType)
    {
        if (_buffer == null) return;

        using (MemoryStream ms = new MemoryStream())
        using (BinaryWriter bw = new BinaryWriter(ms))
        {
            Transform camT = _mainCamera.transform;

            WriteHandData(bw, subsystem.leftHand, camT);
            WriteHandData(bw, subsystem.rightHand, camT);

            _buffer.LatestHands = ms.ToArray();
        }
    }

    private void WriteHandData(BinaryWriter bw, XRHand hand, Transform camT)
    {
        foreach (var jointId in _validJointIds)
        {
            var joint = hand.GetJoint(jointId);

            if (joint.TryGetPose(out Pose pose))
            {
                bw.Write(true); // Valid

                // 1. Local Position (XYZ)
                Vector3 localPos = camT.InverseTransformPoint(pose.position);
                bw.Write(localPos.x);
                bw.Write(localPos.y);
                bw.Write(localPos.z);

                // 2. Depth Distance (Your requested feature)
                float depth = Vector3.Distance(camT.position, pose.position);
                bw.Write(depth);
            }
            else
            {
                // Invalid Joint - Write Zeros
                bw.Write(false);
                bw.Write(0f); bw.Write(0f); bw.Write(0f); // XYZ
                bw.Write(0f); // Depth
            }
        }
    }

    void OnDestroy()
    {
        if (_handSubsystem != null) _handSubsystem.updatedHands -= OnHandsUpdated;
    }
}