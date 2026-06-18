using System.Collections.Generic;
using UnityEngine;
using UnityEngine.XR.Hands;

public class HandSubsystemBinder : MonoBehaviour
{
    // We don't need to reference the hands anymore, 
    // because they find the subsystem automatically.

    void Start()
    {
        Debug.Log("[BINDER] Checking for Hand Subsystem...");

        List<XRHandSubsystem> subsystems = new();
        SubsystemManager.GetSubsystems(subsystems);

        if (subsystems.Count == 0)
        {
            Debug.LogError("[BINDER] XRHandSubsystem NOT FOUND. Check OpenXR Settings.");
            return;
        }

        XRHandSubsystem subsystem = subsystems[0];

        // Just make sure it is running.
        // The Hand Mesh Controllers will detect it automatically.
        if (!subsystem.running)
        {
            subsystem.Start();
            Debug.Log("[BINDER] Subsystem was stopped, started it now.");
        }
        else
        {
            Debug.Log("[BINDER] Subsystem is already running.");
        }
    }
}