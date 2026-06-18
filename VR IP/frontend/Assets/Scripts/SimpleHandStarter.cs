using System.Collections.Generic;
using UnityEngine;
using UnityEngine.XR.Hands;

public class SimpleHandStarter : MonoBehaviour
{
    void Start()
    {
        List<XRHandSubsystem> subsystems = new List<XRHandSubsystem>();
        SubsystemManager.GetSubsystems(subsystems);

        if (subsystems.Count > 0)
        {
            var sub = subsystems[0];
            if (!sub.running) sub.Start();
            Debug.Log("STARTER: Hand Subsystem Started.");
        }
        else
        {
            Debug.LogError("STARTER: No Hand Subsystem found! Check XR Settings.");
        }
    }
}