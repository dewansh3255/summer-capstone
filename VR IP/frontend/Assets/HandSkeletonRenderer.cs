using System.Collections.Generic;
using UnityEngine;
using UnityEngine.XR.Hands;

public class HandSkeletonRenderer : MonoBehaviour
{
    public XRHandSubsystem handSubsystem;
    public bool isLeftHand;
    public GameObject jointPrefab;

    private Dictionary<XRHandJointID, Transform> jointMap =
        new Dictionary<XRHandJointID, Transform>();

    void Start()
    {
        // Create 26 joint visuals
        foreach (var jointId in HandJointOrder.Order)
        {
            GameObject joint = Instantiate(jointPrefab, transform);
            joint.name = jointId.ToString();
            jointMap[jointId] = joint.transform;
        }
    }

    void Update()
    {
        if (handSubsystem == null) return;

        XRHand hand = isLeftHand
            ? handSubsystem.leftHand
            : handSubsystem.rightHand;

        if (!hand.isTracked)
        {
            SetVisible(false);
            return;
        }

        SetVisible(true);

        foreach (var jointId in HandJointOrder.Order)
        {
            XRHandJoint joint = hand.GetJoint(jointId);

            if (!joint.TryGetPose(out Pose pose))
                continue;

            jointMap[jointId].SetPositionAndRotation(
                pose.position,
                pose.rotation
            );
        }
    }

    void SetVisible(bool visible)
    {
        foreach (var t in jointMap.Values)
            t.gameObject.SetActive(visible);
    }
}
