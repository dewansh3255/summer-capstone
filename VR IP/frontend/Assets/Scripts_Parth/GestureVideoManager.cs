using UnityEngine;
using UnityEngine.Video;
using TMPro;

[System.Serializable]
public class AngleData
{
    public string angleName;
    public VideoClip video;

    [TextArea(3, 10)]
    public string instruction;
}

[System.Serializable]
public class GestureGroup
{
    public string gestureName;
    public AngleData[] angles; // ideally size 3
}

public class GestureVideoManager : MonoBehaviour
{
    public VideoPlayer videoPlayer;
    public GestureGroup[] gestureGroups;

    public TextMeshProUGUI instructionText;

    public GameObject generalInstructionsPanel;
    public GameObject videoPanel;

    private int currentGestureIndex = 0;
    private int currentAngleIndex = 0;

    void Start()
    {
        videoPanel.SetActive(false);
        generalInstructionsPanel.SetActive(true);
    }

    public void StartVideos()
    {
        generalInstructionsPanel.SetActive(false);
        videoPanel.SetActive(true);

        currentGestureIndex = 0;
        currentAngleIndex = 0;

        LoadCurrent();
    }

    public void PlayVideo()
    {
        videoPlayer.Stop();
        videoPlayer.Play();
    }

    public void NextVideo()
    {
        videoPlayer.Stop();

        currentAngleIndex++;

        if (currentAngleIndex >= gestureGroups[currentGestureIndex].angles.Length)
        {
            currentAngleIndex = 0;
            currentGestureIndex++;

            if (currentGestureIndex >= gestureGroups.Length)
                return;
        }

        LoadCurrent();
    }

    void LoadCurrent()
    {
        AngleData current = gestureGroups[currentGestureIndex].angles[currentAngleIndex];

        videoPlayer.clip = current.video;
        instructionText.text = current.instruction;
    }
}