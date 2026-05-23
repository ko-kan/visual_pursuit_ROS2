using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Sensor; // sensor_msgs用

public class ImagePublisher : MonoBehaviour
{
    public Camera targetCamera;
    public string topicName = "camera/image_raw";
    private ROSConnection ros;
    private float lastPublishTime;
    public float publishFrequency = 0.1f; // 10Hz

    void Start() {
        ros = ROSConnection.GetOrCreateInstance();
        ros.RegisterPublisher<ImageMsg>(topicName);
    }

    void Update() {
        if (Time.time > lastPublishTime + publishFrequency) {
            PublishImage();
            lastPublishTime = Time.time;
        }
    }

    void PublishImage() {
        if (targetCamera == null) { Debug.LogError("[ImagePublisher] targetCamera が未設定です"); return; }
        if (targetCamera.targetTexture == null) { Debug.LogError("[ImagePublisher] カメラに RenderTexture が設定されていません"); return; }

        // RenderTextureからTexture2Dを作成
        Rect rect = new Rect(0, 0, targetCamera.targetTexture.width, targetCamera.targetTexture.height);
        Texture2D tex = new Texture2D((int)rect.width, (int)rect.height, TextureFormat.RGB24, false);
        
        RenderTexture.active = targetCamera.targetTexture;
        tex.ReadPixels(rect, 0, 0);
        tex.Apply();

        // ROSメッセージの作成
        ImageMsg imageMsg = new ImageMsg {
            header = new RosMessageTypes.Std.HeaderMsg(),
            height = (uint)tex.height,
            width = (uint)tex.width,
            encoding = "rgb8",
            step = (uint)(tex.width * 3),
            data = tex.GetRawTextureData()
        };

        ros.Publish(topicName, imageMsg);
    }
}
