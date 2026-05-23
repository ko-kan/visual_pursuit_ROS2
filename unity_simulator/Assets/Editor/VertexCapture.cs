using UnityEngine;
using UnityEditor;
using System.IO;
using System.Collections.Generic;

public class VertexCapture
{
    private const int ImageSize = 256;
    private const float CameraDistance = 1.5f;
    private const float CameraFov = 40f;
    private const int AzimuthSteps = 8;
    private static readonly float[] ElevationAngles = { -30f, 0f, 30f };

    [MenuItem("Tools/Capture Vertex Images")]
    static void CaptureAll()
    {
        if (!EditorApplication.isPlaying)
        {
            EditorUtility.DisplayDialog("Error", "Play mode で実行してください。", "OK");
            return;
        }

        // Application.dataPath = .../unity_simulator/Assets
        // ../../vertex_db = .../visual_pursuit_ROS2/vertex_db
        string dbRoot = Path.GetFullPath(Path.Combine(Application.dataPath, "../../vertex_db"));

        GameObject camObj = new GameObject("_CaptureCam");
        Camera cam = camObj.AddComponent<Camera>();
        cam.clearFlags = CameraClearFlags.SolidColor;
        cam.backgroundColor = new Color(0.5f, 0.5f, 0.5f);
        cam.fieldOfView = CameraFov;

        RenderTexture rt = new RenderTexture(ImageSize, ImageSize, 24);
        cam.targetTexture = rt;

        int totalSaved = 0;

        for (int vi = 0; vi < 4; vi++)
        {
            GameObject ball = GameObject.Find($"Vertex_{vi}");
            if (ball == null)
            {
                Debug.LogWarning($"Vertex_{vi} が見つかりません。TetraMaker が Start() 済みか確認してください。");
                continue;
            }

            string vertexDir = Path.Combine(dbRoot, $"vertex_{vi}");
            Directory.CreateDirectory(vertexDir);

            // 対象ボール以外の兄弟（他球 + エッジ）を非表示
            var hidden = new List<GameObject>();
            Transform parent = ball.transform.parent;
            if (parent != null)
            {
                foreach (Transform sibling in parent)
                {
                    if (sibling.gameObject != ball)
                    {
                        sibling.gameObject.SetActive(false);
                        hidden.Add(sibling.gameObject);
                    }
                }
            }

            Vector3 center = ball.transform.position;
            int imgIndex = 0;

            foreach (float elev in ElevationAngles)
            {
                for (int ai = 0; ai < AzimuthSteps; ai++)
                {
                    float elevRad = elev * Mathf.Deg2Rad;
                    float azimRad = (ai * 360f / AzimuthSteps) * Mathf.Deg2Rad;

                    Vector3 offset = new Vector3(
                        Mathf.Cos(elevRad) * Mathf.Sin(azimRad),
                        Mathf.Sin(elevRad),
                        Mathf.Cos(elevRad) * Mathf.Cos(azimRad)
                    ) * CameraDistance;

                    camObj.transform.position = center + offset;
                    camObj.transform.LookAt(center);
                    cam.Render();

                    RenderTexture.active = rt;
                    Texture2D tex = new Texture2D(ImageSize, ImageSize, TextureFormat.RGB24, false);
                    tex.ReadPixels(new Rect(0, 0, ImageSize, ImageSize), 0, 0);
                    tex.Apply();
                    RenderTexture.active = null;

                    string filePath = Path.Combine(vertexDir, $"{imgIndex:D3}.png");
                    File.WriteAllBytes(filePath, tex.EncodeToPNG());
                    Object.DestroyImmediate(tex);

                    imgIndex++;
                    totalSaved++;
                }
            }

            // 非表示にしたオブジェクトを元に戻す
            foreach (var obj in hidden)
                obj.SetActive(true);

            Debug.Log($"Vertex_{vi}: {imgIndex} 枚保存 → {vertexDir}");
        }

        cam.targetTexture = null;
        Object.DestroyImmediate(rt);
        Object.DestroyImmediate(camObj);

        EditorUtility.DisplayDialog("完了", $"{totalSaved} 枚の画像を vertex_db/ に保存しました。", "OK");
    }
}
