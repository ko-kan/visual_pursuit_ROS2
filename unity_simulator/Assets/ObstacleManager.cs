using System.Collections.Generic;
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Std;

/// <summary>
/// 障害物の一元管理スクリプト。
///   ① Inspector で障害物リストを編集する
///   ② Play 時にシーン内へ球体（本体 + 安全境界）を自動生成 → 両カメラに映る
///   ③ 障害物情報（中心・安全半径）を ROS トピックへ配信する
///
/// ── ROS メッセージ形式 (topic: environment/obstacles) ──────────────────
///   Float64MultiArray:
///   data = [N, cx0, cy0, cz0, r0,  cx1, cy1, cz1, r1, ...]
///   N = 障害物数, (cx,cy,cz) = 球心 [ROS world frame], r = safetyRadius
///
///   座標変換: Unity world → ROS world = (x_unity, -z_unity, y_unity)
///
/// ── 見た目 ──────────────────────────────────────────────────────────────
///   [本体球]    bodyRadius の不透明球  → 視覚的な障害物
///   [安全境界]  safetyRadius の半透明球 → CBF が守る範囲を可視化
///              ※ bodyRadius ≥ safetyRadius のときは境界球は生成しない
///
/// ── Setup ───────────────────────────────────────────────────────────────
///   1. 任意の空 GameObject（例: ObstacleManager）にこのスクリプトをアタッチ。
///   2. Inspector の Obstacles リストに障害物を追加・設定する。
///   3. ROS ノード (error_control_node.py) は environment/obstacles を購読済み。
/// </summary>
public class ObstacleManager : MonoBehaviour
{
    // ── 障害物エントリー ────────────────────────────────────────────────────

    [System.Serializable]
    public class ObstacleEntry
    {
        [Tooltip("表示名（Inspector 上の識別用）")]
        public string label = "Obstacle";

        [Tooltip("Unity ワールド座標での中心位置")]
        public Vector3 position = Vector3.zero;

        [Tooltip("CBF 安全半径 r_k [m] — ROS 側に送信される")]
        public float safetyRadius = 0.5f;

        [Tooltip("視覚的な障害物本体の半径 [m]。0 = safetyRadius と同じ")]
        public float bodyRadius = 0f;

        [Tooltip("障害物本体の色（不透明）")]
        public Color bodyColor = new Color(0.20f, 0.20f, 0.20f, 1.00f);

        [Tooltip("安全境界球の色（alpha で透過度を調整）")]
        public Color boundaryColor = new Color(1.00f, 0.30f, 0.10f, 0.20f);

        [Tooltip("安全境界球を表示するか")]
        public bool showBoundary = true;
    }

    // ── Inspector ──────────────────────────────────────────────────────────

    [Header("障害物リスト")]
    public List<ObstacleEntry> obstacles = new List<ObstacleEntry>();

    [Header("ROS 配信")]
    [Tooltip("配信トピック名")]
    public string topicName  = "environment/obstacles";
    [Tooltip("配信頻度 [Hz]（障害物が静的なら 5 Hz で十分）")]
    public float  publishHz  = 5f;

    // ── Private ────────────────────────────────────────────────────────────

    ROSConnection _ros;
    float         _nextPublishTime;
    readonly List<GameObject> _roots = new List<GameObject>();  // 生成した可視化オブジェクト

    // =======================================================================
    void Start()
    {
        _ros = ROSConnection.GetOrCreateInstance();
        _ros.RegisterPublisher<Float64MultiArrayMsg>(topicName);

        BuildVisuals();

        Debug.Log($"[ObstacleManager] {obstacles.Count} 個の障害物を初期化しました。");
    }

    void Update()
    {
        if (Time.time < _nextPublishTime) return;
        _nextPublishTime = Time.time + 1f / publishHz;
        Publish();
    }

    void OnDestroy()
    {
        foreach (var root in _roots)
            if (root != null) Destroy(root);
        _roots.Clear();
    }

    // =======================================================================
    // 可視化オブジェクト生成

    void BuildVisuals()
    {
        // 既存の可視化オブジェクトをクリア（再呼び出し対応）
        foreach (var root in _roots)
            if (root != null) Destroy(root);
        _roots.Clear();

        foreach (var obs in obstacles)
        {
            float bRadius = (obs.bodyRadius > 0f) ? obs.bodyRadius : obs.safetyRadius;

            // 親オブジェクト（位置だけ持つ空オブジェクト）
            var root = new GameObject($"[Obstacle] {obs.label}");
            root.transform.SetParent(transform, false);
            root.transform.position = obs.position;

            // ── 本体球（不透明）──────────────────────────────────────────
            var body = CreateSphere("Body", bRadius * 2f, obs.bodyColor, opaque: true);
            body.transform.SetParent(root.transform, false);

            // ── 安全境界球（半透明） ─────────────────────────────────────
            if (obs.showBoundary && obs.safetyRadius > bRadius + 1e-4f)
            {
                var boundary = CreateSphere("SafetyBoundary",
                                            obs.safetyRadius * 2f,
                                            obs.boundaryColor, opaque: false);
                boundary.transform.SetParent(root.transform, false);
            }

            _roots.Add(root);
        }
    }

    // =======================================================================
    // ROS 配信

    void Publish()
    {
        int n = obstacles.Count;
        double[] data = new double[1 + 4 * n];
        data[0] = n;

        for (int i = 0; i < n; i++)
        {
            Vector3 p = obstacles[i].position;

            // Unity world → ROS world: (x, -z, y)
            data[1 + 4 * i + 0] = p.x;
            data[1 + 4 * i + 1] = -p.z;
            data[1 + 4 * i + 2] = p.y;
            data[1 + 4 * i + 3] = obstacles[i].safetyRadius;
        }

        _ros.Publish(topicName, new Float64MultiArrayMsg { data = data });
    }

    // =======================================================================
    // ヘルパー

    /// <summary>コライダーなしの球プリミティブを生成して色を適用する</summary>
    static GameObject CreateSphere(string goName, float diameter,
                                   Color color, bool opaque)
    {
        var go = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        go.name = goName;
        go.transform.localScale = Vector3.one * diameter;

        // 物理コライダーは不要
        var col = go.GetComponent<Collider>();
        if (col != null) Destroy(col);

        ApplyColor(go, color, opaque);
        return go;
    }

    /// <summary>URP / Built-in 両対応のマテリアル適用</summary>
    static void ApplyColor(GameObject go, Color color, bool opaque)
    {
        Renderer r = go.GetComponent<Renderer>();
        if (r == null) return;

        Shader shader =
            Shader.Find("Universal Render Pipeline/Lit") ??
            Shader.Find("Universal Render Pipeline/Simple Lit") ??
            Shader.Find("Standard");

        if (shader == null)
        {
            Debug.LogError("[ObstacleManager] 使用可能なシェーダーが見つかりません。");
            return;
        }

        Material mat = new Material(shader);

        if (opaque)
        {
            if (mat.HasProperty("_BaseColor"))
                mat.SetColor("_BaseColor", color);
            else
                mat.color = color;
        }
        else
        {
            color.a = Mathf.Clamp01(color.a);

            if (mat.HasProperty("_BaseColor"))
            {
                // URP Transparent
                mat.SetColor("_BaseColor", color);
                mat.SetFloat("_Surface", 1f);
                mat.SetFloat("_Blend",   0f);
                mat.SetInt("_SrcBlend",
                    (int)UnityEngine.Rendering.BlendMode.SrcAlpha);
                mat.SetInt("_DstBlend",
                    (int)UnityEngine.Rendering.BlendMode.OneMinusSrcAlpha);
                mat.SetInt("_ZWrite", 0);
                mat.EnableKeyword("_SURFACE_TYPE_TRANSPARENT");
                mat.EnableKeyword("_ALPHAPREMULTIPLY_ON");
                mat.renderQueue =
                    (int)UnityEngine.Rendering.RenderQueue.Transparent;
            }
            else
            {
                // Built-in Fade mode
                mat.color = color;
                mat.SetFloat("_Mode", 2f);
                mat.SetInt("_SrcBlend",
                    (int)UnityEngine.Rendering.BlendMode.SrcAlpha);
                mat.SetInt("_DstBlend",
                    (int)UnityEngine.Rendering.BlendMode.OneMinusSrcAlpha);
                mat.SetInt("_ZWrite", 0);
                mat.DisableKeyword("_ALPHATEST_ON");
                mat.EnableKeyword("_ALPHABLEND_ON");
                mat.DisableKeyword("_ALPHAPREMULTIPLY_ON");
                mat.renderQueue =
                    (int)UnityEngine.Rendering.RenderQueue.Transparent;
            }
        }

        r.material = mat;
    }
}
