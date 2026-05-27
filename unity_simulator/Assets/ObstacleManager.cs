using System.Collections.Generic;
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Std;

/// <summary>
/// 障害物の一元管理スクリプト。
///   ① Inspector で障害物リストを編集する（形状・サイズ・色）
///   ② Play 時にシーン内へ可視化オブジェクトを自動生成 → 両カメラに映る
///   ③ 障害物情報（中心・安全半径・形状）を ROS トピックへ配信する
///
/// ── ROS メッセージ形式 (topic: environment/obstacles) ─────────────────────
///   Float64MultiArray:
///   data = [N, cx0,cy0,cz0, r0, type0,  cx1,cy1,cz1, r1, type1, ...]
///   N    = 障害物数
///   type : 0.0 = 球体 (sphere)  /  1.0 = 垂直円柱 (cylinder, 高さ無限大近似)
///   (cx,cy,cz) = 中心 [ROS world frame]    r = safetyRadius
///
///   座標変換: Unity world → ROS world = (x_unity, -z_unity, y_unity)
///
/// ── 円柱の CBF ──────────────────────────────────────────────────────────────
///   type=cylinder のとき ROS 側は XY 水平面の 2-D 距離を使って
///   l_c,k / l_o,k を計算する（高さ方向は無視 = 無限長柱近似）。
///
/// ── Setup ───────────────────────────────────────────────────────────────────
///   1. 任意の空 GameObject（例: ObstacleManager）にこのスクリプトをアタッチ。
///   2. Inspector の Obstacles リストに障害物を追加・設定する。
/// </summary>
public class ObstacleManager : MonoBehaviour
{
    // ── 形状列挙 ────────────────────────────────────────────────────────────
    public enum ObstacleShape { Cylinder, Sphere, Cone }

    // ── 障害物エントリー ────────────────────────────────────────────────────

    [System.Serializable]
    public class ObstacleEntry
    {
        [Tooltip("表示名（Inspector 上の識別用）")]
        public string label = "Obstacle";

        [Tooltip("形状")]
        public ObstacleShape shape = ObstacleShape.Cylinder;

        [Tooltip("Unity ワールド座標での中心位置")]
        public Vector3 position = Vector3.zero;

        [Tooltip("CBF 安全半径 r_k [m] — ROS 側に送信される")]
        public float safetyRadius = 0.5f;

        [Tooltip("視覚的な障害物本体の半径 [m]。0 = safetyRadius と同じ")]
        public float bodyRadius = 0f;

        [Tooltip("円柱の高さ [m]（視覚のみ。CBF は高さ無視の無限長柱で計算）")]
        public float height = 2.0f;

        [Tooltip("障害物本体の色（不透明）")]
        public Color bodyColor = new Color(0.20f, 0.20f, 0.20f, 1.00f);

        [Tooltip("安全境界の色（alpha で透過度を調整）")]
        public Color boundaryColor = new Color(1.00f, 0.30f, 0.10f, 0.20f);

        [Tooltip("安全境界を表示するか")]
        public bool showBoundary = true;
    }

    // ── Inspector ──────────────────────────────────────────────────────────

    [Header("障害物リスト")]
    public List<ObstacleEntry> obstacles = new List<ObstacleEntry>();

    [Header("ROS 配信")]
    public string topicName = "environment/obstacles";
    public float  publishHz = 5f;

    // ── Private ────────────────────────────────────────────────────────────

    ROSConnection _ros;
    float         _nextPublishTime;
    readonly List<GameObject> _roots = new List<GameObject>();

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
        foreach (var root in _roots) if (root != null) Destroy(root);
        _roots.Clear();
    }

    // =======================================================================
    // 可視化オブジェクト生成

    void BuildVisuals()
    {
        foreach (var root in _roots) if (root != null) Destroy(root);
        _roots.Clear();

        foreach (var obs in obstacles)
        {
            float bRadius = (obs.bodyRadius > 0f) ? obs.bodyRadius : obs.safetyRadius;

            var root = new GameObject($"[Obstacle] {obs.label}");
            root.transform.SetParent(transform, false);
            root.transform.position = obs.position;

            if (obs.shape == ObstacleShape.Cylinder)
            {
                // ── 円柱（本体） ───────────────────────────────────────────
                var body = CreateCylinder("Body", bRadius, obs.height,
                                          obs.bodyColor, opaque: true);
                body.transform.SetParent(root.transform, false);

                // ── 円柱（安全境界） ───────────────────────────────────────
                if (obs.showBoundary && obs.safetyRadius > bRadius + 1e-4f)
                {
                    var boundary = CreateCylinder("SafetyBoundary",
                                                  obs.safetyRadius, obs.height,
                                                  obs.boundaryColor, opaque: false);
                    boundary.transform.SetParent(root.transform, false);
                }
            }
            else if (obs.shape == ObstacleShape.Cone)
            {
                // ── 円錐（本体） ─────────────────────────────────────────
                // 中心が obs.position、底面が下 (y = -height/2)、頂点が上 (y = +height/2)
                var body = CreateCone("Body", bRadius, obs.height,
                                      obs.bodyColor, opaque: true);
                body.transform.SetParent(root.transform, false);

                // ── 円錐（安全境界 = 球体近似） ───────────────────────────
                if (obs.showBoundary && obs.safetyRadius > bRadius + 1e-4f)
                {
                    var boundary = CreateSphere("SafetyBoundary",
                                                obs.safetyRadius * 2f,
                                                obs.boundaryColor, opaque: false);
                    boundary.transform.SetParent(root.transform, false);
                }
            }
            else
            {
                // ── 球体（本体） ───────────────────────────────────────────
                var body = CreateSphere("Body", bRadius * 2f,
                                        obs.bodyColor, opaque: true);
                body.transform.SetParent(root.transform, false);

                // ── 球体（安全境界） ───────────────────────────────────────
                if (obs.showBoundary && obs.safetyRadius > bRadius + 1e-4f)
                {
                    var boundary = CreateSphere("SafetyBoundary",
                                                obs.safetyRadius * 2f,
                                                obs.boundaryColor, opaque: false);
                    boundary.transform.SetParent(root.transform, false);
                }
            }

            _roots.Add(root);
        }
    }

    // =======================================================================
    // ROS 配信
    // フォーマット: [N, cx, cy, cz, r, type,  cx, cy, cz, r, type, ...]
    //   type: 0.0 = sphere, 1.0 = cylinder

    void Publish()
    {
        int n = obstacles.Count;
        double[] data = new double[1 + 5 * n];
        data[0] = n;

        for (int i = 0; i < n; i++)
        {
            Vector3 p = obstacles[i].position;

            // Unity world → ROS world: (x, -z, y)
            data[1 + 5 * i + 0] = p.x;
            data[1 + 5 * i + 1] = -p.z;
            data[1 + 5 * i + 2] = p.y;
            data[1 + 5 * i + 3] = obstacles[i].safetyRadius;
            data[1 + 5 * i + 4] = (obstacles[i].shape == ObstacleShape.Cylinder)
                                   ? 1.0 : 0.0;
        }

        _ros.Publish(topicName, new Float64MultiArrayMsg { data = data });
    }

    // =======================================================================
    // ヘルパー

    /// <summary>
    /// 垂直円柱プリミティブを生成する。
    /// Unity Cylinder: 既定でY軸（= Unity上方向 = ROS世界Z方向）に沿って立つ。
    ///   localScale.x / .z = 直径 = 2 × radius
    ///   localScale.y      = height / 2  （Unity Cylinder は既定で高さ 2）
    /// </summary>
    static GameObject CreateCylinder(string goName, float radius, float height,
                                     Color color, bool opaque)
    {
        var go = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
        go.name = goName;
        go.transform.localScale = new Vector3(radius * 2f, height * 0.5f, radius * 2f);

        var col = go.GetComponent<Collider>();
        if (col != null) Destroy(col);

        ApplyColor(go, color, opaque);
        return go;
    }

    /// <summary>
    /// 円錐メッシュを生成する。
    /// 中心は原点、底面が y = -height/2、頂点が y = +height/2。
    /// CBF は球体近似（type=0.0）で計算される。
    /// </summary>
    static GameObject CreateCone(string goName, float bottomRadius, float height,
                                  Color color, bool opaque)
    {
        var go = new GameObject(goName);
        go.AddComponent<MeshFilter>().mesh = BuildConeMesh(bottomRadius, height, 24);
        go.AddComponent<MeshRenderer>();

        var col = go.GetComponent<Collider>();
        if (col != null) Destroy(col);

        ApplyColor(go, color, opaque);

        // カスタムメッシュは法線の表裏が不定になりやすいので両面描画にする
        var r = go.GetComponent<Renderer>();
        if (r != null && r.material != null)
        {
            r.material.SetInt("_Cull",     0);   // URP: Cull Off
            r.material.SetInt("_CullMode", 0);   // 旧URP命名互換
        }
        return go;
    }

    /// <summary>
    /// 円錐メッシュを生成する。
    ///   vertices[0]              = 頂点 (0, +height/2, 0)
    ///   vertices[1..segments]    = 底面リング (y = -height/2)
    ///   vertices[segments+1]     = 底面中心 (0, -height/2, 0)
    /// </summary>
    static Mesh BuildConeMesh(float bottomRadius, float height, int segments)
    {
        var verts = new Vector3[segments + 2];
        verts[0] = new Vector3(0f, height * 0.5f, 0f);            // 頂点
        for (int i = 0; i < segments; i++)
        {
            float a = 2f * Mathf.PI * i / segments;
            verts[i + 1] = new Vector3(
                bottomRadius * Mathf.Cos(a),
                -height * 0.5f,
                bottomRadius * Mathf.Sin(a));
        }
        verts[segments + 1] = new Vector3(0f, -height * 0.5f, 0f); // 底面中心

        var tris = new int[segments * 6];
        int t = 0;
        for (int i = 0; i < segments; i++)
        {
            int curr = i + 1;
            int next = (i + 1 < segments) ? i + 2 : 1;

            // 側面（頂点 → curr → next）
            tris[t++] = 0;
            tris[t++] = curr;
            tris[t++] = next;

            // 底面（中心 → next → curr、法線が -Y 方向になる巻き順）
            tris[t++] = segments + 1;
            tris[t++] = next;
            tris[t++] = curr;
        }

        var mesh = new Mesh { name = "Cone" };
        mesh.vertices  = verts;
        mesh.triangles = tris;
        mesh.RecalculateNormals();
        mesh.RecalculateBounds();
        return mesh;
    }

    /// <summary>球体プリミティブを生成する。</summary>
    static GameObject CreateSphere(string goName, float diameter,
                                   Color color, bool opaque)
    {
        var go = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        go.name = goName;
        go.transform.localScale = Vector3.one * diameter;

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
            if (mat.HasProperty("_BaseColor")) mat.SetColor("_BaseColor", color);
            else mat.color = color;
        }
        else
        {
            color.a = Mathf.Clamp01(color.a);
            if (mat.HasProperty("_BaseColor"))
            {
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
                mat.renderQueue = (int)UnityEngine.Rendering.RenderQueue.Transparent;
            }
            else
            {
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
                mat.renderQueue = (int)UnityEngine.Rendering.RenderQueue.Transparent;
            }
        }

        r.material = mat;
    }
}
