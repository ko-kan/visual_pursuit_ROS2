using UnityEngine;

/// <summary>
/// DroneCamera の GameObject にアタッチすると、MainCamera 側から
/// ドローンカメラの位置・姿勢が分かる 3D マーカーを自動生成するスクリプト。
///
/// 生成されるマーカー（ローカル座標）
/// ─────────────────────────────────
///   [本体] 濃いグレーの立方体
///   [レンズ] シアンの円柱 → カメラの Z 軸（前方）方向に突き出す
///   [上端マーク] 小さな赤い立方体 → カメラの Y 軸（上）方向に突き出す
///
///        ^ Y(上) ← 赤キューブ
///        |
///   [本体] ─── [レンズ] → Z(前方)
///
/// セットアップ
/// ─────────────
/// 1. DroneCamera の GameObject にこのスクリプトをアタッチ。
/// 2. （推奨）Project Settings > Tags and Layers で
///    ユーザーレイヤーに "DroneMarker" を追加する。
///    → DroneCamera 自身がマーカーを映さなくなる。
///    ※ レイヤーがなくてもマーカーは表示される（自己映り込みあり）。
/// </summary>
[RequireComponent(typeof(Camera))]
public class DroneCameraMarker : MonoBehaviour
{
    // ---- Inspector --------------------------------------------------------

    [Header("マーカーサイズ")]
    [Tooltip("カメラ本体（立方体）の一辺")]
    public float bodySize   = 0.12f;
    [Tooltip("レンズ（円柱）の半径")]
    public float lensRadius = 0.025f;
    [Tooltip("レンズの長さ")]
    public float lensLength = 0.07f;

    [Header("マーカーの色")]
    public Color bodyColor  = new Color(0.15f, 0.15f, 0.15f);
    public Color lensColor  = Color.cyan;
    public Color topColor   = Color.red;

    // ---- Private ----------------------------------------------------------

    private Camera     _cam;
    private GameObject _root;       // マーカー全体の親

    // =======================================================================
    void Start()
    {
        _cam = GetComponent<Camera>();

        // ---- マーカー生成 --------------------------------------------------
        _root      = new GameObject("[DroneCameraMarker]");
        _root.transform.SetParent(transform, false);

        // 本体（立方体）
        GameObject body = CreatePrimitive(PrimitiveType.Cube, "Body");
        body.transform.SetParent(_root.transform, false);
        body.transform.localScale    = Vector3.one * bodySize;
        body.transform.localPosition = Vector3.zero;
        ApplyColor(body, bodyColor);

        // レンズ（円柱を Z 軸方向へ）
        GameObject lens = CreatePrimitive(PrimitiveType.Cylinder, "Lens");
        lens.transform.SetParent(_root.transform, false);
        lens.transform.localRotation = Quaternion.Euler(90f, 0f, 0f); // Y→Z
        lens.transform.localPosition = new Vector3(0f, 0f, bodySize * 0.5f + lensLength * 0.5f);
        lens.transform.localScale    = new Vector3(lensRadius * 2f, lensLength * 0.5f, lensRadius * 2f);
        ApplyColor(lens, lensColor);

        // 上端マーク（小さな赤い立方体を Y 軸上方向へ）
        GameObject top = CreatePrimitive(PrimitiveType.Cube, "TopMark");
        top.transform.SetParent(_root.transform, false);
        float topSize = bodySize * 0.25f;
        top.transform.localScale    = Vector3.one * topSize;
        top.transform.localPosition = new Vector3(0f, bodySize * 0.5f + topSize * 0.5f, 0f);
        ApplyColor(top, topColor);

        // ---- レイヤー設定 --------------------------------------------------
        // "DroneMarker" レイヤーがあれば:
        //   - マーカー全体をそのレイヤーへ移動
        //   - このカメラの cullingMask からそのレイヤーを除外（自己映り込みを防止）
        int layer = LayerMask.NameToLayer("DroneMarker");
        if (layer >= 0)
        {
            SetLayerRecursive(_root, layer);
            _cam.cullingMask &= ~(1 << layer);
            Debug.Log("[DroneCameraMarker] 'DroneMarker' レイヤーを使用。DroneCamera 自身はマーカーを映しません。");
        }
        else
        {
            Debug.LogWarning(
                "[DroneCameraMarker] 'DroneMarker' レイヤーが未登録です。\n" +
                "Project Settings > Tags and Layers にユーザーレイヤー 'DroneMarker' を追加すると、" +
                "DroneCamera 自身がマーカーを映し込まなくなります。");
        }
    }

    // =======================================================================
    void OnDestroy()
    {
        if (_root != null) Destroy(_root);
    }

    // =======================================================================
    // ヘルパー

    /// <summary>コライダーなしの Primitive を作る</summary>
    static GameObject CreatePrimitive(PrimitiveType type, string goName)
    {
        GameObject go = GameObject.CreatePrimitive(type);
        go.name = goName;

        // 物理演算・コライダーは不要
        Collider col = go.GetComponent<Collider>();
        if (col != null) Destroy(col);

        return go;
    }

    /// <summary>URP / Built-in 両対応のマテリアルで単色塗り</summary>
    static void ApplyColor(GameObject go, Color color)
    {
        Renderer r = go.GetComponent<Renderer>();
        if (r == null) return;

        // URP を優先し、見つからなければ Built-in にフォールバック
        Shader shader =
            Shader.Find("Universal Render Pipeline/Lit") ??
            Shader.Find("Universal Render Pipeline/Simple Lit") ??
            Shader.Find("Standard");

        if (shader == null)
        {
            Debug.LogError("[DroneCameraMarker] 使用可能なシェーダーが見つかりません。");
            return;
        }

        Material mat = new Material(shader);

        // URP/Lit は "_BaseColor"、Built-in は "_Color" プロパティを使う
        if (mat.HasProperty("_BaseColor"))
            mat.SetColor("_BaseColor", color);
        else
            mat.color = color;

        r.material = mat;
    }

    /// <summary>オブジェクトとその全子孫のレイヤーを設定</summary>
    static void SetLayerRecursive(GameObject go, int layer)
    {
        go.layer = layer;
        foreach (Transform child in go.transform)
            SetLayerRecursive(child.gameObject, layer);
    }
}
