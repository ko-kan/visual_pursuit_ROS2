using UnityEngine;

public class TetraMaker : MonoBehaviour
{
    public float thickness = 0.05f; // 辺の太さ
    public Material[] edgeMaterials; // ここにさっき作った色を入れます

    void Start()
    {
        // 正四面体の4つの頂点の座標
        Vector3[] vertices = {
            new Vector3(1, 1, 1), new Vector3(1, -1, -1),
            new Vector3(-1, 1, -1), new Vector3(-1, -1, 1)
        };

        // どの頂点とどの頂点を繋ぐか（合計6本の辺）
        int[,] connections = {
            {0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}
        };

        for (int i = 0; i < 6; i++)
        {
            CreateEdge(vertices[connections[i, 0]], vertices[connections[i, 1]], i);
        }
    }

    void CreateEdge(Vector3 start, Vector3 end, int index)
    {
        // シリンダーを生成
        GameObject edge = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
        edge.transform.SetParent(this.transform);

        // 位置を中間地点にする
        edge.transform.position = (start + end) / 2f;
        // 向きを相手の頂点に向ける
        edge.transform.up = end - start;
        // 長さと太さを調整
        float dist = Vector3.Distance(start, end);
        edge.transform.localScale = new Vector3(thickness, dist / 2f, thickness);

        // 色を塗る
        if (index < edgeMaterials.Length)
        {
            edge.GetComponent<Renderer>().material = edgeMaterials[index];
        }
    }
}