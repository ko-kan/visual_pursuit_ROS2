using UnityEngine;

public class TetraMaker : MonoBehaviour
{
    public float edgeThickness = 0.05f;
    public float ballRadius = 0.15f;
    public Material edgeMaterial;

    void Start()
    {
        // Regular tetrahedron, base face parallel to ground (XZ plane), edge length 2.
        // R = 2/sqrt(3), H = 2*sqrt(6)/3; base at y = -H/4, apex at y = 3H/4.
        Vector3[] vertices = {
            new Vector3( 1.1547f, -0.4082f,  0.0000f),
            new Vector3(-0.5774f, -0.4082f,  1.0000f),
            new Vector3(-0.5774f, -0.4082f, -1.0000f),
            new Vector3( 0.0000f,  1.2247f,  0.0000f),
        };

        int[,] connections = {
            {0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}
        };

        for (int i = 0; i < 6; i++)
            CreateEdge(vertices[connections[i, 0]], vertices[connections[i, 1]]);

        for (int i = 0; i < 4; i++)
            CreateBall(vertices[i], i);
    }

    void CreateEdge(Vector3 start, Vector3 end)
    {
        GameObject edge = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
        edge.transform.SetParent(this.transform);
        edge.transform.position = (start + end) / 2f;
        edge.transform.up = end - start;
        float dist = Vector3.Distance(start, end);
        edge.transform.localScale = new Vector3(edgeThickness, dist / 2f, edgeThickness);

        if (edgeMaterial != null)
            edge.GetComponent<Renderer>().material = edgeMaterial;
        else
            edge.GetComponent<Renderer>().material.color = new Color(0.5f, 0.5f, 0.5f);
    }

    // Vertex colors: red, green, blue, yellow — must match vertex_colors in targets.yaml
    static readonly Color[] BallColors = {
        new Color(0.9f, 0.1f, 0.1f),   // vertex 0: red
        new Color(0.1f, 0.8f, 0.1f),   // vertex 1: green
        new Color(0.1f, 0.2f, 0.9f),   // vertex 2: blue
        new Color(0.9f, 0.8f, 0.0f),   // vertex 3: yellow
    };

    void CreateBall(Vector3 position, int vertexIndex)
    {
        GameObject ball = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        ball.name = $"Vertex_{vertexIndex}";
        ball.transform.SetParent(this.transform);
        ball.transform.position = position;
        ball.transform.localScale = Vector3.one * ballRadius * 2f;

        // Shader.Find requires the URP package; falls back gracefully if not found.
        Shader urpLit = Shader.Find("Universal Render Pipeline/Lit");
        Material mat = urpLit != null ? new Material(urpLit)
                                      : new Material(Shader.Find("Standard"));
        mat.SetColor("_BaseColor", BallColors[vertexIndex]);
        mat.color = BallColors[vertexIndex];   // also set for Built-in fallback
        ball.GetComponent<Renderer>().material = mat;
    }

    Texture2D GeneratePatternTexture(int vertexIndex, int size)
    {
        Texture2D tex = new Texture2D(size, size);
        Color[] pixels = new Color[size * size];

        for (int y = 0; y < size; y++)
        {
            for (int x = 0; x < size; x++)
            {
                float u = (float)x / size;
                float v = (float)y / size;
                float cx = u - 0.5f;
                float cy = v - 0.5f;
                float dist = Mathf.Sqrt(cx * cx + cy * cy);

                bool isWhite = false;
                switch (vertexIndex)
                {
                    case 0: // 同心円
                        isWhite = (int)(dist * 12) % 2 == 0;
                        break;
                    case 1: // 十字
                        isWhite = Mathf.Abs(cx) < 0.08f || Mathf.Abs(cy) < 0.08f;
                        break;
                    case 2: // ドットグリッド
                        float gx = u * 5f % 1f - 0.5f;
                        float gy = v * 5f % 1f - 0.5f;
                        isWhite = Mathf.Sqrt(gx * gx + gy * gy) < 0.2f;
                        break;
                    case 3: // チェッカーボード
                        isWhite = (int)(u * 5f) % 2 == (int)(v * 5f) % 2;
                        break;
                }

                pixels[y * size + x] = isWhite ? Color.white : Color.black;
            }
        }

        tex.SetPixels(pixels);
        tex.Apply();
        return tex;
    }
}
