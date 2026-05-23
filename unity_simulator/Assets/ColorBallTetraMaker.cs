using UnityEngine;

public class ColorBallTetraMaker : MonoBehaviour
{
    public float ballRadius = 0.02f;

    // Regular tetrahedron, base face parallel to ground (XZ plane), edge length 0.2 m.
    // Mirrors targets.yaml feature_points in Unity Y-up coords (ROS2 Z-up → swap Y/Z).
    private static readonly Vector3[] vertices = {
        new Vector3( 0.1155f, -0.0408f,  0.0000f),
        new Vector3(-0.0577f, -0.0408f,  0.1000f),
        new Vector3(-0.0577f, -0.0408f, -0.1000f),
        new Vector3( 0.0000f,  0.1225f,  0.0000f),
    };

    // Fully saturated colors (S=1, V=1) rendered via Unlit shader.
    // Unlit: no lighting → every pixel has exactly this color → centroid = sphere center.
    // OpenCV HSV equivalents (S=255, V=255):
    //   vertex 0: red    H=5   → OpenCV H=5
    //   vertex 1: green  H=60  → OpenCV H=60
    //   vertex 2: blue   H=115 → OpenCV H=115
    //   vertex 3: yellow H=30  → OpenCV H=30
    private static readonly Color[] vertexColors = {
        Color.HSVToRGB(  5f / 180f, 1f, 1f),   // red
        Color.HSVToRGB( 60f / 180f, 1f, 1f),   // green
        Color.HSVToRGB(115f / 180f, 1f, 1f),   // blue
        Color.HSVToRGB( 30f / 180f, 1f, 1f),   // yellow
    };

    public float edgeThickness = 0.005f;
    public Color edgeColor = new Color(0.8f, 0.8f, 0.8f);

    private static readonly int[,] edges = {
        {0,1},{0,2},{0,3},{1,2},{1,3},{2,3}
    };

    void Start()
    {
        for (int i = 0; i < edges.GetLength(0); i++)
            CreateEdge(vertices[edges[i, 0]], vertices[edges[i, 1]]);

        for (int i = 0; i < vertices.Length; i++)
            CreateBall(vertices[i], vertexColors[i], i);
    }

    void CreateEdge(Vector3 start, Vector3 end)
    {
        GameObject edge = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
        edge.name = "Edge";
        edge.transform.SetParent(this.transform);
        edge.transform.localPosition = (start + end) / 2f;
        edge.transform.up = (end - start).normalized;
        float dist = Vector3.Distance(start, end);
        edge.transform.localScale = new Vector3(edgeThickness, dist / 2f, edgeThickness);

        Shader shader = Shader.Find("Universal Render Pipeline/Lit");
        if (shader == null) shader = Shader.Find("Standard");
        Material mat = new Material(shader);
        mat.SetColor("_BaseColor", edgeColor);
        mat.color = edgeColor;
        edge.GetComponent<Renderer>().material = mat;
    }

    void CreateBall(Vector3 position, Color color, int index)
    {
        GameObject ball = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        ball.name = $"Vertex_{index}";
        ball.transform.SetParent(this.transform);
        ball.transform.localPosition = position;
        ball.transform.localScale = Vector3.one * ballRadius * 2f;

        Shader shader = Shader.Find("Universal Render Pipeline/Lit");
        if (shader == null) shader = Shader.Find("Standard");
        Material mat = new Material(shader);
        mat.SetColor("_BaseColor", color);
        ball.GetComponent<Renderer>().material = mat;
    }
}
