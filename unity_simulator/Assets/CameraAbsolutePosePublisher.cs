using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Std;

/// <summary>
/// Publishes the camera's absolute pose g_wc ∈ SE(3) as a Float64MultiArray
/// (16 doubles, 4×4 row-major) on topic  camera/absolute_pose.
///
/// Used by error_control_node (Python) for the occlusion-avoidance CBF:
///     p_wo  = p_wc + R_wc @ p_co        (target world position)
///     l_c,k = ‖p_wc − c_k‖ − r_k       (camera clearance to obstacle)
///     l_o,k = ‖p_wo − c_k‖ − r_k       (target  clearance to obstacle)
///
/// ── Coordinate conventions ──────────────────────────────────────────────
///
///   Frame            Handedness   X        Y        Z
///   ───────────────────────────────────────────────────────
///   Unity camera     left         right    up       forward
///   ROS   camera     right        right    down     forward
///   Unity world      left         right    up       (scene)
///   ROS   world      right        right    (scene)  up
///
///   Unity world → ROS world position:
///       p_ros = (x_unity, −z_unity, y_unity)
///
///   Rotation  R_wc  (ROS camera frame → ROS world frame):
///       R_wc = T_WU · R_unity · T_cam
///   where
///       T_WU  = permutation  [x→x, −z→y, y→z]   (Unity world → ROS world)
///       T_cam = diag(1, −1, 1)                    (ROS cam    → Unity cam)
///
///   Expanded (m = Unity localToWorldMatrix 3×3):
///       R_wc = ⎡  m00  −m01   m02 ⎤
///              ⎢ −m20   m21  −m22 ⎥
///              ⎣  m10  −m11   m12 ⎦
///
/// ── Setup ───────────────────────────────────────────────────────────────
///   1. Attach this script to the DroneCamera GameObject.
///   2. Set  poseTopicName = "camera/absolute_pose"  (default).
///   3. In control.yaml, specify obstacle centers in ROS world coordinates:
///         cbf.obstacles[].center: [x, -z_unity, y_unity]
/// </summary>
[RequireComponent(typeof(Camera))]
public class CameraAbsolutePosePublisher : MonoBehaviour
{
    [Header("ROS Topic")]
    [Tooltip("Topic name for camera/absolute_pose (Float64MultiArray, 4×4 row-major SE(3))")]
    public string poseTopicName = "camera/absolute_pose";

    [Header("Publish Rate")]
    [Tooltip("Publication frequency [Hz]")]
    public float publishHz = 30f;

    // -----------------------------------------------------------------------

    ROSConnection _ros;
    float _nextPublishTime = 0f;

    void Start()
    {
        _ros = ROSConnection.GetOrCreateInstance();
        _ros.RegisterPublisher<Float64MultiArrayMsg>(poseTopicName);
    }

    void Update()
    {
        if (Time.time < _nextPublishTime) return;
        _nextPublishTime = Time.time + 1f / publishHz;
        PublishPose();
    }

    void PublishPose()
    {
        // ── Translation: Unity world → ROS world (x, −z, y) ───────────
        Vector3 p = transform.position;
        double px = p.x;
        double py = -p.z;
        double pz = p.y;

        // ── Rotation: R_wc = T_WU · R_unity · T_cam ───────────────────
        // Unity localToWorldMatrix column layout (m.mRC = row R, col C):
        //   col 0: Unity cam X (right)   expressed in Unity world
        //   col 1: Unity cam Y (up)      expressed in Unity world
        //   col 2: Unity cam Z (forward) expressed in Unity world
        Matrix4x4 m = transform.localToWorldMatrix;

        //   T_cam on right (diag 1,−1,1): negate column 1
        //   T_WU  on left  (x→x, −z→y, y→z):
        //       row 0 of result = row 0 of (R_unity·T_cam)
        //       row 1 of result = −row 2 of (R_unity·T_cam)
        //       row 2 of result = row 1 of (R_unity·T_cam)
        double r00 =  m.m00;  double r01 = -m.m01;  double r02 =  m.m02;
        double r10 = -m.m20;  double r11 =  m.m21;  double r12 = -m.m22;
        double r20 =  m.m10;  double r21 = -m.m11;  double r22 =  m.m12;

        // ── Publish 4×4 SE(3) row-major: [[R | p], [0 0 0 1]] ─────────
        var msg = new Float64MultiArrayMsg
        {
            data = new double[]
            {
                r00, r01, r02, px,
                r10, r11, r12, py,
                r20, r21, r22, pz,
                0.0, 0.0, 0.0, 1.0
            }
        };

        _ros.Publish(poseTopicName, msg);
    }
}
