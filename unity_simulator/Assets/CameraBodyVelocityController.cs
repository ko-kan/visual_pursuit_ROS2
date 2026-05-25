using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Geometry;

/// <summary>
/// Camera body-velocity controller for the Visual Pursuit simulation.
///
/// ErrorControlNode (Python) publishes the true camera body velocity
///
///     V^b_wc = -Ad(g_d) · u_c      (eq 7.7 / Fig 7.3, Hatanaka et al. 2015)
///
/// on camera/body_velocity as geometry_msgs/Twist.
/// This script just subscribes to that topic and moves the camera accordingly.
///
/// Topic connections
/// -----------------
///   Sub : camera/body_velocity  (geometry_msgs/Twist)
///           V^b_wc ∈ R^6  [linear.x/y/z, angular.x/y/z]  (ROS convention)
///
/// Coordinate-frame conversion (ROS camera → Unity local)
/// -------------------------------------------------------
///   ROS camera (right-handed): X=right, Y=down,  Z=forward
///   Unity camera (left-handed): X=right, Y=up,    Z=forward
///
///   Linear  velocity : v_unity = ( Vx, -Vy,  Vz )
///   Angular velocity : ω_unity = (-Wx,  Wy, -Wz ) × Rad2Deg
///     (X and Z flipped due to Y-axis reflection → handedness change)
/// </summary>
public class CameraBodyVelocityController : MonoBehaviour
{
    ROSConnection _ros;

    [Header("ROS Topics")]
    public string bodyVelTopicName = "camera/body_velocity";

    [Header("Scale")]
    [Tooltip("Multiplier applied to the linear  part of V^b_wc")]
    public float linearScale  = 1.0f;
    [Tooltip("Multiplier applied to the angular part of V^b_wc")]
    public float angularScale = 1.0f;

    [Header("Safety")]
    [Tooltip("Zero the command if no message arrives within this many seconds")]
    public float velocityTimeout = 0.3f;

    private double[] _Vwc = new double[6];
    private float _lastCmdTime = float.NegativeInfinity;

    void Start()
    {
        _ros = ROSConnection.GetOrCreateInstance();
        _ros.Subscribe<TwistMsg>(bodyVelTopicName, OnBodyVelocity);
    }

    void OnBodyVelocity(TwistMsg msg)
    {
        _Vwc[0] = msg.linear.x;
        _Vwc[1] = msg.linear.y;
        _Vwc[2] = msg.linear.z;
        _Vwc[3] = msg.angular.x;
        _Vwc[4] = msg.angular.y;
        _Vwc[5] = msg.angular.z;
        _lastCmdTime = Time.time;
    }

    void Update()
    {
        // Safety: zero stale commands
        if (Time.time - _lastCmdTime > velocityTimeout)
            System.Array.Clear(_Vwc, 0, 6);

        float dt = Time.deltaTime;

        // Linear: v_unity = (Vx, -Vy, Vz)
        Vector3 linVel = new Vector3(
             (float)_Vwc[0],
            -(float)_Vwc[1],
             (float)_Vwc[2]
        ) * linearScale;

        // Angular: ω_unity = (-Wx, Wy, -Wz) × Rad2Deg
        Vector3 angVelDeg = new Vector3(
            -(float)_Vwc[3],
             (float)_Vwc[4],
            -(float)_Vwc[5]
        ) * angularScale * Mathf.Rad2Deg;

        transform.Translate(linVel    * dt, Space.Self);
        transform.Rotate   (angVelDeg * dt, Space.Self);
    }
}
