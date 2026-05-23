using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Geometry;

public class ROS2ObjController : MonoBehaviour
{
    ROSConnection ros;
    public string topicName    = "/cmd_vel";
    public float linearScale   = 5.0f;
    public float angularScale  = 50.0f;
    // キーを離してからこの秒数後に自動停止 (teleop_twist_keyboard 用)
    public float velocityTimeout = 0.3f;

    private float _linearX;
    private float _angularZ;
    private float _lastMsgTime = float.NegativeInfinity;

    void Start()
    {
        ros = ROSConnection.GetOrCreateInstance();
        ros.Subscribe<TwistMsg>(topicName, OnTwist);
    }

    void OnTwist(TwistMsg msg)
    {
        _linearX     = (float)msg.linear.x;
        _angularZ    = (float)msg.angular.z;
        _lastMsgTime = Time.time;
    }

    void Update()
    {
        // timeout 経過後は速度をゼロにリセット
        if (Time.time - _lastMsgTime > velocityTimeout)
        {
            _linearX  = 0f;
            _angularZ = 0f;
        }

        transform.Translate(Vector3.forward * _linearX  * linearScale  * Time.deltaTime);
        transform.Rotate(   Vector3.up      * -_angularZ * angularScale * Time.deltaTime);
    }
}
