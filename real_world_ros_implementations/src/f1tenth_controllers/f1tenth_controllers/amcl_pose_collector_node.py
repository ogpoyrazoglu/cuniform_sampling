#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped
import datetime
import signal, pickle

class AMCLPoseCollector(Node):
    def __init__(self):
        super().__init__('amcl_pose_collector')
        self.get_logger().info("AMCL Pose Collector Node Started.")
        self.sub = self.create_subscription(PoseWithCovarianceStamped, '/amcl_pose', self.pose_callback, 10)
        self.data = []
        signal.signal(signal.SIGINT, self.save_and_exit)

    def pose_callback(self, msg):
        self.data.append((msg.pose.pose.position.x, msg.pose.pose.position.y))
        self.get_logger().info(f'Pose: x={self.data[-1][0]}, y={self.data[-1][1]}')

    def save_and_exit(self, signum, frame):
        timestamp = datetime.datetime.now().strftime("%m%d_%H%M%s")
        filename = f'trajectory_{timestamp}.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(self.data, f)
        self.get_logger().info(f"Saved to {filename}")
        self.destroy_node()
        rclpy.shutdown()

def main():
    rclpy.init()
    rclpy.spin(AMCLPoseCollector())

if __name__ == '__main__':
    main()
