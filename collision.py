import rospy
import moveit_commander

import sys
import planning_scene_interface
# Initialize the moveit_commander module
moveit_commander.roscpp_initialize(sys.argv)

scene = moveit_commander.PlanningSceneInterface()
import numpy as np
import random
import moveit_msgs.msg
import geometry_msgs.msg



# Initialize a node
rospy.init_node('add_collision_object_node', anonymous=True)

# Create a planning scene interface instance
scene = moveit_commander.PlanningSceneInterface()
rate = rospy.Rate(1)
# Define the pose of the box
while True :
	box_pose = geometry_msgs.msg.PoseStamped()
	box_pose.header.frame_id = "base_link"   # Replace "base_link" with your robot's base frame ID
	box_pose.pose.position.x = 0.5 #random.uniform(0.1, 0.5)   # Replace with the x position of the box
	box_pose.pose.position.y = random.uniform(-0.3, 0.5)    # Replace with the y position of the box
	box_pose.pose.position.z =  0.1 #random.uniform(-0.2, 0.5)     # Replace with the z position of the box
	box_pose.pose.orientation.w = 1.0
	box_pose2 = geometry_msgs.msg.PoseStamped()
	box_pose2.header.frame_id = "base_link"   # Replace "base_link" with your robot's base frame ID
	box_pose2.pose.position.x = random.uniform(0.1, 0.5)    # Replace with the x position of the box
	box_pose2.pose.position.y = random.uniform(-0.3, 0.2)   # Replace with the y position of the box
	box_pose2.pose.position.z = random.uniform(0.1, 0.5)    # Replace with the z position of the box
	box_pose2.pose.orientation.w = 1.0
	# Define the size of the box
	box_size = (0.3, 0.2, 0.1)   # Replace with the size of the box in meters

	# Add the box to the planning scene
	scene.add_box("box1", box_pose, box_size)
	#scene.add_box("box2", box_pose2, box_size)
	rate.sleep()
# Get the names of all the known objects in the scene
object_names = scene.get_known_object_names()
print(object_names)


