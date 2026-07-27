# A single semantic waypoint as stored in the knowledge graph.
string node_id
string scene_id
geometry_msgs/PoseStamped pose
geometry_msgs/PoseStamped navigation_goal_pose
float32[] visual_embedding
string[] detected_objects
ObservationInfo[] observations
string room_label
string configuration_hash
builtin_interfaces/Time creation_timestamp
