import argparse
import os
import pickle
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

from scipy.spatial.transform import Rotation as R
from PIL import Image
from rlbench.backend.utils import image_to_float_array

def _is_stopped(demo, i, obs, stopped_buffer, delta=0.1):
    """
    This code is a modified version of: https://github.com/stepjam/ARM/blob/main/arm/demo_loading_utils.py
    """
    next_is_not_final = i == (len(demo) - 2)
    gripper_state_no_change = (
            i < (len(demo) - 2) and
            (obs.gripper_open == demo[i + 1].gripper_open and
             obs.gripper_open == demo[i - 1].gripper_open and
             demo[i - 2].gripper_open == demo[i - 1].gripper_open))
    small_delta = np.allclose(obs.joint_velocities, 0, atol=delta)
    stopped = (stopped_buffer <= 0 and small_delta and (not next_is_not_final) and gripper_state_no_change)
    return stopped


def compute_keypoints(demo):
    """
    This code is a modified version of: https://github.com/stepjam/ARM/blob/main/arm/demo_loading_utils.py
    """
    keypoints = []

    prev_gripper_open = demo[0].gripper_open
    stopped_buffer = 0
    stopping_delta = 0.1
    for i, obs in enumerate(demo):
        stopped = _is_stopped(demo, i, obs, stopped_buffer, stopping_delta)
        stopped_buffer = 4 if stopped else stopped_buffer - 1

        # If change in gripper, or end of episode.
        last = i == (len(demo) - 1)
        if i != 0 and (obs.gripper_open != prev_gripper_open or last or stopped):
            keypoints.append(i)
        prev_gripper_open = obs.gripper_open

    if len(keypoints) > 1 and (keypoints[-1] - 1) == keypoints[-2]:
        keypoints.pop(-2)

    return keypoints

def project_depth(depth, intrinsics, extrinsics):
    # depth: H x W
    # intrinsics: 3 x 3
    # extrinsics: 4 x 4 (Transformation matrix from camera to world or another frame)
    H, W = depth.shape
    K = intrinsics

    # Create point cloud
    y, x = np.mgrid[0:H, 0:W]
    x = x.flatten()
    y = y.flatten()
    z = depth.flatten()
    points = np.stack([x, y, z, np.ones_like(z)], axis=1)  # Homogeneous coordinates

    constant_x = 1.0 / K[0, 0]
    constant_y = 1.0 / K[1, 1]
    centerX = K[0, 2]
    centerY = K[1, 2]

    points[:, 0] = (points[:, 0] - centerX) * points[:, 2] * constant_x
    points[:, 1] = (points[:, 1] - centerY) * points[:, 2] * constant_y

    # Apply extrinsics transformation
    points = (extrinsics @ points.T).T  # Transform to new coordinate system

    return points[:, :3]  # Return only x, y, z coordinates

def create_frame_from_observation(observation):
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.10)

    # get the rotation matrix
    rotation = observation.gripper_pose[3:]
    rotation = R.from_quat(rotation).as_matrix()

    # translate and rotate the frame
    frame.translate(observation.gripper_pose[:3])
    frame.rotate(rotation, center=observation.gripper_pose[:3])

    return frame

def create_visualization_frames(observations, keypoints):
    origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.30)
    keypoints_frame_list = []
    trajectory = []
    for i, obs in enumerate(observations):
        # add to trajectory
        trajectory.append(obs.gripper_pose[:3])

        # if it is a keypoint
        if i in keypoints:
            # create a frame at the position of the observation and rotate it
            frame = create_frame_from_observation(obs)

            # add the frame to the list
            keypoints_frame_list.append(frame)

    trajectory_frame = o3d.geometry.LineSet()
    trajectory_frame.points = o3d.utility.Vector3dVector(trajectory)
    segments = [[i, i + 1] for i in range(len(trajectory) - 1)]
    trajectory_frame.lines = o3d.utility.Vector2iVector(segments)

    return origin_frame, keypoints_frame_list, trajectory_frame

def get_images_and_point_cloud(data_path, observation, frame_number, camera, color=None):
    # get the path to the images
    path_depth = os.path.join(data_path, camera + "_depth", str(frame_number) + ".png")
    path_rgb = os.path.join(data_path, camera + "_rgb", str(frame_number) + ".png")
    near = observation.misc["%s_camera_near" % camera]
    far = observation.misc["%s_camera_far" % camera]

    color_image = Image.open(path_rgb)
    depth_image = Image.open(path_depth)
    depth_array = image_to_float_array(depth_image, 2**24 - 1)
    depth_array = near + (far - near) * depth_array

    extrinsics = observation.misc["%s_camera_extrinsics" % camera]
    intrinsics = observation.misc["%s_camera_intrinsics" % camera]

    point_cloud = project_depth(depth_array, intrinsics, extrinsics)
    o3d_point_cloud = o3d.geometry.PointCloud()
    o3d_point_cloud.points = o3d.utility.Vector3dVector(point_cloud.reshape(-1, 3))

    if color is None:
        color_array = np.array(color_image)/255
        o3d_point_cloud.colors = o3d.utility.Vector3dVector(color_array.reshape(-1, 3))
    else:
        o3d_point_cloud.colors = o3d.utility.Vector3dVector(np.array([color] * point_cloud.shape[0]))

    return color_image, depth_array, o3d_point_cloud

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--cameras",  nargs='+', type=str, default=["wrist", "front"])
    parser.add_argument("--frame_numbers", nargs='+', type=int, default=[0,-1])
    parser.add_argument("--show_keypoints", type=bool, default=True)
    parser.add_argument("--show_original_colors", type=bool, default=True)
    parser.add_argument("--show_images", type=bool, default=True)

    args = parser.parse_args()

    # create color vector
    colors = [[1, 0, 0],  # Red
              [0, 1, 0],  # Green
              [0, 0, 1],  # Blue
              [1, 1, 0],  # Yellow
              [1, 0, 1],  # Magenta
              [0, 1, 1]]  # Cyan

    args.data_path = os.path.join(args.data_path, args.task, "all_variations", "episodes", "episode"+str(args.episode))

    # set the data path
    observation_path = os.path.join(args.data_path, "low_dim_obs.pkl")

    # load observations
    with open(observation_path, 'rb') as f:
        observations = pickle.load(f)

    # compute keypoints
    keypoints = compute_keypoints(observations)
    print("Found ", len(keypoints), " keypoints with index: ", keypoints)

    # create visualization frames
    origin_frame, keypoints_frame_list, trajectory_frame = create_visualization_frames(observations, keypoints)

    visualizer = o3d.visualization.draw_geometries([origin_frame, trajectory_frame] + keypoints_frame_list)

    frames_to_visualize = args.frame_numbers
    # change -1 to the last frame
    if frames_to_visualize[-1] == -1:
        frames_to_visualize[-1] = len(observations)-1
    if args.show_keypoints:
        # add the keypoints to the frames to visualize
        frames_to_visualize += keypoints
        # sort the frames
        frames_to_visualize = sorted(frames_to_visualize)

    images = []
    for frame_number in frames_to_visualize:
        # get the observation
        obs = observations[frame_number]

        # create the frame
        frame = create_frame_from_observation(obs)

        # get the images and point clouds
        point_clouds = []

        for k, camera in enumerate(args.cameras):

            color_image, depth_image, point_cloud = get_images_and_point_cloud(args.data_path, obs, frame_number, camera, None if args.show_original_colors else colors[k])

            point_clouds.append(point_cloud)
            images.append([color_image, depth_image])

        # visualize the frame
        visualizer = o3d.visualization.draw_geometries([frame, trajectory_frame, origin_frame] + point_clouds)

    # show images using matplotlib in the same figure
    if args.show_images:
        for color_image, depth_image in images:
            fig, axs = plt.subplots(1, 2)
            axs[0].imshow(color_image)
            axs[0].set_title("Color")
            axs[1].imshow(depth_image)
            axs[1].set_title("Depth")
            plt.show()
            plt.close()




