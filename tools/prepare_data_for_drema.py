import copy
import cv2
import shutil
from multiprocessing import Process, Manager

from pyrep.const import RenderMode

from rlbench import ObservationConfig, CameraConfig
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.backend.utils import task_file_to_task_class, image_to_float_array
from rlbench.environment import Environment
import rlbench.backend.task as task

from yarr.utils.video_utils import CircleCameraMotion, TaskRecorder, NeRFTaskRecorder
from pyrep.objects.dummy import Dummy
from pyrep.objects.vision_sensor import VisionSensor

import os
import pickle
from PIL import Image
from rlbench.backend import utils
from rlbench.backend.const import *
import numpy as np

from absl import app
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_string('save_path',
                    '/tmp/rlbench_data/',
                    'Where to save the demos.')
flags.DEFINE_list('tasks', [],
                  'The tasks to collect. If empty, all tasks are collected.')
flags.DEFINE_list('image_size', [128, 128],
                  'The size of the images tp save.')
flags.DEFINE_enum('renderer',  'opengl', ['opengl', 'opengl3'],
                  'The renderer to use. opengl does not include shadows, '
                  'but is faster.')
flags.DEFINE_integer('processes', 1,
                     'The number of parallel processes during collection.')
flags.DEFINE_integer('episodes_per_task', 10,
                     'The number of episodes to collect per task.')
flags.DEFINE_integer('variations', -1,
                     'Number of variations to collect per task. -1 for all.')
flags.DEFINE_bool('all_variations', True,
                  'Include all variations when sampling epsiodes')
flags.DEFINE_integer('num_views', 50,
                        'Number of views to collect per timestep for nerf.')

def check_and_make(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def _set_mask_props(mask_cam: VisionSensor, mask: bool, image_size: list):
    if not mask:
        mask_cam.remove()
    else:
        mask_cam.set_explicit_handling(1)
        mask_cam.set_resolution(image_size)

def _set_rgb_props(rgb_cam: VisionSensor, rgb: bool, depth: bool, image_size: list, render_mode: RenderMode):

    rgb_cam.set_explicit_handling(1)
    rgb_cam.set_resolution(image_size)
    rgb_cam.set_render_mode(render_mode)

def copy_directory(input_path, output_path, gt_masks=False, DEPTH_SCALE = 2 ** 24 - 1):
    images = os.listdir(os.path.join(input_path, "images"))
    os.makedirs(os.path.join(output_path, "images"), exist_ok=True)
    for image in images:
        shutil.copy(os.path.join(input_path, "images", image), os.path.join(output_path, "images", image))

    poses = os.listdir(os.path.join(input_path, "poses"))
    os.makedirs(os.path.join(output_path, "poses"), exist_ok=True)
    for pose in poses:
        shutil.copy(os.path.join(input_path, "poses", pose), os.path.join(output_path, "poses", pose))

    if gt_masks:
        masks = os.listdir(os.path.join(input_path, "masks"))
        os.makedirs(os.path.join(output_path, "object_mask"), exist_ok=True)
        for mask in masks:
            image_mask = cv2.imread(os.path.join(input_path, "masks", mask))
            image_mask = image_mask[:, :, 2]
            cv2.imwrite(os.path.join(output_path, "object_mask", mask), image_mask)

    os.makedirs(os.path.join(output_path, "depth"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "depth_scaled"), exist_ok=True)
    for image in images:
        image_name = image.split(".")[0]

        path_depth_image = os.path.join(input_path, "depths", image)
        # copy the depth image
        shutil.copy(path_depth_image, os.path.join(output_path, "depth", image))
        depth_sim = Image.open(path_depth_image)

        # remove the compression
        depth_sim_array = image_to_float_array(depth_sim, DEPTH_SCALE)
        with open(os.path.join(input_path, "poses", image_name + "_near_far.txt"), "r") as near_far_file:
            lines = near_far_file.read().split("\n")
            far = float(lines[0].split(" ")[1])
            near = float(lines[0].split(" ")[0])
            depth_sim_array = (far - near) * depth_sim_array + near

        # save the depth image
        np.save(os.path.join(output_path, "depth_scaled", image.split(".")[0]+".npy"), depth_sim_array)


def merge_coppelia_labels(path):
    # read the ids from the file
    new_labels = []
    labels_to_remove = []
    new_label = 0
    with open(os.path.join(path, "labels.txt"), "r") as file:
        lines = file.read().split("\n")
        for line in lines:
            if len(line.split(";")) == 1:
                continue
            name, number = line.split(";")

            if "pillar" in name:
                labels_to_remove.append(int(number))
                continue
            elif "square_base" in name:
                new_label = int(number)
            elif "shape_sorter_visual" in name:
                labels_to_remove.append(int(number))
                continue
            elif "shape_sorter" in name:
                new_label = int(number)

            # add the new label
            new_labels.append((name, int(number)))

    # remove the labels from the masks
    masks_path = os.path.join(path, "object_mask")
    for filename in os.listdir(masks_path):
        if filename.endswith(".png"):
            mask = cv2.imread(os.path.join(masks_path, filename), cv2.IMREAD_UNCHANGED)
            for label in labels_to_remove:
                mask[mask == label] = new_label
            cv2.imwrite(os.path.join(masks_path, filename), mask)

    # move the previous file in labels_original.txt
    shutil.move(os.path.join(path, "labels.txt"), os.path.join(path, "labels_original.txt"))

    # save the new labels
    with open(os.path.join(path, "labels.txt"), "w") as file:
        for label in new_labels:
            file.write(f"{label[0]};{label[1]}\n")



def save_demo(demo, example_path, variation, labels=None):

    # Save image data first, and then None the image data, and pickle
    left_shoulder_rgb_path = os.path.join(
        example_path, LEFT_SHOULDER_RGB_FOLDER)
    left_shoulder_depth_path = os.path.join(
        example_path, LEFT_SHOULDER_DEPTH_FOLDER)
    left_shoulder_mask_path = os.path.join(
        example_path, LEFT_SHOULDER_MASK_FOLDER)
    right_shoulder_rgb_path = os.path.join(
        example_path, RIGHT_SHOULDER_RGB_FOLDER)
    right_shoulder_depth_path = os.path.join(
        example_path, RIGHT_SHOULDER_DEPTH_FOLDER)
    right_shoulder_mask_path = os.path.join(
        example_path, RIGHT_SHOULDER_MASK_FOLDER)
    overhead_rgb_path = os.path.join(
        example_path, OVERHEAD_RGB_FOLDER)
    overhead_depth_path = os.path.join(
        example_path, OVERHEAD_DEPTH_FOLDER)
    overhead_mask_path = os.path.join(
        example_path, OVERHEAD_MASK_FOLDER)
    wrist_rgb_path = os.path.join(example_path, WRIST_RGB_FOLDER)
    wrist_depth_path = os.path.join(example_path, WRIST_DEPTH_FOLDER)
    wrist_mask_path = os.path.join(example_path, WRIST_MASK_FOLDER)
    front_rgb_path = os.path.join(example_path, FRONT_RGB_FOLDER)
    front_depth_path = os.path.join(example_path, FRONT_DEPTH_FOLDER)
    front_mask_path = os.path.join(example_path, FRONT_MASK_FOLDER)

    check_and_make(left_shoulder_rgb_path)
    check_and_make(left_shoulder_depth_path)
    check_and_make(left_shoulder_mask_path)
    check_and_make(right_shoulder_rgb_path)
    check_and_make(right_shoulder_depth_path)
    check_and_make(right_shoulder_mask_path)
    check_and_make(overhead_rgb_path)
    check_and_make(overhead_depth_path)
    check_and_make(overhead_mask_path)
    check_and_make(wrist_rgb_path)
    check_and_make(wrist_depth_path)
    check_and_make(wrist_mask_path)
    check_and_make(front_rgb_path)
    check_and_make(front_depth_path)
    check_and_make(front_mask_path)

    for i, obs in enumerate(demo):
        left_shoulder_rgb = Image.fromarray(obs.left_shoulder_rgb)
        left_shoulder_depth = utils.float_array_to_rgb_image(
            obs.left_shoulder_depth, scale_factor=DEPTH_SCALE)
        left_shoulder_mask = Image.fromarray(
            (obs.left_shoulder_mask * 255).astype(np.uint8))
        right_shoulder_rgb = Image.fromarray(obs.right_shoulder_rgb)
        right_shoulder_depth = utils.float_array_to_rgb_image(
            obs.right_shoulder_depth, scale_factor=DEPTH_SCALE)
        right_shoulder_mask = Image.fromarray(
            (obs.right_shoulder_mask * 255).astype(np.uint8))
        overhead_rgb = Image.fromarray(obs.overhead_rgb)
        overhead_depth = utils.float_array_to_rgb_image(
            obs.overhead_depth, scale_factor=DEPTH_SCALE)
        overhead_mask = Image.fromarray(
            (obs.overhead_mask * 255).astype(np.uint8))
        wrist_rgb = Image.fromarray(obs.wrist_rgb)
        wrist_depth = utils.float_array_to_rgb_image(
            obs.wrist_depth, scale_factor=DEPTH_SCALE)
        wrist_mask = Image.fromarray((obs.wrist_mask * 255).astype(np.uint8))
        front_rgb = Image.fromarray(obs.front_rgb)
        front_depth = utils.float_array_to_rgb_image(
            obs.front_depth, scale_factor=DEPTH_SCALE)
        front_mask = Image.fromarray((obs.front_mask * 255).astype(np.uint8))

        left_shoulder_rgb.save(
            os.path.join(left_shoulder_rgb_path, IMAGE_FORMAT % i))
        left_shoulder_depth.save(
            os.path.join(left_shoulder_depth_path, IMAGE_FORMAT % i))
        left_shoulder_mask.save(
            os.path.join(left_shoulder_mask_path, IMAGE_FORMAT % i))
        right_shoulder_rgb.save(
            os.path.join(right_shoulder_rgb_path, IMAGE_FORMAT % i))
        right_shoulder_depth.save(
            os.path.join(right_shoulder_depth_path, IMAGE_FORMAT % i))
        right_shoulder_mask.save(
            os.path.join(right_shoulder_mask_path, IMAGE_FORMAT % i))
        overhead_rgb.save(
            os.path.join(overhead_rgb_path, IMAGE_FORMAT % i))
        overhead_depth.save(
            os.path.join(overhead_depth_path, IMAGE_FORMAT % i))
        overhead_mask.save(
            os.path.join(overhead_mask_path, IMAGE_FORMAT % i))
        wrist_rgb.save(os.path.join(wrist_rgb_path, IMAGE_FORMAT % i))
        wrist_depth.save(os.path.join(wrist_depth_path, IMAGE_FORMAT % i))
        wrist_mask.save(os.path.join(wrist_mask_path, IMAGE_FORMAT % i))
        front_rgb.save(os.path.join(front_rgb_path, IMAGE_FORMAT % i))
        front_depth.save(os.path.join(front_depth_path, IMAGE_FORMAT % i))
        front_mask.save(os.path.join(front_mask_path, IMAGE_FORMAT % i))

        # We save the images separately, so set these to None for pickling.
        obs.left_shoulder_rgb = None
        obs.left_shoulder_depth = None
        obs.left_shoulder_point_cloud = None
        obs.left_shoulder_mask = None
        obs.right_shoulder_rgb = None
        obs.right_shoulder_depth = None
        obs.right_shoulder_point_cloud = None
        obs.right_shoulder_mask = None
        obs.overhead_rgb = None
        obs.overhead_depth = None
        obs.overhead_point_cloud = None
        obs.overhead_mask = None
        obs.wrist_rgb = None
        obs.wrist_depth = None
        obs.wrist_point_cloud = None
        obs.wrist_mask = None
        obs.front_rgb = None
        obs.front_depth = None
        obs.front_point_cloud = None
        obs.front_mask = None

    # Save the low-dimension data
    with open(os.path.join(example_path, LOW_DIM_PICKLE), 'wb') as f:
        pickle.dump(demo, f)

    with open(os.path.join(example_path, VARIATION_NUMBER), 'wb') as f:
        pickle.dump(variation, f)

    if labels is not None:
        f = open(os.path.join(example_path, "labels.txt"), 'w')
        for e in labels:

            f.write(str(e[0]) + ";" + str(e[1]) + "\n")
        f.close()
    # save a dictionary
    converted_demo = []
    for obs in demo:
        converted_obs = {}
        #print("gripper: ", obs.gripper_open, " gripper pose: ", obs.gripper_pose)
        converted_obs["gripper_open"] = obs.gripper_open
        converted_obs["gripper_pose"] = obs.gripper_pose
        converted_obs["gripper_joint_positions"] = obs.gripper_joint_positions
        converted_obs["ignore_collisions"] = obs.ignore_collisions
        converted_obs["task_low_dim_state"] = obs.task_low_dim_state
        converted_obs["joint_positions"] = obs.joint_positions
        converted_obs["joint_velocities"] = obs.joint_velocities
        converted_obs["joint_forces"] = obs.joint_forces
        converted_obs["left_shoulder_camera_intrinsics"] = obs.misc["left_shoulder_camera_intrinsics"]
        converted_obs["left_shoulder_camera_extrinsics"] = obs.misc["left_shoulder_camera_extrinsics"]
        converted_obs["left_shoulder_camera_near"] = obs.misc["left_shoulder_camera_near"]
        converted_obs["left_shoulder_camera_far"] = obs.misc["left_shoulder_camera_far"]
        converted_obs["right_shoulder_camera_intrinsics"] = obs.misc["right_shoulder_camera_intrinsics"]
        converted_obs["right_shoulder_camera_extrinsics"] = obs.misc["right_shoulder_camera_extrinsics"]
        converted_obs["right_shoulder_camera_near"] = obs.misc["right_shoulder_camera_near"]
        converted_obs["right_shoulder_camera_far"] = obs.misc["right_shoulder_camera_far"]
        converted_obs["wrist_camera_intrinsics"] = obs.misc["wrist_camera_intrinsics"]
        converted_obs["wrist_camera_extrinsics"] = obs.misc["wrist_camera_extrinsics"]
        converted_obs["wrist_camera_near"] = obs.misc["wrist_camera_near"]
        converted_obs["wrist_camera_far"] = obs.misc["wrist_camera_far"]
        converted_obs["front_camera_intrinsics"] = obs.misc["front_camera_intrinsics"]
        converted_obs["front_camera_extrinsics"] = obs.misc["front_camera_extrinsics"]
        converted_obs["front_camera_near"] = obs.misc["front_camera_near"]
        converted_obs["front_camera_far"] = obs.misc["front_camera_far"]
        converted_obs["overhead_camera_intrinsics"] = obs.misc["overhead_camera_intrinsics"]
        converted_obs["overhead_camera_extrinsics"] = obs.misc["overhead_camera_extrinsics"]
        converted_obs["overhead_camera_near"] = obs.misc["overhead_camera_near"]
        converted_obs["overhead_camera_far"] = obs.misc["overhead_camera_far"]
        converted_demo.append(converted_obs)

    with open(os.path.join(example_path, "dictionary.pkl"), 'wb') as f:
        pickle.dump(converted_demo, f)


def run(i, lock, task_index, variation_count, results, file_lock, tasks):
    """Each thread will choose one task and variation, and then gather
    all the episodes_per_task for that variation."""

    # Initialise each thread with random seed
    np.random.seed(None)
    num_tasks = len(tasks)

    img_size = list(map(int, FLAGS.image_size))

    obs_config = ObservationConfig()
    obs_config.set_all(True)
    obs_config.right_shoulder_camera.image_size = img_size
    obs_config.left_shoulder_camera.image_size = img_size
    obs_config.overhead_camera.image_size = img_size
    obs_config.wrist_camera.image_size = img_size
    obs_config.front_camera.image_size = img_size

    # Store depth as 0 - 1
    obs_config.right_shoulder_camera.depth_in_meters = False
    obs_config.left_shoulder_camera.depth_in_meters = False
    obs_config.overhead_camera.depth_in_meters = False
    obs_config.wrist_camera.depth_in_meters = False
    obs_config.front_camera.depth_in_meters = False

    # We want to save the masks as rgb encodings.
    obs_config.left_shoulder_camera.masks_as_one_channel = False
    obs_config.right_shoulder_camera.masks_as_one_channel = False
    obs_config.overhead_camera.masks_as_one_channel = False
    obs_config.wrist_camera.masks_as_one_channel = False
    obs_config.front_camera.masks_as_one_channel = False

    print(FLAGS.renderer)
    if FLAGS.renderer == 'opengl':
        obs_config.right_shoulder_camera.render_mode = RenderMode.OPENGL
        obs_config.left_shoulder_camera.render_mode = RenderMode.OPENGL
        obs_config.overhead_camera.render_mode = RenderMode.OPENGL
        obs_config.wrist_camera.render_mode = RenderMode.OPENGL
        obs_config.front_camera.render_mode = RenderMode.OPENGL

    rlbench_env = Environment(
        action_mode=MoveArmThenGripper(JointVelocity(), Discrete()),
        obs_config=obs_config,
        headless=True)
    rlbench_env.launch()

    task_env = None
    tasks_with_problems = results[i] = ''

    while True:
        # Figure out what task/variation this thread is going to do
        with lock:

            if task_index.value >= num_tasks:
                print('Process', i, 'finished')
                break

            my_variation_count = variation_count.value
            t = tasks[task_index.value]
            task_env = rlbench_env.get_task(t)
            var_target = task_env.variation_count()
            if FLAGS.variations >= 0:
                var_target = np.minimum(FLAGS.variations, var_target)
            if my_variation_count >= var_target:
                # If we have reached the required number of variations for this
                # task, then move on to the next task.
                variation_count.value = my_variation_count = 0
                task_index.value += 1

            variation_count.value += 1
            if task_index.value >= num_tasks:
                print('Process', i, 'finished')
                break
            t = tasks[task_index.value]

        task_env = rlbench_env.get_task(t)
        task_env.set_variation(my_variation_count)
        descriptions, obs = task_env.reset()
        
        variation_path = os.path.join(
            FLAGS.save_path, task_env.get_name(), VARIATIONS_FOLDER % my_variation_count)

        check_and_make(variation_path)

        with open(os.path.join(variation_path, VARIATION_DESCRIPTIONS), 'wb') as f:
            pickle.dump(descriptions, f)

        episodes_path = os.path.join(variation_path, EPISODES_FOLDER)
        check_and_make(episodes_path)

        abort_variation = False
        for ex_idx in range(FLAGS.episodes_per_task):

            print('Process', i, '// Task:', task_env.get_name(), '// Variation:', my_variation_count, '// Demo:', ex_idx)
            attempts = 10
            while attempts > 0:
                try:
                    # TODO: for now we do the explicit looping.
                    demo, = task_env.get_demos(
                        amount=1,
                        live_demos=True)

                except Exception as e:
                    attempts -= 1
                    if attempts > 0:
                        continue
                    problem = (
                        'Process %d failed collecting task %s (variation: %d, '
                        'example: %d). Skipping this task/variation.\n%s\n' % (
                            i, task_env.get_name(), my_variation_count, ex_idx,
                            str(e))
                    )
                    print(problem)
                    tasks_with_problems += problem
                    abort_variation = True
                    break
                episode_path = os.path.join(episodes_path, EPISODE_FOLDER % ex_idx)

                with file_lock:
                    save_demo(demo, episode_path)
                    
                break
            if abort_variation:
                break

    results[i] = tasks_with_problems
    rlbench_env.shutdown()


def run_all_variations(i, lock, task_index, variation_count, results, file_lock, tasks):
    """Each thread will choose one task and variation, and then gather
    all the episodes_per_task for that variation."""

    # Initialise each thread with random seed
    np.random.seed(None)
    num_tasks = len(tasks)

    img_size = list(map(int, FLAGS.image_size))

    obs_config = ObservationConfig()
    obs_config.set_all(True)
    obs_config.right_shoulder_camera.image_size = img_size
    obs_config.left_shoulder_camera.image_size = img_size
    obs_config.overhead_camera.image_size = img_size
    obs_config.wrist_camera.image_size = img_size
    obs_config.front_camera.image_size = img_size

    # Store depth as 0 - 1
    obs_config.right_shoulder_camera.depth_in_meters = False
    obs_config.left_shoulder_camera.depth_in_meters = False
    obs_config.overhead_camera.depth_in_meters = False
    obs_config.wrist_camera.depth_in_meters = False
    obs_config.front_camera.depth_in_meters = False

    # We want to save the masks as rgb encodings.
    obs_config.left_shoulder_camera.masks_as_one_channel = False
    obs_config.right_shoulder_camera.masks_as_one_channel = False
    obs_config.overhead_camera.masks_as_one_channel = False
    obs_config.wrist_camera.masks_as_one_channel = False
    obs_config.front_camera.masks_as_one_channel = False

    if FLAGS.renderer == 'opengl':
        obs_config.right_shoulder_camera.render_mode = RenderMode.OPENGL
        obs_config.left_shoulder_camera.render_mode = RenderMode.OPENGL
        obs_config.overhead_camera.render_mode = RenderMode.OPENGL
        obs_config.wrist_camera.render_mode = RenderMode.OPENGL
        obs_config.front_camera.render_mode = RenderMode.OPENGL

    rlbench_env = Environment(
        action_mode=MoveArmThenGripper(JointVelocity(), Discrete()),
        obs_config=obs_config,
        headless=True)
    rlbench_env.launch()

    tasks_with_problems = results[i] = ''

    # ========================================================================
    # nerf data generation
    # ========================================================================
    camera_resolution = [1280, 720]
    nerf_camera_resolution = [128, 128]
    rotate_speed = 0.1
    fps = 30
    num_views = 50 # FLAGS.num_views

    cameras = []
    cameras_mask = []
    cameras_label = []
    cameras_motion = []

    # create set of cameras
    cameras.append(VisionSensor.create(resolution=camera_resolution, render_mode=RenderMode.OPENGL))
    cameras_mask.append(VisionSensor.create(resolution=camera_resolution, render_mode=RenderMode.OPENGL_COLOR_CODED))
    cameras.append(VisionSensor.create(resolution=camera_resolution, render_mode=RenderMode.OPENGL))
    cameras_mask.append(VisionSensor.create(resolution=camera_resolution, render_mode=RenderMode.OPENGL_COLOR_CODED))
    cameras.append(VisionSensor.create(resolution=camera_resolution, render_mode=RenderMode.OPENGL))
    cameras_mask.append(VisionSensor.create(resolution=camera_resolution, render_mode=RenderMode.OPENGL_COLOR_CODED))
    cameras.append(VisionSensor.create(resolution=camera_resolution, render_mode=RenderMode.OPENGL))
    cameras_mask.append(VisionSensor.create(resolution=camera_resolution, render_mode=RenderMode.OPENGL_COLOR_CODED))

    cam_front = VisionSensor('cam_front')
    pose_front = cam_front.get_pose()

    # set pose for cameras
    cameras[0].set_pose(pose_front)
    cameras_mask[0].set_pose(pose_front)
    pose_front[2] += 0.30
    
    cameras[1].set_pose(pose_front)
    cameras_mask[1].set_pose(pose_front)
    pose_front[2] -= 0.5

    cameras[2].set_pose(pose_front)
    cameras_mask[2].set_pose(pose_front)
    pose_front[2] -= 0.25
    pose_front[1] += 0.5

    cameras[3].set_pose(pose_front)
    cameras_mask[3].set_pose(pose_front)

    # set parent for cameras
    for camera in cameras:
        camera.set_parent(Dummy('cam_cinematic_placeholder'))
    for camera in cameras_mask:
        camera.set_parent(Dummy('cam_cinematic_placeholder'))

    # create camera motion
    for camera in cameras:
        cameras_motion.append(CircleCameraMotion(camera, Dummy('cam_cinematic_base'), rotate_speed))
    # ========================================================================

    while True:
        # with lock:
        if task_index.value >= num_tasks:
            print('Process', i, 'finished')
            break

        t = tasks[task_index.value]

        task_env = rlbench_env.get_task(t)
        possible_variations = task_env.variation_count()
        print('Max number of variation', possible_variations)

        # task_recorder = TaskRecorder(task_env, cam_motion, fps=fps)

        variation_path = os.path.join(
            FLAGS.save_path, task_env.get_name(),
            VARIATIONS_ALL_FOLDER)
        check_and_make(variation_path)

        episodes_path = os.path.join(variation_path, EPISODES_FOLDER)
        check_and_make(episodes_path)

        abort_variation = False
        print("EPISODES: ", FLAGS.episodes_per_task)
        for ex_idx in range(min(FLAGS.episodes_per_task, possible_variations)): #range(FLAGS.episodes_per_task):

            attempts = 10

            while attempts > 0:
                try:
                    task_recorder = NeRFTaskRecorder(task_env, cameras_motion, fps=fps, num_views=num_views, masks_cameras=cameras_mask)
                    task_recorder._cam_motion[0].save_pose()
                    variation = ex_idx % possible_variations #np.random.randint(possible_variations)
                    task_env = rlbench_env.get_task(t)
                    task_env.set_variation(variation)
                    descriptions, obs = task_env.reset()
                    task_recorder.record_task_description(descriptions)

                    print('Process', i, '// Task:', task_env.get_name(), '// Variation:', variation, '// Demo:', ex_idx)

                    # TODO: for now we do the explicit looping.
                    demo, = task_env.get_demos(
                        amount=1,
                        live_demos=True,
                        callable_each_step=task_recorder.take_snap
                        )

                except Exception as e:
                    attempts -= 1
                    task_recorder.reset()
                    if attempts > 0:
                        continue
                    problem = (
                        'Process %d failed collecting task %s (variation: %d, '
                        'example: %d). Skipping this task/variation.\n%s\n' % (
                            i, task_env.get_name(), variation, ex_idx,
                            str(e))
                    )
                    print(problem)
                    tasks_with_problems += problem
                    abort_variation = True
                    break


                # record last position
                task_env.get_last_state(callable=task_recorder.take_snap)

                labels = task_env.get_labels()

                episode_path = os.path.join(episodes_path, EPISODE_FOLDER % ex_idx)
                with file_lock:
                    save_demo(demo, episode_path, variation, labels=labels)


                    with open(os.path.join(episode_path, VARIATION_DESCRIPTIONS), 'wb') as f:
                        pickle.dump(descriptions, f)
                    
                    # ========================================================================
                    # nerf data generation
                    # ========================================================================
                    record_file_path = os.path.join(episode_path, 'gs_data')
                    record_nerf_file_path = os.path.join(episode_path, 'nerf_data')
                    task_recorder.save(record_file_path, record_nerf_file_path)
                    task_recorder._cam_motion[0].restore_pose()

                    # convert to format needed by drema
                    first_position = os.path.join(record_file_path, "0")
                    
                    # create directory for drema data
                    drema_path = os.path.join(FLAGS.save_path, "drema_data")
                    os.makedirs(drema_path, exist_ok=True)
                    
                    directory = task_env.get_name()
                    episode = EPISODE_FOLDER % ex_idx
                    
                    copy_directory(first_position, os.path.join(drema_path, directory + "_" + str(EPISODE_FOLDER % ex_idx) + "_start"), True)
                    shutil.copy(os.path.join(episode_path, "dictionary.pkl"), os.path.join(drema_path, directory + "_" + episode + "_start", "dictionary.pkl"))
                    shutil.copy(os.path.join(episode_path, "low_dim_obs.pkl"), os.path.join(drema_path, directory + "_" + episode + "_start", "low_dim_obs.pkl"))
                    shutil.copy(os.path.join(episode_path, "variation_descriptions.pkl"), os.path.join(drema_path, directory + "_" + episode + "_start", "variation_descriptions.pkl")) 
                    shutil.copy(os.path.join(episode_path, "variation_number.pkl"), os.path.join(drema_path, directory + "_" + episode + "_start", "variation_number.pkl"))
                    #if os.path.exists(os.path.join(episode_path, "labels.txt")):
                    shutil.copy(os.path.join(episode_path, "labels.txt"), os.path.join(drema_path, directory + "_" + episode + "_start", "labels.txt"))
                    merge_coppelia_labels(os.path.join(drema_path, directory + "_" + episode + "_start"))
                    print(directory + "_" + episode + "_start")

                break
            if abort_variation:
                break

        # with lock:
        task_index.value += 1

    results[i] = tasks_with_problems
    rlbench_env.shutdown()


def main(argv):

    task_files = [t.replace('.py', '') for t in os.listdir(task.TASKS_PATH)
                  if t != '__init__.py' and t.endswith('.py')]

    if len(FLAGS.tasks) > 0:
        for t in FLAGS.tasks:
            if t not in task_files:
                raise ValueError('Task %s not recognised!.' % t)
        task_files = FLAGS.tasks

    tasks = [task_file_to_task_class(t) for t in task_files]

    manager = Manager()

    result_dict = manager.dict()
    file_lock = manager.Lock()

    task_index = manager.Value('i', 0)
    variation_count = manager.Value('i', 0)
    lock = manager.Lock()

    check_and_make(FLAGS.save_path)

    if FLAGS.all_variations:
        # multiprocessing for all_variations not support (for now)
        run_all_variations(0, lock, task_index, variation_count, result_dict, file_lock, tasks)
    else:
        processes = [Process(
            target=run, args=(i, lock, task_index, variation_count, result_dict, file_lock, tasks))
            for i in range(FLAGS.processes)]
        [t.start() for t in processes]
        [t.join() for t in processes]

    print('Data collection done!')
    for i in range(FLAGS.processes):
        print(result_dict[i])




if __name__ == '__main__':
  app.run(main)
