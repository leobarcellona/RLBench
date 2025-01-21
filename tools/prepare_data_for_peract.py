import argparse
import os
import pickle
import shutil
import random

from rlbench.backend.observation import Observation
from rlbench.demo import Demo

def convert_dictionary_to_rlbench(original_path, scene, episodes, generated_path, cameras, keep_incomplete_demos=True):
    # iterate over th original episodes
    for episode in episodes:

        # set original paths
        episode_path = os.path.join(original_path, episode)
        original_low_dim_obs_path = os.path.join(episode_path, "low_dim_obs.pkl")
        original_description_path = os.path.join(episode_path, "variation_descriptions.pkl")
        original_number_path = os.path.join(episode_path, "variation_number.pkl")

        # set generated paths and check if they exist
        generated_episodes_path = os.path.join(generated_path, scene + "_" + episode + "_start")
        if not os.path.exists(generated_episodes_path):
            continue
        generated_episodes = [x for x in os.listdir(generated_episodes_path) if os.path.isdir(os.path.join(generated_episodes_path, x))]

        # iterate over the generated episodes
        for generated_episode in generated_episodes:

            if generated_episode[:7] != "episode":
                continue

            generated_low_dim_obs_path = os.path.join(generated_episodes_path, generated_episode, "low_dim_obs_generated.pkl")
            generated_low_dim_obs_output = os.path.join(generated_episodes_path, generated_episode, "low_dim_obs.pkl")
            generated_description_output = os.path.join(generated_episodes_path, generated_episode, "variation_descriptions.pkl")
            generated_number_output = os.path.join(generated_episodes_path, generated_episode, "variation_number.pkl")

            # copy the description to the output
            shutil.copyfile(original_description_path, generated_description_output)

            # copy the number to the output
            shutil.copyfile(original_number_path, generated_number_output)

            # load the generated low dim obs and the original low dim obs
            file = open(generated_low_dim_obs_path, "rb")
            generated_demo = pickle.load(file)
            file.close()

            file = open(original_low_dim_obs_path, "rb")
            original_demo = pickle.load(file)
            file.close()

            # the converted demo
            new_demo = []

            # iterate over the generated demo
            for observation in generated_demo:

                # the misc dictionary contains the camera intrinsics, extrinsics, near and far
                misc = {}
                for camera in cameras:
                    d = {}
                    d["%s_camera_intrinsics" % camera] = observation["%s_camera_intrinsics" % camera]
                    d["%s_camera_extrinsics" % camera] = observation["%s_camera_extrinsics" % camera]
                    d["%s_camera_near" % camera] = observation["%s_camera_near" % camera]
                    d["%s_camera_far" % camera] = observation["%s_camera_far" % camera]
                    misc.update(d)

                # create the observation object
                o = Observation(
                    left_shoulder_rgb=None,
                    left_shoulder_depth=None,
                    left_shoulder_point_cloud=None,
                    right_shoulder_rgb=None,
                    right_shoulder_depth=None,
                    right_shoulder_point_cloud=None,
                    overhead_rgb=None,
                    overhead_depth=None,
                    overhead_point_cloud=None,
                    wrist_rgb=None,
                    wrist_depth=None,
                    wrist_point_cloud=None,
                    front_rgb=None,
                    front_depth=None,
                    front_point_cloud=None,
                    left_shoulder_mask=None,
                    right_shoulder_mask=None,
                    overhead_mask=None,
                    wrist_mask=None,
                    front_mask=None,
                    joint_velocities=observation["joint_velocities"],
                    joint_positions=None,
                    joint_forces=observation["joint_forces"],
                    gripper_open=observation["gripper_open"],
                    gripper_pose=observation["gripper_pose"],
                    gripper_matrix=None,
                    gripper_touch_forces=None,
                    gripper_joint_positions=observation["gripper_joint_positions"],
                    task_low_dim_state=None,
                    ignore_collisions=observation["ignore_collisions"],
                    misc=misc)

                # append the observation to the new demo
                new_demo.append(o)

            demo_steps = len(new_demo)
            path_images = os.path.join(generated_episodes_path, generated_episode, cameras[0] + "_rgb")
            number_of_images = len([x for x in os.listdir(path_images) if (os.path.isfile(os.path.join(path_images, x)) and x[-4:] == ".png")])

            # check if the number of steps and the number of images match
            if demo_steps != number_of_images:
                print("Number of steps and number of images do not match: ", demo_steps, number_of_images)
                print(os.path.join(generated_episodes_path, generated_episode))

                if not keep_incomplete_demos:
                    continue

                new_demo = new_demo[:number_of_images]

            # update seed and create the new demo
            new_demo = Demo(new_demo)
            new_demo.random_seed = original_demo.random_seed

            # save the new demo
            with open(generated_low_dim_obs_output, 'wb') as f:
                pickle.dump(new_demo, f)


def merge_generated_data(generated_path, output_path, scenes, starting_index=0):
    """
    Merge the generated data into a single directory containing all the generated
    :param generated_path: path where the generated data is stored. It contains scene as SCENENAME_episodeNUMBER_start
    :param output_path: where the merged data will be stored. There the data will be divided into SCENENAME/all_variations/episodes
    :param scenes: scenes/tasks that are present in the generated data
    :param starting_index: starting index for the merged episodes
    :return: None
    """

    generated_folders = [f for f in os.listdir(generated_path) if os.path.isdir(os.path.join(generated_path, f))]

    # read the tasks and episodes from the generated folders
    tasks = []
    episodes = {}
    for folder in generated_folders:
        task, other = folder.split("_episode")
        episode = int(other.split("_start")[0])
        if task not in tasks:
            tasks.append(task)
            episodes[task] = []

        if episode not in episodes[task]:
            episodes[task].append(episode)

    # sort the episodes and convert them to strings
    for task in tasks:
        episodes[task].sort()
        episodes[task] = [str(e) for e in episodes[task]]

    # remove tasks that are not in the scenes
    tasks = [task for task in tasks if task in scenes]

    # iterate over the tasks
    for task in tasks:
        # create the output directories
        output_index = starting_index
        out_path_task_three = os.path.join(output_path, "three_augmentations", task, "all_variations", "episodes")
        os.makedirs(out_path_task_three, exist_ok=False)

        number = 0
        episode_number_counter = 0
        total_episodes_counter = 0
        all_task_paths = []
        # iterate over the generated episodes
        for episode in episodes[task]:

            total_episodes_counter += 1

            # get the generated episodes
            path = os.path.join(args.path_in, f"{task}_episode{episode}_start")
            gen_episodes = [os.path.join(path, f) for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
            all_task_paths += gen_episodes

            episode_number_counter += 1

            # if we have 5 episodes or we reached the end of the episodes we can generate the output
            if episode_number_counter == 5 or len(episodes[task]) == total_episodes_counter:

                # shuffle the episodes
                random.shuffle(all_task_paths)

                # iterate over the episodes
                for k, episode_path in enumerate(all_task_paths):

                    # copy the episode to the output directory
                    #gen_episode_number = int(episode.split("/")[-1].split("episode")[-1])
                    output_path_episode = os.path.join(out_path_task_three, "episode" + str(output_index + k).zfill(4))
                    os.makedirs(output_path_episode, exist_ok=False)
                    shutil.copytree(episode_path, output_path_episode, dirs_exist_ok=True)

                    # increase the index of the output episodes
                    number += 1

                # reset the counters and epsides
                all_task_paths = []
                episode_number_counter = 0
                # increment
                output_index += 1000

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--original_path', type=str, required=True)
    parser.add_argument('--generated_path', type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument('--scenes', type=str, nargs='+', required=True)
    parser.add_argument("--convert_to_rlbench", type=bool, default=True)
    parser.add_argument('--start_index', type=bool, default=0)
    parser.add_argument("--cameras", type=str, nargs='+', default=["left_shoulder", "right_shoulder", "overhead", "wrist", "front"])
    parser.add_argument("--keep_incomplete_demos", type=bool, default=True)

    args = parser.parse_args()

    if args.convert_to_rlbench:
        for scene in args.scenes:
            original_path = os.path.join(args.original_path, scene, "all_variations", "episodes")
            original_episodes = [x for x in os.listdir(original_path) if os.path.isdir(os.path.join(original_path, x))]
            original_episodes.sort()

            convert_dictionary_to_rlbench(original_path, scene, original_episodes, args.generated_path, args.cameras, keep_incomplete_demos=args.keep_incomplete_demos)





