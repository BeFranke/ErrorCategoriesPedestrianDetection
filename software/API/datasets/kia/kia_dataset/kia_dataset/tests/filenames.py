from collections import defaultdict
import os

def get_all_sample_tokens(base_dir, sequences, company):
    def _get_files(folder):
        fileSet = set() 

        for root, dirs, files in os.walk(folder):
            for fileName in files:
                full_path = os.path.join(root[len(folder)+1:], fileName)
                fileSet.add(full_path)
        return list(fileSet)
    frames = [f for f in _get_files(base_dir) if "sensor/camera/left/png/" in f]
    sample_tokens = []
    for f in frames:
        tokens = f.split("/")
        if sequences is not None:
            sequence = tokens[0].split("_")[3].split("-")[0]
            if int(sequence) not in sequences:
                continue
        if tokens[0].startswith("mv_"):
            if company is None or company == "mv":
                sample_tokens.append(("mv/" + tokens[-1].replace(".png", "")))
        elif tokens[0].startswith("bit_"):
            if company is None or company == "bit":
                sample_tokens.append(("bit/" + tokens[-1].replace(".png", "")))
        else:
            print("Unknown sequence prefix: {}".format(sample_tokens[0].split("_")[0]))
    return sample_tokens

def get_sequence_folder_name(sample_token):
    return sample_token.split("/")[0] + "_results_sequence_" + sample_token.split("-")[2] + "-" + sample_token.split("-")[3]

def check(errors, anno_name, path, base_dir):
    errors[anno_name]
    if not os.path.exists(path):
        if errors[anno_name] == 0:
            print("Error: Cannot find {} file at {}".format(anno_name, path.replace(base_dir, "...")))
        errors[anno_name] += 1


def test(base_dir, sequence, company):
    print("******************")
    print("Checking filenames")
    print("******************")
    print("Printing only first wrong filename per annotation to not spam console.")

    # Sample tokens
    sample_tokens = get_all_sample_tokens(base_dir, [sequence], company)
    if len(sample_tokens) == 0:
        print("Fatal Error: No sample tokens in sequence {} were found.".format(sequence))
        return
    
    sequence_folder = os.path.join(base_dir, get_sequence_folder_name(sample_tokens[0]))
    if not os.path.exists(sequence_folder):
        print("Fatal Error: Sequence folder name does not adhere to naming convention. Cannot find {}".format(sequence_folder))
        return

    errors = defaultdict(int)
    check(errors, "versions", os.path.join(sequence_folder, "versions.txt"), base_dir)
    # Check folder structure
    for sample_token in sample_tokens:
        token_parts = sample_token.split("-")
        frame_token = sample_token.split("/")[1]  # arb-/car- token
        lidar_token = frame_token.replace("camera", "lidar")
        world_token = "world-" + token_parts[2] + "-" + token_parts[3] + "-" + token_parts[4]

        # Network input data
        check(errors, "image-png", os.path.join(sequence_folder, "sensor", "camera", "left", "png", "{}.png".format(frame_token)), base_dir)
        check(errors, "image-exr", os.path.join(sequence_folder, "sensor", "camera", "left", "exr", "{}.exr".format(frame_token)), base_dir)
        check(errors, "depth-exr", os.path.join(sequence_folder, "ground-truth", "depth_exr", "{}.exr".format(frame_token)), base_dir)
        # Lidar NOT as in E1.2.3, but I guess it makes sense to change E1.2.3 here.
        check(errors, "lidar", os.path.join(sequence_folder, "sensor", "lidar", "pcd", "{}.pcd".format(lidar_token)), base_dir)

        # Training Meta-Annotations
        check(errors, "meta-gps-train", os.path.join(sequence_folder, "sensor", "general-globally-per-sequence-train_json", "{}.json".format(world_token)), base_dir)
        check(errors, "meta-gpf-train", os.path.join(sequence_folder, "sensor", "general-globally-per-frame-train_json", "{}.json".format(world_token)), base_dir)
        check(errors, "meta-ops-train", os.path.join(sequence_folder, "sensor", "general-object-per-sequence-train_json", "{}.json".format(world_token)), base_dir)
        check(errors, "meta-opf-train", os.path.join(sequence_folder, "sensor", "general-object-per-frame-train_json", "{}.json".format(world_token)), base_dir)

        # Training Meta-Annotations
        check(errors, "meta-gps-analysis", os.path.join(sequence_folder, "sensor", "general-globally-per-sequence-analysis_json", "{}.json".format(world_token)), base_dir)
        check(errors, "meta-gpf-analysis", os.path.join(sequence_folder, "sensor", "general-globally-per-frame-analysis_json", "{}.json".format(world_token)), base_dir)
        check(errors, "meta-ops-analysis", os.path.join(sequence_folder, "sensor", "general-object-per-sequence-analysis_json", "{}.json".format(world_token)), base_dir)
        check(errors, "meta-opf-analysis", os.path.join(sequence_folder, "sensor", "general-object-per-frame-analysis_json", "{}.json".format(world_token)), base_dir)

        # Training Annotations
        check(errors, "box2d", os.path.join(sequence_folder, "ground-truth", "2d-bounding-box_json", "{}.json".format(frame_token)), base_dir)
        check(errors, "semantic segmentation", os.path.join(sequence_folder, "ground-truth", "semantic-group-segmentation_png", "{}.png".format(frame_token)), base_dir)
        check(errors, "instance segmentation", os.path.join(sequence_folder, "ground-truth", "semantic-instance-segmentation_png", "{}.png".format(frame_token)), base_dir)
        check(errors, "semantic segmentation (lidar)", os.path.join(sequence_folder, "ground-truth", "semantic-group-segmentation_pcd", "{}.pcd".format(frame_token)), base_dir)
        check(errors, "instance segmentation (lidar)", os.path.join(sequence_folder, "ground-truth", "semantic-instance-segmentation_pcd", "{}.pcd".format(frame_token)), base_dir)
        check(errors, "box3d", os.path.join(sequence_folder, "ground-truth", "3d-bounding-box_json", "{}.json".format(world_token)), base_dir)
        check(errors, "skeletons3d", os.path.join(sequence_folder, "ground-truth", "3d-skeletons_json", "{}.json".format(world_token)), base_dir)
        check(errors, "body part segmentation", os.path.join(sequence_folder, "ground-truth", "body-part-segmentation_png", "{}.png".format(frame_token)), base_dir)
        check(errors, "body part segmentation (lidar)", os.path.join(sequence_folder, "ground-truth", "body-part-segmentation_pcd", "{}.pcd".format(frame_token)), base_dir)
    
    print()
    print("******************************")
    print("Filename Erorrs per Annotation")
    print("******************************")
    total_errors = 0
    for k, v in errors.items():
        print("{}: {}".format(k, v))
        total_errors += v

    print()
    print("*********************")
    print("Total Filename Erorrs")
    print("*********************")
    print("Sum: {}".format(total_errors))


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True, type=str)
    parser.add_argument("--company", required=True, type=str)
    parser.add_argument("--sequence", required=True, type=int)
    args = parser.parse_args()
    test(args.data_path, args.sequence, args.company)


if __name__ == "__main__":
    main()
