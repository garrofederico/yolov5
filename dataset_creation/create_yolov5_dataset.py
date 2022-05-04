import os
import sys

sys.path.append(os.path.abspath('.'))
import random
import shutil
import numpy as np
import cv2
import xml.etree.ElementTree as ET
import yaml

def convert_list_to_yolo_format(images_bboxes_list, outdir, cfg):

    for idx, (image_path, bbox_in_image) in enumerate(images_bboxes_list):
        print(image_path)
        image = cv2.imread(image_path)
        # yolo format class_n (x,y,w,h) normalized to 0-1: https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data
        # convert [x-min, y-min, x-max, y-max] to match the yolo format x,y,w,h:
        #get center of coordinates of the object:
        center_x = float(bbox_in_image[0] + (bbox_in_image[2] - bbox_in_image[0]) / 2) / image.shape[1]
        center_y = float(bbox_in_image[1] + (bbox_in_image[3] - bbox_in_image[1]) / 2) / image.shape[0]
        #get width and height of the object:
        bbox_width = float(bbox_in_image[2] - bbox_in_image[0]) / image.shape[1]
        bbox_height = float(bbox_in_image[3] - bbox_in_image[1]) / image.shape[0]
        bbox_yolo = [center_x, center_y, bbox_width, bbox_height]
        category_id = 0
        print('%g %.6f %.6f %.6f %.6f\n' % (category_id, *bbox_yolo))

        if cfg["DEBUG"]:
            # draw a bounding box around the object and display the image:
            bbox_vis = cv2.rectangle(image.copy(), (int(bbox_in_image[0]), int(bbox_in_image[1])), (int(bbox_in_image[2]), int(bbox_in_image[3])), (0, 255, 0), 1)
            # draw circle at the center of the object:
            bbox_vis = cv2.circle(bbox_vis, (int(bbox_in_image[0] + (bbox_in_image[2] - bbox_in_image[0]) / 2), int(bbox_in_image[1] + (bbox_in_image[3] - bbox_in_image[1]) / 2)), 3, (0, 0, 255), -1)
            #display the image
            cv2.imshow("bbox image", bbox_vis)
            cv2.waitKey(0)
        else:
            labels_dir = outdir + 'labels/'
            images_dir = outdir + 'images/'
            #
            os.makedirs(labels_dir, exist_ok=True)
            os.makedirs(images_dir, exist_ok=True)
            #copy images to annotations folder:
            base_name = ("%06d" % idx)
            ext = os.path.splitext(image_path)[1]
            shutil.copyfile(image_path, images_dir + base_name + ext)
            #shutil.copy(image_path, images_dir)
            # create .txt file for each image and write annotations to it:
            with open(labels_dir + base_name + '.txt', 'w') as w:
                w.write('%g %.6f %.6f %.6f %.6f\n' % (category_id, *bbox_yolo))






def get_bboxes_from_annotations(annotations, cfg):
    # Retrieve annotated points from Cvat xml file
    # Returns "format" for the pose:
    n_keypoints = cfg["N_KEYPOINTS"]
    tree = ET.parse(annotations)
    root = tree.getroot()
    bboxes_from_annotations = []
    for c in root[2:]:
        points = []
        for point in c[:n_keypoints]:
            x, y = point.attrib['points'].split(',')
            x = float(x)
            y = float(y)
            points.append([x, y])

        points = np.array(points, dtype=np.float32)
        # bounding box format [x-min, y-min, x-max, y-max]
        bbox_in_pixels = [np.min(points[:, 0]), np.min(points[:, 1]),
                          np.max(points[:, 0]), np.max(points[:, 1])]
        # add padding to bounding box:
        bbox_in_pixels[0] -= cfg["BBOX_PADDING"]
        bbox_in_pixels[1] -= cfg["BBOX_PADDING"]
        bbox_in_pixels[2] += cfg["BBOX_PADDING"]
        bbox_in_pixels[3] += cfg["BBOX_PADDING"]

        bboxes_from_annotations.append(bbox_in_pixels)
    return bboxes_from_annotations


def split_and_shuffle_img_list(cfg):
    out_path = cfg["OUTPUT_PATH"]
    images_poses_list = join_bboxes_and_images_list(cfg)

    # if cfg["FOR_INFERENCE"]:
    #     # images_bboxes_list.sort()
    #     test_list = images_bboxes_list
    #     convert_list_to_bop_format(test_list, out_path + 'testing/', cfg)
    # else:
    # TODO: Make splits from cfg
    random.seed(42)
    random.shuffle(images_poses_list)
    train_spl = int(len(images_poses_list) * 0.8 + 0.5)
    test_spl = int(len(images_poses_list) * 0.9 + 0.5)
    train_list = images_poses_list[:train_spl]
    test_list = images_poses_list[train_spl:test_spl]
    val_list = images_poses_list[test_spl:]

    convert_list_to_yolo_format(test_list, out_path + 'test/', cfg)
    convert_list_to_yolo_format(train_list, out_path + 'train/', cfg)
    convert_list_to_yolo_format(val_list, out_path + 'val/', cfg)


def join_bboxes_and_images_list(cfg):
    # Join images and bboxes from the same image
    images_poses_list = []

    for input_image_set, annotations in zip(cfg["INPUT_PATH"], cfg["ANNOTATIONS_PATH"]):
        # get list of poses
        bboxes_list = get_bboxes_from_annotations(annotations, cfg)
        # get list of images
        img_list = [input_image_set + f
                    for f in os.listdir(input_image_set)
                    if f.endswith('.png') or f.endswith('.jpg')
                    ]
        img_list.sort()
        # join two lists
        images_and_bboxes = list(zip(img_list, bboxes_list))
        images_poses_list.extend(images_and_bboxes)
    return images_poses_list


def get_config_file(config_file):
    assert os.path.exists(config_file)
    with open(config_file, "r") as stream:
        try:
            cfg = yaml.safe_load(stream)

        except yaml.YAMLError as exc:
            print(exc)
    file_name = os.path.basename(config_file)
    config_name = os.path.splitext(file_name)[0] + '/'
    cfg["OUTPUT_PATH"] += config_name
    # create output directory:
    os.makedirs(cfg["OUTPUT_PATH"], exist_ok=True)
    # save config file into .txt file for reference
    with open(cfg["OUTPUT_PATH"] + "readme.txt", 'w') as w:
        w.write("Config file parameters used for the creation of the dataset: \n\n")
        for key, value in cfg.items():
            w.write(f"{key}: {value} \n")

    print(f"creating dataset from the {file_name}.yaml file...\n")
    return cfg


def main():
    config_file = 'configs/repere4_5vids.yaml'

    cfg = get_config_file(config_file)

    split_and_shuffle_img_list(cfg)


if __name__ == "__main__":
    main()
