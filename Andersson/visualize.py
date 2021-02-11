from os import replace
import numpy as np
import matplotlib.pyplot as plt
import pprint
import os
import glob
import json
from PIL import Image


raw_path = u"\\\\?\\" + "C:\\Users\\Sánta Péter\\Desktop\\BME\\Diplomatervezés\\datasets\\Andersson\\kinect gait raw dataset"
json_path = u"\\\\?\\" + "C:\\Users\\Sánta Péter\\Desktop\\BME\\Diplomatervezés\\datasets\\Andersson\\kinect_gait_json_dataset\\"
train_path = json_path + "train"
val_path = json_path + "val"
test_path = json_path + "test"


# Need to eliminate them
less_than_5_vids = ["Person002", "Person015", "Person158", "Person164"]

# Need to eliminate the most noisy sequences of them
more_than_5_vids = [
    {"person": "Person003",
     "delete_idx": 4},
    {"person": "Person034",
     "delete_idx": 1},
    {"person": "Person036",
     "delete_idx": 3},
    {"person": "Person052",
     "delete_idx": 5},
    {"person": "Person053",
     "delete_idx": 5},
    {"person": "Person074",
     "delete_idx": 3},
    {"person": "Person096",
     "delete_idx": 6}]


categories = [
    {
    "supercategory": "person", 
    "id": 1, 
    "name": "person",
    "keypoints": [
        'Head',                 # 0
        'Shoulder-Center',      # 1
        'Shoulder-Right',       # 2
        'Shoulder-Left',        # 3
        'Elbow-Right',          # 4
        'Elbow-Left',           # 5
        'Wrist-Right',          # 6
        'Wrist-Left',           # 7
        'Hand-Right',           # 8
        'Hand-Left',            # 9
        'Spine',                # 10
        'Hip-centro',           # 11
        'Hip-Right',            # 12
        'Hip-Left',             # 13
        'Knee-Right',           # 14
        'Knee-Left',            # 15
        'Ankle-Right',          # 16
        'Ankle-Left',           # 17
        'Foot-Right',           # 18
        'Foot-Left'             # 19
    ],
    "skeleton": [
        [0, 1],                                 # Head - Shoulder_Center
        [1, 2], [2, 4], [4, 6], [6, 8],         # Right arm
        [1, 3], [3, 5], [5, 7], [7, 9],         # Left arm
        [1, 10], [10, 11],                      # Shoulder_Center - Spine - Hip_centro
        [11, 12], [12, 14], [14, 16], [16, 18], # Right leg
        [11, 13], [13, 15], [15, 17], [17, 19]  # Left leg
    ]
    }
]

camera = [
    {
        "focalx": 485,
        "focaly": 485,
        "height_orig": 480,
        "width_orig": 640,
        "height_ntu": 1080,
        "width_ntu": 1920
    }
]


'''
first_person_keypoints = {
'Head': [-0.02845094, 0.7953885, 2.586351],
'Shoulder-Center': [-0.01007896, 0.6364418, 2.609512],
'Shoulder-Right': [0.02592668, 0.4859544, 2.466846],
'Shoulder-Left': [0.02815875, 0.3025993, 2.432657],
'Elbow-Right': [0.07640454, 0.283133, 2.377785],
'Elbow-Left': [0.02133219, 0.1101851, 2.342309],
'Wrist-Right': [-0.003952064, 0.0485933, 2.342648],
'Wrist-Left': [-0.1129728, 0.07186447, 2.563852],
'Hand-Right': [-0.03491365, -0.01741201, 2.354896],
'Hand-Left': [-0.1714047, -0.01992598, 2.667762],
'Spine': [-0.07590929, 0.1866629, 2.535528],
'Hip-centro': [-0.05240498, 0.08996442, 2.56602],
'Hip-Right': [-0.0622596, 0.05532872, 2.572145],
'Hip-Left': [-0.05218426, 0.04130767, 2.553447],
'Knee-Right': [0.04352561, -0.3369331, 2.533142],
'Knee-Left': [-0.04563818, -0.3681221, 2.716675],
'Ankle-Right': [0.08793103, -0.6665676, 2.52232],
'Ankle-Left': [-0.001235061, -0.6409979, 2.735974],
'Foot-Right': [0.03853168, -0.7267708, 2.505344],
'Foot-Left': [0.05054331, -0.6851594, 2.794123]}
'''


# Transforming coordinates and upscaling them to the NTU resolution
def transform_coordinates(x, y, z, track_id, video_id):
    global camera

    width_ratio = camera[0]["width_ntu"] / camera[0]["width_orig"]
    height_ratio = camera[0]["height_ntu"] / camera[0]["height_orig"]

    if float(z) == 0:
        z = "3.93457"
        print(f'Z=0 ! PERSON: {track_id}, VIDEO: {video_id}')

    u = float(x)*camera[0]["focalx"]/float(z) + camera[0]["width_orig"]//2
    v = camera[0]["height_orig"]//2 - float(y)*camera[0]["focaly"]/float(z)

    u = u*width_ratio
    v = v*height_ratio

    return u, v


# Creating the coco dictionary of a single video
def create_coco_dictionary(file):
    f = open(file, "r")

    # Getting person (track) id and video id
    _, person, video = file.rsplit('\\', 2)
    track_id = int(person[-3:])
    video_id = int(video.rsplit('.')[0])

    # Reading all lines from the txt file
    lines = f.readlines()

    num_frames = int(len(lines) / len(categories[0]["keypoints"]))
    

    # Creating a simple dictionary of the transformed coordinates
    counter = 0
    raw_dict_keypoints = {}
    raw_dict_keypoints[counter] = {}
    for line in lines:        
        line = line.replace('\n', '')
        cat, x, y, z = line.split(';')

        u, v = transform_coordinates(x = x, y = y, z = z, track_id = track_id, video_id = video_id)

        raw_dict_keypoints[counter][cat] = [u, v]

        if cat == 'Foot-Left' and counter < (num_frames - 1):
            counter += 1
            raw_dict_keypoints[counter] = {}


    # Creating the coco "annotations" dictionary list
    coco_images = []
    coco_annotations = []
    for frame, raw_dict in raw_dict_keypoints.items():
        coco_frame_list_keypoints = []
        for cat, raw_list_keypoints in raw_dict.items():
            coco_frame_list_keypoints.append(raw_list_keypoints[0])       # x
            coco_frame_list_keypoints.append(raw_list_keypoints[1])       # y
            coco_frame_list_keypoints.append(1)                           # v (visible) ==> in this dataset it is always 1
        
        coco_annotation_frame_dict = {}
        coco_annotation_frame_dict["keypoints"] = coco_frame_list_keypoints
        coco_annotation_frame_dict["track_id"] = track_id
        coco_annotation_frame_dict["image_id"] = int(f"1{track_id:03d}{video_id:03d}{frame:04d}")
        coco_annotation_frame_dict["id"] = int(f"1{track_id:03d}{video_id:03d}{frame:04d}00")
        coco_annotation_frame_dict["scores"] = []
        coco_annotation_frame_dict["category_id"] = 1
        coco_annotations.append(coco_annotation_frame_dict)


        coco_image_frame_dict = {}
        coco_image_frame_dict["has_no_densepose"] = True
        coco_image_frame_dict["is_labeled"] = True
        coco_image_frame_dict["file_name"] = None
        coco_image_frame_dict["nframes"] = num_frames
        coco_image_frame_dict["frame_id"] = int(f"1{track_id:03d}{video_id:03d}{frame:04d}")
        coco_image_frame_dict["vid_id"] = f"1{track_id:03d}{video_id:03d}"
        coco_image_frame_dict["id"] = int(f"1{track_id:03d}{video_id:03d}{frame:04d}")
        coco_images.append(coco_image_frame_dict)
        

    coco_dictionary = {}
    coco_dictionary["images"] = coco_images
    coco_dictionary["annotations"] = coco_annotations
    coco_dictionary["categories"] = categories
    coco_dictionary["camera"] = camera


    print(f'NUMBER OF FRAMES: {counter}')
    return coco_dictionary


def show_skeleton(dict_keypoints, save = False):

    fig = plt.figure()

    for frame_dict in dict_keypoints["annotations"]:
        person_id = frame_dict["track_id"] 
        video_id = int(str(frame_dict["image_id"])[4:7])
        frame_id = int(str(frame_dict["image_id"])[-4:])

        img_height = dict_keypoints['camera'][0]["height_ntu"]
        img_width = dict_keypoints['camera'][0]["width_ntu"]
        
        skeleton_img = np.zeros((img_height, img_width, 3))
    
        kps = np.array(frame_dict["keypoints"])
        xs = kps[0::3]
        ys = kps[1::3]
    
        plt.clf()
        plt.plot(xs, ys,'o', markersize=6, markerfacecolor='white', markeredgecolor='white', markeredgewidth=1)
        for sk in [dict_keypoints["categories"][0]["skeleton"][index] for index in [0, 9, 10]]:
            plt.plot(xs[sk], ys[sk], linewidth=3, color='steelblue')
        for sk in [dict_keypoints["categories"][0]["skeleton"][index] for index in [1, 2, 3, 4, 11, 12, 13, 14]]:
            plt.plot(xs[sk], ys[sk], linewidth=3, color='salmon')
        for sk in [dict_keypoints["categories"][0]["skeleton"][index] for index in [5, 6, 7, 8, 15, 16, 17, 18]]:
            plt.plot(xs[sk], ys[sk], linewidth=3, color='lightseagreen')

        if save:
            plt.axis('off')
            plt.imshow(skeleton_img)
            plt.savefig(f"../../datasets/Andersson/visualizations/images/Person{person_id:03d}/video{video_id:03d}/person{person_id:03d}video{video_id:03d}frame{frame_id:03d}.png", dpi=None, facecolor='black', edgecolor='black')
        else:
            plt.axis('off')
            plt.imshow(skeleton_img)
            plt.pause(.000000001)
            plt.draw()


def create_gif(imgs_path, gif_path, gif_name):
    ims_path = glob.glob(imgs_path + '*.png')

    ims = []
    for im_path in ims_path:
        ims.append(Image.open(im_path))

    ims[0].save(gif_path + gif_name + '.gif', format='GIF', append_images=ims[1:], save_all=True, duration=40, loop=1)


def create_json_dataset(raw_path, train_path, val_path, test_path, less_than_5_vids, more_than_5_vids): 
    np.random.seed(42)

    more_than_5_vids_persons = [more_than_5_vids[i]["person"] for i in range(len(more_than_5_vids))]
    
    for root, dirs, files in os.walk(raw_path):#+"\\"+"Person003"):
        person = root.rsplit("\\", 1)[1]
        person_idx = person[-3:]
        
        if person in less_than_5_vids:
            continue
        elif person in more_than_5_vids_persons:
            delete_idx = int(*(more_than_5_vids[i]["delete_idx"] for i in range(len(more_than_5_vids)) if more_than_5_vids[i]["person"] == person)) - 1
            del files[delete_idx]

        val_idx, test_idx = None, None
        if files != []:
            file_idxs = [int(file.rsplit(".")[0]) for file in files]
            val_idx, test_idx = np.random.choice(file_idxs, 2, replace=False)
        
        print(person, val_idx, test_idx)
        for f in files:
            coco_dictionary = create_coco_dictionary(root + "\\" + f)

            file_idx = int(f.rsplit(".")[0])
            path = None
            if file_idx == val_idx:
                path = val_path
            elif file_idx == test_idx:
                path = test_path
            else:
                path = train_path
            
            save_path = path + "\\" + person_idx + f"{file_idx:03d}.json"
            with open(save_path, 'w') as fp:
                json.dump(coco_dictionary, fp, indent=4)
                
seaborn => distplot, jointplot

if __name__ == "__main__":
    
    create_json_dataset(raw_path=raw_path, train_path=train_path, val_path=val_path, test_path=test_path, less_than_5_vids=less_than_5_vids, more_than_5_vids=more_than_5_vids)

    '''
    # 71: video002, 147: video004
    for person_id in [71, 147]:
        for video_id in range(1, 6):
            skeleton_raw_path = f"../../datasets/Andersson/kinect gait raw dataset/Person{person_id:03d}/{video_id}.txt"
            skeleton_images_path = f"../../datasets/Andersson/visualizations/images/Person{person_id:03d}/video{video_id:03d}/"
            gif_path = f"../../datasets/Andersson/visualizations/gifs/Person{person_id:03d}/"
            gif_name = f"person{person_id:03d}video{video_id:03d}"

            coco_dictionary = create_coco_dictionary(skeleton_raw_path)
            
            #print(coco_dictionary['annotations'])
            #show_skeleton(dict_keypoints=coco_dictionary)#, save=True)
            #create_gif(imgs_path=skeleton_images_path, gif_path=gif_path, gif_name=gif_name)
    '''



