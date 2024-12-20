import cv2
import numpy as np
import imgaug.augmenters as iaa
import labelme
import json
from labelme import utils
import os
import numpy as np
import glob
import base64
import json
import time
from imgaug.augmentables.polys import Polygon, PolygonsOnImage

from imgaug import augmenters as iaa



import json

import numpy as np

import cv2

import imgaug as ia
from imgaug.augmentables.polys import Polygon, PolygonsOnImage
from utils import *

def make_polys(json_file):
    with open(json_file, "r") as js:
        json_data = json.load(js)

    polys = []
    # Iterate over the shapes in the JSON data
    # If json_data[shapes] is empty, stop augmenting
    for shape in json_data['shapes']:
        # This assert might be overkill but better safe that sorry ...
        assert shape['shape_type'] == "polygon"
        polys.append(Polygon(shape['points'], label=shape['label']))

    img_shape = (json_data['imageHeight'], json_data['imageWidth'], 3)
    polys_oi = PolygonsOnImage(polys, shape=img_shape)
    return(polys_oi)

def POI2labelmejson(img_name,poly, json_name) -> None:
    # convert the PolygonsOnImage object to list of coordinates
    polygons = poly.polygons
    coords = [[(x, y) for (x, y) in polygon.exterior] for polygon in polygons]
    labels = [polygon.label for polygon in polygons]
    print("labels are>>>>>>>>>>", labels)

    # create the labelme-compatible dictionary
    labelme_dict = {
        "version": "3.16.4", #4.5.7
        "flags": {},
        "shapes": [
            {
                "label": label_name,#"polygon",
                "line_color": None,
                "fill_color": None,
                "points": coord_list,
                "group_id": None,
                "shape_type": "polygon",
                "flags": {}
            } for label_name, coord_list in zip(labels,coords)
        ],
        "lineColor": [0,225,0,128],
        "fillColor": [225,0,0,128],
        "imagePath": f"{img_name}",
        "imageData": base64.b64encode(open(img_name, "rb").read()).decode('utf-8'),
        "imageHeight": 2048,
        "imageWidth": 2048
    }

    # save the labelme-compatible dictionary to a file
    with open(f"{json_name}.json", "w") as f:
        json.dump(labelme_dict, f, cls = NumpyEncoder)
    return None





def augmentation(img_path, json_path, my_augmenter,num_aug, aug_img_path):
    quokka_img = cv2.imread(img_path)

    polys_oi = make_polys(json_path)

    # If you pass the arguments, it returns 2 elements
    # - The augmented Image
    # - The augmented polygons 
    augmented = my_augmenter(image = quokka_img, polygons = polys_oi)
    [type(x) for x in augmented]


    # So you can make a bunch of augmented image/polygon pairs
    augmented_list = [my_augmenter(image = quokka_img, polygons = polys_oi) for _ in range(num_aug)]

    i =0
    for img, poly in augmented_list:
        cv2.imwrite(aug_img_path+f"{img_path.split('/')[-1].split('.')[0]}_aug_{i}.png", img)

        POI2labelmejson(img_name=aug_img_path+f"{img_path.split('/')[-1].split('.')[0]}_aug_{i}.png",
                        poly=poly, json_name=aug_img_path+ f"{img_path.split('/')[-1].split('.')[0]}_aug_{i}" )
        i+=1



if __name__ == "__main__":

    my_augmenter = iaa.Sequential([
    iaa.GaussianBlur((0.1, 5)),
    iaa.Fliplr(0.5),
    iaa.Flipud(0.5),
    # iaa.Rotate((-15,15)),
    iaa.Affine(rotate=(-15, 15), fit_output = True),#, fit_output = True
    # iaa.AdditiveGaussianNoise(scale=(0, 0.05*255), per_channel=False), #f
    # iaa.CoarseDropout(0.1, size_percent=0.5),  #f
    # iaa.BlendAlpha((0.0, 1.0),iaa.Affine(rotate=(-25, 25)),per_channel=0.5), #f
    # iaa.MotionBlur(k=(5,15), angle=[-20, 20]), #f
    iaa.MultiplyAndAddToBrightness(add=(-25, 10)), ##? #mul=(0.5, 1.0), #f
    # iaa.SigmoidContrast(gain=(10, 30), cutoff=(0.0, 0.2)),##?
    # iaa.LinearContrast((0.4, 1.0)),
    # iaa.AllChannelsCLAHE(),
    iaa.Affine(shear=(-25, 25), fit_output = True),
    # iaa.AveragePooling([2, 8],keep_size=False),
    # iaa.FastSnowyLandscape(lightness_threshold=[128, 200],lightness_multiplier=(0.5, 1.0))
    ])


    NUM_AUG =10

    img_files = sorted(glob.glob("/mnt/data/needle_images_only/needle_images/*.jpg"))
    json_files = sorted(glob.glob("/mnt/data/needle_images_only/needle_images/*.json"))

    
    for img_file, json_file in zip(sorted(img_files), sorted(json_files)):
        start_time = time.time()
        print("[Manna log: img_file]: ", img_file)
        print("[Manna log: json_file]: ", json_file)
        augmentation(img_file, json_file, my_augmenter, NUM_AUG, "/mnt/data/needle_images_only/aug_needle_images/")
        end_time = time.time()
        print("Elapsed time: {:.2f} seconds".format(end_time - start_time))
        print("\n\n")

