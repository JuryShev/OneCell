import json
import supervisely as sly
from imutils import face_utils
import dlib
import cv2
import glob
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
import copy

p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

path_img='./img_save/*.jpg'
IMAGE_FILES = glob.glob(path_img)
def show_detect_face(img):
    plt.imshow(img)
    plt.show()

def detect_point(image):


    rects = detector(image, 0)

    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(image, rect)
        shape = face_utils.shape_to_np(shape)
        return shape, rect


def draw_point(image, shape, size_point=7,color=(0, 255, 0), rect=None):
    for (x, y) in shape:
        cv2.circle(image, (x, y), size_point, color, -1)
        if rect is not None:
            cv2.rectangle(image, (rect.right(), rect.top()), (rect.left(), rect.bottom()), (255, 25, 0), 6)
    return image


def detect_face(image):
    shape, rect=detect_point(image)
    image=draw_point(image, shape, rect=rect)
    show_detect_face(image)


def recove_img_data(data, index, reshape_w, reshape_h):

    img = data.iloc[index].Image
    img = list(map(lambda x:int(x),img.split(' ')))
    img = np.array(img).reshape(reshape_h, reshape_w)
    return img


for idx, file in enumerate(IMAGE_FILES):
    image = cv2.imread(file)
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    point, _=detect_point(img_rgb)


figure_json = {
    "points": {
        "exterior": [

        ],
        "interior": []
    }
}

figure_json_ = {
    "points": {
        "exterior": [
            [200, 100]
        ],
        "interior": []
    }
}
scene_tag = sly.TagMeta("scene", sly.TagValueType.ANY_STRING)
points=[]
for i in point:
    figure_json["points"]["exterior"].clear()
    figure_json["points"]["exterior"].append(i.tolist())
    tag = sly.Tag(scene_tag, value="indoor")
    figure = sly.Point.from_json(figure_json)
    face_class = sly.ObjClass("cat", sly.Point, color=[0, 255, 0])
    point_send = sly.Label(figure, face_class)
    points.append(point_send)


api = sly.Api(server_address="https://app.supervise.ly", token="")

my_teams = api.team.get_list()
print(f"I'm a member of {len(my_teams)} teams")

# get first team and workspace
team = my_teams[0]
workspace = api.workspace.get_list(team.id)[0]

project = api.project.create(workspace.id, "face", change_name_if_conflict=True)
dataset = api.dataset.create(project.id, "face_detect", change_name_if_conflict=True)

print(f"Project {project.id} with dataset {dataset.id} are created")




project_meta = sly.ProjectMeta(obj_classes=[face_class], tag_metas=[scene_tag])

api.project.update_meta(project.id, project_meta.to_json())

image_info = api.image.upload_path(dataset.id, name="9.jpg", path="img_save/9.jpg")



ann = sly.Annotation(img_size=( 1467, 1920,), labels=points, img_tags=[tag])
api.annotation.upload_ann(image_info.id, ann)

print("ok")