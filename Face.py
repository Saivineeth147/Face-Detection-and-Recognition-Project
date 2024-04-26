'''
Notes:
1. All of your implementation should be in this file. This is the ONLY .py file you need to edit & submit. 
2. Please Read the instructions and do not modify the input and output formats of function detect_faces() and cluster_faces().
3. If you want to show an image for debugging, please use show_image() function in helper.py.
4. Please do NOT save any intermediate files in your final submission.
'''

import cv2
import numpy as np
import os
import sys
import math

import face_recognition

from typing import Dict, List
from utils import show_image

'''
Please do NOT add any imports. The allowed libraries are already imported for you.
'''


def detect_faces(img: np.ndarray) -> List[List[float]]:
    """
    Args:
        img : input image is an np.ndarray represent an input image of shape H x W x 3.
            H is the height of the image, W is the width of the image. 3 is the [R, G, B] channel (NOT [B, G, R]!).

    Returns:
        detection_results: a python nested list.
            Each element is the detected bounding boxes of the faces (may be more than one faces in one image).
            The format of detected bounding boxes a python list of float with length of 4. It should be formed as
            [topleft-x, topleft-y, box-width, box-height] in pixels.
    """
    detection_results: List[List[float]] = []  # Please make sure your output follows this data format.

    # Add your code here. Do not modify the return and input arguments.
    faces = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    flags = cv2.CASCADE_SCALE_IMAGE
    elements = faces.detectMultiScale(gray, 1.1, 5, flags)
    for (x, y, bw, bh) in elements:
        bounding_boxes = [float(x), float(y), float(bw), float(bh)]
        detection_results.append(bounding_boxes)
    return detection_results


def cluster_faces(imgs: Dict[str, np.ndarray], K: int) -> List[List[str]]:
    """
    Args:
        imgs : input images. It is a python dictionary
            The keys of the dictionary are image images (without path).
            Each value of the dictionary is an np.ndarray represent an input image of shape H x W x 3.
            H is the height of the image, W is the width of the image. 3 is the [R, G, B] channel (NOT [B, G, R]!).
        K: Number of clusters.
    Returns:
        cluster_results: a python list where each elemnts is a python list.
            Each element of the list a still a python list that represents a cluster.
            The elements of cluster list are python strings, which are image filenames (without path).
            Note that, the final filename should be from the input "imgs". Please do not change the filenames.
    """
    cluster_results: List[List[str]] = [[]] * K  # Please make sure your output follows this data format.

    # Add your code here. Do not modify the return and input arguments.

    face_list = []
    images = []
    for key, value in imgs.items():
        bounding_boxes = face_recognition.face_locations(value)
        face_encoding = face_recognition.face_encodings(value, bounding_boxes)
        face_list.append(face_encoding)
        images.append(key)

    face_data = np.array(face_list)
    center_data = centers(face_data, K)
    labels = kmeans_clustering(face_data, K, center_data, 10)

    for i in range(K):
        arr = []
        for j in range(len(list(labels))):
            if i == labels[j]:
                arr.append(images[j])

        cluster_results[i] = arr

    return cluster_results

    '''
If your implementation requires multiple functions. Please implement all the functions you design under here.
But remember the above 2 functions are the only functions that will be called by task1.py and task2.py.
'''


# Your functions. (if needed)

def kmeans_clustering(face_data, k, centers, attempts):

    points = cDistances(face_data, centers)
    center_arr = np.array([np.argmin(p) for p in points])

    for _ in range(attempts):
        centers = []
        for j in range(k):
            temp_centers = np.mean(face_data[center_arr == j], axis=0)
            centers.append(temp_centers)
        points = cDistances(face_data, centers)
        center_arr = np.array([np.argmin(p) for p in points])

    return center_arr


def cDistances(col1, col2):
    centers = []

    for val1 in range(len(col1)):
        for val2 in range(len(col2)):
            distance = np.linalg.norm(col1[val1][0] - col2[val2][0])
            centers.append(distance)

    centers = np.reshape(centers, (len(col1), len(col2)))

    return centers


def centers(face_data, k):

    clusters = [face_data[np.random.choice(range(k))]]

    for index in range(k - 1):
        distances = []
        for _, pt in enumerate(face_data):
            close_point = np.argmin(np.sqrt(np.sum((pt - clusters) ** 2, axis=1)))
            distances.append(close_point)
        distances = np.array(distances)
        new_clusters = face_data[np.argmax(distances), :]
        clusters.append(new_clusters)

    return clusters
