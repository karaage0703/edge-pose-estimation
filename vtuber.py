#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import time
import argparse

import cv2 as cv
import os
import numpy as np
import urllib.request
import onnxruntime


def run_inference(onnx_session, input_size, image):
    image_width, image_height = image.shape[1], image.shape[0]

    # Pre process:Resize, BGR->RGB, Reshape, float32 cast
    input_image = cv.resize(image, dsize=(input_size, input_size))
    input_image = cv.cvtColor(input_image, cv.COLOR_BGR2RGB)
    input_image = input_image.reshape(-1, input_size, input_size, 3)
    input_image = input_image.astype('float32')

    # Inference
    input_name = onnx_session.get_inputs()[0].name
    output_name = onnx_session.get_outputs()[0].name
    outputs = onnx_session.run([output_name], {input_name: input_image})

    keypoints_with_scores = outputs[0]
    keypoints_with_scores = np.squeeze(keypoints_with_scores)

    # Postprocess:Calc Keypoint
    keypoints = []
    scores = []
    for index in range(17):
        keypoint_x = int(image_width * keypoints_with_scores[index][1])
        keypoint_y = int(image_height * keypoints_with_scores[index][0])
        score = keypoints_with_scores[index][2]

        keypoints.append((keypoint_x, keypoint_y))
        scores.append(score)

    return keypoints, scores


def face_overlay(image, image_tmp, cascade):
    # image padding
    padding_size = int(image.shape[1] / 2)
    padding_img = cv.copyMakeBorder(image, padding_size, padding_size , padding_size, padding_size, cv.BORDER_CONSTANT, value=(0,0,0))
    image_tmp = cv.copyMakeBorder(image_tmp, padding_size, padding_size , padding_size, padding_size, cv.BORDER_CONSTANT, value=(0,0,0))
    image_tmp = image_tmp.astype('float64')

    # face detect
    image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    facerect = cascade.detectMultiScale(image_gray, scaleFactor=1.1, minNeighbors=1, minSize=(1, 1))

    # face overlay
    if len(facerect) > 0:
        for rect in facerect:
            face_size = rect[2] * 2
            face_pos_adjust = int(rect[2] * 0.5)
            face_img = cv.imread('./model/karaage_icon.png', cv.IMREAD_UNCHANGED)
            face_img = cv.resize(face_img, (face_size, face_size))
            mask = face_img[:,:,3]
            mask = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
            mask = mask / 255.0
            face_img = face_img[:,:,:3]

            image_tmp[rect[1]+padding_size-face_pos_adjust:rect[1]+face_size+padding_size-face_pos_adjust,
                      rect[0]+padding_size-face_pos_adjust:rect[0]+face_size+padding_size-face_pos_adjust] *= 1 - mask
            image_tmp[rect[1]+padding_size-face_pos_adjust:rect[1]+face_size+padding_size-face_pos_adjust,
                      rect[0]+padding_size-face_pos_adjust:rect[0]+face_size+padding_size-face_pos_adjust] += face_img * mask

    image_tmp = image_tmp[padding_size:padding_size+image.shape[0], padding_size:padding_size+image.shape[1]]
    image_tmp = image_tmp.astype('uint8')

    return image_tmp


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        default='model/model_float32.onnx',
    )
    parser.add_argument(
        "--input_size",
        type=int,
        default=192,
        choices=[192, 256],
    )
    parser.add_argument("--keypoint_score", type=float, default=0.3)

    args = parser.parse_args()
    model_path = args.model
    input_size = args.input_size
    keypoint_score_th = args.keypoint_score

    # Initialize video capture
    cap = cv.VideoCapture(2)

    # download required files
    url_model = 'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_alt.xml'
    path_model = './model/haarcascade_frontalface_alt.xml'

    url_face_image = 'https://raw.githubusercontent.com/karaage0703/karaage_icon/master/karaage_icon.png'
    path_face_image = './model/karaage_icon.png'


    is_file = os.path.isfile(path_model)
    if not is_file:
        print('model file downloading...')
        urllib.request.urlretrieve(url_model, path_model)

    is_file = os.path.isfile(path_face_image)
    if not is_file:
        print('face image file downloading...')
        urllib.request.urlretrieve(url_face_image, path_face_image)

    cascade = cv.CascadeClassifier('./model/haarcascade_frontalface_alt.xml')

    # Load model
    onnx_session = onnxruntime.InferenceSession(model_path)

    while True:
        # Capture read
        ret, frame = cap.read()
        if not ret:
            break
        image_org = copy.deepcopy(frame)

        # Inference execution
        keypoints, scores = run_inference(
            onnx_session,
            input_size,
            frame,
        )

        # Draw
        image_tmp = draw_body(
            frame,
            keypoint_score_th,
            keypoints,
            scores,
        )

        out_img = face_overlay(image_org, image_tmp, cascade)

        key = cv.waitKey(1)
        if key == 27:  # ESC
            break
        cv.imshow('MoveNet(singlepose) Demo', out_img)

    cap.release()
    cv.destroyAllWindows()


def draw_body(
    image,
    keypoint_score_th,
    keypoints,
    scores,
):
    HUMAN_COLOR = (255, 0, 0)

    image_width, image_height = image.shape[1], image.shape[0]
    body_line_size = int(image_width * 0.05)

    # Create black background
    debug_image = np.zeros((image_height, image_width, 3), np.uint8)

    connect_list = [
        [5, 6, HUMAN_COLOR],  # left shoulder → right shoulder
        [5, 7, HUMAN_COLOR],  # left shoulder → left elbow
        [7, 9, HUMAN_COLOR],  # left elbow → left wrist
        [6, 8, HUMAN_COLOR],  # right shoulder → right elbow
        [8, 10, HUMAN_COLOR],  # right elbow → right wrist
        [11, 12, HUMAN_COLOR],  # left hip → right hip
        [5, 11, HUMAN_COLOR],  # left shoulder → left hip
        [11, 13, HUMAN_COLOR],  # left hip → left knee
        [13, 15, HUMAN_COLOR],  # left knee → left ankle
        [6, 12, HUMAN_COLOR],  # right shoulder → right hip
        [12, 14, HUMAN_COLOR],  # right hip → right knee
        [14, 16, HUMAN_COLOR],  # right knee → right ankle
    ]

    # Connect Line
    for (index01, index02, color) in connect_list:
        if scores[index01] > keypoint_score_th and scores[
                index02] > keypoint_score_th:
            point01 = keypoints[index01]
            point02 = keypoints[index02]
            cv.line(debug_image, point01, point02, color, body_line_size)


    # fill body
    if scores[5] > keypoint_score_th and scores[6] > keypoint_score_th and scores[
        11] > keypoint_score_th and scores[12] > keypoint_score_th:
        cv.rectangle(debug_image, keypoints[5], keypoints[12], HUMAN_COLOR, -1)

    return debug_image


if __name__ == '__main__':
    main()
