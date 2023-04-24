#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import argparse

import cv2 as cv
import numpy as np
import onnxruntime
import time
import sys

import pygame
import pygame.midi

pygame.init()
pygame.midi.init()

# time param
start_time = 0.0
dot_line = 0

# music setting
volume = 127
note_list = []

HUMAN_COLOR = (255, 0, 0)

# midi setup
for i in range(pygame.midi.get_count()):
    print(pygame.midi.get_device_info(i))
    interf, name, input_dev, output_dev, opened = pygame.midi.get_device_info(i)
    if output_dev and b'NSX-39 ' in name:
        print('midi id=' + str(i))
        midi_output = pygame.midi.Output(i)

    if output_dev and b'UM-1' in name:
        print('midi id=' + str(i))
        midi_output = pygame.midi.Output(i)

    if output_dev and b'InstaChord' in name:
        print('midi id=' + str(i))
        midi_output = pygame.midi.Output(i)

try:
    midi_output.set_instrument(1, 2) # inst: MIDI instrument No, ch: MIDI channel from 0 ex: 2 -> 3
except:
    print('Not found MIDI device')
    sys.exit()


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)
    parser.add_argument('--fullscreen', type=bool, default=False)

    args = parser.parse_args()

    return args


def get_pentatonic_scale(note):
    # C
    if note % 5 == 0:
        out_note = note // 5 * 12

    # D#
    if note % 5 == 1:
        out_note = note // 5 * 12 + 3

    # F
    if note % 5 == 2:
        out_note = note // 5 * 12 + 5

    # G
    if note % 5 == 3:
        out_note = note // 5 * 12 + 7

    # A#
    if note % 5 == 4:
        out_note = note // 5 * 12 + 10

    out_note += 60
    while out_note > 127:
        out_note -= 128

    return out_note


def skeleton_sequencer(src):
    global start_time
    global dot_line
    global note_list

    # parameters
    speed = 0.5
    d_circle = 30

    image_h, image_w = src.shape[:2]

    h_max = int(image_h / d_circle)
    w_max = int(image_w / d_circle)

    # create blank image
    npimg_target = np.zeros((image_h, image_w, 3), np.uint8)
    dot_color = [[0 for i in range(h_max)] for j in range(w_max)]

    # make dot information from ndarray
    for y in range(0, h_max):
        for x in range(0, w_max):
            dot_color[x][y] = src[y * d_circle][x * d_circle]

    # move dot
    while time.time() - start_time > speed:
        start_time += speed
        dot_line += 1
        if dot_line > w_max - 1:
            dot_line = 0

        # sound off
        for note in note_list:
            midi_output.note_off(note, volume, 2)

        # sound on
        note_list = []

        for y in range(0, h_max):
            if dot_color[dot_line][y].tolist() == list(HUMAN_COLOR):
                note_list.append(get_pentatonic_scale(y))

        for note in note_list:
            midi_output.note_on(note, volume, 2)

    # draw dot
    for y in range(0, h_max):
        for x in range(0, w_max):
            center = (int(x * d_circle + d_circle * 0.5), int(y * d_circle + d_circle * 0.5))
            if x == dot_line:
                if dot_color[dot_line][y].tolist() == list(HUMAN_COLOR):
                    cv.circle(npimg_target, center, int(d_circle / 2), [
                        255 - (int)(dot_color[x][y][0]), 255 - (int)(dot_color[x][y][1]), 255 - (int)(dot_color[x][y][2])],
                        thickness=-1, lineType=8, shift=0)
                else:
                    cv.circle(npimg_target, center, int(d_circle / 2), [
                        255, 255, 255], thickness=-1, lineType=8, shift=0)
            else:
                cv.circle(npimg_target, center, int(d_circle / 2), [
                    (int)(dot_color[x][y][0]), (int)(dot_color[x][y][1]), (int)(dot_color[x][y][2])],
                    thickness=-1, lineType=8, shift=0)

    return npimg_target


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


def main():
    global start_time

    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # Load model
    model_path = './model/model_float32.onnx'
    keypoint_score_th = 0.3
    onnx_session = onnxruntime.InferenceSession(model_path)

    window_name = 'Skeleton Sequencer'
    if args.fullscreen:
        cv.namedWindow(window_name, cv.WINDOW_NORMAL)
        cv.setWindowProperty(window_name, cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)

    start_time = time.time()
    while True:
        # camera capture
        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)  # ミラー表示
        debug_image = copy.deepcopy(image)

        # detection
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        # Inference execution
        keypoints, scores = run_inference(
            onnx_session,
            192,
            image,
        )

        # Draw
        debug_image = draw_debug(
            debug_image,
            keypoint_score_th,
            keypoints,
            scores,
        )

        image_ss = skeleton_sequencer(debug_image)

        key = cv.waitKey(1)
        if key == 27:  # ESC
            break

        # cv.imshow(window_name, debug_image)
        cv.imshow(window_name, image_ss)

    for note in note_list:
        midi_output.note_off(note, volume, 2)

    midi_output.close()
    cap.release()
    cv.destroyAllWindows()


def draw_debug(
    image,
    keypoint_score_th,
    keypoints,
    scores,
):
    image_width, image_height = image.shape[1], image.shape[0]
    face_size = int(image_width * 0.3)
    # face_offset = int(image_width * 0.05)
    body_line_size = int(image_width * 0.1)
    # hand_line_size = int(image_width * 0.1)

    debug_image = copy.deepcopy(image)

    connect_list = [
        [0, 1, HUMAN_COLOR],  # nose → left eye
        [0, 2, HUMAN_COLOR],  # nose → right eye
        [1, 3, HUMAN_COLOR],  # left eye → left ear
        [2, 4, HUMAN_COLOR],  # right eye → right ear
        [0, 5, HUMAN_COLOR],  # nose → left shoulder
        [0, 6, HUMAN_COLOR],  # nose → right shoulder
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

    # Draw face
    if scores[0] > keypoint_score_th:
        cv.circle(debug_image, keypoints[0], 5, HUMAN_COLOR, face_size)

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
