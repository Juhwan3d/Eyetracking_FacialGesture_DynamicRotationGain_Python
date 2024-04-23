import mediapipe as mp
from mediapipe.tasks import python
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

import cv2
import numpy as np
import time

model_path = 'C:/Users/juhwan/SCD/models/face_landmarker.task'

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(255, 235, 0))

RESULT = None


def draw_landmarks_live(frame):
    face_landmarks_list = RESULT.face_landmarks
    annotated_frame = np.copy(frame)

    for idx, lms in enumerate(face_landmarks_list):
        face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        face_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in lms
        ])

        solutions.drawing_utils.draw_landmarks(
            image=annotated_frame,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=drawing_spec,
            connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_tesselation_style())
        solutions.drawing_utils.draw_landmarks(
            image=annotated_frame,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=drawing_spec,
            connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_contours_style())
        solutions.drawing_utils.draw_landmarks(
            image=annotated_frame,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=drawing_spec,
            connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_iris_connections_style())

    return annotated_frame


# Create a face landmarker instance with the live stream mode:
def result_callback(result: FaceLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global RESULT
    RESULT = result
    if not result.face_landmarks:
        return

    landmarks = result.face_landmarks[0]

    img_h, img_w, img_c = output_image.height, output_image.width, output_image.channels
    face_2d, face_3d = [], []

    for idx, lm in enumerate(landmarks):
        if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
            x, y = int(lm.x * img_w), int(lm.y * img_h)

            face_2d.append([x, y])
            face_3d.append([x, y, lm.z])

    # Get 2d Coord
    face_2d = np.array(face_2d, dtype=np.float64)
    face_3d = np.array(face_3d, dtype=np.float64)
    focal_length = 1 * img_w

    cam_matrix = np.array([[focal_length, 0, img_h / 2],
                           [0, focal_length, img_w / 2],
                           [0, 0, 1]])
    distortion_matrix = np.zeros((4, 1), dtype=np.float64)

    success, rotation_vec, translation_vec = cv2.solvePnP(objectPoints=face_3d, imagePoints=face_2d,
                                                          cameraMatrix=cam_matrix, distCoeffs=distortion_matrix)
    # Getting rotational of face
    rmat, jac = cv2.Rodrigues(rotation_vec)
    angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

    x = np.degrees(angles[0])
    y = np.degrees(angles[1])
    z = np.degrees(angles[2])

    # Here based on axis rot angle is calculated
    # if y < -10:
    #     print("Looking Right")
    # elif y > 10:
    #     print("Looking Left")
    # elif x < -10:
    #     print("Looking Down")
    # elif x > 10:
    #     print("Looking Up")
    # else:
    #     print("Looking Forward")
    # print(f"x: {x}, y: {y}, z: {z}")


options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_faces=1,
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=True,
    result_callback=result_callback)


def main():
    with FaceLandmarker.create_from_options(options) as landmarker:
        # The landmarker is initialized. Use it here.
        # ...
        cap = cv2.VideoCapture(0)
        frame_count = 0
        fps = cap.get(cv2.CAP_PROP_FPS)

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            timestamp_ms = int(round(time.time() * 1000))
            frame = mp_image.numpy_view()
            # Calculate the timestamp of the current frame
            # frame_count += 1
            # timestamp_ms = int(frame_count / fps * 1000)

            landmarker.detect_async(mp_image, timestamp_ms)

            if not isinstance(RESULT, type(None)):
                if not RESULT.face_landmarks:
                    print("Where your face?")
                else:
                    frame = draw_landmarks_live(frame)
            else:
                print(f"RESULT Type: {type(RESULT)}")

            cv2.imshow('SCD', cv2.flip(frame, 1))
            if cv2.waitKey(5) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()
    return


if __name__ == '__main__':
    main()
