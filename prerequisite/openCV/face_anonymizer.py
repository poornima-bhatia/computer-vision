import os
import cv2

import argparse
import mediapipe as mp

def process_img(img , face_detection):
    H, W, _ = img.shape
    img_rgb = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
    out = face_detection.process(img)
    if out.detections:  # Check if detections is not empty
        for detection in out.detections:
            location_data = detection.location_data
            bbox = location_data.relative_bounding_box
            x1, y1, w, h = bbox.xmin, bbox.ymin, bbox.width, bbox.height
            
            x1 = int(x1 * W)
            y1 = int(y1 * H)
            w = int(w * W)
            h = int(h * H)
            
            # img = cv2.blur(img , (10 , 10))
            img[y1:y1+h , x1:x1+w , :] = cv2.blur(img[y1:y1+h , x1:x1+w , :],(10,10))
    return img 

args = argparse.ArgumentParser()
args.add_argument("--mode", default='webcam')
args.add_argument("--filePath", default=None)
args = args.parse_args()

output_dir = './prerequisite/openCV/data/output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Read image
# img_path = "./prerequisite/openCV/data/testImg.png"
# img = cv2.imread(img_path)

# H, W, _ = img.shape

#Face Detection
# mp_face_detection = mp.solutions.face_detection

# with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
#     img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     out = face_detection.process(img_rgb)

#     if out.detections:  # Check if detections is not empty
#         for detection in out.detections:
#             location_data = detection.location_data
#             bbox = location_data.relative_bounding_box
#             x1, y1, w, h = bbox.xmin, bbox.ymin, bbox.width, bbox.height
            
#             x1 = int(x1 * W)
#             y1 = int(y1 * H)
#             w = int(w * W)
#             h = int(h * H)
            
#             img = cv2.rectangle(img, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)

#     # Display the image with detected faces
#     cv2.imshow('Detected Faces', img)
#     cv2.waitKey(0)  # Wait for a key press to close the window
#     cv2.destroyAllWindows()  # Close all OpenCV windows

#Blur Faces
# mp_face_detection = mp.solutions.face_detection

# with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
#     img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     out = face_detection.process(img_rgb)

#     if out.detections:  # Check if detections is not empty
#         for detection in out.detections:
#             location_data = detection.location_data
#             bbox = location_data.relative_bounding_box
#             x1, y1, w, h = bbox.xmin, bbox.ymin, bbox.width, bbox.height
            
#             x1 = int(x1 * W)
#             y1 = int(y1 * H)
#             w = int(w * W)
#             h = int(h * H)
            
#             # img = cv2.blur(img , (10 , 10))
#             img[y1:y1+h , x1:x1+w , :] = cv2.blur(img[y1:y1+h , x1:x1+w , :],(10,10))
#     # Display the image with detected faces
#     cv2.imshow('Blurred Faces', img)
#     cv2.waitKey(0)  # Wait for a key press to close the window
#     cv2.destroyAllWindows()  # Close all OpenCV windows

mp_face_detection = mp.solutions.face_detection
with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
    if args.mode in ["image"]:
        # read image
        img = cv2.imread(args.filePath)
        img = process_img(img , face_detection)
        # save images
        cv2.imwrite(os.path.join(output_dir, 'output.png') , img)
    elif args.mode in ["video"]:
        capture = cv2.VideoCapture(args.filePath)
        ret , frame = capture.read()
        output_video = cv2.VideoWriter(os.path.join(output_dir , 'output.mp4') , cv2.VideoWriter_fourcc(*'MP4V'), 25 , (frame.shape[1] , frame.shape[0]))
        while ret:
            frame = process_img(frame , face_detection)
            output_video.write(frame)
            ret , frame = capture.read()
        capture.release()
        output_video.release()
    elif args.mode in ['webcam']:
        capture = cv2.VideoCapture(0)

        ret, frame = capture.read()
        while ret:
            frame = process_img(frame, face_detection)

            cv2.imshow('frame', frame)
            cv2.waitKey(25)

            ret, frame = capture.read()

        capture.release()