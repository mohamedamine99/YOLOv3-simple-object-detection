
import cv2
import os
import time
import imageio

def preprocess_img_for_detection(net, img, size=(320, 320)):
    """
    This function preprocesses an input image for object detection using a specified YOLOv3 or YOLOv3-tiny
    DNN model. The image is resized to the specified size and converted into a blob. The blob is then set
    as the input for the DNN model. The function returns the output of the DNN model after forward pass.

    Parameters:
        net: cv2.dnn_Net object
        YOLOv3 or YOLOv3-tiny DNN model.

        img: numpy.ndarray
        Input image for object detection.

        size: tuple, optional
        Size to which the input image is resized. Default value is (320, 320).

    Returns:
        outputs: numpy.ndarray
        Output of the DNN model after forward pass.


    """
    # Convert the input image into a blob
    blob = cv2.dnn.blobFromImage(img, 1 / 255, size, [0, 0, 0], 1, crop=False)

    # Set the blob as the input for the DNN model
    net.setInput(blob)
    layersNames = net.getLayerNames()

    # Perform forward pass through the DNN model
    output_layers_idx = net.getUnconnectedOutLayers()[0] - 1
    outputNames = [(layersNames[idx - 1]) for idx in net.getUnconnectedOutLayers()]
    # print(outputNames)
    outputs = net.forward(outputNames)

    # Return the output of the DNN model after forward pass
    return outputs


def detectObjects(img, outputs, score_threshold=0.8, NMS_threshold=0.5):
    """
    This function takes an input image and the output of a YOLOv3 or YOLOv3-tiny DNN model after forward pass,
    detects objects in the image and draws bounding boxes around the objects. It also writes the class label and
    confidence score for each object inside the bounding box.

    Parameters:
        img: numpy.ndarray
        Input image for object detection.

        outputs: numpy.ndarray
        Output of the YOLOv3 or YOLOv3-tiny DNN model after forward pass.

        score_threshold: float, optional
            Minimum confidence score required for an object to be considered for detection. Default value is 0.8.

        NMS_threshold: float, optional
            Non-maximum suppression threshold for eliminating overlapping bounding boxes. Default value is 0.5.

        Returns:
            img: numpy.ndarray
            Input image with bounding boxes and class labels drawn around the detected objects.

    """
    # Get the shape of the input image
    hT, wT, cT = img.shape

    # Create empty lists to store the bounding boxes, class IDs and confidence scores for detected objects
    bbox = []
    classIds = []
    confs = []

    # Loop over each output of the DNN model after forward pass
    for output in outputs:
        # Loop over each detection in the output
        for det in output:
            # Extract the class ID, confidence score and bounding box coordinates from the detection
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > score_threshold:
                w, h = int(det[2] * wT), int(det[3] * hT)
                x, y = int((det[0] * wT) - w / 2), int((det[1] * hT) - h / 2)
                bbox.append([x, y, w, h])
                classIds.append(classId)
                confs.append(float(confidence))

    # Perform non-maximum suppression to eliminate overlapping bounding boxes
    indices = cv2.dnn.NMSBoxes(bbox, confs, score_threshold, NMS_threshold)

    # Loop over each index in the indices list
    for i in indices:
        # Get the bounding box coordinates, class label and confidence score for the current index
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)
        cv2.putText(img, f'{labels[classIds[i]].upper()} {int(confs[i] * 100)}%',
                    (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Return the input image with bounding boxes and class labels drawn around the detected objects
    return img


def detect_objects_in_videos_directory(net, videos_path, save_dir, size=(416, 416),
                                       score_threshold=0.5, NMS_threshold=0.4):
    """
    Detects objects in videos in the specified directory using the given model and saves the resulting video files
    to the specified directory.

    Args:
    - net: the neural network model to use for object detection
    - videos_path: the path to the directory containing the videos to process
    - save_dir: the path to the directory where the processed videos will be saved
    - size: the size to resize the frames to before passing them to the neural network
    - score_threshold: the confidence threshold below which detected objects will be discarded
    - NMS_threshold: the Non-Maximum Suppression (NMS) threshold for removing overlapping bounding boxes

    Returns:
    - None
    """

    for video_file in os.listdir(videos_path):
        cap = cv2.VideoCapture(os.path.join(videos_path, video_file))

        width = int(cap.get(3))  # get `width`
        height = int(cap.get(4))  # get `height`
        print((width, height))
        save_file = os.path.join(save_dir, video_file[:-4] + ".avi")

        # define an output VideoWriter  object
        out = cv2.VideoWriter(save_file,
                              cv2.VideoWriter_fourcc(*"MJPG"),
                              20, (width, height))

        # Check if the webcam is opened correctly
        if not cap.isOpened():
            print("Error opening video stream or file")

        # Read the video frames
        while cap.isOpened():
            ret, frame = cap.read()

            # If the frame was not read successfully, break the loop
            if not ret:
                print("Error reading frame")
                print((width, height))
                break
            beg = time.time()

            # Capture the video frame
            # by frame
            outputs = preprocess_img_for_detection(net, frame, size)

            # Generate and then overlay the model heatmap for the current frame
            frame = detectObjects(frame, outputs, score_threshold, NMS_threshold)
            # end = time.time()
            # fps = 1/(end - beg)
            # frame = cv2.putText(frame, f"FPS = {fps}", (20,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            # append frame to the video file
            out.write(frame)

            # the 'q' button is set as the
            # quitting button you may use any
            # desired button of your choice

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # After the loop release the cap
        cap.release()
        out.release()


def GIF_from_vid(vid_file, gif_file, fps=20, skip=2):
    """
    Creates a GIF file from a video file.

    Parameters:
        vid_file (str): The path to the video file.
        gif_file (str): The path to save the generated GIF file.
        fps (int, optional): The frames per second to be used in the GIF file. Defaults to 20.
        skip (int, optional): The number of frames to skip in the video file. Defaults to 2.

    Returns:
        None
    """

    # initialize frame counter
    i = 0

    cap = cv2.VideoCapture(vid_file)
    width = int(cap.get(3))  # get `width`
    height = int(cap.get(4))  # get `height`

    # Create a writer object to write the frames to a GIF file
    writer = imageio.get_writer(gif_file, mode='I', fps=fps)

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        print("Error opening video stream or file")

    # Read the video frames
    while cap.isOpened():
        ret, frame = cap.read()

        # If the frame was not read successfully, break the loop
        if not ret:
            print("Error reading frame")
            break

        # Increment the frame counter
        i += 1

        # Skip frames if necessary based on the skip parameter
        if (i % skip == 0):
            continue

        # add current RGB frame to the GIF file
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        writer.append_data(frame)

    # Close the reader and writer objects
    writer.close()
    cap.release()



