import cv2
import os
import numpy as np
import tensorflow as tf
from PIL import Image

# ## Object detection imports
# Here are the imports from the object detection module.
from utils import label_map_util
from utils import visualization_utils_only_needed as vis_util

#variables for the detection
min_score_threshold = 0.6
MAX_FACES_TO_DETECT = 1
ready_to_detect = True

def num_faces_detected(faces_scores):
    """returns the amount of faces detected based on the amount of scores in the scores list that are bigget than the threshold"""
    num_faces = 0
    for x in range(0,MAX_FACES_TO_DETECT):
        if faces_scores[x] >= min_score_threshold:
          num_faces += 1
    return num_faces   

def prepare_face_detection_model():
    """loads the detection model I trained"""
    # What model is going to be used.
    MODEL_NAME = 'face_graph'

    # Path to frozen detection graph. This is the actual model that is used for the object(face) detection.
    PATH_TO_MODEL = MODEL_NAME + '/frozen_inference_graph.pb'

    # List of the labels that the model can identify (notains 1 label - face).
    PATH_TO_LABELS = os.path.join('training', 'face-detection.pbtxt')

    #amount of classes the model can identify (1 which is face).
    NUM_CLASSES = 1

    #loading the frozen model (weights saved) into memory
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_MODEL, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    #map of all the labels (1 - face)
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    #create list of dictionaries with 'id' and 'name' keys, for example: {'0' : 'face'}
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    #create dictionary with 'id' and category to this id
    category_index = label_map_util.create_category_index(categories)

    print("detection ready")
    return detection_graph, category_index

def detect_faces_in_frame(img, detection_graph, category_index, sess):
    """detects faces in the frame and returns the boxes in the face's location"""
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    img_np_expanded = np.expand_dims(img, axis=0)
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # Each box represents a part of the image where a particular object was detected.
    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    # Actual detection.
    (boxes, scores, classes, num_detections) = sess.run([boxes, scores, classes, num_detections],
    feed_dict={image_tensor: img_np_expanded})
    
    #make the detection bounding box square because facenet requires, 
    # and adding padding for extra safety(so all face will be in the box)
    image_pil = Image.fromarray(np.uint8(img)).convert('RGB')
    im_width, im_height = image_pil.size
    for box in boxes:
        box[0][0] = box[0][0]*im_height
        box[0][1] = box[0][1]*im_width
        box[0][2] = box[0][2]*im_height
        box[0][3] = box[0][3]*im_width
        deltaY = abs(box[0][0] - box[0][2])
        deltaX = abs(box[0][1] - box[0][3])     
        diff = deltaY - deltaX
        divided_diff = diff/2
        box[0][1] -= divided_diff
        box[0][3] += divided_diff
        
    #visualize the bounding boxes on the frame
    return vis_util.visualize_boxes_and_labels_on_image_array(
    img,
    np.squeeze(boxes),
    np.squeeze(classes).astype(np.int32),
    np.squeeze(scores),
    category_index,
    use_normalized_coordinates=False,
    skip_labels=True,
    skip_scores=True,
    line_thickness=8,
    min_score_thresh=min_score_threshold), boxes[0], scores[0]

def webcam_face_recognizer():
    """
    Runs a loop that extracts images from the computer's webcam and checks if it contains a face.
    If it contains a face, it will update the flags in the files so that the recognition could start
    """
    global ready_to_detect
    #create camera window and set input
    cv2.namedWindow("preview")
    cam_input = cv2.VideoCapture(0)

    #loop for going frame after frame
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            while cam_input.isOpened():
                
                _, frame = cam_input.read()
                img = np.copy(frame)

                if ready_to_detect == True:
                    img, face_boxes, face_scores = detect_faces_in_frame(img, detection_graph, category_index, sess)  #detect faces in the frame
                    ready_to_recognize_file = open("varThread\\ready_recognize.txt", "r") #checks if we even need to check whether faces detected, 
                                                                                          #as if it's recognizing we do not need to send the information
                    if int(ready_to_recognize_file.read(1)) == 0: #not recognizing
                        ready_to_recognize_file.close()
                        num_detected_faces = num_faces_detected(face_scores) #how many faces detected
                        if num_detected_faces > 0:
                            np.savez('varThread\\vars_to_pass.npz', img = frame, face_boxes = face_boxes[:num_detected_faces]) #save the image array in file
                            ready_to_recognize_file = open("varThread\\ready_recognize.txt", "r+") #make the ready to recognize 'flag' true - 1
                            ready_to_recognize_file.seek(0, 0)
                            ready_to_recognize_file.write("1")
                            ready_to_recognize_file.close()
                    ready_to_recognize_file.close()        
                key = cv2.waitKey(10) #variable to check if exit key is pressed
                cv2.imshow("preview", img) #show the video input with the detections on screen
                
                if key == 27: # exit on ESC
                    is_quit_file = open("varThread\\is_quit.txt", "w") 
                    is_quit_file.write('1')
                    is_quit_file.close()
                    break

            cv2.destroyWindow("preview")


#code starts
#initalize the flags in the files to be: don't recognize 
ready_to_recognize_file = open("varThread\\ready_recognize.txt", "w")
ready_to_recognize_file.write("0")
ready_to_recognize_file.close()

detection_graph, category_index = prepare_face_detection_model() #initialize model
webcam_face_recognizer() #start detection            