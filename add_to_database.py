import cv2
import os
import numpy as np
import tensorflow as tf
from PIL import Image
import tkinter as tk
from tkinter import filedialog

# ## Object detection imports
# Here are the imports from the object detection module.
from utils import label_map_util
from utils import visualization_utils_only_needed as vis_util

#variables for the detection
min_score_threshold = 0.6
MAX_FACES_TO_DETECT = 1
MARGIN = 20
ready_to_detect = True

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

def num_faces_detected(faces_scores):
    """returns the amount of faces detected based on the amount of scores in the scores list that are bigget than the threshold"""
    num_faces = 0
    for x in range(0,MAX_FACES_TO_DETECT):
        if faces_scores[x] >= min_score_threshold:
          num_faces += 1
    return num_faces   

def webcam_face_recognizer():
    """
    Runs a loop that extracts images from the computer's webcam and determines whether or not
    it contains faces.
    If it contains a face, it will give the option to save the image.
    """
    #create camera window and set input
    cv2.namedWindow("preview")
    choose_directory_image =  cv2.imread("project_images/choose_directory.png")
    cv2.imshow("preview", choose_directory_image)
    #open file explorer to choose database directory for saving the image
    root = tk.Tk()  # open tkinter widget
    root.withdraw() # Close the root window
    path = filedialog.askdirectory()
    if path == None:
        path = '/database_images'
    cam_input = cv2.VideoCapture(0)

    #loop for going frame after frame
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            while cam_input.isOpened():
                
                _, frame = cam_input.read()
                img = np.copy(frame)

                key = cv2.waitKey(25)      
                
                if key == 27: # exit on ESC
                    break
                
                if ready_to_detect == True:
                    img, face_boxes, face_scores = detect_faces_in_frame(img, detection_graph, category_index, sess)  #detect faces in the frame
                    num_detected_faces = num_faces_detected(face_scores) #checks if we had detected any face
                    if num_detected_faces > 0:
                        key = cv2.waitKey(15) #variable to check if enter key is pressed
                        if key == 13: # stop on enter key
                            key = 0
                            vis_util.add_text_on_image(img, "If you like the image press enter to save. \nEsc if otherwise.")
                            while (key != 27 and key != 13):
                                cv2.imshow("preview", img) #show the video input with the detections on screen
                                key = cv2.waitKey(15) #variable to check if enter key is pressed
                                if key == 13:
                                    img = np.copy(frame)
                                    name_done = False
                                    name = ""
                                    while(name_done == False): #writing the name of the person
                                        img_with_name = np.copy(img)
                                        img_with_name = vis_util.add_text_on_image(img_with_name, "Enter a name:" + name)
                                        cv2.imshow("preview", img_with_name)
                                        key = cv2.waitKey(15) 
                                        if(key == 27):
                                            break
                                        elif(key == 13):
                                            name_done = True
                                        elif(key == 8):
                                            if(name != ""):
                                                name = name[:-1]   
                                        elif(key == -1):
                                            pass         
                                        else:
                                            name += chr(key) 
                                    if name_done == True:    
                                        #crop the face from the image
                                        cropped_img = frame[int(face_boxes[0][0]) - MARGIN:int(face_boxes[0][2])+MARGIN, 
                                        int(face_boxes[0][1])-MARGIN:int(face_boxes[0][3])+MARGIN] 
                                        cropped_img = cv2.resize(cropped_img, (96, 96)) #resize to 96*96 so it will match FaceNet's input and be small
                                        cv2.imwrite(str(path) + "/" + name  +".jpg", cropped_img ) #save image
                                    else:
                                        break        
                  
                cv2.imshow("preview", img) #show the video input with the detections on screen

            cv2.destroyWindow("preview")


#code start
detection_graph, category_index = prepare_face_detection_model() #initialize
webcam_face_recognizer() #start detecting            