from keras import backend as K
K.set_image_data_format('channels_first')
import time
import cv2
import os
import glob
import numpy as np
import tensorflow as tf
from fr_utils import *
from inception_blocks_v2 import *

#variables for the model
PADDING = 50
MAX_FACES_TO_DETECT = 10
MIN_DIST_THRESHOLD = 0.7

def triplet_loss(y_true, y_pred, alpha = 0.3):
    """
    Implementation of the triplet loss as defined by formula (3)
    
    Arguments:
    y_pred -- python list containing three objects:
            anchor -- the encodings for the anchor images, of shape (None, 128)
            positive -- the encodings for the positive images, of shape (None, 128)
            negative -- the encodings for the negative images, of shape (None, 128)
    
    Returns:
    loss -- real number, value of the loss
    """
    
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
    
    # Step 1: Compute the (encoding) distance between the anchor and the positive, you will need to sum over axis=-1
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), axis=-1)
    # Step 2: Compute the (encoding) distance between the anchor and the negative, you will need to sum over axis=-1
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis=-1)
    # Step 3: subtract the two previous distances and add alpha(learning variable).
    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
    # Step 4: Take the maximum of basic_loss and 0.0. Sum over the training examples.
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0))
    
    return loss

def prepare_database():
    database = {}

    # load all the images of individuals to recognize into the database
    for file in glob.glob("database_images/*"):
        identity = os.path.splitext(os.path.basename(file))[0] #
        database[identity] = img_path_to_encoding(file, FRmodel) #encode to embedding
    print("finished data")
    return database


def process_frame(img, frame, boxes):
    """
    Determine whether the current frame contains the faces of people from our database
    """
    faces = boxes
    # Loop through all the faces detected and determine whether or not they are in the database
    identities = []

    for (y, x, h, w) in faces: #add padding
        x1 = int(x-PADDING)
        y1 = int(y-PADDING)
        x2 = int(w+PADDING)
        y2 = int(h+PADDING)

        identity = find_identity(frame, x1, y1, x2, y2)

        if identity is not None:
            identities.append(identity)

    if identities != []:
        welcome_users(identities)

    #set flag to not recognize until face detected again
    ready_to_recognize_file = open("varThread\\ready_recognize.txt", "r+")
    ready_to_recognize_file.seek(0,0)
    ready_to_recognize_file.write('0')
    ready_to_recognize_file.close()
    
def find_identity(frame, x1, y1, x2, y2):
    """
    Determine whether the face contained within the bounding box exists in our database

    x1,y1_____________
    |                 |
    |                 |
    |_________________x2,y2

    """
    height, width, channels = frame.shape
    # The padding is necessary since the OpenCV face detector creates the bounding box around the face and not the head
    part_image = frame[max(0, y1):min(height, y2), max(0, x1):min(width, x2)]
    
    return who_is_it(part_image, database, FRmodel)

def who_is_it(image, database, model):
    """
    Arguments:
    image -- array that represents the image
    database -- database containing image encodings along with the name of the person on the image
    model -- FaceNet model
    
    Returns:
    min_dist -- the minimum distance between image_path encoding and the encodings from the database
    identity -- string, the name prediction for the person on image_path
    """
    encoding = img_to_encoding(image, model) #encodes the current face image 
    
    min_dist = 100
    identity = None
    
    # Loop over the database dictionary's names and encodings.
    for (name, db_enc) in database.items():
        
        # Compute L2 distance between the target "encoding" and the current "emb" from the database.
        dist = np.linalg.norm(db_enc - encoding)

        print('distance for %s is %s' %(name, dist))

        # If this distance is less than the min_dist, then set min_dist to dist, and identity to name
        if dist < min_dist:
            min_dist = dist
            identity = name
    
    if min_dist > MIN_DIST_THRESHOLD: #check if distance is bigger than the threshold
        return None
    else:
        return str(identity)

def welcome_users(identities):
    """ Create file that is used for checking if person is recognized.
        Prints the name of the recognized person. 
    """
    welcome_message = 'Hello'
    is_recognized_file = open('varThread\\recognized.check', 'w')
    is_recognized_file.write('1')
    is_recognized_file.close()
    if len(identities) == 1:
        welcome_message += '%s!' % identities[0]
        print(welcome_message)
    else:
        for identity_id in range(len(identities)-1):
            welcome_message += '%s, ' % identities[identity_id]
        welcome_message += 'and %s, ' % identities[-1]
        welcome_message += '!'
        print(welcome_message)

def is_ready_to_recognize():
    """Checks file and return True if flag 1 and False when flas is 0"""
    time.sleep(0.01)
    ready_to_recognize_file = open("varThread\\ready_recognize.txt", "r")
    if int(ready_to_recognize_file.read(1)) == 1:
        ready_to_recognize_file.close()
        is_recognized_file = open('varThread\\recognized.check', 'r')
        is_recognized = is_recognized_file.read(1)
        if is_recognized == '0':
            is_recognized_file.close()
            return True    
        else:
            is_recognized_file.close()
            return False    
    else:
        ready_to_recognize_file.close()
        return False    


#code starts here
#initalizing model and database
FRmodel = faceRecoModel(input_shape=(3, 96, 96))
print("shaped_model")
FRmodel.compile(optimizer = 'adam', loss = triplet_loss, metrics = ['accuracy'])
print("compiled model")
load_weights_from_FaceNet(FRmodel)
print("loaded FaceNet model")
database = prepare_database()

do_not_quit = True        

while(do_not_quit == True):
    if is_ready_to_recognize(): #check if ready to recognize
        #load all the variables from the files
        recognize_variables = np.load("varThread\\vars_to_pass.npz")
        img = recognize_variables['img']
        frame = np.copy(img)
        boxes = recognize_variables['face_boxes']
        process_frame(img, frame, boxes)
    is_quit_file = open("varThread\\is_quit.txt", "r") 
    if int(is_quit_file.read(1)) == 1: #check if ready to quit
        is_quit_file.close()
        do_not_quit = False
        print("quiting recognition")


