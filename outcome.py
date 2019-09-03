import os  
import cv2
import time

def show_opened_door():
    """Shows the door opened"""
    opened_door = cv2.imread("project_images/opened_door.png")
    cv2.imshow("outcome", opened_door)
    timer = 0
    start_time = time.time()
    #after starting, the door is opened for 3 seconds
    while(timer <= 3):
        key = cv2.waitKey (25)
        if key == 27:
            exit()
        timer = time.time() - start_time
        

def show_door():
    """Draws the door in another window"""
    cv2.namedWindow("outcome") #create window
    closed_door = cv2.imread("project_images/closed_door.png") #load image
    while True:
        cv2.imshow("outcome", closed_door) #show the door closed
        #check if person recognized
        is_recognized_file = open('varThread\\recognized.check', 'r+')
        is_recognized = is_recognized_file.read(1)
        if is_recognized == '1':
            #set that person has been recognized to not and show the open door
            show_opened_door()
            is_recognized_file.seek(0,0)
            is_recognized_file.write('0')
            is_recognized_file.close()
        else:
            is_recognized_file.close()    
            #check if need to quit
            is_quit_file = open("varThread\\is_quit.txt", "r+") 
            if int(is_quit_file.read(1)) == 1:
                is_quit_file.seek(0,0)
                is_quit_file.write('0')
                is_quit_file.close()
                exit()
        key = cv2.waitKey (25)
        if key == 27: #exit on hitting Esc
            exit()


#code starts
#making sure that the recognized flag in 0 in the beginning so the door is closed.
is_recognized_file = open('varThread\\recognized.check', 'w')
is_recognized_file.write('0')
is_recognized_file.close()
show_door() #start showing the door