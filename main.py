import os                                                                       
from multiprocessing import Pool
from multiprocessing import Process       
import cv2        
import time                                                                              

#tuple of the names of the files to run in the processes.                                                                               
processes = ('detect.py', 'outcome.py', 'recognize.py', 'add_to_database.py')                                    
                                                                                                                             
def run_process(process):   
    """Runs the python file"""                                                          
    os.system('python {}'.format(process))                                       
                                                                                
def get_key_pressed():
    """Return the key pressed by the user if it's 1,2 or Esc"""
    key = 0
    while (key != 27 and key != ord('1') and key != ord('2')):
        key = cv2.waitKey(10)
    return key    

if __name__ == "__main__":

    recognize_runs = False
    
    #initalize the flags in the files to be: don't exit 
    is_quit_file = open("varThread\\is_quit.txt", "w") 
    is_quit_file.write('0')
    is_quit_file.close()
    
    #run recognize to minimize load waiting time                
    recognize_proc = Process(target=run_process, args=(processes[2:3]))             
                     
    start_image = cv2.imread("project_images/start_image.png") #load image 
    
    while True:
        cv2.namedWindow("main") #create window "main"
        cv2.imshow("main", start_image) #show image 
        key = get_key_pressed()  
        if key == ord('1'): 
            cv2.destroyWindow("main") #close the window
            #run the first 2 files in *processes* tuple
            #in different processes so they run at the same time.
            if recognize_runs:
                pool = Pool(processes=2)                                                        
                pool.map(run_process, processes[:2])  
            else:
                recognize_proc.start() 
                recognize_runs = True
                pool = Pool(processes=2)                                                        
                pool.map(run_process, processes[:2])          
        elif key == ord('2'):
            cv2.destroyWindow("main")#close the window
            #run the add_to_database.py file in different process.
            if not recognize_runs:
                recognize_proc.start()     
                recognize_runs = True
            pool = Pool(processes=1)
            pool.map(run_process, processes[3:]) 
        elif key == 27: #Esc exit
            if recognize_runs:
                recognize_proc.terminate()
            is_quit_file = open("varThread\\is_quit.txt", "w") 
            is_quit_file.write('1')
            is_quit_file.close()
            break 
        
           