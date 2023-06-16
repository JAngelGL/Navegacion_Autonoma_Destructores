#!/usr/bin/env python3
import os
import sys
import rospy
import cv2
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import Image

sys.path.append(sys.path[0]+'/yolov5')
from detection import detector 

class DetectStop:
    def __init__(self):
        rospy.init_node('detect_stop_node', anonymous=True)
        ############################### SUBCRIBERS ##################################
        self.image_sub = rospy.Subscriber("/video_source/raw", Image, self.imgmsg_to_cv2)
        

        ############################### PUBLISHERSS ##################################
        self.image_pub = rospy.Publisher("/red_neuronal", Image, queue_size=1)
        
        self.pub_semaforo = rospy.Publisher("/semaforo", String, queue_size=1)
        self.pub_signal = rospy.Publisher("/signal", String, queue_size=1)
        
        self.image_received = 0 #Flag to indicate that we have already received an image

         # Load model
        weights = "/home/puzzlebot/catkin_ws/src/red_neuronal/src/yolo.pt" # adjust the path to your system! 
        print("Loading network")
        
        self.yolo = detector(weights, 0.5)

    def imgmsg_to_cv2(self, msg):
        self.image = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
        self.image_received = 1
        return self.image
        
    def cv2_to_imgsmg(self, cv_image, encoding= "bgr8"):
        height, width = cv_image.shape[:2]
        
        img_array = np.array(cv_image, dtype = np.uint8).flatten()
        
        img_msg = Image()
        img_msg.height = height
        img_msg.width = width
        img_msg.encoding = encoding
        img_msg.is_bigendian = False
        img_msg.step = 3 * len(cv_image[0])
        img_msg.data = img_array.tobytes()
        
        return img_msg
        
        
    def run(self):
        rate = rospy.Rate(50)  # 50 Hz
        while not rospy.is_shutdown():
            t = rospy.get_time()
            if self.image_received:   
                pred = self.yolo.detect(self.image) # access the library to change thresholds and other hyperparameters 
                for i, det in enumerate(pred):
                    #print(det) # the output of this call is a table with N detections, the 4 elements of a bounding box, the confidence and the class ID
                    print(rospy.get_time()-t)
                    size = det.size()
                    size_a = np.asarray(size)
                    if size_a[0] == 0:
                        print("Tensor is empty")
                    else:
                        parte1 = det[0]
                        if size_a[0] > 1:
                            parte2 = det[1]
                        else:
                            parte2 = None
                        self.enclose_object(parte1, parte2)
        rate.sleep()

    def enclose_object(self, parte1, parte2):
        clases = {
            0 : 'give_way',
            1 : 'green',
            2 :  'left',
            3 : 'red',
            4 : 'right',
            5 : 'stop',
            6 : 'straight',
            7 : 'work',
            8 : 'yellow'
        }
        
        parte1_np = np.asarray(parte1)
        parte2_np = np.asarray(parte2)
        #print(parte1)
        #print(parte2)
        
        x1, y1, w1, h1, pred1, class_id1 = parte1_np
        
        r1_x1 = w1 - x1
        print("r1_x1: " + str(r1_x1) + clases[class_id1] + "pred" + str(pred1))
        r1_y1 = h1 - y1        
        print("r1_y1: " + str(r1_y1) + clases[class_id1] + "pred" + str(pred1))
        
        #Caution signals and movement signals
        #Signal
        #Give way distancia de deteccion de 30 a 26 cm
        if 14.0 <= r1_x1 <= 18.4 and 27.0 <= r1_y1 <= 34.8 and class_id1 == 0 and pred1 > 0.60:
            cv2.rectangle(self.image, (int(x1), int(y1)), (int(w1), int(h1)), (0, 0, 255), 1)
            cv2.putText(self.image, clases[class_id1], (int(x1), int(y1 - 2)), cv2.FONT_HERSHEY_SIMPLEX,0.2, (0, 0, 255), 1)
            self.pub_signal.publish(clases[class_id1])
            
            #Left distancia de deteccion de 28 a 24 cm
        elif 14.0 <= r1_x1 <= 19.1 and 31.2 <= r1_y1 <= 42.5 and class_id1 == 2 and pred1 > 0.81:
            cv2.rectangle(self.image, (int(x1), int(y1)), (int(w1), int(h1)), (0, 0, 255), 1)
            cv2.putText(self.image, clases[class_id1], (int(x1), int(y1 - 2)), cv2.FONT_HERSHEY_SIMPLEX,0.2, (0, 0, 255), 1)
            self.pub_signal.publish(clases[class_id1])
            
            #Right distancia de deteccion de 28 a 25 cm 
        elif 13.5 <= r1_x1 <= 19.5 and 30.5 <= r1_y1 <= 38.3 and class_id1 == 4 and pred1 > 0.35:
            cv2.rectangle(self.image, (int(x1), int(y1)), (int(w1), int(h1)), (0, 0, 255), 1)
            cv2.putText(self.image, clases[class_id1], (int(x1), int(y1 - 2)), cv2.FONT_HERSHEY_SIMPLEX,0.2, (0, 0, 255), 1)
            self.pub_signal.publish(clases[class_id1])
            
            #Stop distancia de deteccion de 23 a 20 cm 
        elif 14.0 <= r1_x1 <= 20.0 and 30.5 <= r1_y1 <= 40.1 and class_id1 == 5 and pred1 > 0.89:
            cv2.rectangle(self.image, (int(x1), int(y1)), (int(w1), int(h1)), (0, 0, 255), 1)
            cv2.putText(self.image, clases[class_id1], (int(x1), int(y1 - 2)), cv2.FONT_HERSHEY_SIMPLEX,0.2, (0, 0, 255), 1)
            self.pub_signal.publish(clases[class_id1])
            
            #Straight distancia de deteccion de 28 a 24.5 aprox cm 
        elif 14.4 <= r1_x1 <= 19.0 and 27.0 <= r1_y1 <= 39.5 and class_id1 == 6 and pred1 > 0.80:
            cv2.rectangle(self.image, (int(x1), int(y1)), (int(w1), int(h1)), (0, 0, 255), 1)
            cv2.putText(self.image, clases[class_id1], (int(x1), int(y1 - 2)), cv2.FONT_HERSHEY_SIMPLEX,0.2, (0, 0, 255), 1)
            self.pub_signal.publish(clases[class_id1])
            
            #Work distancia de deteccion de 30 a 26 cm
        elif 2.0 <= r1_x1 <= 19.5 and  26.8 <= r1_y1 <= 33.2 and class_id1 == 7 and pred1 > 0.75:
            cv2.rectangle(self.image, (int(x1), int(y1)), (int(w1), int(h1)), (0, 0, 255), 1)
            cv2.putText(self.image, clases[class_id1], (int(x1), int(y1 - 2)), cv2.FONT_HERSHEY_SIMPLEX,0.2, (0, 0, 255), 1)
            self.pub_signal.publish(clases[class_id1])
            
        #Semaforo
        # Verde distancia de deteccion de 34 a 26 cm
        elif 2.5 <= r1_x1 <= 20.9 and 49.7 <= r1_y1 <= 76.4 and class_id1 == 1 and pred1 > 0.70:
            cv2.rectangle(self.image, (int(x1), int(y1)), (int(w1), int(h1)), (255, 0, 0), 1)
            cv2.putText(self.image, clases[class_id1], (int(x1), int(y1 - 2)), cv2.FONT_HERSHEY_SIMPLEX,0.2, (255, 0, 0), 1)
            self.pub_semaforo.publish(clases[class_id1])
        
        #Rojo distancia de deteccion de 30 a 26 cm
        elif 2.5 <= r1_x1 <= 20.9 and 33.0 <= r1_y1 <= 62.5 and class_id1 == 3 and pred1 > 0.38:
            cv2.rectangle(self.image, (int(x1), int(y1)), (int(w1), int(h1)), (255, 0, 0), 1)
            cv2.putText(self.image, clases[class_id1], (int(x1), int(y1 - 2)), cv2.FONT_HERSHEY_SIMPLEX,0.2, (255, 0, 0), 1)
            self.pub_semaforo.publish(clases[class_id1])
            
        #Amarillo distancia de deteccion de 30 a 26 cm
        elif 2.5 <= r1_x1 <= 20.9 and 33.0 <= r1_y1 <= 69.0 and class_id1 == 8 and pred1 > 0.57:
            cv2.rectangle(self.image, (int(x1), int(y1)), (int(w1), int(h1)), (255, 0, 0), 1)
            cv2.putText(self.image, clases[class_id1], (int(x1), int(y1 - 2)), cv2.FONT_HERSHEY_SIMPLEX,0.2, (255, 0, 0), 1)
            self.pub_semaforo.publish(clases[class_id1])

        else:
            parte1_np = None
                
        if np.all(parte2_np) == None:
            pass
        else:
            x2, y2, w2, h2, pred2, class_id2 = parte2_np  
            
            r1_x2 = w2 - x2
            r1_y2 = h2 - y2
            #Caution signals and movement signals
            #Signal
            #Give way distancia de deteccion de 30 a 26 cm
            if 14.0 <= r1_x2 <= 18.4 and 27.0 <= r1_y2 <= 34.8 and class_id2 == 0 and pred2 > 0.60:
                cv2.rectangle(self.image, (int(x2), int(y2)), (int(w2), int(h2)), (0, 0, 255), 1)
                cv2.putText(self.image, clases[class_id2], (int(x2), int(y2 - 2)), cv2.FONT_HERSHEY_SIMPLEX,0.2, (0, 0, 255), 1)
                self.pub_signal.publish(clases[class_id2])
                
                #Left distancia de deteccion de 28 a 24 cm
            elif 14.0 <= r1_x2 <= 19.1 and 31.2 <= r1_y2 <= 42.5 and class_id2 == 2 and pred2 > 0.81:
                cv2.rectangle(self.image, (int(x2), int(y2)), (int(w2), int(h2)), (0, 0, 255), 1)
                cv2.putText(self.image, clases[class_id2], (int(x2), int(y2 - 2)), cv2.FONT_HERSHEY_SIMPLEX,0.2, (0, 0, 255), 1)
                self.pub_signal.publish(clases[class_id2])
                
                #Right distancia de deteccion de 28 a 25 cm 
            elif 13.5 <= r1_x2 <= 19.5 and 30.5 <= r1_y2 <= 38.3 and class_id2 == 4 and pred2 > 0.35:
                cv2.rectangle(self.image, (int(x2), int(y2)), (int(w2), int(h2)), (0, 0, 255), 1)
                cv2.putText(self.image, clases[class_id2], (int(x2), int(y2 - 2)), cv2.FONT_HERSHEY_SIMPLEX,0.2, (0, 0, 255), 1)
                self.pub_signal.publish(clases[class_id2])
                
                #Stop distancia de deteccion de 23 a 20 cm 
            elif 14.0 <= r1_x2 <= 20.0 and 35.4 <= r1_y2 <= 40.1 and class_id2 == 5 and pred2 > 0.89:
                cv2.rectangle(self.image, (int(x2), int(y2)), (int(w2), int(h2)), (0, 0, 255), 1)
                cv2.putText(self.image, clases[class_id2], (int(x2), int(y2 - 2)), cv2.FONT_HERSHEY_SIMPLEX,0.2, (0, 0, 255), 1)
                self.pub_signal.publish(clases[class_id2])
                
                #Straight distancia de deteccion de 28 a 24.5 aprox cm 
            elif 14.4 <= r1_x2 <= 19.0 and 34.2 <= r1_y2 <= 39.5 and class_id2 == 6 and pred2 > 0.80:
                cv2.rectangle(self.image, (int(x2), int(y2)), (int(w2), int(h2)), (0, 0, 255), 1)
                cv2.putText(self.image, clases[class_id2], (int(x2), int(y2 - 2)), cv2.FONT_HERSHEY_SIMPLEX,0.2, (0, 0, 255), 1)
                self.pub_signal.publish(clases[class_id2])
                
                #Work distancia de deteccion de 30 a 26 cm
            elif 2.0 <= r1_x2 <= 19.5 and  26.8 <= r1_y2 <= 33.2 and class_id2 == 7 and pred2 > 0.80:
                cv2.rectangle(self.image, (int(x2), int(y2)), (int(w2), int(h2)), (0, 0, 255), 1)
                cv2.putText(self.image, clases[class_id2], (int(x2), int(y2 - 2)), cv2.FONT_HERSHEY_SIMPLEX,0.2, (0, 0, 255), 1)
                self.pub_signal.publish(clases[class_id2])
                
            #Semaforo
            # Verde distancia de deteccion de 34 a 26 cm
            elif 2.5 <= r1_x2 <= 20.9 and 33.0 <= r1_y2 <= 76.4 and class_id2 == 1 and pred2 > 0.38:
                cv2.rectangle(self.image, (int(x2), int(y2)), (int(w2), int(h2)), (255, 0, 0), 1)
                cv2.putText(self.image, clases[class_id2], (int(x2), int(y2 - 2)), cv2.FONT_HERSHEY_SIMPLEX,0.2, (255, 0, 0), 1)
                self.pub_semaforo.publish(clases[class_id2])
            
            #Rojo distancia de deteccion de 30 a 26 cm
            elif 2.5 <= r1_x2 <= 15.9 and 33.0 <= r1_y2 <= 62.5 and class_id2 == 3 and pred2 > 0.38:
                cv2.rectangle(self.image, (int(x2), int(y2)), (int(w2), int(h2)), (255, 0, 0), 1)
                cv2.putText(self.image, clases[class_id2], (int(x2), int(y2 - 2)), cv2.FONT_HERSHEY_SIMPLEX,0.2, (255, 0, 0), 1)
                self.pub_semaforo.publish(clases[class_id2])
                
            #Amarillo distancia de deteccion de 30 a 26 cm
            elif 2.5 <= r1_x2 <= 16.7 and 33.0 <= r1_y2 <= 69.0 and class_id2 == 8 and pred2 > 0.57:
                cv2.rectangle(self.image, (int(x2), int(y2)), (int(w2), int(h2)), (255, 0, 0), 1)
                cv2.putText(self.image, clases[class_id2], (int(x2), int(y2 - 2)), cv2.FONT_HERSHEY_SIMPLEX,0.2, (255, 0, 0), 1)
                self.pub_semaforo.publish(clases[class_id2])

            else:
                parte1_np = None
                   
        res = self.cv2_to_imgsmg(self.image)
        self.image_pub.publish(res)
         

if __name__ == "__main__":
    try:
        detect_stop = DetectStop()
        detect_stop.run()
    except rospy.ROSInterruptException:
        pass
