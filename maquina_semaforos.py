#!/usr/bin/env python

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Twist
from std_msgs.msg import String

class line_follower():
    def __init__(self):
        rospy.on_shutdown(self.shutdown)
        rate = rospy.Rate(50)
        self.cv_bridge = CvBridge()

        ##PUBLICADORES
        self.image_sub = rospy.Subscriber('/video_source/raw', Image, self.image_callback)
        rospy.Subscriber('/signal', String, self.signal_callback)
        rospy.Subscriber('/semaforo', String, self.semaforo_callback)

        ##SUSCRIPTOR
        self.robot_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.twist_msg = Twist()

        self.signal = "start"
        self.semaforo = "null"

        #Marco de Referencia
        self.x = 40  # Coordenada x del borde izquierdo
        self.y = 150 # Coordenada y del borde superior
        self.width = 80
        self.height = 45 

        self.linear_vel = 0.067 # Velocidad lineal del robot
        

        self.flagdetection=False 
        self.flagactivado=True


        self.flagdirection=True #supone que hay una senal de izquierda
        
        # Controlador PID
        self.u_w = [0.0, 0.0]
        self.e_w = [0.0, 0.0, 0.0]

        self.kp_w = 0.024
        self.ki_w = 0.0052
        self.kd_w = 0.00051

        self.w_max = 1.5
        
        self.delta_t = 1.0 / 50.0

        self.image_received_flag = 0

        self.K1_w = self.kp_w + self.delta_t * self.ki_w + self.kd_w / self.delta_t
        self.K2_w = - self.kp_w - 2.0 * self.kd_w / self.delta_t
        self.K3_w = self.kd_w / self.delta_t
        
        self.kernel = np.ones((5, 5), np.uint8)

        while not rospy.is_shutdown() :
            if (self.image_received_flag and self.flagactivado== True) :
                self.image_received_flag = 0
                cv_image = self.frame

                self.roi = cv_image[self.y:self.y+self.height, self.x:self.x+self.width]

                self.roi = cv2.GaussianBlur(self.roi, (15, 15), 0)
                eroded = cv2.erode(self.roi, self.kernel, iterations = 1)
                dilated = cv2.erode(eroded, self.kernel, iterations = 1)

                gray_roi = cv2.cvtColor(dilated, cv2.COLOR_BGR2GRAY)

                _, binary_roi = cv2.threshold(gray_roi, 100, 125, cv2.THRESH_BINARY_INV)


                # Buscar los contornos en la imagen binaria
                self.contours, _ = cv2.findContours(binary_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                self.estado = self.signal
                self.estado_semaforo = self.semaforo

                self.flag_once = False
                self.flag_turn = False

                # Si se encuentran contornos
                if self.estado == "start" and (self.estado_semaforo == "null" or self.estado_semaforo == "green"): 
                    self.line_follower(1)
                    print("follow line")

                elif self.estado == "give_way" and self.flag_once == False:
                    print("cambio a give_way")
                    self.give_way_state(2)
                    self.flag_once = True

                elif self.estado == "work":
                    print("cambio a work")
                    self.work_state(2)	
                    
                elif self.semaforo == "red":
                    print("estado red")
                    self.line_follower(3)
                    if self.semaforo != "red":
                        self.estado = "start"
                        self.semaforo = "green"

                elif self.estado == "left" and self.estado_semaforo == "green":#funciona
                    print("cambio a left")
                    self.turn_left(1)

                elif self.estado == "left" and self.estado_semaforo == "yellow":#funciona
                    print("cambio a left vel mitad")
                    self.turn_left(4)              
                    
                elif self.estado == "right"and self.estado_semaforo == "green":#funciona
                    print("cambio a right")
                    self.turn_right(1)

                elif self.estado == "right"and self.estado_semaforo == "yellow": #funciona
                    print("cambio a right vel mitad")
                    self.turn_right(4)       

                elif (self.estado == "straight" and self.estado_semaforo == "green"): #funciona
                    print("cambio a straight")
                    self.forward_state(1) #el 1 para mantener la vel normal  

                elif (self.estado == "straight" and self.estado_semaforo == "yellow"):#funciona
                    print("cambio a straight vel mitad")
                    self.forward_state(3) ##el 3 es para dividir la velocidad
                                         
                elif (self.estado == "stop"):
                    print("cambio a stop")
                    self.stop_state()
                           
            cv2.waitKey(1)
            rate.sleep()

    def image_callback(self, msg):
        try: 
            self.frame = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8") 
            self.image_received_flag = 1 
        except CvBridgeError as e: 
            print(e)

    def signal_callback(self, msg):
        self.signal = msg.data

    def semaforo_callback(self, msg):
        self.semaforo = msg.data

    def line_follower (self, m):
        line_contour = max(self.contours, key=cv2.contourArea)

        cv2.drawContours(self.roi, [line_contour], -1, (0, 0, 255), thickness=2)

        # Calcular el centro del contorno
        M = cv2.moments(line_contour)
        if M['m00'] > 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            cv2.circle(self.roi, (cx, cy), 5, (0, 255, 0), -1)
            # Controlador PID
            self.e_w[0] = 40 - cx
            self.u_w[0] = self.K1_w * self.e_w[0] + self.K2_w * self.e_w[1] + self.K3_w * self.e_w[2] + self.u_w[1]
            self.e_w[2] = self.e_w[1]
            self.e_w[1] = self.e_w[0]
            self.u_w[1] = self.u_w[0]

            if self.u_w [0] > self.w_max:
                self.u_w[0] = self.w_max
            elif self.u_w[0] < -self.w_max:
                self.u_w[0] = -self.w_max

            # Control de movimiento
            angular_z = self.u_w[0]
            linear_x = self.linear_vel

            # Publicar los comandos de velocidad
            if m == 3:
                self.twist_msg.linear.x = 0.0
                self.twist_msg.angular.z = 0.0
            else:
                self.twist_msg.linear.x = linear_x/m
                self.twist_msg.angular.z = angular_z/m 
            #print(twist_msg)


            self.robot_vel_pub.publish(self.twist_msg)

            print("lineal", self.twist_msg.linear.x)
            print("angular",self.twist_msg.angular.z)

    def turn_left (self,mod):
        print(self.estado)
        # Mantener el movimiento constante durante 1.5 segundos
        for _ in range(22 * mod):
            self.twist_msg.linear.x = 0.2 / mod
            self.twist_msg.angular.z = 0
            self.robot_vel_pub.publish(self.twist_msg)
            rospy.sleep(0.1)

        print("hora de la vuelta")

        # Realizar la vuelta durante 5 segundos
        for _ in range(39):
            self.twist_msg.linear.x = 0.0
            self.twist_msg.angular.z = 0.4
            self.robot_vel_pub.publish(self.twist_msg)

            rospy.sleep(0.1)

        print("acaba vuelta + empuje")

        for _ in range(12):
            self.twist_msg.linear.x = 0.1
            self.twist_msg.angular.z = 0
            self.robot_vel_pub.publish(self.twist_msg)
            rospy.sleep(0.1)

        self.signal = "start"
        self.semaforo = "null"
        
    def turn_right (self,mod):
        # Mantener el movimiento constante durante 1.5 segundos
        for _ in range(22 * mod):
            self.twist_msg.linear.x = 0.195 / mod
            self.twist_msg.angular.z = 0
            self.robot_vel_pub.publish(self.twist_msg)
            rospy.sleep(0.1)

        print("hora de la vuelta")

        # Realizar la vuelta durante 5 segundos
        for _ in range(44):
            self.twist_msg.linear.x = 0.0
            self.twist_msg.angular.z = -0.35 
            self.robot_vel_pub.publish(self.twist_msg)

            rospy.sleep(0.1)

        print("acaba vuelta + empuje")

        for _ in range(6):
            self.twist_msg.linear.x = 0.192 
            self.twist_msg.angular.z = 0.0
            self.robot_vel_pub.publish(self.twist_msg)
            rospy.sleep(0.1)

        self.signal = "start"
        self.semaforo = "null"

    def give_way_state (self, mod):
        print("Give_way")
        # Mantener el movimiento constante durante 5 segundos
        for _ in range(45):
            self.line_follower(mod)	#se pone el 2 para reducir la velocidad a la mitad
            rospy.sleep(0.1)
        print("termina Give_way")

        self.signal = "start"
        self.semaforo = "null"

    def work_state (self, mod):
        print("Work")
        # Mantener el movimiento constante durante 5 segundos
        for _ in range(18):
            self.line_follower(mod)	#se pone el 2 para reducir la velocidad a la mitad
            rospy.sleep(0.1)
        print("termina Work")

        self.signal = "start"
        self.semaforo = "null"

    def forward_state (self, mod):
        print("straight")
        for _ in range(35 * mod):
            self.twist_msg.linear.x = 0.2 / mod
            self.twist_msg.angular.z = 0.0
            self.robot_vel_pub.publish(self.twist_msg)
            rospy.sleep(0.1)

        print("termina straight")
        self.signal = "start"
        self.semaforo = "null"


    def stop_state (self):
        print("Senal de stop")

        # Mantener el movimiento constante durante 10 segundos
        
        for _ in range(10):
            self.twist_msg.linear.x = 0.0
            self.twist_msg.angular.z = 0.0
            self.robot_vel_pub.publish(self.twist_msg)
            rospy.sleep(0.1)
        
        print("termina stop")
        
        self.signal = "stop"
        self.semaforo = "null"

    def shutdown(self):
        self.twist_msg.linear.x = 0.0
        self.twist_msg.angular.z = 0.0
        self.robot_vel_pub.publish(self.twist_msg)       
       

if __name__ == "__main__":
    print("Initialized")
    rospy.init_node("detec_line", anonymous=True)
    line_follower()
