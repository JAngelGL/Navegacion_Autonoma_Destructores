# Navegacion_Autonoma_Destructores
Realizado por

* Elizabeth Díaz Lerma A01703363
* Ramón Marínez Delgado A01702917
* Emiliano Mendoza Nieto A01706083
* José Ángel García López A01275108

Para correr lo contenido en este repositorio se deben correr los siguientes comandos en la terminal

Se debe configurar el puerto conectado a la Hackerboard
$ chmod 666 /dev/ttyUSB0

Luego
$ roslaunch  puzzlebot_autostart puzzlebot_autostart.launch
$ roslaunch ros_deep_learning video_source_ros1.launch input_width:=160 input_height:=160 input_codec:=“HEVC”

Para correr los nodos asumimos que ya tiene un paquete de ros creado y que archivos .py ya son ejecutables

$ rosrun red_neuronal test_test.py
$ rosrun red_neuronal maquina_semaforos.py


