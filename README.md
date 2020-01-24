# Emotion-Detection
One of the easiest way to do emotion detection
Steps to be followed for object Detection using Tensorflow in Windows
Gathering images
	
1)Gathering images

2)Convert extensions
All images to be in jpg format

3) Rename all images

4) Label images
https://github.com/tzutalin/labelImg ( download the folder for the use of labelling)
Command to be used: 
conda install pyqt=5 
pyrcc5 -o resources.py resources.qrc
python labelImg.py

5) Split images manually randomly ( test : 20% , train: 80%)

6) Installing TensorFlow-GPU 
Command: 
pip install --upgrade tensorflow-gpu

7)Install the packages
pillow, lxml, jupyter, matplotlib, opencv, cython

8) Download faster_rcnn_inception_v2_coco
This is the model we use for training.
https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md

9) Configure environment variable
This has to be done by going to the environment variable location In the system and add the below path’s to the python path.

PYTHONPATH variable must be created that points to the directories
\models 
\models\research 
\models\research\slim

10)Compile Protobufs
To run the command in the windows powershell.
From within TensorFlow/models/research/
Get-ChildItem object_detection/protos/*.proto | foreach {protoc "object_detection/protos/$($_.Name)" —python_out=.}

11) The labelled Images are stored in .XML format we need to convert them into .tfrecord 

We need to activate tensorflow_gpu then
For converting we have to steps
A)	.XML to .CSV
  From object detection folder we need to run these commands
       python xml_to_csv.py -i images\train -o images\train_labels.csv
       python xml_to_csv.py -i images\test -o images\test_labels.csv
B)	.CSV to .tfrecord
   From research folder we need to run these commands
       python generate_tfrecord.py --csv_input=object_detection\images\test_labels.csv --			image_dir=object_detection\images\test --output_path=test.record
       python generate_tfrecord.py --csv_input=object_detection\images\train_labels.csv --i			image_dir=object_detection\images\train —output_path=train.record

Now Make sure the .tfrecord files are in the /object-detection/images folder

12) Create a label map and edit the training configuration file.
This file will have a note of how many classes we have

13) Configure object detection training pipeline 

cd C:\tensorflow1\models\research\object_detection\samples\configs copy faster_rcnn_inception_v2_pets.config past it in \training dir and edit it

a -
In the model section change num_classes to number of different classes

b -
fine_tune_checkpoint : C:/tensorflow1/models/research/object_detection/faster_rcnn_inception_v2_coco_2018_01_28/model.ckpt

c -
In the train_input_reader section change input_path and label_map_path as : 
Input_path : C:/tensorflow1/models/research/object_detection/train.record 
Label_map_path: C:/tensorflow1/models/research/object_detection/training/labelmap.pbtxt

d -
In the eval_config section change num_examples as : 
Num_examples = number of files in \images\test directory.

e -
In the eval_input_reader section change input_path and label_map_path as :
Input_path : C:/tensorflow1/models/research/object_detection/test.record 
Label_map_path: C:/tensorflow1/models/research/object_detection/training/labelmap.pbtxt

F- num_steps: no of steps we need to train


14) Run the training command
From Research folder

python train.py --logtostderr --train_dir=object_detection/training/ —pipeline_config_path=object_detection/training/faster_rcnn_resnet101_atrous_coco.config


15) once the training is done we will have our model saved in the training folder which is inside object-detection folder.

16) Export Inference Graph

python export_inference_graph.py --input_type image_tensor --pipeline_config_path object_detection/training/faster_rcnn_inception_v2_pets.config --trained_checkpoint_prefix object_detection/training/model.ckpt-xxxxx --output_directory object_detection/inference_graph

Xxxx- will be the no of steps we have trained for
