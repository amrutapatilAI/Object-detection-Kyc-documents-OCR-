# Object-detection-Kyc-documents-OCR-
This reipository is mainly to learn about object detection.How to train our models with YOLO-Darknet.I came across use case of OCR of KYC documents,So trying to give a shot to give back to the community from which i personally learned lot. 

# object detection -

Basically object detection is specifing perticlar object in an image.What probelm statement we had was to extarct required text from provided kyc documents.(eg:-pan/aadhaar/voter etc.).
We went for YOLO (You Only Look Once )Algorithm.YOLO is extremely fast for real time multi object detection algorithm. 

# Training-

So train our model we need to have Darknet installed .You can follow following commands
```
 
 git clone https://github.com/pjreddie/darknet 
 
 cd darknet
 
 make
```

You can follow their official website https://pjreddie.com/darknet/yolo/
# Steps to follow for training.
## 1-preparation of training data (Laballing):-

   
   We have to prepare out training data.What we do is laballing.We have to specifiy in exatctly which ara of image we are looking for.There are multiple tools available for image labelling.but I went for microsofts visual object tagging tool (VOTT).
you can follow their official github link https://github.com/microsoft/VoTT 

## 2-Training yolo on VOC data(VOTT gives you option in which format you want to save your lablled data eg:-JSON,pascal VOC):-

   For tarining your yolo model on darknet you need to take care of some of mandetory files.We are going to train our model by using allready trained model on imagenet we call it as TRANSFER LERANING. We will use convolutional weights from darknet53 model.
   
   to download those weights you can use wget https://pjreddie.com/media/files/darknet53.conv.74
   
   When you will install darknet you will have some folders like cfg,backup,data.
   
   cfg folder contains all yolo configuration files with .cfg extention.also files with .data extention which contains information like path of your train.txt file also path of lable.names file. 
exaple of .data file:-

```
  1 classes= 2 #number of classes 
  2 train  = <path-to>/train.txt
  3 valid  = <path-to->/est.txt
  4 names = data/labels.names
  5 backup = backup #path where to save weights of model
  
  #NOTE-it's not necessary you have save your weights in backup folder only
```
Now the main task is wrting you own customised yolov3.cfg file (Note-you can give any name you wish to give)
In current use case which is detect important text from kyc documents-I had lablled data with only one label text.So my kyc.names file contained only one lable 'text'.
Now consider follwing part of cfg file.We will understand what it means
```
[net]
# Training
batch=64
subdivisions=8
width=416
height=416
channels=3
momentum=0.9
decay=0.0005
````
 batch=64 indicates ***batch-size*** Training process is basically iterrative process,which iteratively updates weights of a neural network.which is mainly depends on how many mistakes it is making on trainig dataset.
 eg:- here bath-size is 64 which means 64 images are used in one iteration.
 But sometimes we don't have enough memory to interate 64 images single time ,to overcome from this darknet has provided us ***subdivisions*** facility  which devides single batch in parts.
 training dataset images will be resized to the size we have mentioned against ***width & height. channel =3*** indicates that we are procissing coloured images. 
 ***momentum*** is actully penalizing  parameter to penalize large weight changes during iterations.
 
 ```
 learning_rate=0.001
burn_in=1000
max_batches = 5200
policy=steps
steps=4160,4680
scales=.1,.1
```
***learning_rate*** Typically this is a number between 0.01 and 0.0001.
***max_batches** this number should not less than ****no_of_classes into 2000**** here in our case we have only one class.
***steps*** set steps to 80% and 90% of max_batches

**So what more changes we need to do ?** we have make changes according to our use case in ***[yolo]*** layers in available neural netowork which apprears 3 times in the file.so here is what we need to change :-
```
[convolutional]
size=1
stride=1
pad=1
filters=18
activation=linear

[yolo]
mask = 3,4,5
anchors = 10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326
classes=1
num=9
jitter=.3
ignore_thresh = .5
truth_thresh = 1
random=1
```
Thre is one formula to dicide number of filters we should have -***filters=(classes + 5)x3***


Now we are all set to start train our own model.what we have to do is to run following command.

***withoud GPU***

***./darknet detector train cfg/kyc.data cfg/yolov3.cfg darknet53.conv.74***

***with GPU***

***./darknet detector train cfg/coco.data cfg/yolov3.cfg darknet53.conv.74 -gpus 0,1,2,3***

and that's it.with this our training will start.
