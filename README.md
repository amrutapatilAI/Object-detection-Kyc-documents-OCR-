# Object-detection-Kyc-documents-OCR-
This reipository is mainly to learn about object detection.How to train our models with YOLO-Darknet.I came across use case of OCR of KYC documents,So trying to give a shot to give back to the community from which i personally learned lot. 

# object detection -

Basically object detection is specifing perticlar object in an image.What probelm statement we had was to extarct required text from provided kyc documents.(eg:-pan/aadhaar/voter etc.).
We went for YOLO (You Only Look Once )Algorithm.YOLO is extremely fast for real time multi object detection algorithm. 

# Training-

So train our model we need to have Darknet installed .You can follow following commands
##################################################################################################
 
 git clone https://github.com/pjreddie/darknet 
 
 cd darknet
 
 make
##################################################################################################

You can follow their official website https://pjreddie.com/darknet/yolo/
# Steps to follow for training.
1-preparation of training data (Laballing):-

   
   We have to prepare out training data.What we do is laballing.We have to specifiy in exatctly which ara of image we are looking for.There are multiple tools available for image labelling.but I went for microsofts visual object tagging tool (VOTT).
you can follow their official github link https://github.com/microsoft/VoTT 

2-Training yolo on VOC data(VOTT gives you option in which format you want to save your lablled data eg:-JSON,pascal VOC):-

   For tarining your yolo model on darknet you need to take care of some of mandetory files.We are going to train our model by using allready trained model on imagenet we call it as TRANSFER LERANING. We will use convolutional weights from darknet53 model.
   
   to download those weights you can use wget https://pjreddie.com/media/files/darknet53.conv.74
   
   When you will install darknet you will have some folders like cfg,backup,data.
   
   cfg folder contains all yolo configuration files with .cfg extention.also files with .data extention which contains information like path of your train.txt file also path of lable.names file. 
exaple of .cfg file:-
