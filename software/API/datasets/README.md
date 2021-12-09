# Folder structure

In your `input` folder please add the datasets in the specified format which you can find below.

### COCO Datasets
We expect that this __`datasets`__ folder contains a subfolder with the name __`coco`__. All types of COCO datasets for 
the different years should be stored right here. For example, if you want to store the images for the COCO Detection 
dataset of 2017 you should put all the files in a folder called __`2017`__. In this folder there should be an 
__`annotations`__ folder that contains the json files for the train and val datasets. The name of the files should be 
"instances_<MODE><DATASET_NAME>.json". For instance the train dataset annotations file for 2017 should be called 
__`instances_train2017.json`__. The images will be stored in an __`<MODE><DATASET_NAME>`__ folder within the 
<DATASET_NAME> folder. If you want to remove certain image ids from the datasets you can add a __`blacklists`__ folder 
which contains txt files with a similar name to the annotations file (e.g.__`img_ids_blacklist_train2017.txt`__). We 
wrote a short python script that should create a blacklist for you COCO dataset. Some images have no detections
at all which is not compatible during training time. Furthermore, some COCO detections have a zero width/height which 
might lead to an infinity loss because of the logarithm.  

##### Config:
 - __dataset__: *'coco'*
 - __evaltype__: *'coco'* or *'pascalvoc'*
 - __pascal_voc_root__: *'../../input/coco/'*
 - __image_set__: e.g. *'2017'*
 
All in all, the folder structure should look like the following example:

##### Example:
    .
    ├── ...
    ├── coco  
    │   ├── 2017  <IMAGE_SET>
    │   │   ├── train2017
    │   │   │   ├── 00001.png
    │   │   │   ├── ...
    │   │   │   └── 99999.png
    │   │   │
    │   │   ├── val2017
    │   │   │   ├── 00001.png
    │   │   │   ├── ...
    │   │   │   └── 99999.png
    │   │   │
    │   │   ├── annotations
    │   │   │   ├── instances_train2017.json
    │   │   │   └── instances_val_2017.json
    │   │   │
    │   │   ├── blacklists
    │   │   │   ├── img_ids_blacklist_train2017.txt
    │   │   │   └── img_ids_blacklist_val2017.txt
    │   │   │
    │   │   └── ...  
    │   │
    │   ├── 2018 <IMAGE_SET>
    │   │   ├── train2018
    │   │   │   ├── image_00001.png
    │   │   │   ├── ...
    │   │   │   └── image_99999.png
    │   │   │
    │   │   ├── val2018
    │   │   │   ├── 00001.png
    │   │   │   ├── ...
    │   │   │   └── 99999.png
    │   │   │
    │   │   ├── annotations
    │   │   │   ├── instances_train2018.json
    │   │   │   └── instances_val_2018.json
    │   │   │
    │   │   ├── blacklists
    │   │   │   ├── img_ids_blacklist_train2018.txt
    │   │   │   └── img_ids_blacklist_val2018.txt
    │   │   │
    │   │   └── ...  
    │   └── ... 
    └── ... 
    
### Pascal Voc Datasets
Here again we need to enforce a certain folder structure to allow our PascalVocDataset class to efficiently load the 
images and annoations from disk. Similar to the COCO dataset we expect to find a __`pascalvoc`__ folder that contains
subfolders with the names of the datasets at hand. For the VOCdevkit this would be VOC2007 or VOC2012 and corresponds to
the image_set in the config.yaml. The Pascal Voc dataset for 2007 should be stored in a folder called 
__`2007`__ or __`VOC2007`__. Here again we expect to find an __`Annotations`__ folder and an __`JPEGImages`__ folder 
with the given Pascal Voc files. Compared to the COCO structure, we now have an __`ImageSets`__ folder which contains a 
subfolder __`Main`__ which then contains the the train.txt or val.txt files with the ids
of the images that belong to the dataset split. Please find an example below. If you use the `VOCdevkit` you can simply 
rename the folder to `pascalvoc` and place it in the input folder. In the config.yaml file, set the image_set to __VOC2007__ or __VOC2012__ and the mode
to one of the txt files in the ImageSets > Main folder.

##### Config:
 - __dataset__: *'pascalvoc'*
 - __evaltype__: *'coco'* or *'pascalvoc'*
 - __pascal_voc_root__: *'../../input/pascalvoc/'*
 - __image_set__: *'VOC2007'* or *'VOC2012"*

##### Example:
    .
    ├── ...
    ├── pascalvoc
    │   ├── 2007 <IMAGE_SET>
    │   │   ├── JPEGImages
    │   │   │   ├── 00001.jpg
    │   │   │   ├── ...
    │   │   │   └── 99999.jpg
    │   │   │
    │   │   ├── Annotations
    │   │   │   ├── 00001.xml
    │   │   │   ├── ...
    │   │   │   └── 99999.xml
    │   │   │
    │   │   ├── ImageSets
    │   │   │   ├── Main
    │   │   │   │   ├── train.txt <MODE>
    │   │   │   │   └── val.txt <MODE>
    │   │   │   └── ... 
    │   │   └── ...  
    │   │
    │   ├── 2012 <IMAGE_SET>
    │   │   ├── JPEGImages
    │   │   │   ├── 00001.jpg
    │   │   │   ├── ...
    │   │   │   └── 99999.jpg
    │   │   │
    │   │   ├── Annotations
    │   │   │   ├── 00001.xml
    │   │   │   ├── ...
    │   │   │   └── 99999.xml
    │   │   │
    │   │   ├── ImageSets
    │   │   │   ├── Main
    │   │   │   │   ├── train.txt <MODE>
    │   │   │   │   └── val.txt <MODE>
    │   │   │   └── ...  
    │   │   └── ...  
    │   └── ... 
    └── ... 
    
### A2D2 Datasets
For the Audi dataset we simply reuse the same structure as the original dataset.
Thus, we have to add a new folder `a2d2` to the `input` folder. Compared to the
COCO and VOC dataset we do not have datasets for multiple years. Therefore, we
simply add the `camera_lidar_semantic_bboxes` folder to the `a2d2` folder. The 
__image_set__ parameter in the config would be `camera_lidar_semantic_bboxes`.
Within that folder we expect to find the sequence folders. The splits for training, validation or
testing are part of the A2D2Dataset class. The dataset splits consist of the sequence names.

##### Config:
 - __dataset__: *'a2d2'*
 - __evaltype__: *'coco'* or *'pascalvoc'*
 - __pascal_voc_root__: *'../../input/a2d2/'*
 - __image_set__: *'camera_lidar_semantic_bboxes"*
 
##### Example:
    {HOME}/kia/input/
    ├── ...
    ├── a2d2  
    │   ├── camera_lidar_semantic_bboxes
    │   │   │   ├── 20180807_145028
    │   │   │   │   ├── camera
    │   │   │   │   │   ├── cam_front_center
    │   │   │   │   │   │   ├── 20180807145028_camera_frontcenter_000000091.png
    │   │   │   │   │   │   ├── 20180807145028_camera_frontcenter_000000091.json
    │   │   │   │   │   │   ├── ...
    │   │   │   │   │   │   ├── 20180807145028_camera_frontcenter_000000127.json
    │   │   │   │   │   │   └── 20180807145028_camera_frontcenter_000000127.png
    │   │   │   │   │   └── ...
    │   │   │   │   ├── label3D
    │   │   │   │   │   ├── cam_front_center
    │   │   │   │   │   │   ├── 20180807145028_label3D_frontcenter_000000091.json
    │   │   │   │   │   │   ├── ...
    │   │   │   │   │   │   └── 20180807145028_label3D_frontcenter_000000127.json
    │   │   │   │   │   └── ...
    │   │   │   │   └── ...
    │   │   │   ├── 20180810_142822
    │   │   │   ├── 20180925_101535
    │   │   │   └── ...
    │   │   └── ...  
    │   └── ... 
    └── ... 
