# Error Categorization for Pedestrian Detection

## Introduction

This repository contains a framework for a fine-grained evaluation of the performance of pedestrian detectors, 
based on error categorization.
8 categories are evaluated, 5 for false negatives and 3 for false positives.
On each category, the **Filtered Log-Average Miss Rate** (FLAMR) is reported.
Additionally, various plotting scripts are supplied to plot MR-FPPI curves, heatmaps and various histograms.

## Directory Structure

```
project
│   README.md
└───ErrorVisualizationTool
│   │  {various internal files}
│   │  run.py     
│
└───evaluation
│   │   main.py
│   └─── API
│       │ cfg_eval.yaml
│       │ {various internal files}
│   └─── plotting
│       │ heatmaps.py
│       │ plot_lines.py
│       │ vis.py
│   
└─── input
│   └─── datasets
│   └─── dt
│   └─── gt
└─── output
    └─── {evaluation-reports will be placed here}
```

## Requirements

TODO

## Datasets-folder
``input/datasets`` is meant for datasets. At very least, the evaluated dataset should contain ground truth as
**instance segmentation maps** and **semantic segmentation maps**.
For the error visualization tool to work, it also needs to contain the RGB-images of the dataset.
For example, to use cityscapes/citypersons, it should be placed like this:
```
project
└─── input
│   └─── datasets
│        └─── cityscapes
│             └─── gtBboxCityPersons
│             └─── gtFine
│             └─── leftImg8bit
...
```


## JSON Input Format

To evaluate a model on a dataset, 2 json files are to be supplied by the user:

### Ground Truth File

Path: ``input/gt/{split}_{dataset_name}.json``

This is a json file summarizing the dataset, loosely based on the MS-COCO format with some extra keys.
The format is explained below:
```
{
    "images": [image],
    "annotations": [annotations],
    "categories":  [
        {
            "id": 0,
            "name": "ignore"
        },
        {
            "id": 1,
            "name": "pedestrian"
        }
    ]
}
```

An entry in the list of images should look like this:

```
{
    "height"        :   int,
    "width"         :   int
    "id":           :   int (1-based),
    "im_name"       :   string (filename without extension or path),
    "instance_map"  :   string (exact path to the corresponding instance map relative to the datasets-folder),
    "semantic_map"  :   string (exact path to the corresponding semantic map relative to the datasets-folder)
}
```


An entry in the list of annotations should look like this:

```
{
    "bbox"          :   [x,y,width,height : int],
    "category_id"   :   int,
    "height"        :   int,
    "id"            :   int (1-based),
    "ignore"        :   0 or 1,
    "image_id"      :   int (1-based),
    "instance_id"   :   int,
    "vis_bbox"      :   [x,y,width,height : int],
    "vis_ratio"     :   float 
}
```

**It is very important that the instance id given in the annotation corresponds to the actual instance id used in the 
instance map!**

### Detection File

Path:``input/dt/{any-name}/{model_name}.json``

This file gives the detections of the model in unmodified MS-COCO format:

```
[{
    "image_id"      : int,
    "category_id"   : int,
    "bbox"          : [x,y,width,height],
    "score"         : float,
}]
```

(Source: https://cocodataset.org/#format-results)

## Running an evaluation

Run ``python3 evaluation/main.py dt_folder gt_file [--config CONFIG] [--out OUT]``.
The arguments are explained below:

``dt_folder`` (REQUIRED): relative to ``input/dt``, specifies the name of the folder containing one or more detection 
files in the JSON format specified above. If the folder contains multiple detection files, each of them will be
evaluated (the file name is used as model name in the results).


``gt_file`` (REQUIRED): relative to ``input/gt``, specifies the name of the json file containing the object-detection 
ground truth like specified above. 


``CONFIG`` (OPTIONAL):  specifies a path to a config.yaml file giving teh parameters of the evaluation. 
By default, will use ``evaluation/API/cfg_eval.yaml``, which contains carefully chosen default values.

``OUT`` (OPTIONAL):  specifies a path to save the output files. By default, creates a time-stamped folder
in ``output``.

## Output Format

TODO

## Error Visualization Tool

TODO

## Parameters

TODO


        