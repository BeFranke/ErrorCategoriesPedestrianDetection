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
│   ecpd.yaml
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
See ``requirements.txt``.

## Datasets-folder
``input/datasets`` is meant for datasets. **The datasets are only needed for the Error VisualizationTool to function, 
for the evaluation all data is read from the supplied json files.**
Currently, the error visualization tool only supports the *citypersons* dataset, 
which would have to be placed like this:
```
project
└─── input
│   └─── datasets
│        └─── cityscapes
│             └─── leftImg8bit
...
```


## JSON Input Format

To evaluate a model on a dataset, 2 json files are to be supplied by the user:

### Ground Truth File

Path: ``input/gt/{split}_{dataset_name}.json``

This is a json file summarizing the dataset, loosely based on the MS-COCO format with some extra keys.
An example file for citypersons is already provided.
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
    "im_name"       :   string (filename without extension or path)
}
```


An entry in the list of annotations should look like this:

```
{
    "bbox"              :   [x,y,width,height : int],
    "height"            :   int,
    "id"                :   int (1-based),
    "ignore"            :   0 or 1,
    "image_id"          :   int (1-based),
    "vis_bbox"          :   [x,y,width,height : int],
    "vis_ratio"         :   float,
    "crowd_occl_ratio"  :   float,
    "env_occl_ratio"    :   float,
    "inst_vis_ratio"    :   float
}
```

Compared to the MS-COCO format, the following keys are added:

- ``crowd_occl_ratio``: Ratio of semantic pedestrian pixels inside the bounding box that belong to other pedestrians to pedestrian pixels that belong to the referenced pedestrian
- ``env_occl_ratio``: Area occupied by potentially occluding objects inside the bounding box over area of bounding box
- ``inst_vis_ratio``: Area occupied by pixels belonging to the actual pedestrian over area of bounding box

Other keys on any level of the json structure may be specified and will be ignored by the framework.

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

Other keys on any level of the json structure may be specified and will be ignored by the framework.

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

By default, each evaluation will create a new, tim-stamped folder in ```output/```, containing the following 
subdirectories:

```
project
└─── output
│   └─── {YYYYMMDD-hhmmss}
│        │      results.csv
│        └───   figures
│        └───   plotting-raw
│        └───   raw
...
```

The evaluation results are saved to ``results.csv``, where each line will contain the model name 
(= the name of the detection json file minus the file extension), the evaluation setting as int
(currently, reasonable=0 and all=4 are supported), LAMR. and a number of metrics over the error categories.
For each detection json file present in the specified input folder, the ``results.csv`` will contain one line.


``plotting-raw`` and ``raw`` contain raw information that is used by the plotting scripts and the 
error visualization tool. ``figures`` is the folder where the plotting scripts will place the generated plots, if run.

## Error Visualization Tool

TODO

## Plotting Scripts

TODO

## Parameters

TODO


        