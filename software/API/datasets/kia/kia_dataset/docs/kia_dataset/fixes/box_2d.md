[Back to Overview](../../README.md)



# kia_dataset.fixes.box2d

This is an executable script, that fixes the 2d bounding boxes in the kia dataset.

```bash
kia_fix_box_2d --data_path D:\\KIA-datasets\\extracted [--debug_mode true]
```

The script fixes the class attribute error and adds an estimated occlusion and depth field.
The corrected boxes are stored in '2d-bounding-box-fixed_json'.

## Authors and Contributors
* Philipp Heidenreich (Opel), Lead-Developer
* Michael Fürst (DFKI)


---
### *def* **fix_bounding_boxes_2d**(data_path, debug_mode=False, estimate_depth=True)

Loads all boxes from all tranches and fixes the 2D bounding boxes.

* **data_path**: (str) An absolute path to the dataset, e.g. ('D:\\KIA-datasets\\extracted').


---
### *def* **fix_class_id**(tranche, boxes)

*(no documentation found)*

---
### *def* **fix_class_id_and_instance_id**(tranche, boxes, path, seq_number)

*(no documentation found)*

---
### *def* **post_proc_estimate_occlusion_and_depth**(tranche, boxes, path, filename, estimate_depth)

*(no documentation found)*

---
### *def* **write_boxes**(sequence_path, filename, boxes)

*(no documentation found)*

---
### *def* **visualize_boxes**(boxes, image)

*(no documentation found)*

---
### *def* **main**()

*(no documentation found)*

