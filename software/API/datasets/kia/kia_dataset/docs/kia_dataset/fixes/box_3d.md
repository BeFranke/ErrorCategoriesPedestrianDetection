[Back to Overview](../../README.md)



# kia_dataset.fixes.box_3d

This is an executable script, that fixes the 3d bounding boxes in the kia dataset.

```bash
kia_fix_box_2d --data_path D:\\KIA-datasets\\extracted [--debug_mode true]
```

The script fixes the class attribute error and adds an estimated occlusion and depth field.
The corrected boxes are stored in '2d-bounding-box-fixed_json'.

## Authors and Contributors
* Michael Fürst (DFKI), Lead-Developer
* Philipp Heidenreich (Opel)


---
### *def* **fix_bounding_boxes_3d**(data_path, debug_mode=False)

Loads all boxes from all tranches and fixes the 3D bounding boxes.

* **data_path**: (str) An absolute path to the dataset, e.g. ('D:\\KIA-datasets\\extracted').


---
### *def* **fix_class_id**(tranche, boxes)

*(no documentation found)*

---
### *def* **fix_class_id_and_instance_id**(tranche, boxes, path, seq_number)

*(no documentation found)*

---
### *def* **fix_center_and_size**(tranche, boxes)

*(no documentation found)*

---
### *def* **convert_legacy_3d_box_format**(box)

*(no documentation found)*

---
### *def* **write_boxes**(sequence_path, filename, boxes)

*(no documentation found)*

---
### *def* **main**()

*(no documentation found)*

