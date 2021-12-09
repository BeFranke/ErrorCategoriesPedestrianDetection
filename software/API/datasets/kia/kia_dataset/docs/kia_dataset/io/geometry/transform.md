[Back to Overview](../../../README.md)



# kia_dataset.io.geometry.transform

> Geometric transformations between coordinate systems.

## Authors and Contributors
* Michael Fürst (DFKI), Lead-Developer

**WARNING: This code is from dfki-dtk and might be removed in the future.**


---
---
## *class* **Transform**(object)

*(no documentation found)*

---
### *def* **identity**()

*(no documentation found)*

---
### *def* **inverse**(*self*)

*(no documentation found)*

---
### *def* **R**(*self*)

*(no documentation found)*

---
### *def* **t**(*self*)

*(no documentation found)*

---
### *def* **q**(*self*)

*(no documentation found)*

---
### *def* **axis**(*self*)

*(no documentation found)*

---
### *def* **angle**(*self*)

*(no documentation found)*

---
### *def* **radians**(*self*)

*(no documentation found)*

---
### *def* **degrees**(*self*)

*(no documentation found)*

---
### *def* **then**(*self*, transform)

*(no documentation found)*

---
### *def* **then_rotate**(*self*, R=None, q=None)

*(no documentation found)*

---
### *def* **apply_to_point**(*self*, point, single_point=True)

Apply transform to a pointcloud of shape (3, N).

* **point**: Return a pointcloud of shape (3, N).
* **single_point**: Boolean if a single point should be processed. (Output then is only (3,).)
* **returns**: The transformed pointcloud of shape (3, N).


---
### *def* **apply_to_direction**(*self*, direction)

*(no documentation found)*

---
### *def* **apply_to_orientation**(*self*, orientation: Union[Quaternion, Sequence[float]]) -> Quaternion

*(no documentation found)*

---
### *def* **unsafe_q**(*self*)

*(no documentation found)*

