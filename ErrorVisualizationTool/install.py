import os

# simply create symlinks
try:
    os.symlink("../eval/", "eval")
except FileExistsError:
    print("Symlink to eval folder exists!")
try:
    os.symlink("../../../../input/datasets/cityscapes", "cityscapes")
except FileExistsError:
    print("Symlinks to cityscapes exists!")
