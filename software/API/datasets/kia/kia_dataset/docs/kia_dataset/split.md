[Back to Overview](../README.md)



# kia_dataset.split

> The official dataset split.

## Installation/Setup

Follow the instructions in the main readme.

## Constants

There are three constants which each contain a List[str] of the sequence names which are allowed for use in the respective split.

The official splits per tranche per company:
* `TRAIN_{company}_TRANCHE_{num}`: Can be used to train neural networks.
* `VAL_{company}_TRANCHE_{num}`: Can be used to validate the neural network during training.
* `TEST_{company}_TRANCHE_{num}`: This split must not be used during development and can only be used (ideally) one time once the development is done.

The official splits for the releases:
* `TRAIN_RELEASE_{num}`: Can be used to train neural networks.
* `VAL_RELEASE_{num}`: Can be used to validate the neural network during training.
* `TEST_RELEASE_{num}`: This split must not be used during development and can only be used (ideally) one time once the development is done.


