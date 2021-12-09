from datasets.a2d2.a2d2_bbox_dataset import A2D2BboxDataset


class A2D2SemanticDataset(A2D2BboxDataset):

    def __init__(self, config, image_set, mode, augmentation) -> None:
        super().__init__(config, image_set, mode, augmentation)

    def _get_class_names(self) -> list:
        return ["background", 'Car', 'Small Vehicle', 'Tractor', 'Bicycle',
                'Pedestrian', 'Truck', 'Utility Vehicle']

    def _get_label_type(self):
        return "label2D"

    def _get_sequence_splits(self):
        return {
            "train": ["20180807_145028", "20180925_135056", "20181107_132300",
                      "20181108_091945", "20181204_154421", "20180810_142822",
                      "20181008_095521", "20181107_132730", "20181108_103155",
                      "20181204_170238", "20180925_101535", "20181016_082154",
                      "20181107_133258", "20181108_123750", "20181204_191844",
                      "20180925_112730", "20181016_095036", "20181107_133445",
                      "20181108_141609", "20180925_124435", "20181016_125231"],
            "val": ["20181108_084007"],
            "test": ["20181204_135952"],
            "mini": ["mini"],
        }
