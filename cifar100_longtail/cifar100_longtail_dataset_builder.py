"""cifar100_longtail dataset."""

import random
from collections import Counter
from typing import cast

import tensorflow_datasets as tfds
from tensorflow_datasets.image_classification.cifar import (
    _CIFAR_IMAGE_SHAPE,
    Cifar100,
)

_VERSION = tfds.core.Version("0.0.1")
_RELEASE_NOTES = {
    "0.0.1": "Initial dataset",
}

N_CLASSES = 100
N_HEAD_CLASSES_LIST = [100, 50, 25]


class Cifar100LongtailConfig(tfds.core.BuilderConfig):
    """BuilderConfig for Cifar100Longtail."""

    def __init__(
        self, *, n_head_classes=N_CLASSES, n_instances_per_tail=50, **kwargs
    ):
        super().__init__(**kwargs)
        self.n_head_classes = n_head_classes
        self.n_instances_per_tail = n_instances_per_tail


def _make_builder_configs():
    config_list = []
    for n_head_classes in N_HEAD_CLASSES_LIST:
        name_str = f"head_{n_head_classes}"
        description_str = f"num head classes = {n_head_classes} / {N_CLASSES}"
        config_list.append(
            Cifar100LongtailConfig(
                name=name_str,
                version=_VERSION,
                release_notes=_RELEASE_NOTES,
                description=description_str,
                n_head_classes=n_head_classes,
            )
        )
    return config_list


class LongtailFilter:
    def __init__(
        self,
        n_head_classes: int,
        n_classes: int,
        n_tail_instances: int,
        seed: int,
    ) -> None:
        self.n_tail_classes = n_classes - n_head_classes

        self.n_tail_instances = n_tail_instances
        self.accepted_counter: Counter[int] = Counter()

        self.seed = seed
        self.class_ordering = list(range(n_classes))

        random.seed(seed)
        random.shuffle(self.class_ordering)

    def filter(self, record) -> bool:
        label = record["label"]
        label_idx = self.class_ordering[label]

        # Filter if in tail class + already accepted enough
        if label_idx < self.n_tail_classes:
            if self.accepted_counter[label_idx] >= self.n_tail_instances:
                return True
            else:
                self.accepted_counter[label_idx] += 1

        return False


class Builder(Cifar100):
    """DatasetBuilder for cifar100_label_noise dataset."""

    BUILDER_CONFIGS = _make_builder_configs()
    SEED = 42

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            description=(
                "The CIFAR-10 dataset consists of 60000 32x32 colour "
                "images in 10 classes, with 6000 images per class. There "
                "are 50000 training images and 10000 test images."
            ),
            features=tfds.features.FeaturesDict(
                {
                    "id": tfds.features.Text(),
                    "image": tfds.features.Image(shape=_CIFAR_IMAGE_SHAPE),
                    "label": tfds.features.ClassLabel(num_classes=N_CLASSES),
                    "coarse_label": tfds.features.ClassLabel(num_classes=20),
                }
            ),
            supervised_keys=("image", "label"),
        )

    def _split_generators(self, dl_manager):
        return super()._split_generators(dl_manager)

    def _generate_examples(self, split_prefix, filepaths):
        gen_fn = super()._generate_examples(split_prefix, filepaths)

        build_config = cast(Cifar100LongtailConfig, self.builder_config)
        longtail_filter = LongtailFilter(
            build_config.n_head_classes,
            N_CLASSES,
            build_config.n_instances_per_tail,
            seed=self.SEED,
        )

        for key, example in gen_fn:
            if longtail_filter.filter(example):
                continue

            yield key, example
