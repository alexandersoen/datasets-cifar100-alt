"""cifar100_noisy dataset.

TODO: Coarse labels are not noise.
"""

import random
from typing import cast

import tensorflow_datasets as tfds
from tensorflow_datasets.image_classification.cifar import (
    _CIFAR_IMAGE_SHAPE,
    Cifar100,
    CifarInfo,
)

_VERSION = tfds.core.Version("0.0.1")
_RELEASE_NOTES = {
    "0.0.1": "Initial dataset",
}

N_CLASSES = 100
N_CLASSES_NOISY_LIST = [0, 10, 25]


class Cifar100NoisyConfig(tfds.core.BuilderConfig):
    """BuilderConfig for Cifar100Noisy."""

    def __init__(self, *, n_classes_noisy=0, **kwargs):
        """BuilderConfig for Imagenet2012Corrupted.

        Args:
          num_class_noisy: integer, number of classes with label noise
          **kwargs: keyword arguments forwarded to super.
        """
        super().__init__(**kwargs)
        self.n_classes_noisy = n_classes_noisy


def _make_builder_configs():
    """Construct a list of BuilderConfigs.

    Construct a list of 95 Imagenet2012CorruptedConfig objects, corresponding to
    the 15 + 4 corruption types, with each type having 5 severities.

    Returns:
      A list of 95 Imagenet2012CorruptedConfig objects.
    """
    config_list = []
    for n_classes_noisy in N_CLASSES_NOISY_LIST:
        name_str = f"label_noise_{n_classes_noisy}"
        description_str = (
            f"num classes with uniform label noise = {n_classes_noisy}"
        )
        config_list.append(
            Cifar100NoisyConfig(
                name=name_str,
                version=_VERSION,
                release_notes=_RELEASE_NOTES,
                description=description_str,
                n_classes_noisy=n_classes_noisy,
            )
        )
    return config_list


def add_label_noise(record, n_classes_noisy, n_classes):
    if record["label"] < n_classes_noisy:
        record["label"] = random.randrange(n_classes)

    return record


class Builder(Cifar100):
    """DatasetBuilder for cifar100_label_noise dataset."""

    BUILDER_CONFIGS = _make_builder_configs()
    SEED = 42

    @property
    def _cifar_info(self):
        return CifarInfo(
            name=self.name,
            url="https://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz",
            train_files=["train.bin"],
            test_files=["test.bin"],
            prefix="cifar-100-binary/",
            label_files=["coarse_label_names.txt", "fine_label_names.txt"],
            label_keys=["coarse_label", "label"],
        )

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
        random.seed(self.SEED)

        build_config = cast(Cifar100NoisyConfig, self.builder_config)

        for key, example in gen_fn:
            example = add_label_noise(
                example, build_config.n_classes_noisy, N_CLASSES
            )
            yield key, example
