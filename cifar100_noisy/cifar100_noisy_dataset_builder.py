"""cifar100_noisy dataset.

TODO: Coarse labels are not noise.
"""

import random
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
N_NOISY_CLASSES_LIST = [0, 10, 25]


class Cifar100NoisyConfig(tfds.core.BuilderConfig):
    """BuilderConfig for Cifar100Noisy."""

    def __init__(self, *, n_noisy_classes=0, **kwargs):
        """BuilderConfig for Imagenet2012Corrupted.

        Args:
          num_class_noisy: integer, number of classes with label noise
          **kwargs: keyword arguments forwarded to super.
        """
        super().__init__(**kwargs)
        self.n_noisy_classes = n_noisy_classes


def _make_builder_configs():
    """Construct a list of BuilderConfigs.

    Construct a list of 95 Imagenet2012CorruptedConfig objects, corresponding to
    the 15 + 4 corruption types, with each type having 5 severities.

    Returns:
      A list of 95 Imagenet2012CorruptedConfig objects.
    """
    config_list = []
    for n_noisy_classes in N_NOISY_CLASSES_LIST:
        name_str = f"label_noise_{n_noisy_classes}"
        description_str = (
            f"num classes with uniform label noise = {n_noisy_classes}"
        )
        config_list.append(
            Cifar100NoisyConfig(
                name=name_str,
                version=_VERSION,
                release_notes=_RELEASE_NOTES,
                description=description_str,
                n_noisy_classes=n_noisy_classes,
            )
        )
    return config_list


class NoiseGenerator:
    def __init__(
        self,
        n_noisy_classes: int,
        n_classes: int,
        seed: int,
    ) -> None:
        self.n_noisy_classes = n_noisy_classes
        self.n_classes = n_classes

        self.seed = seed
        self.class_ordering = list(range(n_classes))

        random.seed(seed)
        random.shuffle(self.class_ordering)

    def add_noise(self, record):
        label = record["label"]
        label_idx = self.class_ordering[label]

        if label_idx < self.n_noisy_classes:
            record["label"] = random.randrange(self.n_classes)

        return record


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
        random.seed(self.SEED)

        build_config = cast(Cifar100NoisyConfig, self.builder_config)
        noisy_generator = NoiseGenerator(
            build_config.n_noisy_classes,
            N_CLASSES,
            seed=self.SEED,
        )

        for key, example in gen_fn:
            example = noisy_generator.add_noise(example)

            yield key, example
