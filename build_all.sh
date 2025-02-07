#!/bin/bash

tfds build --file_format=array_record cifar100_noisy/
tfds build --file_format=array_record cifar100_longtail/
