# CIFAR100 Alternative Datasets

Select corruptions of CIFAR100 dataset.

## Build instructions

Utilize the `tfds` CLI.

```bash
tfds build cifar100_noisy --file_format=array_record
```

Although optional, `array_record` is recommended. Allows for indexing, which makes things less painful if you need to switch to torch.

Can also use the bash file `build_all.sh`.
