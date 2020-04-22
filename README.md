# Ottertune Clustering

Project to improve cluster and metric pruning in [OtterTune](https://www.cs.cmu.edu/~ggordon/van-aken-etal-parameters.pdf) for [CS645 Database Design and Implementation](http://avid.cs.umass.edu/courses/645/s2020/index.html).

## Setup

1.  To setup test data, unzip the `project3_dataset.tar.gz` file, e.g. `tar zxf project3_dataset.tar.gz`. You should have a folder named `project3_dataset` in the root directory of this project,
2.  To setup the conda environment (see guide on Miniconda installation [here](https://docs.conda.io/en/latest/miniconda.html)), run `conda env create -f environment.yml`,

NOTE: If you add packages to conda, delete the original `environment.yml` file and run `conda env export --from-history | grep -v "^prefix: " > environment.yml` to generate a new one with the addition you have made.

3.  After extracting the data and setting up the environment, run `python3 partition.py --num-sets=n` to create **n** sets of partitions of data for training and validation, and the results will be written in the data folder as `workload_partition_i.csv` where `i = 1...n`. The class method `Dataset.build_dataset_from_partition()` can then be used to build a pair of training and validation set from the partitions. 
