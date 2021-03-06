# Ottertune Clustering

Project to improve cluster and metric pruning in [OtterTune](https://www.cs.cmu.edu/~ggordon/van-aken-etal-parameters.pdf) for [CS645 Database Design and Implementation](http://avid.cs.umass.edu/courses/645/s2020/index.html). The [Github repository](https://github.com/treble-maker123/ottertune) for this project can be found here.

To run the pipeline end-to-end, simply run `bash run_pipeline.sh`. Note that the script contains various hyperparamteres.

The test.csv in the root directory is generated using the GaussianRandomProjection and K-Means clustering as the metric-pruning method.

## Setup

1.  To setup test data, unzip the `project3_dataset.tar.gz` file, e.g. `tar zxf project3_dataset.tar.gz`. You should have a folder named `project3_dataset` in the root directory of this project,
2.  To setup the conda environment (see guide on Miniconda installation [here](https://docs.conda.io/en/latest/miniconda.html)), run `conda env create -f environment.yml`,

NOTE: If you add packages to conda, delete the original `environment.yml` file and run `conda env export --from-history | grep -v "^prefix: " > environment.yml` to generate a new one with the addition you have made.
