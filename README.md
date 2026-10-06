Functions and notes for the CNNPZ estimator

- `cnnpz/` is the package: catalogue reading, feature construction, ensemble training and prediction, and qp output
- The notebooks (Cardinal and BPZ tests, the PZ data challenge task sets) live on the `notebooks` branch

Data on nersc:
`/global/cfs/cdirs/lsst/groups/PZ/users/qhang/cnnpz_data/`
Under this folder contains:
`bpz_mock_training_set.parquet`  `cardinal/`  `pop-cosmos-data/`
the first file is used in the bpz notebook, the second folder contains the cardinal training and test data, the third contains the pop cosmos noisy data for pre-training.

To-do's to make pz challenge submission:

- Need to convert the output format (currently mean + std for the ensemble model) to a PDF in qp ensemble format

**Note**

I cleaned up the notebooks by moving functions out of them and calling the functions from the imported python file - this may mean that some of the cells will break due to missing arguements etc., but hopefully they are easy to trace down!
