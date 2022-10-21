## Unofficial Implementation of "Liu, W., Li, A., Wang, X., Yuan, M., Chen, Y., Zheng, C., & Li, X. (2022). A Neural Beamspace-Domain Filter for Real-Time Multi-Channel Speech Enhancement. Symmetry, 14(6), 1081."


- [Dependencies](#dependencies)
- [Data Generation](#data-generation)
- [Network Training](#network-training)
- [Results Computation](#results-computation)

### Dependencies
- Python, it has been tested with version 3.10.6
- Numpy, tqdm, matplotlib
- torch 1.12.1
- torchaudio 0.12.1

### Content
- compute_results.py  --> Compute separation results on audio tracks using trained models.
- data_lib.py --> Contains utilities for building pytorch dataset.
- network_lib.py --> Contains network architecture.
- network_modules.py --> Contains modules used to build network architecture.
- train.py --> Trains the model.
- train_lib.py --> Contains utilities for training the networks.
