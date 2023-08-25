## BG Removal Models

### Models
1 model added:<br>
  <b>u2net</b>: currently in production, no changes to original code, trained on our dataset

More models can be added under `src`. Code Structure should be similar to u2net.
Models should be exported to onnx format for production use.

-----

### Datasets
A sample of 15000 datapoints has been added under `datasets` which is pushed to gcloud using dvc. 
Custom datasets can be added similarly.

Any script used for dataset preparation should be under `scripts`.

-----

### Testing
Test cases are added to `tests/SIVI-TEST`. It also contains the ideal masks for the images. Model output should be close the masks provided.

