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


### Added the following lines of code due to attribute error
<b> AttributeError: </b>  module 'PIL.Image' has no attribute 'Resampling'

Added in 
1. train_loghtning.py
2. predict.py

Ref : https://stackoverflow.com/questions/71738218/module-pil-has-not-attribute-resampling


if not hasattr(PIL.Image, 'Resampling'):  # Pillow<9.0
    PIL.Image.Resampling = PIL.Image

