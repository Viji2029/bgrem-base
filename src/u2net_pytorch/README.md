OG model: [u2net](https://github.com/xuebinqin/U-2-Net)

`./train.sh` is entrypoint

To export, run `export.py`, change paths in the file
NOTE: For export, we need to change the model code - the model has to return `sigmoid` output
Comments is there in `export.py` and `u2net.py` files on where to make the changes

This model is currently in production, performs well but unable to still match our results on latest dataset.

Before running training run the command `pip uninstall -y torch_xla`

## Model

No changes to arch, same as original. Only during export we have the changes
We've just wrapped it in pytorch lightning to make training easier

## Data
Production is trained on `bgrem_duts_plus`
We've tried other datasets as well as the latest one (`bgrem_duts_shopify_unsp_31k`), didn't perform well on our test cases

### Pre-post processing
Training happens on square images, we pad with 0 to fill remaining - default code itself
Logic being in mask, 0 is background, so padding with background itself, and it has to predict background back - 0

## Hyperparams
Don't remember much on hyperparams for this, mostly we ran using default itself
