## DVC Setup

#### To setup DVC run the following command
`pip install dvc==2.19.0 dvc[gs]==0.0.1`

#### Add any dataset to DVC, run this comand from the root of the repository.
`dvc add datasets/[datasets_name]`<br>
`dvc push datasets/[datasets_name]`<br>
 This will stage a .dvc file. Then push the .dvc file to Github.

#### Pull the sample dataset
`dvc pull datasets/bgrem_duts_shopify_unsp_31k_sampled`