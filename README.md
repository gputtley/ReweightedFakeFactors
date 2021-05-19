# ReweightedFakeFactors

This module is used to derive reweighted fake factors and apply them to trees.

## Making BDT files

To make the relevant reweighted fake factor BDTs run this command.

``` bash
python scripts/ff_reweight_ml.py --channel=mt --year=2017
```
This will output the unnormalised fake factor BDTs into the BDTs folder as well as the normalisation json.

## Applying reweighted fake factors to trees

To apply the reweighted fake factors to a single root file run this command.

``` bash
python scripts/add_ff_reweight_to_trees.py --input_location=/vols/cms/gu18/Offline/output/MSSM/mssm_2017_v10 --output_location=/vols/cms/gu18/Offline/output/MSSM/mssm_2017_v10_new --filename=SingleMuonB_mt_2017.root --channel=mt --year=2017 --splitting=100000 --offset=0
```
The offset and splitting is used to fix memory problems. You may have to loop over offsets depending on the size of the tree. You will then need to hadd the relevant outputs with the following command.

``` bash
hadd -f /vols/cms/gu18/Offline/output/MSSM/mssm_2017_v10_new/SingleMuonB_mt_2017.root /vols/cms/gu18/Offline/output/MSSM/mssm_2017_v10_new/SingleMuonB_mt_2017_*
```
To batch add reweighted fake factors to a folder of root files run this command.

``` bash
python scripts/batch_add_ff_reweight_to_trees.py --input_location=/vols/cms/gu18/Offline/output/MSSM/mssm_2017_v10 --output_location=/vols/cms/gu18/Offline/output/MSSM/mssm_2017_v10_new
```
To hadd the resulting output run this command.

``` bash
python scripts/batch_add_ff_reweight_to_trees.py --output_location=/vols/cms/gu18/Offline/output/MSSM/mssm_2017_v10_new --hadd
```

## Links

Reweighted with Boosed Decision Trees: [article](https://arogozhnikov.github.io/2015/10/09/gradient-boosted-reweighter.html), [paper](https://arxiv.org/pdf/1608.05806.pdf), [notebook](https://github.com/arogozhnikov/hep_ml/blob/master/notebooks/DemoReweighting.ipynb), [github repository](https://github.com/arogozhnikov/hep_ml) 


