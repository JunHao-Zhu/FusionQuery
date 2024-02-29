# **FusionQuery**

Python implementation of FusionQuery in paper [FusionQuery: On-demand Fusion Queries over Multi-source
 Heterogeneous Data](https://github.com/JunHao-Zhu/FusionQuery/blob/main/technical_report.pdf).

## Dependencies
 
* Python 3.8
* sentence-transformers 2.2.2
* faiss-gpu 1.7.2
* numpy 1.23.1
* pytorch 1.12.1

## Datasets

This repo contains two datasets, _Movie_ and _Book_. We released KG version of these 
two datasets in the `data`. Each data source is stored in three files. Entities in source n, 
are stored in `ent_ids_n`, relations are stored in `rel_ids_n` and triples are stored in 
`triples_n`. The queries conducted on the datasets are stored in `query.json`.

More datasets can be found in this [web](http://lunadong.com/fusionDataSets.htm)

## Run code
Perform the entire workflow of FusionQuery. 
```shell
python main.py --data_root "./data/movie" \
--data_name movie \
--fusion_model FusionQuery \
--types JSON KG CSV \
--iters 20 \
--thres_for_query 0.9 \
--thres_for_fusion 0.4
```
The more detailed information about arguments is listed as follows.

|Arguments|Explainations|Default|
|----|----|----|
|`--data_root`|root path of data|`../data/movie`|
|`--data_name`|data name used in the current experiment|`movie`|
|`--fusion_model`|data fusion methods used in the framework (e.g., FusionQuery, DART, CASE, etc.)|`FusionQuery`|
|`--types`|data types used in the current experiment (a list)|`JSON KG CSV`|
|`--iters`|maximum iterations for convergence|`20`|
|`--thres_for_query`|initial matching threshold $\tau$|`0`|
|`--thres_for_fusion`|threshold for data veracity|`0.5`|
|`--gpu`|the gpu device id|`0`|
|`--seed`|random seed|`2021`|