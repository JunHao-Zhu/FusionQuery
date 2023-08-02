# **FusionQuery: On-demand Fusion Queries over Multi-source Heterogeneous Data**

FusionQuery, an efficient on-demand fusion query framework, 
which consists of the query stage and the fusion stage. which consists of the query stage 
and the fusion stage. During the fusion stage, we develop an EM-style algorithm, which 
iteratively updates data veracity and source trustworthiness; furthermore, we design an 
incremental estimation method of source trustworthiness to address the lackness of 
sufficient observations.

## Requirements

* python 3.8
* sentence-transformers 2.2.2
* faiss-gpu 1.7.2
* numpy 1.23.1
* pytorch 1.12.1

## Dataset

We conduct experiments on two datasets, _Movie_ and _Book_. We released KG version of these 
two datasets in the `data`. Each data source is stored in three files. Entities in source n, 
are stored in `ent_ids_n`, relations are stored in `rel_ids_n` and triples are stored in 
`triples_n`. The queries conducted on the datasets are stored in `query.json`.

## Setting

Hyperparameters of the data fusion methods used in our experiments are presented in `config.json`.

## Run code
Perform the entire workflow of FusionQuery. 
```shell
python main.py --data_root "./data/movie" \
--data_name movie \
--fusion_model FusionQuery \
--types JSON KG CSV \
--iters 20 \
--thres_for_query 0.9 \
--thres_for_fusion 0.5
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