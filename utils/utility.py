import os
import pickle as pkl
from tqdm import tqdm
import torch

from fusion.baseline import *
from query.linegraph_match import *
from fusion.fusion import *


DATASET = {"movie": {"JSON": (1, 5), "KG": (5, 10), "CSV": (10, 14)},
           "book": {"JSON": (1, 4), "CSV": (4, 7), "XML": (7, 11)},
           "flight": {"CSV": (1, 11), "JSON": (11, 21)},
           "stock": {"CSV": (1, 11), "JSON": (11, 21)}}


def set_random_seed(seed):
    if seed != -1:
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def query_search(data_file, data_graph, query_id, matcher: LineGraphQuerier, lm, line_transform=True):
    query_path = os.path.join(data_file, "query.json")
    query_dict = {"query_id": query_id,
                  "query_path": query_path}
    qry_g = GraphSet(is_query=True, **query_dict)
    qry_g = LineGraph(qry_g, lm=lm, is_query=True) if line_transform else qry_g

    query_cands = {}
    for src_id, src_g in enumerate(data_graph):
        query_ans = matcher.query(qry_g, src_g, source_quality=0.97)
        query_cands[src_id] = query_ans
    return query_cands


def prepare_graph(data_file, data_name, types, lm, line_transform=True):
    cache_dir = os.path.join(data_file, "linegraph")
    if os.path.exists(cache_dir):
        src_graphs = load_graph(cache_dir, data_name, types)
        return src_graphs
    else:
        src_graphs = []
        data_file = os.path.join(data_file, "data2kg")
        os.makedirs(cache_dir, exist_ok=True)
        for type in types:
            (sid, eid) = DATASET[data_name][type]
            for src_id in tqdm(range(sid, eid), desc="loading sources"):
                id2entity = os.path.join(data_file, "ent_ids_{}".format(src_id))
                id2relation = os.path.join(data_file, "rel_ids_{}".format(src_id))
                triples = os.path.join(data_file, "triples_{}".format(src_id))

                graph_dict = {"entity_path": id2entity,
                            "relation_path": id2relation,
                            "triple_path": triples}

                graph = GraphSet(is_query=False, **graph_dict)
                if line_transform:
                    graph = LineGraph(graph, lm=lm, is_query=False)
                src_graphs.append(graph)
                with open(os.path.join(cache_dir, "source_graph_{}.pkl".format(src_id)), "wb") as g_f:
                    pkl.dump(graph, g_f)
        return src_graphs


def prepare_query(data_file, qry_num, lm, line_transform=True):
    qry_set = []
    for qry_id in range(qry_num):
        query_path = os.path.join(data_file, "query.json")
        query_dict = {"query_id": qry_id,
                      "query_path": query_path}
        qry_g = GraphSet(is_query=True, **query_dict)
        qry_g = LineGraph(qry_g, lm=lm, is_query=True) if line_transform else qry_g
        qry_set.append(qry_g)
        del qry_g
    return qry_set


def load_graph(data_root, data_name, types):
    src_graphs = []
    for type in types:
        (sid, eid) = DATASET[data_name][type]
        for src_id in range(sid, eid):
            graph_path = os.path.join(data_root,
                                      "source_graph_{}.pkl".format(src_id))
            with open(graph_path, "rb") as g_f:
                src_g = pkl.load(g_f)
            src_graphs.append(src_g)
    return src_graphs


def load_fusion_model(model_name, source_num, config):
    if model_name == "FusionQuery":
        fusioner = EMFusioner(source_num=source_num, **config[model_name])
    elif model_name == "CASE":
        fusioner = CASEFusion(source_num=source_num, **config[model_name])
    elif model_name == "DART":
        fusioner = DARTFusion(source_num=source_num, **config[model_name])
    elif model_name == "LTM":
        fusioner = LTMFusion(source_num=source_num, **config[model_name])
    elif model_name == "TruthFinder":
        fusioner = TruthFinder(source_num=source_num, **config[model_name])
    elif model_name == "MajorityVoter":
        fusioner = MajorityVoter(source_num=source_num)
    else:
        fusioner = None
    return fusioner
