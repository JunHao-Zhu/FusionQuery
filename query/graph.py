import json
import faiss
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer


class GraphSet:
    indicator = ["what", "where", "when", "who", "dummy node"]

    def __init__(self, is_query=True, **kwargs):
        self._vertex_set = {}
        self._edge_set = {}
        self.edge_type = {}
        self.answer = [] if is_query else None
        self.tgt_node = None

        if not is_query:
            with open(kwargs["entity_path"], 'r') as ent_file:
                for line in ent_file:
                    e_id, ent = line.strip().split("\t")
                    self._vertex_set[int(e_id)] = ent
            with open(kwargs["relation_path"], 'r') as rel_file:
                for line in rel_file:
                    r_id, rel = line.strip().split("\t")
                    self.edge_type[int(r_id)] = rel
            with open(kwargs["triple_path"], 'r') as trip_file:
                for line in trip_file:
                    h_id, r_id, t_id = line.strip().split("\t")
                    self._edge_set[(int(h_id), int(t_id))] = int(r_id)
        else:
            query_id = kwargs["query_id"]
            with open(kwargs["query_path"], 'r') as query_file:
                query_data = json.load(query_file)
            query_json = query_data[query_id]
            query_graph = query_json["queryGraph"]
            for e_id, ent in query_graph["ent_ids"].items():
                self._vertex_set[int(e_id)] = ent
                if ent in GraphSet.indicator:
                    self.tgt_node = int(e_id)
            for r_id, rel in query_graph["rel_ids"].items():
                self.edge_type[int(r_id)] = rel
            for line in query_graph["triples"]:
                h_id, r_id, t_id = line.strip().split('-')
                self._edge_set[(int(h_id), int(t_id))] = int(r_id)
            self.answer.extend(query_json["answers"])

    @property
    def vertex_set(self):
        return self._vertex_set

    @property
    def edge_set(self):
        return self._edge_set

    def adjacent_set(self):
        vertex_num = len(self._vertex_set)
        adjacency = [[] for _ in range(vertex_num)]

        for (v1, v2) in self._edge_set:
            adjacency[v1].append((v1, v2))
            adjacency[v2].append((v1, v2))
        return adjacency

    def neighbor(self, vertex_id):
        assert vertex_id < len(self._vertex_set), "vertex id is out of index!"

        adjacency = self.adjacent_set()
        adj_edges = adjacency[vertex_id]
        neighbors = []
        for i in range(len(adj_edges)):
            v1, v2 = adj_edges[i][0], adj_edges[i][1]
            if v1 != vertex_id:
                neighbors.append(v1)
            elif v2 != vertex_id:
                neighbors.append(v2)
            else:
                exit()
        return neighbors


class LineGraph:
    def __init__(self, graph: GraphSet, lm, is_query=True):
        self.graph = graph
        self.is_query = is_query
        self.node_idx = {}
        self.node_attr = []
        self.edge_set = None
        self.tgt_nodes = None
        self.s_emb = []
        self.o_emb = []
        self.rel_category = {}
        self.lm = lm
        self.embed_size = self.lm.get_sentence_embedding_dimension()
        self.p_s_faiss = {}  # faiss.IndexFlatL2(self.lm.get_sentence_embedding_dimension())
        self.p_o_faiss = {}  # faiss.IndexFlatL2(self.lm.get_sentence_embedding_dimension())
        self.p_emb_faiss = faiss.IndexFlatL2(self.embed_size)
        self.initialize()

    @property
    def vertices(self):
        return self.node_idx

    def initialize(self):
        self.rel2embed()
        if self.is_query:
            self.triple2node_in_qry()
        else:
            self.rel_category = {rel: [] for rel in self.graph.edge_type}
            self.p_s_faiss = {rel: faiss.IndexFlatL2(self.embed_size) for rel in self.graph.edge_type}
            self.p_o_faiss = {rel: faiss.IndexFlatL2(self.embed_size) for rel in self.graph.edge_type}
            self.triple2node_in_src()

    @torch.no_grad()
    def rel2embed(self):
        rel2emb = self.lm.encode(list(self.graph.edge_type.values()),
                                 show_progress_bar=False,
                                 convert_to_tensor=True)
        self.rel2emb = F.normalize(rel2emb, dim=-1).cpu()
        torch.cuda.empty_cache()
        if not self.is_query:
            self.p_emb_faiss.add(self.rel2emb.numpy())

    @torch.no_grad()
    def triple2node_in_qry(self):
        assert self.is_query, "line graph transition for query graph"
        self.tgt_nodes = []
        for nid, (n1, n2) in enumerate(self.graph.edge_set.keys()):
            if n1 == self.graph.tgt_node:
                node_emb = self.lm.encode(self.graph.vertex_set[n2],
                                          show_progress_bar=False,
                                          convert_to_tensor=True)
                node_emb = F.normalize(node_emb, dim=-1).cpu()
                self.s_emb.append(None)
                self.o_emb.append(node_emb.numpy())
                self.tgt_nodes.append(0)
            elif n2 == self.graph.tgt_node:
                node_emb = self.lm.encode(self.graph.vertex_set[n1],
                                          show_progress_bar=False,
                                          convert_to_tensor=True)
                node_emb = F.normalize(node_emb, dim=-1).cpu()
                self.s_emb.append(node_emb.numpy())
                self.o_emb.append(None)
                self.tgt_nodes.append(1)
            else:
                node_triple = [self.graph.vertex_set[n1], self.graph.vertex_set[n2]]
                node_emb = self.lm.encode(node_triple,
                                          show_progress_bar=False,
                                          convert_to_tensor=True)
                node_emb = F.normalize(node_emb, dim=-1).cpu()
                self.s_emb.append(node_emb[0].numpy())
                self.o_emb.append(node_emb[1].numpy())
                self.tgt_nodes.append(None)
            self.node_idx[nid] = (n1, n2)
        torch.cuda.empty_cache()

    @torch.no_grad()
    def triple2node_in_src(self):
        assert not self.is_query, "line graph transition for source graph"
        for nid, (n1, n2) in enumerate(self.graph.edge_set.keys()):
            node_triple = [self.graph.vertex_set[n1], self.graph.vertex_set[n2]]
            node_emb = self.lm.encode(node_triple,
                                      show_progress_bar=False,
                                      convert_to_tensor=True)
            node_emb = F.normalize(node_emb, dim=-1).cpu()
            self.s_emb.append(node_emb[0])
            self.o_emb.append(node_emb[1])
            self.node_idx[nid] = (n1, n2)
            self.rel_category[self.graph.edge_set[(n1, n2)]].append(nid)
        for p in self.graph.edge_type.keys():
            lnid = self.rel_category[p]
            self.p_s_faiss[p].add(torch.stack(self.s_emb, dim=0).numpy()[lnid])
            self.p_o_faiss[p].add(torch.stack(self.o_emb, dim=0).numpy()[lnid])
            self.rel_category[p] = np.array(self.rel_category[p])

    def structure_complete(self, cand_nodes: set):
        self.edge_set = set()
        for idx, ln1 in enumerate(cand_nodes):
            line_node_1 = self.node_idx[ln1]
            for ln2 in cand_nodes[idx + 1:]:
                if ln2 == ln1:
                    continue
                line_node_2 = self.node_idx[ln2]
                if len(set(line_node_1) & set(line_node_2)) > 0:
                    self.edge_set.add(tuple(sorted((ln1, ln2))))
        return self.edge_set
