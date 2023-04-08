import math
import numpy as np

from query.graph import GraphSet, LineGraph

from typing import Union


class LineGraphQuerier:
    def __init__(self, device="cpu"):
        self.device = device

    def match(self, **kwargs):
        trustworthy = kwargs["source_quality"]
        threshold = math.sqrt(2 - 2 * trustworthy)
        answer_pair = []
        rel_in_qry = list(self.qry_g.graph.edge_set.values())
        rel_dist, rel_map = self.src_g.p_emb_faiss.search(self.qry_g.rel2emb[rel_in_qry].numpy(), k=1)
        for nid, _ in self.qry_g.vertices.items():
            tgt_id = self.qry_g.tgt_nodes[nid]
            s_emb, o_emb = self.qry_g.s_emb[nid], self.qry_g.o_emb[nid]
            mapped_rel = rel_map[nid].item()
            if s_emb is not None and o_emb is None:
                s_emb = s_emb[np.newaxis, :]
                res_num, dist, lnid = self.src_g.p_s_faiss[mapped_rel].range_search(s_emb, threshold)
            else:
                o_emb = o_emb[np.newaxis, :]
                res_num, dist, lnid = self.src_g.p_o_faiss[mapped_rel].range_search(o_emb, threshold)
            # else:
            #     pass
            sorted_idx = np.argsort(dist)
            lnid = lnid[sorted_idx][-300:]
            dist = dist[sorted_idx][-300:]
            matched_ln = self.src_g.rel_category[mapped_rel][lnid]
            dist = 1 - dist ** 2 / 2
            for idx, score in zip(matched_ln, dist):
                tgt_node = self.src_g.vertices[idx][tgt_id]
                answer_pair.append((self.src_g.graph.vertex_set[tgt_node], score))
        return answer_pair

    def query(self, subg: Union[GraphSet, LineGraph], orig: Union[GraphSet, LineGraph], source_quality=0.8):
        self.src_g = orig
        self.qry_g = subg

        res = self.match(source_quality=source_quality)
        return res
