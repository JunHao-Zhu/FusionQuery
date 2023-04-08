import time
from typing import Union
import logging

from query.graph import LineGraph
from query.linegraph_match import LineGraphQuerier
from utils.statistic import Statistics


class FusionQuery:
    def __init__(self, aggregator, src_graphs, threshold, veracity_thres, device):
        self.src_graphs = src_graphs
        self.match_thresh = threshold
        self.veracity_thres = veracity_thres
        self.matcher = LineGraphQuerier(device=device)
        self.aggregator = aggregator
        self.statistic = Statistics()

    def query(self, qry_g: Union[LineGraph]):
        qry_cand = {}
        for src_id, src_g in enumerate(self.src_graphs):
            qry_ans = self.matcher.query(qry_g, src_g, self.match_thresh[src_id])
            qry_cand[src_id] = qry_ans
        return qry_cand

    def fusion(self, qry_cand):
        self.aggregator.prepare_for_fusion(qry_cand)
        self.match_thresh = self.aggregator.iterate_fusion(threshold=self.match_thresh)
        return self.aggregator.ans_set, self.aggregator.veracity

    def evaluate(self, qry_set:[LineGraph], timing=True):
        for qid, qry_g in enumerate(qry_set):
            qry_start_t = time.time()
            qry_ans = self.query(qry_g)
            qry_end_t = time.time()
            each_qry_t = qry_end_t - qry_start_t
            fus_start_t = time.time()
            fus_ans, fus_val = self.fusion(qry_ans)
            fus_end_t = time.time()
            each_fus_t = fus_end_t - fus_start_t
            if timing:
                self.statistic.timing(each_qry_t, each_fus_t)
            if (qid + 1) % 20 == 0:
                logging.info("Record @ {}-th query".format(qid + 1))
                logging.info("runtime (s): {:.4f}".format(self.statistic.fusion_time))
                self.statistic.judge(qry_g.graph.answer, fus_ans, fus_val,
                                     thresh=self.veracity_thres, record_once=True)
            else:
                self.statistic.judge(qry_g.graph.answer, fus_ans, fus_val,
                                     thresh=self.veracity_thres)
