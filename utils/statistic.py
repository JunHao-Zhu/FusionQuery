from collections import Counter
import logging


class Statistics:
    def __init__(self):
        self.examples = 0
        self.tps = 0
        self.tns = 0
        self.fps = 0
        self.fns = 0
        self.query_time = 0
        self.fusion_time = 0

    def dice_dist(self, str1, str2):
        cnt1 = Counter(str1)
        cnt2 = Counter(str2)
        unions = cnt1 & cnt2
        return 2 * sum(unions.values()) / (len(str1) + len(str2))

    def judge(self, truth, answers, veracity, thresh, record_once=False):
        tps, tns, fps, fns = 0, 0, 0, 0
        for aid, ans in enumerate(answers):
            flag = False
            if veracity[aid] >= thresh:
                for gt in truth:
                    if self.dice_dist(ans.lower(), gt.lower()) > .5:
                        tps += 1
                        flag = True
                        break
                if not flag:
                    fps += 1
            else:
                for gt in truth:
                    if self.dice_dist(ans.lower(), gt.lower()) > .5:
                        fns += 1
                        flag = True
                        break
                if not flag:
                    tns += 1
        self.update(tps, tns, fps, fns)
        if record_once:
            prec = 100 * tps / max(tps + fps, 1)
            rec = 100 * tps / max(tps + fns, 1)
            f1 = 2 * prec * rec / max(prec + rec, 1)
            acc = 100 * (tps + tns) / max(tps + tns + fps + fns, 1)
            logging.info("| f1: {f1:7.2f} | prec: {prec:7.2f} | rec: {rec:7.2f} | acc: {acc:7.2f} |".format(
                f1=f1, prec=prec, rec=rec, acc=acc))

    def update(self, tps=0, tns=0, fps=0, fns=0):
        examples = tps + tns + fps + fns
        self.tps += tps
        self.tns += tns
        self.fps += fps
        self.fns += fns
        self.examples += examples

    def f1(self):
        prec = self.precision()
        recall = self.recall()
        return 2 * prec * recall / max(prec + recall, 1)

    def precision(self):
        return 100 * self.tps / max(self.tps + self.fps, 1)

    def recall(self):
        return 100 * self.tps / max(self.tps + self.fns, 1)

    def accuracy(self):
        return 100 * (self.tps + self.tns) / self.examples

    def timing(self, each_qry_time, each_fus_time):
        self.query_time += each_qry_time
        self.fusion_time += each_fus_time

    def print_stat_info(self):
        logging.info((' | F1: {f1:7.2f} | Prec: {prec:7.2f} | Rec: {rec:7.2f} | Acc: {acc:7.2f} || '
                      'Query time: {q_time:7.2f} | Fusion time: {f_time:7.2f}').format(
                      f1=self.f1(),
                      prec=self.precision(),
                      rec=self.recall(),
                      acc=self.accuracy(),
                      q_time=self.query_time,
                      f_time=self.fusion_time))
