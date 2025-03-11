import os
import logging
import argparse
import json
from datetime import datetime
from sentence_transformers import SentenceTransformer

from FusionQuery.framework import FusionQuery
from utils.utility import *


def set_logger(name):
    log_dir = "./logger"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join('./logger', name)
    
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def parse_args(args=None):
    parser = argparse.ArgumentParser(description='fusion query')
    parser.add_argument('--data_root', type=str, default="./data/movie",
                        help='dataset name')
    parser.add_argument("--data_name", type=str, default="movie",
                        help="saved root")
    parser.add_argument("--lm_path", type=str, default="../../.cache/huggingface/transformers/sentence-bert")
    parser.add_argument("--fusion_model", type=str, default="FusionQuery",
                        help="select from (FusionQuery, CASE, DART, LTM, TruthFinder, MajorityVoter)")
    parser.add_argument("--types", nargs='+', required=True)
    parser.add_argument("--iters", type=int, default=20, help="iteration for EM algorithm")
    parser.add_argument("--thres_for_query", type=float, default=0.9, help="threshold the for query stage")
    parser.add_argument("--thres_for_fusion", type=float, default=0.5, help="threshold for value veracity")
    parser.add_argument("--gpu", type=int, default=0, help="gpu device")
    parser.add_argument("--seed", type=int, default=2021, help='random seed')
    return parser.parse_args()


def main():
    args = parse_args()
    data_root = args.data_root
    data_name = args.data_name
    qry_num = 210 if data_name == "movie" else 100
    types = args.types
    lm_path = args.lm_path
    device = "cuda:{}".format(args.gpu) if args.gpu != -1 else "cpu"
    max_iters = args.iters
    fusion_model = args.fusion_model
    query_thres = args.thres_for_query
    veracity_thres = args.thres_for_fusion
    with open("config.json", 'r') as config_file:
        config = json.load(config_file)
    if max_iters > 0:
        config["max_iters"] = max_iters

    set_random_seed(args.seed)
    cur_time = ' ' + datetime.now().strftime("%F %T")
    type_name = '-'.join(types)
    logger_name = args.data_name + '-' + fusion_model + '-time-' + type_name + '-' + cur_time.replace(':', '-')
    set_logger(logger_name)
    logging.info(args)
    logging.info(config[fusion_model])
    logging.info("seed: {}".format(args.seed))

    logging.info("{:*^50}".format("[Data Preparation]"))
    lang_model = SentenceTransformer(lm_path).to(device)
    src_g = prepare_graph(data_root, data_name, types, lm=lang_model, line_transform=True)
    src_num = len(src_g)
    logging.info("The number of data sources: {}".format(src_num))
    logging.info("{:*^50}".format("[Query Preparation]"))
    qry_g = prepare_query(data_root, qry_num=qry_num, lm=lang_model)
    logging.info("The number of queries: {}".format(qry_num))
    logging.info("{:*^50}".format("[END]"))
    fusion_model = load_fusion_model(fusion_model, src_num, config)
    pipeline = FusionQuery(aggregator=fusion_model,
                           src_graphs=src_g,
                           threshold=[query_thres] * src_num,
                           veracity_thres=veracity_thres,
                           device=device)
    pipeline.evaluate(qry_g, timing=True)
    pipeline.statistic.print_stat_info()


if __name__ == '__main__':
    main()
