import numpy as np
import json
from dataset import ANETCaptions
from utils import * 

class ANETCaptions:
    def __init__(self, gt_file, pred_file, verbose=False):
        self.pred_keys = ['results', 'version', 'external_data']
        self.verbose = verbose
        self.gts, self.gt_vids = self.load_ground_truth(gt_file)
        self.preds = self.load_prediction(pred_file)
        self.tokenizer = PTBTokenizer()

    def load_ground_truth(self, filenames):
        if self.verbose: print(f"| Loading ground truths: {filenames}.")
        if isinstance(filenames, str):
            filenames = [filenames]
        gt_vids = set()
        gt = defaultdict(dict)
        gts = []
        for filename in filenames:
            with open(filename, "r") as f:
                _gt = json.load(f) 
            gt_vids.update(_gt.keys())
            gts.append(_gt)
        for vid in gt_vids:
            t, s = [], []
            for _g in gts:
                if vid not in _g: continue
                t += _g[vid]["timestamps"]
                s += _g[vid]["sentences"]
            sort_t, sort_s = list(zip(*sorted(zip(t, s), key=lambda x: x[0][0])))
            gt[vid]["timestamps"] = sort_t
            gt[vid]["sentences"] = sort_s
        if self.verbose:
            print(f"stats:\n\t n_files: {len(filenames)}, n_videos: {len(gt_vids)}")
        return dict(gt), gt_vids 

    def load_prediction(self, filename):
        if self.verbose: print(f"\n| Loading predictions: {filename}.")
        with open(filename, 'r') as f:
            pred = json.load(f)
        # If the json file doesn’t have enough attribute
        if not all([key in pred.keys() for key in self.pred_keys]):
            raise IOError('Please input a correct format prediction file.')
        results = {}
        for vid in pred['results']:
            #if vid not in self.gt_vids: continue
            results[vid] = sorted(pred["results"][vid], key=lambda x: x["timestamp"][0])
        self.check_videos(self.gt_vids, results.keys())
        return results

    def preprocess(self):
        if self.verbose: print("\n| Preprocessing captions...")
        p_spliter = [0]
        g_spliter = [0]
        times = {}
        cur_preds = {}
        cur_gts = {}
        for i, vid in enumerate(self.gt_vids): 
            cur_preds.update({j+p_spliter[-1]:[{"caption": remove_nonascii(p["sentence"])}] for j,p in enumerate(self.preds[vid])})
            cur_gts.update({j+g_spliter[-1]:[{"caption": remove_nonascii(p)}] for j,p in enumerate(self.gts[vid]["sentences"])})
            times[i] = [p["timestamp"] for p in self.preds[vid]]
            p_spliter.append(p_spliter[-1] + len(times[i]))
            g_spliter.append(g_spliter[-1] + len(self.gts[vid]["sentences"]))
        tokenize_preds = self.tokenizer.tokenize(cur_preds)
        tokenize_gts = self.tokenizer.tokenize(cur_gts)
        for i, vid in enumerate(self.gt_vids): 
            _p = [tokenize_preds[j] for j in range(p_spliter[i],p_spliter[i+1])]
            _g = [tokenize_gts[j] for j in range(g_spliter[i],g_spliter[i+1])]
            self.preds[vid] = {"timestamps":times[i], "sentences":_p}
            self.gts[vid]["sentences"] = _g

    def check_videos(self, gold_vid, pred_vid):
        not_appear = set(gold_vid) - set(pred_vid)
        if len(not_appear) > 0:
            print((f"Warning: some videos in ground truth file are not appeared in prediction file!\n"
                f"\t{len(not_appear)} videos are not predicted: {not_appear}"))
        self.gt_vids = list(set(gold_vid) & set(pred_vid))

