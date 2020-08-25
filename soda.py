#!/uer/bin/env python
import argparse
import json
from tqdm import tqdm
import numpy as np

from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
#from nlpeval.bert_score import BertScore

from dataset import ANETCaptions
from utils import iou, remove_nonascii 

class SODA:
    def __init__(self, data, soda_type="c", tious=None, scorer="Meteor", verbose=False):
        #self.data = data
        self.preds = data.preds
        self.gts = data.gts
        self.gt_vids = data.gt_vids
        self.soda_type = soda_type
        self.tious = [0.0] if tious is None else tious
        self.tokenizer = PTBTokenizer()
        self.scorer = eval(scorer)()
        self.scorer_name = scorer
        self.verbose = verbose

        if soda_type == "a":    # averaging F-measure scores with IoU threshold = 0.9, 0.7, 0.5, 0.3
            self.soda_func = self.soda_a
        elif soda_type == "b":  # F-measure, where IoU threshold is set to 0.
            self.soda_func = self.soda_b
        elif soda_type == "c":  # F-measure, utilizing the IoU x METEOR score
            self.soda_func = self.soda_c
        else:
            raise NotImplementedError

    def calc_iou_matrix(self, preds, golds):
        #print(preds["timestamps"], gt["timestamps"])
        return np.array([[iou(pred,ct) for pred in preds["timestamps"]] for ct in golds['timestamps']])

    def calc_score_matrix(self, preds, golds):
        # Reformat to fit the input of pycocoevalcap scorers.
        p_sent, g_sent = preds["sentences"], golds["sentences"]
        res = {index:p for index, p in enumerate(p_sent)}
        gts = [{index:g for index in range(len(p_sent))} for i,g  in enumerate(g_sent)]
        return np.array([self.scorer.compute_score(res ,gt)[1] for gt in gts])

    def evaluate(self):
        if self.verbose: print(f"\n| Running SODA {self.soda_type}.")
        tious = self.tious
        all_scores = np.zeros((len(self.tious)))
        n_pred = 0
        n_gold = 0
        for vid in tqdm(self.gt_vids, disable=not self.verbose):
            pred = self.preds[vid]
            gold = self.gts[vid]
            # create matrix
            _iou = self.calc_iou_matrix(pred, gold)
            scores = self.calc_score_matrix(pred, gold)
            for i, tiou in enumerate(self.tious):
                iou = np.copy(_iou)    
                iou[iou<tiou] = 0.0
                max_score, pairs = self.soda_func(iou, scores)
                all_scores[i] += max_score
            (n_g, n_p) = scores.shape
            n_pred += n_g
            n_gold += n_p

        precision = all_scores / n_pred
        recall = all_scores / n_gold
        f1 = 2 * precision * recall / (precision + recall)
        # average scores across all the tIoUs
        if self.verbose:
            for i, tiou in enumerate(tious):
                partial_result = {self.scorer_name: [precision[i], recall[i], f1[i]]}
                print_score(partial_result, description=f"tIoU: {tiou}")

        final_scores = [np.mean(precision), np.mean(recall), np.mean(f1)]
        result = {self.scorer_name: final_scores}
        print_score(result)
        return result
        
    def soda_a(self, iou, scores):
        _, pairs = self.chased_dp_assignment(iou)
        r,c = (*zip(*pairs),)
        max_score = np.sum(scores[r,c])
        return max_score, pairs
    
    def soda_b(self, iou, scores):
        # same as soda_a
        _, pairs = self.chased_dp_assignment(iou)
        r,c = (*zip(*pairs),)
        max_score = np.sum(scores[r,c])
        return max_score, pairs

    def soda_c(self, iou, scores):
        max_score, pairs = self.chased_dp_assignment(iou*scores)
        return max_score, pairs

    def chased_dp_assignment(self, scores):
        """ 
        Run dp matching
        Recurrence:  
            dp[i,j] = 
                max(dp[i-1,j], dp[i-1,j-1] + scores[i,j], dp[i,j-1])
        """
        M, N = scores.shape
        dp = - np.ones((M,N))
        path = np.zeros((M,N))
        def transition(i,j):
            if dp[i,j] >= 0: return dp[i,j]
            elif i==0 and j==0:                                                                              
                state = [-1, -1, scores[i,j]]
            elif i==0:
                state = [-1, transition(i, j-1), scores[i,j]]
            elif j==0:
                state = [transition(i-1,j), -1, scores[i,j]]
            else:
                state = [transition(i-1, j), transition(i, j-1), transition(i-1, j-1) + scores[i,j]]
            dp[i,j] = np.max(state)
            path[i,j] = np.argmax(state)
            return dp[i,j]
        def get_pairs(i, j):
            p = np.where(path[i][:j+1]==2)[0]
            if i!=0 and len(p)==0: return get_pairs(i-1, j)
            elif i==0 or p[-1]==0: return  [(i, p[-1])]
            else: return get_pairs(i-1, p[-1]-1) + [(i, p[-1])]
        N, M = scores.shape
        max_score = transition(N-1, M-1)
        pairs = get_pairs(N-1, M-1)
        return max_score, pairs

def print_score(result, description="SODA result"):
    prf = ["precision", "recall", "f1_score"]
    print('-' * 80)
    print(description)
    print('-' * 80)
    for scorer_name, score in result.items():
        print(f'| scorer:{scorer_name}')
        for k,v in zip(prf, score):
            print(f"\t{k}:{v*100:2.4f}")

def main(args):
    # Call coco eval
    data = ANETCaptions(gt_file=args.references,
                             pred_file=args.prediction,
                             verbose=args.verbose,
                             )
    data.preprocess()
    if args.soda_type == 'a':
        tious = args.tious
    else:
        tious = None
    evaluator = SODA(data,
                    soda_type=args.soda_type, 
                    tious=tious, 
                    scorer=args.metric, 
                    verbose=args.verbose
                    )
    evaluator.evaluate() 

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--prediction', type=str, required=True, default='sample.json',
                        help='system output file with json format for ActivityNet Challenge')
    parser.add_argument('-r', '--references', type=str, nargs='+', default=['./data/val_1.json', './data/val_2.json'],
                        help='reference files with ground truth captions')
    parser.add_argument('-m', '--metric', type=str, default="Meteor", choices=['Meteor', 'Cider',  'BertScore'],
                        help='choice evaluation metrics for SODA')
    parser.add_argument('-s', '--soda_type', type=str, default="c", choices=['a', 'b',  'c'],
                        help='choice evaluation metrics for SODA')
    parser.add_argument('--tious', type=float,  nargs='+', default=[0.3, 0.5, 0.7, 0.9],
                        help='list of the tIoUs (only for SODA-a)')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='print details')
    args = parser.parse_args()

    main(args)


