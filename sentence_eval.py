from bert_score import score, load_model

class BertScore:
    def __init__(self, lang="en", model_type="bert-large-uncased"):
        self.lang = lang
        self.model_type=model_type
        self.model = load_model(model_type=model_type, lang=lang)

    def compute_score(self, gts, res):
        assert gts.keys()==res.keys()
        # convert dict to list of str
        cands = list(map(lambda x:x[0], res.values()))
        refs = list(map(lambda x:x[0], gts.values()))
        (P, R, F), hashname = score(cands, refs, model=self.model, model_type=self.model_type, lang=self.lang, return_hash=True)
        #print(f'{hashname}: P={P.mean().item():.6f} R={R.mean().item():.6f} F={F.mean().item():.6f}')
        F = F.numpy()
        return F.mean(), F

    def method(self):
        return "BertScore"
