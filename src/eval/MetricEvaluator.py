# pip install rouge-score
# pip install bleu
# https://pypi.org/project/bleu/
# https://pypi.org/project/rouge-score/
from dataclasses import dataclass
from bleu import list_bleu, multi_list_bleu
from rouge_score import rouge_scorer
from collections import defaultdict

@dataclass
class MetricEvaluator:
    rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)


    def calc_score(self, method_type, reference: list, hypothesis: list):
        if method_type.lower() == "rouge":
            # loop through all reference sentences and hypothesis to get larger dictionary of all rouge scores
            if type(reference) == list:
                rouge_scores = defaultdict(list)
                for hypo, ref in zip(hypothesis, reference):
                    score = self.rouge_scorer.score(ref, hypo)
                    for key, val in score.items():
                        rouge_scores[key].append(val)
                return rouge_scores
            else:
                return self.rouge_scorer.score(reference, hypothesis)
        elif method_type.lower() == "blue":
            # TODO: This code below is wrong. There are several BLUE python packages, including from nltk.translate.bleu_score import sentence_bleu
            # and a huggingface library equivalent
            # use if only one hypothesis
            if len(hypothesis) == 1:
                assert len(reference) == 1
                blue_scores = list_bleu([reference], hypothesis)
                return blue_scores
            # use multi_list_bleu if multiple hypotheses
            elif type(hypothesis)== list:
                # Case 1: only have one hypothesis
                # if len(reference) == reference == 1:
                blue_scores = multi_list_bleu([reference], [hypothesis])
                return blue_scores
            else:
                raise NotImplementedError()
            # blue_score = list_bleu([[reference[0]]], [hypothesis[0]])
            # return blue_score
        else:
            raise NotImplementedError()

evaluator = MetricEvaluator()
# ref = ['The quick brown fox jumps over the lazy dog', "Reference sentence"]
# hypothesis = [ 'The quick brown dog jumps on the log.', "Totally unrelated hypothesis"]
# rouge_scores = evaluator.calc_score("rouge", ref, hypothesis)
# print(f"Rouge Scores: {rouge_scores}")
# blue_sfores =  evaluator.calc_score("blue", ref, hypothesis)
# print(f"blue_sfores: {blue_sfores}")

# from bleu import list_bleu
# ref = ['it is a white cat .',
#              'wow , this dog is huge .']
# hyp = ['it is a white kitten .',
#             'wowww , the dog is huge !']

# ref = ['it is a white cat .', "it is a black elaphant"]
# hyp = ['it is a white kitten .']
# blue_scores =  evaluator.calc_score("blue", ref, hyp*2)
# print(blue_scores)

# print(multi_list_bleu([ref, ref1], [hyp, hyp1], detok=False))
# print("done")