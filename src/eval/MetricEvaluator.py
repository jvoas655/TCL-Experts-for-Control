# pip install rouge-score
# pip install bleu
# https://pypi.org/project/bleu/
# https://pypi.org/project/rouge-score/
from dataclasses import dataclass
from bleu import list_bleu, multi_list_bleu
from rouge_score import rouge_scorer
from collections import defaultdict
from nltk.translate.bleu_score import sentence_bleu


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
            # Note There are several BLUE python packages, including from nltk.translate.bleu_score import sentence_bleu
            # and a huggingface library equivalent
            if type(reference) == list:
                ref_list = reference
            elif type(reference) == str:
                ref_list = [reference]
            if type(hypothesis) == list:
                hypo_list = hypothesis
            elif type(hypothesis) == str:
                hypo_list = [hypothesis]
            bleu_scores = defaultdict(list)
            for ref, hyp in zip(ref_list, hypo_list):
                ref_split = [ref.split()]
                hyp_split = hyp.split()
                
                #  Calculate BLEU 1-Gram
                one_gram =  sentence_bleu(ref_split, hyp_split, weights=(1, 0, 0, 0))
                print('Cumulative 1-gram: %f' % one_gram)
                bleu_scores["one-gram"].append(one_gram)

                #  Calculate BLEU 2-Gram
                two_gram =  sentence_bleu(ref_split, hyp_split, weights=(0.5, 0.5, 0, 0))
                print('Cumulative 2-gram: %f' % two_gram )
                bleu_scores["two-gram"].append(two_gram)

                #  Calculate BLEU 3-Gram
                three_gram =  sentence_bleu(ref_split, hyp_split, weights=(0.333, 0.333, .333, 0))
                print('Cumulative 3-gram: %f' % three_gram )
                bleu_scores["three_gram"].append(three_gram)

                #  Calculate BLEU 4-Gram
                four_gram =  sentence_bleu(ref_split, hyp_split, weights=(0.25, 0.25, .25, .25))
                print('Cumulative 4-gram: %f' % four_gram )
                bleu_scores["four-gram"].append(four_gram)
                
            return bleu_scores
            # if len(hypothesis) == 1:
            #     assert len(reference) == 1
            #     blue_scores = list_bleu([reference], hypothesis)
            #     return blue_scores
            # # use multi_list_bleu if multiple hypotheses
            # elif type(hypothesis)== list:
            #     # Case 1: only have one hypothesis
            #     # if len(reference) == reference == 1:
            #     blue_scores = multi_list_bleu([reference], [hypothesis])
            #     return blue_scores
            # else:
            #     raise NotImplementedError()
            # blue_score = list_bleu([[reference[0]]], [hypothesis[0]])
            # return blue_score
        else:
            raise NotImplementedError()

evaluator = MetricEvaluator()
ref = ['The quick brown fox jumps over the lazy dog', "Reference sentence"]
hypothesis = [ 'The quick brown dog jumps on the log.', "Totally unrelated hypothesis"]
rouge_scores = evaluator.calc_score("rouge", ref, hypothesis)
print(f"Rouge Scores: {rouge_scores}")
blue_scores =  evaluator.calc_score("blue", ref, hypothesis)
print(f"blue_sfores: {blue_scores}")