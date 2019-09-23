from collections import Counter
from nltk.translate import bleu_score as nltkbleu

def f1_score(pred_items, gold_items):
    """
    Compute precision, recall and f1 given a set of gold and prediction items.
    :param pred_items: iterable of predicted values
    :param gold_items: iterable of gold values
    :return: f1
    """
    common = Counter(gold_items) & Counter(pred_items)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_items)
    recall = 1.0 * num_same / len(gold_items)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def bleu(guess, answer):
    """Compute approximate BLEU score between guess and answer."""
    return nltkbleu.sentence_bleu(
        [answer],
        guess,
        smoothing_function=nltkbleu.SmoothingFunction(epsilon=0.001).method1
    )