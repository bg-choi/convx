import regex
import unicodedata
import string
import numpy as np
from collections import Counter

class SimpleTokenizer(object):
    ALPHA_NUM = r'[\p{L}\p{N}\p{M}]+'
    NON_WS = r'[^\p{Z}\p{C}]'

    def __init__(self):
        """
        Args:
            annotators: None or empty set (only tokenizes).
        """
        self._regexp = regex.compile(
            '(%s)|(%s)' % (self.ALPHA_NUM, self.NON_WS),
            flags=regex.IGNORECASE + regex.UNICODE + regex.MULTILINE
        )

    def tokenize(self, text, uncased=False):
        matches = [m for m in self._regexp.finditer(text)]
        if uncased:
            tokens = [m.group().lower() for m in matches]
        else:
            tokens = [m.group() for m in matches]
        return tokens


def _normalize(text):
    return unicodedata.normalize('NFD', text)


def match_answer(answers, text, tokenizer=SimpleTokenizer()) -> bool:
    """Check if a document contains an answer string."""
    text = _normalize(text)
    text = tokenizer.tokenize(text, uncased=True)

    for answer in answers:
        answer = _normalize(answer)
        answer = tokenizer.tokenize(answer, uncased=True)
        if answer == text:
            return True
    
    return False

def get_exact_match_score(outputs,answers):
    """
    outputs: [string1,string2]
    answers: [
                [string1_1,string1_2],
                [string2_1,string2_2]
             ]
    """
    import numpy as np
    assert len(outputs) == len(answers)
    if not isinstance(answers[0],list):
        answers = [[x] for x in answers]
    exact_match_scores = []
    for output, answer in zip(outputs, answers):
        if match_answer(answer,output): # EM evaluation
            exact_match_scores.append(1.0)
        else:
            exact_match_scores.append(0.0)
        

    exact_match = round(sum(exact_match_scores)/len(outputs), 4)

    return exact_match, exact_match_scores


def has_answer(answers, text, tokenizer=SimpleTokenizer()) -> bool:
    """Check if a document contains an answer string."""
    text = _normalize(text)
    text = tokenizer.tokenize(text, uncased=True)

    for answer in answers:
        answer = _normalize(answer)
        answer = tokenizer.tokenize(answer, uncased=True)
        for i in range(0, len(text) - len(answer) + 1):
            if answer == text[i: i + len(answer)]:
                return True
    return False

def get_substring_match_score(outputs,answers):
    """
    outputs: [string1,string2]
    answers: [
                [string1_1,string1_2],
                [string2_1,string2_2]
             ]
    """
    import numpy as np
    assert len(outputs) == len(answers)
    if not isinstance(answers[0],list):
        answers = [[x] for x in answers]
    substring_match_scores = []
    answer_lengths = []
    for output,answer in zip(outputs,answers):
        if has_answer(answer,output): # MATCH evaluation
            substring_match_scores.append(1.0)
        else:
            substring_match_scores.append(0.0)
        
        answer_lengths.append(len(output.split()))

    substring_match = round(sum(substring_match_scores)/len(outputs), 4)
    lens = round(np.mean(answer_lengths), 4)

    return substring_match,substring_match_scores


def rougel_score(prediction, ground_truth):
    from rouge import Rouge
    rouge = Rouge()
    # no normalization
    try:
        scores = rouge.get_scores(prediction, ground_truth, avg=True)
    except ValueError:  # "Hypothesis is empty."
        return 0.0
    return scores["rouge-l"]["f"]


def rl(prediction, ground_truths):
    return max([rougel_score(prediction, gt) for gt in ground_truths])

def normalize_answer(s):
    def remove_articles(text):
        return regex.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)

    return f1

def f1(prediction, ground_truths):
    return max([f1_score(prediction, gt) for gt in ground_truths])

def eval_truthfulqa(outputs,answers):

    f1_scores = []
    rl_scores = []
    for output,answer in zip(outputs,answers):
        f1_scores.append(f1(output, answer))
        rl_scores.append(rl(output, answer))

    F1 = round(np.mean(f1_scores), 4)
    RL = round(np.mean(rl_scores), 4)

    return F1, f1_scores, RL, rl_scores