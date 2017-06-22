import numpy as np
from collections import defaultdict
import coref

def mention_rank(markables,i,feats,weights):
    """ return top scoring antecedent for markable i

    :param markables: list of markables
    :param i: index of current markable to resolve
    :param feats: feature function
    :param weights: weight defaultdict
    :returns: index of best scoring candidate (can be i)
    :rtype: int
    """
    maxScore = float('-inf')
    maxPos = 0
    for pos in range(i + 1):
        features = feats(markables, pos, i)
        score = sum([features[key] * weights[key] for key in features])
        if score > maxScore:
            maxScore = score
            maxPos = pos
    return maxPos

def compute_instance_update(markables,i,true_antecedent,feats,weights):
    """Compute a perceptron update for markable i.
    This function should call mention_rank to determine the predicted antecedent,
    and should make an update if the true antecedent and predicted antecedent *refer to different entities*

    Note that if the true and predicted antecedents refer to the same entity, you should not
    make an update, even if they are different.

    :param markables: list of markables
    :param i: current markable
    :param true_antecedent: ground truth antecedent
    :param feats: feature function
    :param weights: defaultdict of weights
    :returns: defaultdict of updates
    :rtype: defaultdict
    """
    pred_antecedent = mention_rank(markables,i,feats,weights)

    update = defaultdict(float)

    pred_features = feats(markables, pred_antecedent, i)
    true_features = feats(markables, true_antecedent, i)
    if (markables[pred_antecedent]['entity'] != markables[true_antecedent]['entity'] or ((true_antecedent < i and pred_antecedent == i) or (pred_antecedent < i and true_antecedent == i))):
        for key in true_features:
            update[key] = true_features[key] - (pred_features[key] if key in pred_features else 0.0)
        for key in pred_features:
            update[key] = (true_features[key] if key in true_features else 0.0) - pred_features[key]

    return update

def train_avg_perceptron(markables,features,N_its=20):
    weights = defaultdict(float)
    tot_weights = defaultdict(float)
    weight_hist = []
    T = 0
    
    for it in xrange(N_its):
        num_wrong = 0 #helpful but not required to keep and print a running total of errors
        for document in markables:
            true_antecedents = coref.get_true_antecedents(document)
            for pos in range(len(document)):
                update = compute_instance_update(document, pos, true_antecedents[pos], features, weights)
                if update:
                    for key in update:
                        weights[key] += update[key]
                    num_wrong += 1

                for key in weights:
                    tot_weights[key] += weights[key]
                T += 1
        print num_wrong,

        # update the weight history
        weight_hist.append(defaultdict(float))
        for feature in tot_weights.keys():
            weight_hist[it][feature] = tot_weights[feature] / T

    return weight_hist

# helpers
def make_resolver(features,weights):
    return lambda markables : [mention_rank(markables,i,features,weights) for i in range(len(markables))]
        
def eval_weight_hist(markables,weight_history,features):
    scores = []
    for weights in weight_history:
        score = coref.eval_on_dataset(make_resolver(features,weights),markables)
        scores.append(score)
    return scores
