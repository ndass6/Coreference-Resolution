import itertools
import coref_rules
from nltk import wordnet

pronoun_list=['it','he','she','they','this','that']
poss_pronoun_list=['its','his','her','their']
oblique_pronoun_list=['him','her','them']
def_list=['the','this','that','these','those']
indef_list=['a','an','another']

def minimal_features(markables,a,i):
    """Compute a minimal set of features for antecedent a and mention i

    :param markables: list of markables for the document
    :param a: index of antecedent
    :param i: index of mention
    :returns: features
    :rtype: dict
    """
    f = dict()
    if a == i:
        f['new-entity'] = 1
    else:
        if coref_rules.exact_match(markables[a], markables[i]):
            f['exact-match'] = 1
        if coref_rules.match_last_token(markables[a], markables[i]):
            f['last-token-match'] = 1
        if coref_rules.match_on_content(markables[a], markables[i]):
            f['content-match'] = 1
        if coref_rules.overlap(markables[a], markables[i]):
            f['crossover'] = 1
    return f


def pronoun_feature(markables, a, i):
    """Compute an advanced set of features for antecedent a and mention i

    :param markables: list of markables for the document
    :param a: index of antecedent
    :param i: index of mention
    :returns: features
    :rtype: dict
    """
    f = dict()
    if coref_rules.exact_match_no_pronouns(markables[a], markables[i]):
        f['pronoun'] = 1
    return f

def distance_features(x,a,i,
                      max_mention_distance=10,
                      max_token_distance=10):
    """compute a set of distance features for antecedent a and mention i

    :param x: markable list for document
    :param a: antecedent index
    :param i: mention index
    :param max_mention_distance: upper limit on mention distance
    :param max_token_distance: upper limit on token distance
    :returns: feature dict
    :rtype: dict
    """
    f = dict()
    if a != i:
        mention_distance = min(i - a, max_mention_distance)
        token_distance = min(x[i]['start_token'] - x[a]['end_token'], max_token_distance)
        f['mention-distance-' + str(mention_distance)] = 1
        f['token-distance-' + str(token_distance)] = 1
    return f
    
###### Feature combiners ######

def make_feature_union(feat_func_list):
    """return a feature function that is the union of the feature functions in the list

    :param feat_func_list: list of feature functions
    :returns: feature function
    :rtype: function
    """
    def f_out(x, a, i):
        combinedFeatures = {}
        for feat_func in feat_func_list:
            features = feat_func(x, a, i)
            for feature in features:
                combinedFeatures[feature] = features[feature]
        return combinedFeatures
    return f_out

def make_feature_cross_product(feat_func1,feat_func2):
    """return a feature function that is the cross-product of the two feature functions

    :param feat_func1: a feature function
    :param feat_func2: a feature function
    :returns: another feature function
    :rtype: function
    """
    def f_out(x, a, i):
        crossedFeatures = {}
        features1 = feat_func1(x, a, i)
        features2 = feat_func2(x, a, i)
        for key1 in features1:
            for key2 in features2:
                crossedFeatures[key1 + "-" + key2] = features1[key1] * features2[key2]
        return crossedFeatures
    return f_out

def make_bakeoff_features():
    """return a feature function for the bakeoff

    :returns: another feature function
    :rtype: function
    """
    def f_out(x,a,i):
        return make_feature_union([make_feature_cross_product(pronoun_feature, distance_features),
            make_feature_cross_product(pronoun_feature, minimal_features),
            make_feature_cross_product(minimal_features, distance_features),
            minimal_features, pronoun_feature, distance_features])(x, a, i)
    return f_out