### Rule-based coreference resolution  ###########
# Lightly inspired by Stanford's "Multi-pass sieve"
# http://www.surdeanu.info/mihai/papers/emnlp10.pdf
# http://nlp.stanford.edu/pubs/conllst2011-coref.pdf

import nltk

pronouns = ['I','me','mine','you','your','yours','she','her','hers','he','him','his','it','its','they','them','their','theirs','this','those','these','that','we','our','us','ours']
downcase_list = lambda toks : [tok.lower() for tok in toks]

############## Pairwise matchers #######################

def exact_match(m_a,m_i):
    """return True if the strings are identical

    :param m_a: antecedent markable
    :param m_i: referent markable
    :returns: True if the strings are identical
    :rtype: boolean
    """
    return downcase_list(m_a['string'])==downcase_list(m_i['string'])

def exact_match_no_pronouns(m_a,m_i):
    """return True if strings are identical and are not pronouns

    :param m_a: antecedent markable
    :param m_i: referent markable
    :returns: True if the strings are identical and are not pronouns
    :rtype: boolean
    """
    match = exact_match(m_a, m_i)
    if match:
        if len(m_a['string']) > 1:
            return True
        if m_a['string'][0] in pronouns:
            return False
        if m_i['string'][0] in pronouns:
            return False
        return True
    else:
        return False

def match_last_token(m_a,m_i):
    """return True if final token of each markable is identical

    :param m_a: antecedent markable
    :param m_i: referent markable
    :rtype: boolean
    """
    return m_a['string'][-1].lower() == m_i['string'][-1].lower()

def match_last_token_no_overlap(m_a,m_i):
    """
    :param m_a: antecedent markable
    :param m_i: referent markable
    :returns: True if final tokens match and strings do not overlap
    :rtype: boolean
    """
    return match_last_token(m_a, m_i) and not overlap(m_a, m_i)

def overlap(m_a, m_i):
    """
    :param m_a: antecedent markable
    :param m_i: referent markable
    :returns: True if the position ranges overlap
    :rtype: boolean
    """
    return len(set(range(m_a['start_token'], m_a['end_token'])).intersection(range(m_i['start_token'], m_i['end_token']))) > 0

def match_on_content(m_a, m_i):
    """
    :param m_a: antecedent markable
    :param m_i: referent markable
    :returns: True if all match on all "content words" (defined by POS tag) and markables do not overlap
    :rtype: boolean
    """
    content_words = ['NN', 'NNS', 'NNP', 'NNPS', 'CD', 'PRP', 'PRP$']
    m_a_content_words = downcase_list([m_a['string'][i] for i, tag in enumerate(m_a['tags']) if tag in content_words])
    m_i_content_words = downcase_list([m_i['string'][i] for i, tag in enumerate(m_i['tags']) if tag in content_words])
    return m_a_content_words == m_i_content_words and not overlap(m_a, m_i)

########## helper code

def most_recent_match(markables,matcher):
    """given a list of markables and a pairwise matcher, return an antecedent list
    assumes markables are sorted

    :param markables: list of markables
    :param matcher: function that takes two markables, returns boolean if they are compatible
    :returns: list of antecedent indices
    :rtype: list
    """
    antecedents = range(len(markables))
    for i,m_i in enumerate(markables):
        for a,m_a in enumerate(markables[:i]):
            if matcher(m_a,m_i):
                antecedents[i] = a
    return antecedents

def make_resolver(pairwise_matcher):
    """convert a pairwise markable matching function into a coreference resolution system, which generates antecedent lists

    :param pairwise_matcher: function from markable pairs to boolean
    :returns: function from markable list and word list to antecedent list
    :rtype: function

    The returned lambda expression takes a list of words and a list of markables. 
    The words are ignored here. However, this function signature is needed because
    in other cases, we want to do some NLP on the words.
    """
    return lambda markables : most_recent_match(markables,pairwise_matcher)