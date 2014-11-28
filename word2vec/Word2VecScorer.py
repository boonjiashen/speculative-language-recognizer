import numpy as np

class Word2VecScorer(object):
    """Scorer for a word2vec or doc2vec model, using Google's set of analogous
    comparisons.
    """

    def __init__(self, model):
        "Takes in a word2vec or doc2vec model"
        self.comparisons = []  # a list of 4-ples
        self._build_comparisons(model)

    def _build_comparisons(self, model,
            evaluation_filename='data/questions-words.txt'):
        """Build the list of comparisons appropriate for this model.

        We only make use of grammar comparisons since it's the only categories
        where our training data has the required vocabulary. We also ignore
        comparisons if any of the four items in the comparison are not in the
        model's vocabulary.
        """

        # Each line is a comparison 'x y X Y' i.e. x is to y what X is to Y EXCEPT
        # for categories, which are lines that start with a colon, e.g.
        #: city-in-state
        #: family
        #: gram1-adjective-to-adverb
        #: gram2-opposite
        with open(evaluation_filename) as fid:
            lines = fid.read().splitlines()

        # Grab all test-able comparisons as a list of 4-ples
        vocab = set(model.vocab.keys())
        for line in lines:

            # Skip category headers
            if line[0] == ':':
                continue

            # Don't bother testing if any of these words aren't in vocab
            tokens = line.split()
            if not set(tokens) <= vocab:
                continue

            self.comparisons.append(tokens)

    def score(self, model, topn, percentage=True, return_correct_comparisons=False):
        """Score a model by how many correct comparisons it makes from the
        set of analogies

        Score is zero if there are no comparison to be made.

        topn - correct answer should be within top #N predictions
        """

        # Start comparisons
        correct_inds = []  # Inds of comparisons that were predicted right
        for ci, comparison in enumerate(self.comparisons):

            # E.g. 'stupid clever slow fast'
            # which implies stupid is to clever what slow is to fast.
            # We want to predict 'fast' from the first three words
            pos1, neg1, pos2, neg2 = comparison

            try:

                # Make predictions (these come with confidence levels)
                predictions_and_confidence = model.most_similar(
                        positive=[pos1, pos2], negative=[neg1], topn=topn)
                predictions = zip(*predictions_and_confidence)[0]

                # Update performance metrics
                if neg2 in predictions:
                    correct_inds.append(ci)
            except KeyError:
                continue

        score = len(correct_inds) if self.comparisons else 0
        if percentage:
            score = float(score) / len(self.comparisons)
        correct_comparisons = [self.comparisons[i] for i in correct_inds]

        if return_correct_comparisons:
            return score, correct_comparisons
        else:
            return score

    def mean_similarity(self, model):
        """1 number metric on how well a word2vec model is doing.
        
        The higher the better.
        """

        model.init_sims()

        def get_norm_vector(word):
            "Get normalized word vector of a word in a model"
            return model.syn0norm[model.vocab[word].index]

        similarities = []  # How well model does for each comparison
        for pos1, neg1, pos2, neg2 in self.comparisons:

            # Get offsets pos1-neg1 and pos2-neg2
            # These vectors should be similar in a well-trained model
            offsets = [get_norm_vector(pos) - get_norm_vector(neg)
                    for pos, neg in [(pos1, neg1), (pos2, neg2)]]

            # Get similarity between the two offsets
            similarity = np.dot(offsets[0], offsets[1])

            similarities.append(similarity)

        mean_similarity = np.mean(similarities) if similarities else 0

        return mean_similarity

