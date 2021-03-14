import nltk
import torch
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')


def extract_phrases(my_tree, phrase):
    """
      This method is extract the valid phrases according to the grammar from the parse tree of a sentence.
      Parameters
      ----------
      my_tree: This is the parse tree of the sentence
      phrases: Non terminals of the grammar rules. This basically defines the structure of phrases we are trying to extract

      Returns
      -------
      a list of tokens
    """
    my_phrases = []
    if my_tree.label() in phrase:
        my_phrases.append(my_tree.copy(True))

    for child in my_tree:

        if type(child) is nltk.Tree:

            list_of_phrases = extract_phrases(child, phrase)
            if len(list_of_phrases) > 0:
                my_phrases.extend(list_of_phrases)

    return my_phrases


def custom_tokenizer(text):
    """
      This method tokenizes the input text
      Parameters
      ----------
      text: The sentence to be tokenized

      Returns
      -------
      a list of tokens
    """
    grammar = """NP: {<RB>*<DT>?(<JJ>|<JJS>|<JJR>)*(<NN>|<NNP>|<NNS>)+}
               RBJJ:{(<RB>|<RBR>|<RBS>)+(<JJ>|<JJS>|<JJR>)+}
               JJ: {<JJ>}
               JJS: {<JJS>}
               JJR: {<JJR>}
               VB: {<VB>}
               VBG: {<VBG>}
               VBN: {<VBN>}
               VBP: {<VBP>}
               VBZ: {<VBZ>}
               VBD: {<VBD>}
               MD: {<MD>}
               RB: {<RB>}
               RBR: {<RBR>}
               RBS: {<RBS>}
               PRP: {<PRP>}
               IN: {<IN>}
               CC: {<CC>}
                """
    cp = nltk.RegexpParser(grammar)
    sentence = nltk.pos_tag(nltk.tokenize.word_tokenize(text))
    tree = cp.parse(sentence)
    list_of_noun_phrases = extract_phrases(tree,
                                           ['NP', 'VBD', 'IN', 'VB', 'VBN', 'VBP', 'VBZ', 'RBR', 'RB', 'RBS', 'PRP',
                                            'JJ', 'JJS', 'JJR', 'RBJJ', 'CC'])
    tokens = []
    for phrase in list_of_noun_phrases:
        tokens.append("_".join([x[0] for x in phrase.leaves()]))
    return tokens


def create_embeddings(word2idx, weights,embeddingbag):
    """
      This method is used to create embedding representations for the phrases.
      Parameters
      ----------
      word2idx: A mapping from words to index
      weights: GloVe word vectors for individual words.
      embeddingbag: an instance of EmbeddingBag to average the word vectors.

      Returns
      -------
      word embeddings for all the words including the phrases.
    """
    new_weights=weights.detach().clone()
    for word,index in list(word2idx.items()):
        if '_' in word:
            tokens = word.split('_')
            token_id=[]
        for token in tokens:
            token_id.append(word2idx[token])
        inputs = torch.LongTensor([token_id])
        new_vec = embeddingbag(inputs)
        new_weights[index] = new_vec
        token_id=[]
    return new_weights
