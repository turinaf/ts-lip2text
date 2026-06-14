import nltk
import string
from nltk.corpus import cmudict
from g2p_en import G2p

nltk.download('cmudict')
nltk.download('averaged_perceptron_tagger_eng')
d = cmudict.dict()

def word_to_phoneme(word):
    word = word.lower().strip(string.punctuation)
    
    if word in d:
        return d[word][0]
    else:
        phonemes = []
        for char in word:
            phonemes.append(char)
        return phonemes
g2p = G2p()

# print(word_to_phoneme("Thi is a sentence"))
print(g2p("Speech"))

