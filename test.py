


import spacy
from spacy import displacy

sentences = [
    'a rusted fire hydrant sitting in the middle of a forest .',
    'a person riding a surf board on a wave',
    'a man riding a wave on a surfboard in the ocean .'
]


nlp = spacy.load("en_core_web_md")
doc = nlp(sentences[0])
displacy.serve(doc, style="dep")