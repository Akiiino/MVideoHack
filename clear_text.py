import pymorphy2
import re

class Clearer:
    def __init__(self):
        self.morph = pymorphy2.MorphAnalyzer()
        
    def work(self, text):
        try:
            text = text.lower()
        except AttributeError as e:
            return ''
        letters_only = re.sub("([^а-я]+)", " ", text)
        letters_only = letters_only.strip()
        if letters_only == '':
            return ''
        text = letters_only.split()
        newtext = ' '.join([self.morph.parse(word)[0].normal_form for word in text])
        return newtext