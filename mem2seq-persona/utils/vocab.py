import spacy
import ftfy

class SpacyLowerTokenizer:
    def __init__(self):
        self.tokenizer = spacy.load('en', disable=['parser', 'tagger', 'ner', 'textcat'])

    def __call__(self, string):
        string = ftfy.fix_text(string)
        words = [t.text.strip() for t in self.tokenizer(string)]
        words = [w.lower() for w in words if w]

        return words

class CustomVocab:
    we = '</w>'

    pad_token = '<pad>'
    sos_token = '<sos>'
    eos_token = '<eos>'
    
    # Special tokens to denote persona, speaker1, speaker2.
    p_token = '<p>'
    s1_token = '<s1>'
    s2_token = '<s2>'
    
    # Special token for pointer.
    sentinel = '$$$$'

    @staticmethod
    def from_files(vocab_path, codes_path, *args, **kwargs):
        with open(vocab_path, 'r', encoding='utf-8') as vocab_file:
            vocab = [t.strip() for t in vocab_file.readlines()]

        with open(codes_path, 'r', encoding='utf-8') as codes_file:
            codes = [c.strip() for c in codes_file.readlines()]

            if codes[0].startswith('#version'):
                codes = codes[1:]

            codes = [tuple(c.split()) for c in codes if c]

        return CustomVocab(vocab, codes, *args, **kwargs)

    @staticmethod
    def get_pairs(string):
        if len(string) < 2:
            return set()

        return set(zip(string[:-1], string[1:]))

    def __init__(self, vocab, codes, tokenizer=SpacyLowerTokenizer()):
        #TODO: add check for special tokens
        self.spec_tokens = [CustomVocab.pad_token, CustomVocab.sos_token, CustomVocab.eos_token,
                            CustomVocab.p_token, CustomVocab.s1_token, CustomVocab.s2_token,
                            CustomVocab.sentinel]
        
        vocab = self.spec_tokens + vocab
        
        self.token2id = {t: i for i, t in enumerate(vocab)}
        self.id2token = {i: t for i, t in enumerate(vocab)}
        self.bpe_ranks = dict(zip(codes, range(len(codes))))
        self.tokenizer = tokenizer
        self.cache = {}
        
        self.pad_id = self.token2id[self.pad_token]
        self.sos_id = self.token2id[self.sos_token]
        self.eos_id = self.token2id[self.eos_token]
        
        self.p_id = self.token2id[self.p_token]
        self.s1_id = self.token2id[self.s1_token]
        self.s2_id = self.token2id[self.s2_token]
        
        self.sentinel_id = self.token2id[self.sentinel]

    def __len__(self):
        return len(self.token2id)

    def _bpe(self, token):
        if token in self.cache:
            return self.cache[token]

        word = tuple(token[:-1]) + (token[-1] + CustomVocab.we,)
        pairs = CustomVocab.get_pairs(word)

        if not pairs:
            return (token + CustomVocab.we,)

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break

            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word)-1 and word[i+1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1

            word = tuple(new_word)

            if len(word) == 1:
                break
            else:
                pairs = CustomVocab.get_pairs(word)

        self.cache[token] = word

        return word

    def string2ids(self, string):
        tokens = self.tokenizer(string)
        bpe_tokens = sum([self._bpe(t) for t in tokens], tuple())
        ids = [self.token2id[t] for t in bpe_tokens if t in self.token2id]

        return ids


    def ids2string(self, ids):
        bpe_tokens = [self.id2token[id] for id in ids]
    
        return ''.join(bpe_tokens).replace(CustomVocab.we, ' ')
    
def merge_bpe(words):
        for i, word in enumerate(words):
            if word == '<eos>' and i < len(words) - 1:
                words = words[:i+1]
        return ''.join(words).replace(CustomVocab.we, ' ')