import re
import nltk
nltk.download('averaged_perceptron_tagger_eng', quiet=True)
class PhonemeConverter:
    def __init__(self):
        try:
            from g2p_en import G2p
            self.g2p = G2p()
        except ImportError:
            raise ImportError("pip install g2p_en")

        arpabet = [
            "<pad>", "<unk>",
            "AA","AE","AH","AO","AW","AY",
            "B","CH","D","DH","EH","ER","EY",
            "F","G","HH","IH","IY","JH","K","L",
            "M","N","NG","OW","OY","P","R","S",
            "SH","T","TH","UH","UW","V","W","Y","Z","ZH"
        ]

        self.vocab = {p:i for i,p in enumerate(arpabet)}

    def normalize_text(self, text):
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        return text

    def remove_stress(self, phoneme):
        return ''.join([c for c in phoneme if not c.isdigit()])

    def text_to_phonemes(self, text):
        text = self.normalize_text(text)
        phonemes = self.g2p(text)

        indices = []

        for p in phonemes:
            base = self.remove_stress(p)
            indices.append(self.vocab.get(base, self.vocab["<unk>"]))

        return phonemes, indices

    @property
    def vocab_size(self):
        return len(self.vocab)