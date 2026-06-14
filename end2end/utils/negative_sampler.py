import random

class NegativeSampler:
    def __init__(self, all_transcripts):
        self.all_transcripts = [t for t in all_transcripts if isinstance(t, str) and t.strip()]

    def _random_other(self, positive_transcript, candidates):
        if not candidates:
            return positive_transcript
        neg = random.choice(candidates)
        tries = 0
        while neg == positive_transcript and tries < 20:
            neg = random.choice(candidates)
            tries += 1
        return neg

    def _token_overlap(self, a, b):
        a_set = set(a.split())
        b_set = set(b.split())
        if not a_set and not b_set:
            return 0.0
        denom = len(a_set | b_set)
        return (len(a_set & b_set) / denom) if denom > 0 else 0.0

    def _hard_negative(self, positive_transcript):
        # Approximate hard negative: pick the most lexically similar transcript
        # from a random candidate subset to keep this lightweight.
        candidates = [t for t in self.all_transcripts if t != positive_transcript]
        if not candidates:
            return positive_transcript

        sampled = random.sample(candidates, k=min(64, len(candidates)))
        sampled.sort(key=lambda t: self._token_overlap(positive_transcript, t), reverse=True)
        return sampled[0]

    def _phoneme_similar_negative(self, positive_transcript):
        """Pick a transcript with high character bigram overlap as a phonetic proxy.

        Dataset-agnostic: works for GRID word transcripts and digit-string
        transcripts without relying on any hardcoded vocabulary.
        """
        candidates = [t for t in self.all_transcripts if t != positive_transcript]
        if not candidates:
            return positive_transcript

        def bigram_overlap(a, b):
            a_bg = set(a[i:i + 2] for i in range(len(a) - 1))
            b_bg = set(b[i:i + 2] for i in range(len(b) - 1))
            denom = len(a_bg | b_bg)
            return len(a_bg & b_bg) / denom if denom > 0 else 0.0

        sampled = random.sample(candidates, k=min(64, len(candidates)))
        sampled.sort(key=lambda t: bigram_overlap(positive_transcript, t), reverse=True)
        return sampled[0]
        
    def sample(self, positive_transcript):
        strategy = random.choice(['hard', 'wrong', 'shuffled', 'phoneme_similar', 'random'])
        candidates = self.all_transcripts

        if strategy == 'hard':
            return self._hard_negative(positive_transcript)
        
        if strategy == 'wrong' or strategy == 'random':
            return self._random_other(positive_transcript, candidates)
            
        elif strategy == 'shuffled':
            words = positive_transcript.split()
            random.shuffle(words)
            return " ".join(words)
            
        elif strategy == 'phoneme_similar':
            # Use character bigram overlap as a phonetic proxy — works for both
            # word-based (GRID) and digit-based transcripts, without hardcoding
            # any vocabulary.
            return self._phoneme_similar_negative(positive_transcript)
            
        return positive_transcript