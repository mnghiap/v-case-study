import torch

class Vocab:
    """
    Class to manage the vocabulary and the list of entity.
    Constraints: each entity is just a token, for simplicity

    Args:
        vocab_txt: Path to a vocab.txt file, where each line has
            the format <SYMBOL> <INDEX>
        entity_list_txt: Path to entity_list.txt file, where each line
            is an <ENTITY NAME>
        sb_symbol: Sentence boundary symbol <SB>. We don't distinguish
            between BOS and EOS for simplicity
    """

    def __init__(self, vocab_txt, entity_list_txt, sb_symbol, blank_symbol, entity_boundary_symbol):
        self.symbol_to_idx = dict()
        self.idx_to_symbol = dict()
        self.entity_idxs = []
        self.sb_symbol = sb_symbol
        self.blank_symbol = blank_symbol
        self.entity_boundary_symbol = entity_boundary_symbol

        with open(vocab_txt, "r") as vocab:
            for line in vocab.readlines():
                symbol, idx = line.strip().split()
                idx = int(idx)
                self.symbol_to_idx[symbol] = idx
                self.idx_to_symbol[idx] = symbol

        with open(entity_list_txt, "r") as entity_list:
            for line in entity_list.readlines():
                self.entity_idxs.append(self.symbol_to_idx[line.strip()])

        self.sb_idx = self.symbol_to_idx[sb_symbol]
        self.blank_idx = len(self.symbol_to_idx)
        self.entity_boundary_idx = self.symbol_to_idx[entity_boundary_symbol]
        self.vocab_size_no_blank = len(self.symbol_to_idx)

    
    def generate_entity_mask(self):
        """
        Generate a torch mask a boolean torch mask of size V+1 (blank counted),
        where the True indices are indices of the entities.

        Returns:
            mask: Indices with True are indicies of the entities (V+1)
        """
        mask = torch.full((self.vocab_size_no_blank+1,), fill_value=False, dtype=torch.bool)
        entity_idxs = torch.tensor(self.entity_idxs, dtype=torch.long)
        mask.scatter_(0, entity_idxs, True)
        return mask
