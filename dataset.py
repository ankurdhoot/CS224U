import torch
import random
import os
import json

DATA_HOME = os.path.join("data", "nlidata")

SNLI_HOME = os.path.join(DATA_HOME, "snli_1.0")

class NLIExample(object):
    """For processing examples from SNLI or MultiNLI.
    Parameters
    ----------
    d : dict
        Derived from a JSON line in one of the corpus files. Each
        key-value pair becomes an attribute-value pair for the
        class.
    """
    def __init__(self, d):
        for k, v in d.items():
            setattr(self, k, v)

    def __str__(self):
        return repr(self)

    def __repr__(self):
        d = {k: v for k, v in self.__dict__.items() if not k.startswith('__')}
        return """"NLIExample({})""".format(d)
    
        
    
class SNLIDataset(torch.utils.data.Dataset):
    """SNLI Dataset."""

    def __init__(self, src_filename, samp_percentage=None):
        """
        Args:
            src_filename : string
                Path to the SNLI Dataset
            samp_percentage : float or None
                If not None, randomly sample approximately this percentage
                of lines.
        """
        self.src_filename = src_filename
        self.samp_percentage = samp_percentage
        self.examples = []
        for line in open(src_filename, encoding='utf8'):
            if (not self.samp_percentage) or random.random() <= self.samp_percentage:
                d = json.loads(line)
                ex = NLIExample(d)
                gold_label = getattr(ex, 'gold_label')
                if gold_label != '-':
                    self.examples.append([ex.sentence1, ex.sentence2, ex.gold_label])
        

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]
    
class SNLITrainDataset(SNLIDataset):
    
    def __init__(self, samp_percentage=None):
        """
        Args:
            samp_percentage : float or None
                If not None, randomly sample approximately this percentage
                of lines.
        """
    
        src_filename = os.path.join(
                SNLI_HOME, "snli_1.0_train.jsonl")

        super(SNLITrainDataset, self).__init__(src_filename, samp_percentage)
        
        
class SNLIDevDataset(SNLIDataset):
    
    def __init__(self, samp_percentage=None):
        """
        Args:
            samp_percentage : float or None
                If not None, randomly sample approximately this percentage
                of lines.
        """
    
        src_filename = os.path.join(
                SNLI_HOME, "snli_1.0_dev.jsonl")

        super(SNLIDevDataset, self).__init__(src_filename, samp_percentage)
        
class SNLITestDataset(SNLIDataset):
    
    def __init__(self, samp_percentage=None):
        """
        Args:
            samp_percentage : float or None
                If not None, randomly sample approximately this percentage
                of lines.
        """
    
        src_filename = os.path.join(
                SNLI_HOME, "snli_1.0_test.jsonl")

        super(SNLIDevDataset, self).__init__(src_filename, samp_percentage)
    