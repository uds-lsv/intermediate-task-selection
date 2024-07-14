import abc
from collections import OrderedDict
import numpy as np

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# We do not implement `process` for tasks that its answer in natural language
# such as HellaSWAG, 


"""Defines functions to process the outputs to make them ready for the evaluation."""

def string_to_float(string, default=-1., **unused_kwargs):
  """Converts string to float, using default when conversion not possible."""
  try:
    return float(string)
  except ValueError:
    return default


class PostProcessor(abc.ABC): 
    """Postprocess the predictions and labels to make them suitable for
    evaluation."""
    def __init__(self, tokenizer, ignore_pad_token_for_loss):
       self.tokenizer = tokenizer 
       self.ignore_pad_token_for_loss = ignore_pad_token_for_loss 

    def process(self, preds, labels, data_info=None):
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        if self.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        # Some simple post-processing
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [label.strip() for label in decoded_labels]

        # Attribute `text2LabelId` is required for performing class label in text training.
        # preds and labels will be mapped to original class labels, e.g. 0, 1.
        if hasattr(self, "text2LabelId"):
            logger.info(f"before mapping: decoded_preds {decoded_preds}")
            logger.info(f"before mapping: decoded_labels {decoded_labels}")
            # use differeent index
            decoded_preds =  [ self.text2LabelId[tok] if tok in self.text2LabelId else "-101" for tok in decoded_preds ]
            decoded_labels = [ self.text2LabelId[tok] if tok in self.text2LabelId else "-100" for tok in decoded_labels ]

        return decoded_preds, decoded_labels 


class CR(PostProcessor):
    text2LabelId = {"negative": "0", 
                    "postive": "1"}


class MRPC(PostProcessor):
    text2LabelId = {"not_equivalent": "0",
                    "equivalent": "1"}


class COLA(PostProcessor):
    text2LabelId = {"unacceptable":"0", 
                    "acceptable":  "1"}


class SST2(PostProcessor):
    text2LabelId = {"negative": "0", 
                    "positive": "1"}


class STSB(PostProcessor):
    text2LabelId = {
        'zero': '0',
        'one': '1',
        'two': '2',
        'three': '3',
        'four': '4',
        'five': '5',
        'six': '6',
        'seven': '7',
        'eight': '8',
        'nine': '9',
        'point': '.'
    }

    def process(self, preds, labels, data_info):
        """Redefine the process fn.
        """
        def map_text_to_floating_number(inputs):
            """Mapping floading text (text) inot number (string).

            preds = ["three point two", "three", "three xk1o"]
            preds = ["3.2"            , "3"    , "3"]
            """
            new_preds = list()
            for pred in inputs:
                tmp = [ self.text2LabelId[tok] for tok in pred.split(" ") if tok in self.text2LabelId]
                new_preds.append("".join(tmp))
            logger.info(f"new preds: {new_preds}")
            return new_preds

        # Avoide use the `super().process()` direcly
        if isinstance(preds, tuple):
            preds = preds[0]
        
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        if self.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        # Some simple post-processing
        preds = [pred.strip() for pred in decoded_preds]
        labels = [label.strip() for label in decoded_labels]

        logger.info(f"before mapping: decoded_preds {preds}")
        logger.info(f"before mapping: decoded_labels {labels}")    
        preds =  map_text_to_floating_number(preds)
        labels = map_text_to_floating_number(labels)
        return preds, labels


class QQP(PostProcessor):
    text2LabelId = {"not_duplicate": "0",
                    "duplicate": "1"}


class MNLI(PostProcessor):
    text2LabelId = {"entailment": "0",
                    "neutral":    "1",
                    "contradiction": "2"}
    

class QNLI(PostProcessor):
    text2LabelId = {"entailment": "0", 
                    "not_entailment": "1"}
    

class RTE(PostProcessor):
    text2LabelId = {"entailment":"0", 
                    "not_entailment":"1"}
    

class SuperGLUEBoolQ(PostProcessor):
    text2LabelId = {"False": "0", 
                    "True": "1"}


class SuperGLUECB(PostProcessor):
    text2LabelId = {"entailment":    "0", 
                    "contradiction": "1",
                    "neutral": "2"}


class SuperGLUECOPA(PostProcessor):
    text2LabelId = {"choice1":"0",
                    "choice2":"1",
                    }


class MultiRC(PostProcessor):
    text2LabelId = {"False": "0", 
                    "True": "1",
                    }
    
    def process(self, preds, labels, data_info=None):
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        if self.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        # Some simple post-processing
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [label.strip() for label in decoded_labels]

        # Attribute `text2LabelId` is required for performing class label in text training.
        # preds and labels will be mapped to original class labels, e.g. 0, 1.
        if hasattr(self, "text2LabelId"):
            logger.info(f"before mapping: decoded_preds {decoded_preds}")
            logger.info(f"before mapping: decoded_labels {decoded_labels}")
            decoded_preds =  [ self.text2LabelId[tok] if tok in self.text2LabelId else "-101" for tok in decoded_preds ]
            decoded_labels = [ self.text2LabelId[tok] if tok in self.text2LabelId else "-100" for tok in decoded_labels ]
            
        decoded_preds = [{"group": info["group"], "value":pred} \
            for info, pred in zip(data_info, decoded_preds)]
        decoded_labels = [{"group": info["group"], "value": label}\
            for info, label in zip(data_info, decoded_labels)] 
        
        return decoded_preds, decoded_labels 

class SuperGLUEWIC(PostProcessor):
    text2LabelId = {"False": "0",
                    "True": "1",
                    }


class SuperGLUEWSCFixed(PostProcessor):
    text2LabelId = {"False": "0", 
                    "True": "1"}


class Record(PostProcessor):
    def process(self, preds, labels, data_info):
        preds, labels = super().process(preds, labels, data_info) 
        labels = [info["answers"] for info in data_info]
        return preds, labels 


class CXC(PostProcessor):
    text2LabelId = {
        'zero': '0',
        'one': '1',
        'two': '2',
        'three': '3',
        'four': '4',
        'five': '5',
        'six': '6',
        'seven': '7',
        'eight': '8',
        'nine': '9',
        'point': '.'
    }

    def process(self, preds, labels, data_info):
        """Redefine the process fn.
        """
        def map_text_to_floating_number(inputs):
            """Mapping floading text (text) inot number (string).

            preds = ["three point two", "three", "three xk1o"]
            preds = ["3.2"            , "3"    , "3"]
            """
            new_preds = list()
            for pred in inputs:
                tmp = [ self.text2LabelId[tok] for tok in pred.split(" ") if tok in self.text2LabelId]
                new_preds.append("".join(tmp))
            logger.info(f"new preds: {new_preds}")
            return new_preds

        # Avoide use the `super().process()` direcly
        if isinstance(preds, tuple):
            preds = preds[0]
        
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        if self.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        # Some simple post-processing
        preds = [pred.strip() for pred in decoded_preds]
        labels = [label.strip() for label in decoded_labels]

        logger.info(f"before mapping: decoded_preds {preds}")
        logger.info(f"before mapping: decoded_labels {labels}")    
        # preds =  map_text_to_floating_number(preds)
        # labels = map_text_to_floating_number(labels)
        return preds, labels


class Squad(PostProcessor):
    def process(self, preds, labels, data_info):
        preds, labels = super().process(preds, labels, data_info) 
        labels = [info["answers"] for info in data_info]
        return preds, labels 

class Drop(PostProcessor):
    def process(self, preds, labels, data_info):
        preds, labels = super().process(preds, labels, data_info) 
        labels = [info["answers"] for info in data_info]
        return preds, labels 



POSTPROCESSOR_MAPPING = OrderedDict(
    [
        ('squad', Squad),
        ("cr", CR),
        ("mrpc", MRPC),
        ("cola", COLA),
        ("sst2", SST2),
        ("qnli", QNLI),
        ("rte", RTE),
        ("mnli", MNLI),
        ("qqp", QQP),
        ("stsb", STSB),
        ("superglue-boolq", SuperGLUEBoolQ),
        ("superglue-copa", SuperGLUECOPA),
        ('superglue-multirc', MultiRC),
        ('superglue-cb', SuperGLUECB),
        ('superglue-record', Record),
        ("superglue-wic", SuperGLUEWIC),
        ("superglue-wsc.fixed", SuperGLUEWSCFixed),
        ("cxc", CXC),
        ('drop', Drop),
    ]
)




class AutoPostProcessor:
    @classmethod
    def get(self, task, tokenizer, ignore_pad_token_for_loss):
        if task in POSTPROCESSOR_MAPPING:
            return POSTPROCESSOR_MAPPING[task](tokenizer, ignore_pad_token_for_loss)
        return PostProcessor(tokenizer, ignore_pad_token_for_loss)
