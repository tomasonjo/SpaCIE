import pickle
import shelve
from typing import List, Dict, Optional, Set, Tuple
import requests

import spacy
import torch
from classy.data.data_drivers import QASample
from classy.pl_modules.base import ClassyPLModule
from classy.utils.lightning import (
    load_prediction_dataset_conf_from_checkpoint,
    load_classy_module_from_checkpoint,
)
from spacy import Language

from spacy.tokens import Doc, Span


def build_context(
    candidates: List[str], answer: Optional[str]
) -> Tuple[str, Optional[int], Optional[int]]:
    context = ""
    answer_start, answer_end = None, None

    for candidate in candidates:

        if answer is not None and candidate == answer:
            answer_start = len(context)
            answer_end = answer_start + len(candidate)

        candidate = candidate + " . "
        context += candidate

    return context, answer_start, answer_end


def load_checkpoint(checkpoint_path: str, device: int) -> ClassyPLModule:
    model = load_classy_module_from_checkpoint(checkpoint_path)
    if device >= 0:
        model.to(torch.device(device))
    model.freeze()
    return model


def load_mentions_inventory() -> Dict:
    inventory_stores = dict()
    return inventory_stores


def call_wiki_api(item):
    try:
        url = f"https://www.wikidata.org/w/api.php?action=wbsearchentities&search={item}&language=en&format=json"
        data = requests.get(url).json()
        return list(set([el['display']['label']['value'] for el in data['search']]))
    except:
        return None


def annotate_doc(annotated_samples: List[QASample]):
    for annotated_sample in annotated_samples:
        start_index, end_index = annotated_sample.predicted_annotation
        annotated_sample.ne._.disambiguated_entity = annotated_sample.context[
            start_index:end_index
        ]


@ Language.factory(
    "extend",
    default_config={
        "checkpoint_path": None,
        "tokens_per_batch": 2000,
        "device": 0,
    },
)
class ExtendComponent:

    def __init__(
        self,
        nlp,
        name,
        checkpoint_path: str,
        tokens_per_batch: int,
        device: int,
    ):
        assert checkpoint_path is not None, ""
        self.model = load_checkpoint(checkpoint_path, device)
        self.dataset_conf = load_prediction_dataset_conf_from_checkpoint(
            checkpoint_path
        )
        self.token_batch_size = tokens_per_batch
        self.mentions_inventory = load_mentions_inventory()

    def get_candidates(self, text):
        if self.mentions_inventory.get(text):
            return self.mentions_inventory[text]
        else:
            candidates = call_wiki_api(text)
            self.mentions_inventory[text] = candidates
            return candidates

    def _samples_from_doc(self, doc: Doc) -> List[QASample]:
        samples = []
        doc_tokens = [token.text for token in doc]
        for named_entity in doc._.ents:
            candidates = self.get_candidates(named_entity.text)
            context, _, _ = build_context(candidates, answer=None)
            left_tokens = doc_tokens[: named_entity.start]
            right_tokens = doc_tokens[named_entity.end:]
            question = " ".join(
                left_tokens + ["{", named_entity.text, "}"] + right_tokens
            )
            samples.append(
                QASample(context, question,
                         candidates=candidates, ne=named_entity)
            )
        return samples

    def __call__(self, doc: Doc) -> Doc:
        input_samples = self._samples_from_doc(doc)
        annotated_samples = self.model.predict(
            input_samples, self.dataset_conf, token_batch_size=self.token_batch_size
        )
        annotate_doc(annotated_samples)
        return doc


Span.set_extension("disambiguated_entity", default=None)
