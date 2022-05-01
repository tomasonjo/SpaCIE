from requests import head
import hashlib
from spacy import Language
import spacy
from typing import List

from spacy.tokens import Doc, Span

import re

from transformers import pipeline


def extract_triplets(text):
    """
    Function to parse the generated text and extract the triplets
    """
    triplets = []
    relation, subject, relation, object_ = '', '', '', ''
    text = text.strip()
    current = 'x'
    for token in text.replace("<s>", "").replace("<pad>", "").replace("</s>", "").split():
        if token == "<triplet>":
            current = 't'
            if relation != '':
                triplets.append(
                    {'head': subject.strip(), 'type': relation.strip(), 'tail': object_.strip()})
                relation = ''
            subject = ''
        elif token == "<subj>":
            current = 's'
            if relation != '':
                triplets.append(
                    {'head': subject.strip(), 'type': relation.strip(), 'tail': object_.strip()})
            object_ = ''
        elif token == "<obj>":
            current = 'o'
            relation = ''
        else:
            if current == 't':
                subject += ' ' + token
            elif current == 's':
                object_ += ' ' + token
            elif current == 'o':
                relation += ' ' + token
    if subject != '' and relation != '' and object_ != '':
        triplets.append(
            {'head': subject.strip(), 'type': relation.strip(), 'tail': object_.strip()})

    return triplets


@ Language.factory(
    "rebel",
    requires=["doc.sents"],
    default_config={
        "model_name": "Babelscape/rebel-large",
        "device": -1,
    },
)
class RebelComponent:
    def __init__(
        self,
        nlp,
        name,
        model_name: str,
        device: int,
    ):
        assert model_name is not None, ""
        self.triplet_extractor = pipeline(
            "text2text-generation", model=model_name, tokenizer=model_name, device=device)
        # Instantiate an index
        self.index = 0
        # Register custom extension on the Doc
        if not Doc.has_extension("rel"):
            Doc.set_extension("rel", default={})
        if not Doc.has_extension("ents"):
            Doc.set_extension("ents", default=[])

    def get_index(self):
        self.index += 1
        return self.index

    def _generate_triplets(self, sent: Span) -> List[dict]:
        output_ids = self.triplet_extractor(sent.text, return_tensors=True, return_text=False)[
            0]["generated_token_ids"]["output_ids"]
        extracted_text = self.triplet_extractor.tokenizer.batch_decode(
            output_ids[0])
        extracted_triplets = extract_triplets(extracted_text[0])
        return extracted_triplets

    def set_annotations(self, doc: Doc, triplets: List[dict]):
        ents = list()

        for triplet in triplets:
            # get substring to spacy span
            head_span = re.search(
                " ".join([f"\\b{el}\\b" for el in triplet['head'].split(" ")]), doc.text)
            tail_span = re.search(
                " ".join([f"\\b{el}\\b" for el in triplet['tail'].split(" ")]), doc.text)
            # If regex doesn't match, then skip the relationship
            if not head_span or not tail_span:
                continue
            # get spacy span
            head_span = doc.char_span(
                head_span.start(), head_span.end())
            tail_span = doc.char_span(
                tail_span.start(), tail_span.end())
            # Sometimes the match doesn't work, not sure why
            if not head_span or not tail_span:
                continue
            # Remove self-loops (relationships that start and end at the entity)
            if head_span == tail_span:
                continue

            index = hashlib.sha1("".join(
                [head_span.text, tail_span.text, triplet['type']]).encode('utf-8')).hexdigest()
            # Add relationship
            if index not in doc._.rel:
                doc._.rel[index] = {"relation": triplet["type"],
                                    "head_span": head_span, "tail_span": tail_span}
            # Add entity
            print(head_span)
            print(ents)
            if not any([e.text == head_span.text for e in ents]):
                ents.append(head_span)
            if not any([e.text == tail_span.text for e in ents]):
                ents.append(tail_span)

            doc._.ents.extend(ents)
            """
            # Attach processed entities to doc.ents
            try:
                # this works with non-overlapping spans
                
            except Exception as e:
                print(e)
                # filter the overlapping spans, keep the (first) longest one
                doc.ents = spacy.util.filter_spans(ents)
            """

    def __call__(self, doc: Doc) -> Doc:
        for sent in doc.sents:
            sentence_triplets = self._generate_triplets(sent)
            self.set_annotations(doc, sentence_triplets)
        return doc
