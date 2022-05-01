import spacy
import os
import crosslingual_coreference
import extend_component
import rebel_component

DEVICE = -1


dir_path = os.path.dirname(os.path.realpath(__file__))

extend_config = dict(
    checkpoint_path=dir_path +
    "/extend-longformer-large/2021-10-22/09-11-39/checkpoints/best.ckpt",
    device=DEVICE,
    tokens_per_batch=4000,
)

"""
# Add coreference resolution model
coref = spacy.load('en_core_web_sm', disable=[
                   'ner', 'tagger', 'parser', 'attribute_ruler', 'lemmatizer'])
coref.add_pipe(
    "xx_coref", config={"chunk_size": 2500, "chunk_overlap": 2, "device": DEVICE})
"""

# Define rel extraction model

rel_ext = spacy.load('en_core_web_sm', disable=[
                     'ner', 'lemmatizer', 'attribute_rules', 'tagger'])
rel_ext.add_pipe("rebel", config={
    'device': DEVICE,  # Number of the GPU, -1 if want to use CPU
    'model_name': 'Babelscape/rebel-large'}  # Model used, will default to 'Babelscape/rebel-large' if not given
)

rel_ext.add_pipe("extend", after="rebel", config=extend_config)


input_text = """
Albert Einstein(/ˈaɪnstaɪn / EYEN-styne; [6] German: [ˈalbɛʁt ˈʔaɪnʃtaɪn](listen); 14 March 1879 – 18 April 1955) was a German-born theoretical physicist, [7] widely acknowledged to be one of the greatest and most influential physicists of all time. Einstein is best known for developing the theory of relativity, but he also made important contributions to the development of the theory of quantum mechanics. Relativity and quantum mechanics are together the two pillars of modern physics.[3][8] His mass–energy equivalence formula E = mc2, which arises from relativity theory, has been dubbed "the world's most famous equation".[9] His work is also known for its influence on the philosophy of science.[10][11] He received the 1921 Nobel Prize in Physics "for his services to theoretical physics, and especially for his discovery of the law of the photoelectric effect", [12] a pivotal step in the development of quantum theory. His intellectual achievements and originality resulted in "Einstein" becoming synonymous with "genius".[13]
"""

#coref_text = coref(input_text)._.resolved_text

doc = rel_ext(input_text)


for value, rel_dict in doc._.rel.items():
    print(f"{value}: {rel_dict}")

disambiguated_entities = [(ent.text, ent._.disambiguated_entity)
                          for ent in doc._.ents]

print(disambiguated_entities)
