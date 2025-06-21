from scipy.spatial.distance import cosine
from ai.dtos.animal_embedding import AnimalEmbedding

def calc_similarity(origin: AnimalEmbedding, compare: AnimalEmbedding):
    face = (1 - cosine(origin.face, compare.face)) * 0.9
    eye_0 = (1 - cosine(origin.eye_0, compare.eye_0)) * 0.025
    eye_1 = (1 - cosine(origin.eye_1, compare.eye_1)) * 0.025
    nose = (1 - cosine(origin.nose, compare.nose)) * 0.05
    return face + eye_0 + eye_1 + nose