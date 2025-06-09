from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import re

# Corpus original
corpus_original = [
    "Eu não gostei do produto e o produto parece ruim.",
    "O produto parece bom.",
    "O produto parece ruim.",
    "Parece muito ruim esse produto."
]

# Pré-processamento: remover pontuação, converter para minúsculas, normalizar espaços
def preprocessar(frase):
    frase = frase.lower()  # minúsculas
    frase = re.sub(r'([^\w\s])', r' ', frase)  # substitui pontuação por espaço
    frase = re.sub(r'\s+', ' ', frase).strip()
    return frase

corpus = [preprocessar(frase) for frase in corpus_original]

# Função para extrair n-gramas e gerar dataframe
def gerar_vetores_ngramas(corpus, n):
    vectorizer = CountVectorizer(ngram_range=(1, n), lowercase=True)
    result = vectorizer.fit_transform(corpus)
    df = pd.DataFrame(result.toarray(), columns=vectorizer.get_feature_names_out())
    return df

# Loop para n = 1, 2, 3
for n in range(1, 4):
    print(f"\n🔹 Vetores para n-grama com n={n}:")
    df_ngram = gerar_vetores_ngramas(corpus, n)
    print(df_ngram)
