"""Neighbourhood-conditional pooling: token weights, and the pooling that consumes them.

Two modules, one question each. `encode` gets contextual token embeddings out of
the frozen sentence-transformers encoder and pools them, with mean pooling as the
literal `gamma = 0` case. `weights` computes `w_t^(i) = (1 - df_t^(i)) ** gamma`
from an episode's neighbourhood, plus the two global-statistic incumbents
(corpus-wide df, and SIF) that the local rule has to beat to claim anything.

    from brain.embed.pooling import shared_token_cache, local_weights, pool_weighted

    cache = shared_token_cache()
    docs = cache.docs(texts)
    v_full = mean_pool(docs)                     # today's vectors, byte for byte
    order = neighbour_indices(v_full, k=8)
    rows = local_weights(docs, token_sets(docs), order, gamma=1.0)
    v_dist, stats = pool_weighted(docs, rows)

Nothing here modifies `brain.embed.embedder`: the token embeddings come out of
the same forward pass the sentence vector comes from, via the encoder's own
`output_value="token_embeddings"` read-out.
"""

from brain.embed.pooling.encode import (
    TOKEN_BATCH_SIZE,
    TokenCache,
    TokenDoc,
    mean_pool,
    pool_weighted,
    reset_shared_token_cache,
    shared_token_cache,
)
from brain.embed.pooling.weights import (
    SIF_DEFAULT_A,
    distinctiveness,
    global_document_frequency,
    global_weights,
    local_document_frequency,
    local_weights,
    neighbour_indices,
    remove_first_principal_component,
    sif_weights,
    token_sets,
    unigram_probabilities,
)

__all__ = [
    "TOKEN_BATCH_SIZE",
    "TokenCache",
    "TokenDoc",
    "mean_pool",
    "pool_weighted",
    "reset_shared_token_cache",
    "shared_token_cache",
    "SIF_DEFAULT_A",
    "distinctiveness",
    "global_document_frequency",
    "global_weights",
    "local_document_frequency",
    "local_weights",
    "neighbour_indices",
    "remove_first_principal_component",
    "sif_weights",
    "token_sets",
    "unigram_probabilities",
]
