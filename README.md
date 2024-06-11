# Replicating Toy Models of Universality

For my March 2024 AI Safety Fundamentals project, I replicated Bilal Chughtai's paper [A Toy Model of Universality](https://arxiv.org/pdf/2302.03025). Due to time and skill constraints, I was only able to test the logit attributions (Section 5.1) and the embeddings and unembeddings (Section 5.2) using MLP models trained on the group C_113, but this was sufficient to confirm the overall thesis of the paper.

## Background

One of the largest fields of AI alignment research is mechanistic interpretability: the study of the internals of deep learning models. The hope of mechanistic interpretability is that, by understanding how those models work, we will be able to better determine when they are or aren’t aligned, or even deliberately steer them towards alignment.

Much of the tractability of this approach may rely on [universality](https://distill.pub/2020/circuits/zoom-in/#claim-3): the hypothesis that similar models will contain similar features. If universality holds, we’ll be able to understand a wide variety of models by studying just a few; if it doesn’t, we’ll have to start over from scratch every time. This hypothesis can be further divided into weak universality (models implement similar algorithms) and strong universality (models implement those algorithms in the same way).

Bilal Chughtai’s paper [A Toy Model of Universality](https://arxiv.org/pdf/2302.03025) tests both of these hypotheses by training one-layer models to do group composition (eg modular arithmetic). This turns out to have a relatively simple algorithm - in the embedding, the model can convert the inputs into equivalent matrices (eg turning modular arithmetic into rotation matrices); in the hidden layer, the model can multiply those matrices together; and in the unembedding, the model can compute the trace of the multiplication of that product and the inverse of every possible result. This trace is maximized when that last multiplication is the identity, or equivalently, when the product and the possible result are the same. The map between the group and the matrices is called a representation, so this algorithm is called the Group Composition via Representations (GCR) algorithm.

Notably, most groups have multiple representations.
