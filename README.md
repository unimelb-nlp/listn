# Lexicon Induction with Socio-Temporal Nuance (De Kock, ACL 2025)

In-group language is an important signifier of group dynamics. This paper proposes a novel method for inducing lexicons of in-group language, which incorporates its socio-temporal context. Existing methods for lexicon induction do not capture the evolving nature of in-group language, nor the social structure of the community. Using dynamic word and user embeddings trained on conversations from online anti-women communities, our approach outperforms prior methods for lexicon induction. We develop a test set for the task of lexicon induction and a new lexicon of manosphere language, validated by human experts, which quantifies the relevance of each term to a specific sub-community at a given point in time. Finally, we present novel insights on in-group language which illustrate the utility of this approach.

---

This repo contains code to train the Cerberus architecture using Pytorch and Pytorch Lightning. The "lexicons" folder contains two files: 
# gold_standard.csv 
Our manually validated test set, containing verified positive (1) and negative (0) samples, capturing whether the word represents valid in-group language within the manosphere according to a domain expert.

# lexicon_with_scores.csv: 
A lexicon of 938 manosphere-related words, with scores corresponding to their relevance to each of the 7 communities included in our study. This data was used to create Figure 2 and Table 3 in the paper. **NOTE**: we provide scores for all words in our own lexicon as well as lexica from prior work, as described in Table 1. This is reflected in the "origin" column. We include only lexical innovations, whereas other works have opted to include standard terms as well.

Full details are provided in the paper. 
