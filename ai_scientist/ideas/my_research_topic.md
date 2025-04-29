# Title: VLM Embeddings for OOD Detection: Promises and Pitfalls in Fine-grained Classification

## Keywords
vision-language models, embeddings, out-of-distribution detection, fine-grained classification, failure modes

## TL;DR
While vision-language model embeddings show promise for OOD detection in fine-grained image classification, significant challenges remain in their practical application, particularly in complex domains like aircraft recognition.

## Abstract
Recent advances in vision-language models (VLMs) have sparked enthusiasm about their potential for out-of-distribution (OOD) detection in fine-grained image classification tasks. The rich semantic embeddings produced by these models theoretically offer a strong foundation for distinguishing known from unknown classes without explicit training on unknown samples. However, our empirical investigation using aircraft classification as a test case reveals several significant challenges that limit practical application. 

First, we demonstrate that VLM embeddings often struggle to maintain sufficient separation between semantically similar manufacturer classes (e.g., Boeing vs. Airbus), with inter-class similarities frequently overlapping intra-class distributions. Second, we observe performance degradation when moving from binary classification to fine-grained variant recognition, with accuracy dropping 25-40% despite strong reported performance on benchmark datasets. Third, our analysis reveals that OOD detection capabilities vary dramatically depending on the semantic distance between known and unknown classes, with nearby but excluded classes proving especially problematic.

Most importantly, we identify inconsistencies in embedding quality across different instances of the same class, causing unreliable similarity metrics that compromise OOD detection reliability. Through controlled experiments varying prompt engineering strategies, model architecture, and threshold selection approaches, we characterize these failure modes and propose a hybrid approach that combines embedding-based similarity with confidence calibration techniques. Our findings highlight the gap between theoretical capabilities and practical performance of VLM embeddings for OOD detection in specialized domains, offering actionable insights for researchers and practitioners seeking to deploy these models in real-world applications.