# Downloaded Papers

13 papers (all valid PDFs, downloaded from arXiv). Grouped by role. Papers marked
**★ deep-read** were read in full (all chunks) — see `../literature_review.md` for
detailed notes.

## Core: sampling at intermediate layers & gradient estimators

1. **★ Estimating or Propagating Gradients Through Stochastic Neurons** — `1308.3432_bengio2013_estimating_propagating_gradients_stochastic_neurons.pdf`
   - Bengio, Léonard, Courville — 2013
   - **THE foundational paper for sampling at hidden units.** Bernoulli-samples
     activations `h_i ~ sigmoid(a_i)` at intermediate layers; introduces/compares
     4 gradient estimators: Straight-Through (best), unbiased REINFORCE + min-var
     baseline, Stochastic-Times-Smooth, noisy rectifier. MNIST conditional computation.

2. **★ Auto-Encoding Variational Bayes (VAE)** — `1312.6114_kingma2013_auto_encoding_variational_bayes.pdf`
   - Kingma, Welling — 2013
   - The reparameterization trick: sample `z = mu + sigma*eps` at the latent
     (intermediate) layer so gradients flow. ELBO = reconstruction − KL. MNIST, Frey Faces.

3. **★ Categorical Reparameterization with Gumbel-Softmax** — `1611.01144_jang2016_gumbel_softmax_categorical_reparameterization.pdf`
   - Jang, Gu, Poole — 2016/2017
   - Differentiable **categorical/softmax sampling** at intermediate layers via the
     Gumbel-Max trick + temperature-τ softmax relaxation. ST variant for hard samples.
   - **Most direct match to the hypothesis** (softmax sampling = categorical sampling).

4. **★ Neural Discrete Representation Learning (VQ-VAE)** — `1711.00937_vandenoord2017_vqvae_neural_discrete_representation.pdf`
   - van den Oord, Vinyals, Kavukcuoglu — 2017
   - Discrete codebook at the intermediate layer (nearest-neighbor quantization +
     straight-through + commitment loss); autoregressive prior to *sample* codes.
     CIFAR-10, ImageNet, audio, video.

5. **The Concrete Distribution** — `1611.00712_maddison2016_concrete_distribution.pdf`
   - Maddison, Mnih, Teh — 2016
   - Independent, concurrent continuous relaxation of discrete variables (the
     density behind Gumbel-Softmax). Theoretical foundation + bias/variance analysis.

6. **Stochastic Backpropagation and Approximate Inference in Deep Generative Models** — `1401.4082_rezende2014_stochastic_backprop_deep_generative.pdf`
   - Rezende, Mohamed, Wierstra — 2014
   - Concurrent-with-VAE derivation of reparameterized ("stochastic back-propagation")
     gradients for deep latent Gaussian models; second-order / general rules.

7. **ProbAct: A Probabilistic Activation Function** — `1905.10761_lee2019_probact_probabilistic_activation.pdf`
   - Lee et al. — 2019
   - A **stochastic activation function**: output is *sampled* from a per-element
     mean+variance distribution at every layer. Very close operationalization of
     the hypothesis as a drop-in activation. Reports regularization/generalization gains.

## Stochasticity as regularization / Bayesian inference

8. **Dropout as a Bayesian Approximation** — `1506.02142_gal2016_dropout_bayesian_approximation.pdf`
   - Gal, Ghahramani — 2016 — dropout = sampling a Bernoulli mask over activations =
     approximate Bayesian inference; MC-dropout for uncertainty. Baseline for "sampling activations."

9. **Variational Dropout and the Local Reparameterization Trick** — `1506.02557_kingma2015_variational_dropout_local_reparameterization.pdf`
   - Kingma, Salimans, Welling — 2015 — turns weight uncertainty into per-activation
     local noise; variance ∝ 1/batch. Learnable dropout rates.

10. **Weight Uncertainty in Neural Networks (Bayes by Backprop)** — `1505.05424_blundell2015_weight_uncertainty_bayes_by_backprop.pdf`
    - Blundell et al. — 2015 — reparameterized sampling of *weights* (vs activations);
      relevant contrast for where to inject stochasticity.

11. **Noisy Activation Functions** — `1603.00391_gulcehre2016_noisy_activation_functions.pdf`
    - Gülçehre et al. — 2016 — inject noise into saturating activations so gradients
      flow; noise as exploration + optimization aid (not only regularization).

12. **Deep Variational Information Bottleneck** — `1612.00410_alemi2016_deep_variational_information_bottleneck.pdf`
    - Alemi et al. — 2016 — stochastic intermediate representation trained with an
      IB objective (compress input, keep label info); uses reparameterized sampling
      at the bottleneck. Robustness/generalization framing for sampled activations.

13. **Stochastic Activation Pruning (SAP)** — `1803.01442_dhillon2018_stochastic_activation_pruning.pdf`
    - Dhillon et al. — 2018 — at test time, *sample* which activations to keep
      (probabilities ∝ magnitude). Stochastic activations for adversarial robustness.
      (See also the erratum on its adversarial evaluation in the literature review.)

---

### Chunked PDFs
`pages/` holds 3-page PDF chunks + manifests for the 4 deep-read papers
(1308.3432, 1312.6114, 1611.01144, 1711.00937), produced with
`.claude/skills/paper-finder/scripts/pdf_chunker.py`.

### Search provenance
Papers were surfaced via the paper-finder service (214 unique candidates across 4
queries) and an arXiv API fallback; the full ranked candidate lists are in
`../paper_search_results/`. The 13 above were selected as the highest-relevance
foundational + directly-on-topic works and downloaded from arXiv for reliable PDFs.
