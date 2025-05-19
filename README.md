# Autoformalizing-Euclidean-Geometry

# Autoformalizing Euclidean Geometry

## 📚 Overview  
A neuro‑symbolic system that converts informal Euclidean geometry statements into fully machine‑verifiable Lean proofs by combining SMT solvers with LLMs.

## 🎯 Objectives  
- Automate formalization of text‑based geometric proofs.  
- Infer and complete diagrammatic constraints automatically.  
- Evaluate correctness via a new LeanEuclid benchmark.

## 🛠 Architecture  
1. **Diagrammatic Inference:** SMT solver fills in missing spatial relations.  
2. **Textual Formalization:** GPT‑4/GPT‑4V generates Lean proof scripts from cleaned text.  
3. **Semantic Evaluation:** Automated checks in Lean ensure logical soundness.

![Architecture Diagram](./docs/architecture.png)

## 📂 Datasets & Benchmark  
- **UniGeo**: 200+ geometry problems with formal annotations.  
- **Euclid’s Elements**: 48 classic theorems.  
- **LeanEuclid**: Combined benchmark suite for evaluation.

## 🚀 Results  
- 85% semantic accuracy (GPT‑4) on LeanEuclid.  
- End‑to‑end pipeline processes a proof in ~2 minutes.  
- Identified diagram‑heavy edge cases and proposed solver enhancements.

## 🔮 Future Work  
- Improve SMT solver diagram inference accuracy.  
- Fine‑tune LLM on geometric corpora for better precision.  
- Extend to non‑Euclidean geometries and more complex theorems.

## 📖 References  
See [arXiv:2405.17216](https://arxiv.org/abs/2405.17216) for full paper.

