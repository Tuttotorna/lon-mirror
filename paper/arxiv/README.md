# OMNIA — arXiv Submission Notes

This folder contains an **arXiv-ready LaTeX source** for the OMNIA paper.

## Main file
- `main.tex` (compile this file)

## arXiv compatibility
- Uses standard `article` class
- No BibTeX
- No custom fonts
- UTF-8 encoding
- `\pdfoutput=1` set

## How arXiv should compile
Select:
- **Primary TeX file:** `main.tex`
- **Compiler:** pdfLaTeX

## Included / excluded files
- Included: `paper/arxiv/main.tex`
- Optional data/code live in the repository root and are **not required** for compilation.
- Large JSON/raw files are intentionally excluded from arXiv.

## Reproducibility
All experiments, metrics (TruthΩ, PBII), and raw outputs referenced in the paper
are available in the public repository:

https://github.com/Tuttotorna/lon-mirror

## License / authorship
Author: Massimiliano Brighindi  
Signature: MB-X.01 / Omniabase±

This paper describes a **measurement layer**, not a decision or generation system.

