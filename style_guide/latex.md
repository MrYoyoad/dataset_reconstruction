# LaTeX / Thesis Conventions

Only project- and Weizmann-specific decisions are documented here. Generic LaTeX practice (`\input`, `booktabs`, `cleveref`, `latexmk -pdf`, etc.) is assumed.

LaTeX in this project is compiled via **Overleaf**, not locally — don't suggest `latexmk` invocations or local toolchain debugging.

## Weizmann Thesis Proposal

- **Document class**: Weizmann thesis class if available, otherwise `report` / `book`
- **Page size**: A4 (Israeli/European standard)
- **Font**: Computer Modern (default) or Times New Roman if department requires
- **Line spacing**: 1.5 or double for body — check department requirements
- **Margins**: typically 2.5 cm all sides
- **Language**: English primary with Hebrew abstract. Use `babel` or `polyglossia` for bilingual support.
- **Bibliography**: BibTeX/BibLaTeX with department-specified style (else `apalike` or `plainnat`)

## Thesis Proposal Structure

1. Title page (institution, department, title, author, advisor, date)
2. Abstract (English + Hebrew)
3. Table of Contents
4. Introduction / Background
5. Literature Review
6. Proposed Research / Methodology
7. Preliminary Results (if any)
8. Expected Contributions
9. Timeline
10. Bibliography

## Beamer (Academic Presentations)

- **Theme**: Metropolis (modern, clean) or department preference
- **Font**: Fira Sans (Metropolis default)
- Section divider slides between major sections — helps audience track progress in long talks
- Appendix slides as a "pressure valve" — move detailed Q&A backup content there rather than bloating the main flow
- All [guardrails.md](guardrails.md) rules apply, especially D1-D6 (density) and F1-F4 (freshness)
