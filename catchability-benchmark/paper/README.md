# FR-B3 ICLR manuscript workspace

This is an anonymous results-blocked skeleton, not a submission-ready paper.
Red `PENDING` markers prevent unknown factorial or learned-transfer conclusions
from being silently converted into claims.

## Official format

ICLR 2027 requires at most nine main-text pages at submission, excluding
references, with appendices after the bibliography. Download the official style
archive from:

<https://media.iclr.cc/Conferences/ICLR2027/iclr-2027-style-files.zip>

Place `iclr2027_conference.sty` and the accompanying official files in this
directory. Until then, `main.tex` uses a local article fallback for structural
compilation only.

## Compile

```bash
cd catchability-benchmark/paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Main-text page budget

| Section | Target pages |
|---|---:|
| Abstract and introduction | 1.5 |
| Related work | 0.75 |
| Dimensionless formulation | 0.75 |
| Scripted benchmark methods | 1.25 |
| Learned-transfer methods | 1.25 |
| Results | 2.25 |
| Limitations, reproducibility, ethics | 1.0 |
| Total | 8.75 |

The mandatory AI-use disclosure does not count toward the page limit under the
2027 author guidelines. The authors must update it to match the complete final
research workflow.

## Result insertion gate

Do not remove a result placeholder until:

1. the corresponding immutable output passes its completeness/provenance gate;
2. the frozen analysis finishes without manual edits;
3. every plotted number traces to the machine-readable analysis;
4. the conclusion states the observed result, including negative or null
   outcomes.
