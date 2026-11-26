# Build every technical doc in this folder. `make` rebuilds each PDF whose .tex
# changed; `make clean` removes LaTeX aux files. Requires latexmk (TeX Live).
# duck_app_*.tex / duck_anim_*.tex are fragments written by app/bench_mac.py / app/bench_anim.py (\input by the docs), not documents
TEX := $(filter-out duck_app_%.tex duck_anim_%.tex duck_flight_%.tex,$(wildcard *.tex))
PDF := $(TEX:.tex=.pdf)

LATEXMK := latexmk -pdf -interaction=nonstopmode -halt-on-error

all: $(PDF)

%.pdf: %.tex
	$(LATEXMK) $<

clean:
	latexmk -C

.PHONY: all clean
