filename=main

all: ${filename}.log
	pdflatex ${filename}.tex
	bibtex ${filename}
	pdflatex ${filename}.tex

submission:
	gs -sDEVICE=pdfwrite -dNOPAUSE -dBATCH -dSAFER -dPrinted=false -dFirstPage=1 -dLastPage=12 -sOutputFile=submission.pdf ${filename}.pdf

appendix:
	gs -sDEVICE=pdfwrite -dNOPAUSE -dBATCH -dSAFER -dPrinted=false -dFirstPage=13 -sOutputFile=supplement.pdf ${filename}.pdf

supplementary:
	gs -sDEVICE=pdfwrite -dNOPAUSE -dBATCH -dSAFER -dPrinted=false -dFirstPage=13 -sOutputFile=supplement.pdf ${filename}.pdf
	rm -f supplement.zip
	zip supplement.zip 'supplement.pdf'
	zip -r supplement.zip code -i '*.py'
	zip -r supplement.zip code -i '*.txt'
	zip -r supplement.zip code -i '*.md'
	zip -r supplement.zip code -i '*.ipynb'

arxiv:
	rm -f arxiv.zip
	latexpand --empty-comments --keep-includes --expand-bbl ${filename}.bbl ${filename}.tex > arxiv.tex
	zip arxiv.zip 'arxiv.tex' 'icml2023_dp4ml.sty' 'icml2023.bst' 'utils.tex'
	zip -r arxiv.zip fig -i 'fig/*.tex' 'tables/workshop_uci_table.tex'
	zip -r arxiv.zip tables -i 'tables/workshop_uci_table.tex'

jpeg:
	find . -iname 'MATCH-FILES-HERE*.png' | while read i; do echo "Convert $$i to $${i%.*}.jpg"; convert "$$i" -sampling-factor 4:2:0 -strip -quality 85 -interlace JPEG -colorspace RGB "$${i%.*}.jpg"; done
