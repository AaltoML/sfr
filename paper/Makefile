filename=main
aux_dir=.aux

paper:
	latexmk -pdf -interaction=nonstopmode -auxdir=${aux_dir} -outdir=${aux_dir} ${filename}.tex
	mv ${aux_dir}/${filename}.pdf ${filename}.pdf

clean:
	rm -rf ${aux_dir}

submission:
	gs -sDEVICE=pdfwrite -dNOPAUSE -dBATCH -dSAFER -dPrinted=false -dFirstPage=1 -dLastPage=13 -sOutputFile=submission.pdf ${filename}.pdf

appendix:
	gs -sDEVICE=pdfwrite -dNOPAUSE -dBATCH -dSAFER -dPrinted=false -dFirstPage=14 -sOutputFile=supplement.pdf ${filename}.pdf

supplementary:
        gs -sDEVICE=pdfwrite -dNOPAUSE -dBATCH -dSAFER -dPrinted=false -dFirstPage=14 -sOutputFile=supplement.pdf ${filename}.pdf
        rm -f supplement.zip
        zip supplement.zip 'supplement.pdf'
        zip -r supplement.zip code -r 'code/'

arxiv:
        rm -f arXiv.zip
        latexpand --empty-comments --keep-includes --expand-bbl ${filename}.bbl ${filename}.tex > arxiv.tex
        zip arXiv.zip 'arxiv.tex' 'neurips_2023.sty'
        zip -r arXiv.zip fig -i 'fig/*.tex'
