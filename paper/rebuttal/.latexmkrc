$pdf_mode = 1;        # tex -> pdf

@default_files = ('main.tex');

$pdflatex="pdflatex -interaction=nonstopmode %O %S";
$out_dir = '.aux';
$aux_dir = '.aux';
