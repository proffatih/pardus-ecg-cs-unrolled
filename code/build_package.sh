#!/usr/bin/env bash
# build_package.sh — assemble the SPL resubmission package from the results.
#   merge results -> stats/tables/macros -> figures -> compile -> PDFs -> zips
set -euo pipefail
cd "$(dirname "$0")"
BASE=..
SUB=$BASE/resubmission/SUBMISSION
MS=$BASE/resubmission/manuscript
export CV_FILE=cv_final.csv

echo ">>> merge + statistics + tables + macros"
true
python3 stats_v2.py | tail -25
python3 make_supplement.py
python3 make_numbers.py

echo ">>> figures"
if [ "${SKIP_FIGS:-0}" = "1" ]; then echo "    (atlandi: SKIP_FIGS=1)"; else python3 make_figures2.py; fi

echo ">>> compile manuscript"
cd "$MS"
for f in manuscript supplementary; do
  pdflatex -interaction=nonstopmode $f.tex >/dev/null 2>&1 || true
  pdflatex -interaction=nonstopmode $f.tex >/dev/null 2>&1 || true
  echo "    $f.pdf: $(pdfinfo $f.pdf 2>/dev/null | awk '/Pages/{print $2}') pages, \
$(grep -c '^! ' $f.log || true) errors, \
$(grep -c 'undefined' $f.log || true) undefined"
done
cd - >/dev/null

echo ">>> package files"
mkdir -p "$SUB/figures"
cp "$MS/manuscript.pdf" "$SUB/manuscript.pdf"
cp "$MS/manuscript.pdf" "$SUB/double.pdf"
cp "$MS/manuscript.tex" "$SUB/manuscript.tex"
cp "$MS/supplementary.pdf" "$SUB/supplementary.pdf"
cp "$MS"/fig_*.pdf "$SUB/figures/" 2>/dev/null || true

echo ">>> plain abstract"
python3 ../../../_TAKIP/make_plain_abstract.py \
   "Asus_Pardus_0027_ECG_CS/IEEE_Signal_Processing_Letters/resubmission/manuscript/manuscript.tex" \
   2>/dev/null | tail -2 || true
cp "$MS/abstract_plain.txt" "$SUB/" 2>/dev/null || true
cp "$MS/abstract_plain.rtf" "$SUB/" 2>/dev/null || true

echo ">>> cover letter + response PDFs"
for d in cover_letter response_to_reviewers; do
  pandoc "$SUB/$d.md" -o "$SUB/$d.pdf" --pdf-engine=xelatex \
    -V mainfont="DejaVu Serif" -V geometry:margin=1in 2>/dev/null
  echo "    $d.pdf: $(pdfinfo "$SUB/$d.pdf" | awk '/Pages/{print $2}') pages"
done



echo ">>> single-file supplementary archive (portal allows one file)"
T=$(mktemp -d)
cp "$MS/supplementary.pdf" "$T/Supplementary_Material.pdf"
cp -r "$BASE/code_v2" "$T/code"; cp -r "$BASE/results_v2" "$T/results"
rm -rf "$T/code/__pycache__"
rm -f "$T/results/"*BROKEN* "$T/results/"*_old* "$T/results/"*smoke* 2>/dev/null || true
cp "$SUB/README_SI.txt" "$T/README.txt" 2>/dev/null || true
(cd "$T" && zip -qr /tmp/_si.zip .) && mv /tmp/_si.zip "$SUB/supplementary.zip"
rm -rf "$T"
echo "    supplementary.zip: $(unzip -l "$SUB/supplementary.zip" | tail -1 | awk '{print $2}') files"

echo ">>> merged cover letter + response (portal has no Supporting Document slot)"
pdfunite "$SUB/cover_letter.pdf" "$SUB/response_to_reviewers.pdf" \
         "$SUB/cover_letter_with_response.pdf"
echo "    cover_letter_with_response.pdf: $(pdfinfo "$SUB/cover_letter_with_response.pdf" | awk '/Pages/{print $2}') pages"

echo ">>> reference check"
cd ../../..
python3 _TAKIP/verify_refs.py \
  "Asus_Pardus_0027_ECG_CS/IEEE_Signal_Processing_Letters/resubmission/manuscript/manuscript.tex" | tail -3

echo ">>> LaTeX flat zip"
python3 _TAKIP/make_latex_zip.py \
  "Asus_Pardus_0027_ECG_CS/IEEE_Signal_Processing_Letters/resubmission/manuscript/manuscript.tex" \
  --name SPL_resubmission || true

echo "=== PACKAGE READY ==="
ls -la "Asus_Pardus_0027_ECG_CS/IEEE_Signal_Processing_Letters/resubmission/SUBMISSION/"
