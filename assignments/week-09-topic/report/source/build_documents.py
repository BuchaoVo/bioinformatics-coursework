"""Rebuild the PDF and Word deliverables from source (run from any directory)."""
from pathlib import Path
import subprocess,sys
ROOT=Path(__file__).resolve().parents[2]
P=ROOT/'report'/'source';OUT=ROOT/'report'/'generated';OUT.mkdir(exist_ok=True)
subprocess.run([sys.executable,str(P/'assemble.py')],check=True,cwd=ROOT)
for stem,src,wordsrc in [
 ('week09_ensemble_learning_course_paper','manuscript_pdf.md','manuscript_word.md'),
 ('week09_worksheet_solutions_and_plot_guide','supplement_pdf.md','supplement.md')]:
    common=['pandoc','-f','markdown+tex_math_single_backslash','--no-highlight']
    subprocess.run(common+[str(P/src),'--template',str(P/'template.tex'),'--pdf-engine=xelatex','-o',str(OUT/(stem+'.pdf'))],check=True,cwd=ROOT)
    subprocess.run(common+[str(P/wordsrc),'-o',str(OUT/(stem+'.docx'))],check=True,cwd=ROOT)
    subprocess.run([sys.executable,str(P/'style_docx.py'),str(OUT/(stem+'.docx'))],check=True,cwd=ROOT)
print('Deliverables:',OUT)
