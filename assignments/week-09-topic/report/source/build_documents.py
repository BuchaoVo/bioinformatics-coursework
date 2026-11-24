"""Rebuild the PDF and Word deliverables from source (run from any directory)."""
from pathlib import Path
import subprocess,sys
ROOT=Path(__file__).resolve().parents[2]
P=ROOT/'report'/'source';OUT=ROOT/'report'/'generated';OUT.mkdir(exist_ok=True)
subprocess.run([sys.executable,str(P/'assemble.py')],check=True,cwd=ROOT)
for stem,src,wordsrc in [
 ('第九讲_集成学习_课程论文','manuscript_pdf.md','manuscript_word.md'),
 ('第九讲_A-H答案与实验绘图指南','supplement_pdf.md','supplement.md')]:
    common=['pandoc','-f','markdown+tex_math_single_backslash','--no-highlight']
    subprocess.run(common+[str(P/src),'--template',str(P/'template.tex'),'--pdf-engine=xelatex','-o',str(OUT/(stem+'.pdf'))],check=True,cwd=ROOT)
    subprocess.run(common+[str(P/wordsrc),'-o',str(OUT/(stem+'.docx'))],check=True,cwd=ROOT)
    subprocess.run([sys.executable,str(P/'style_docx.py'),str(OUT/(stem+'.docx'))],check=True,cwd=ROOT)
print('Deliverables:',OUT)
