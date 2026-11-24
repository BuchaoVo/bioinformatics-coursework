from pathlib import Path
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4,landscape
from PIL import Image
from reportlab.lib.units import cm
ROOT=Path(__file__).resolve().parents[2]
F=ROOT/'figures';OUT=ROOT/'report'/'thyroid_analysis_figure_atlas.pdf'
W,H=landscape(A4);c=canvas.Canvas(str(OUT),pagesize=(W,H),pageCompression=1)
P=sorted(F.glob('*.png'))
# one cover page
c.setFillColorRGB(.11,.28,.39);c.rect(0,H-3.7*cm,W,3.7*cm,fill=1,stroke=0)
c.setFont('Helvetica',22);c.setFillColorRGB(1,1,1);c.drawString(2*cm,H-2.25*cm,'Thyroid Diagnostic Codes: Model Analysis Atlas')
c.setFillColorRGB(.13,.25,.34);c.setFont('Helvetica',13);c.drawString(2*cm,H-5*cm,'UCI thyroid0387: 9,172 records and 33 analysis figures')
for k,s in enumerate(['01 Data: diagnosis codes, missingness, clinical features, and co-occurrence','02 Statistics: Mann-Whitney, Fisher-BH, PCA, and laboratory profiles','03 Models: three-fold CV, ROC/PR, heatmaps, radar plots, and bootstrap intervals','04 Interpretation: SHAP, permutation importance, ablation, thresholds, and subgroups']):
 c.setFont('Helvetica',11);c.drawString(2.3*cm,H-(7+k*1.1)*cm,s)
c.setFont('Helvetica',10);c.drawString(2.3*cm,1.8*cm,'Target: presence of additional historical diagnosis codes. Associations do not imply causality.');c.showPage()
margin=1.6*cm
for i,p in enumerate(P):
 c.setFont('Helvetica',12);c.setFillColorRGB(.12,.25,.34);c.drawString(margin,H-0.75*cm,'Analysis atlas / Full-resolution PNG files are available in figures/')
 im=Image.open(p);iw,ih=im.size
 availableW=W-2*margin;availableH=H-4.2*cm;s=min(availableW/iw,availableH/ih)
 rw=iw*s;rh=ih*s;x=(W-rw)/2;y=H-1.5*cm-rh
 if y<2.8*cm:y=2.8*cm
 c.drawImage(str(p),x,y,width=rw,height=rh,preserveAspectRatio=True,mask='auto')
 nm=p.stem.replace('_',' ')
 c.setFont('Helvetica',11);c.setFillColorRGB(.1,.19,.25);c.drawCentredString(W/2,1.75*cm,nm[:65])
 c.setFont('Helvetica',8);c.setFillColorRGB(.4,.4,.4);c.drawCentredString(W/2,.65*cm,f'{i+2} / {1+len(P)}')
 c.showPage()
c.save();print(OUT,'figures',len(P))
