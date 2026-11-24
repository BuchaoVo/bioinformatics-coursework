from pathlib import Path
from docx import Document
from docx.shared import Cm,Pt,RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.enum.text import WD_ALIGN_PARAGRAPH

ROOT=Path(__file__).resolve().parents[2]

def setfont(obj,size=10.5,bold=None):
    obj.font.name='Times New Roman';obj.font.size=Pt(size)
    if bold is not None:obj.font.bold=bold
    el=obj._element
    pr=el.get_or_add_rPr()
    fonts=pr.find(qn('w:rFonts'))
    if fonts is None:fonts=OxmlElement('w:rFonts');pr.insert(0,fonts)
    fonts.set(qn('w:eastAsia'),'宋体')

def style(path):
    d=Document(path)
    for sec in d.sections:
        sec.page_width=Cm(21);sec.page_height=Cm(29.7)
        sec.top_margin=sec.bottom_margin=Cm(1.9);sec.left_margin=sec.right_margin=Cm(1.9)
        sec.header_distance=sec.footer_distance=Cm(.9)
        h=sec.header.paragraphs[0];h.text='第九讲课程作业 | 集成学习与 Bootstrap';h.alignment=WD_ALIGN_PARAGRAPH.LEFT
        for r in h.runs:setfont(r,8);r.font.color.rgb=RGBColor.from_string('536879')
        f=sec.footer.paragraphs[0];f.alignment=WD_ALIGN_PARAGRAPH.CENTER
        rr=f.add_run();fld=OxmlElement('w:fldSimple');fld.set(qn('w:instr'),'PAGE');rr._r.addnext(fld)
    for name in ['Normal','Body Text','First Paragraph']:
        if name in d.styles:
            st=d.styles[name];setfont(st,10.5);st.paragraph_format.line_spacing=1.2;st.paragraph_format.space_after=Pt(5)
    for name,sz in [('Title',18),('Heading 1',14),('Heading 2',12),('Heading 3',11)]:
        if name in d.styles:
            st=d.styles[name];setfont(st,sz,True);st.font.color.rgb=RGBColor.from_string('142B40')
            st.paragraph_format.space_before=Pt(11);st.paragraph_format.space_after=Pt(6)
    for p in d.paragraphs:
        if p.style.name.startswith('Heading'):p.paragraph_format.keep_with_next=True
        if p.text.startswith(('图1','图2','图3','图4','图5','图6','图7','图8','补图S1')):
            for r in p.runs:setfont(r,9)
        if p.text.startswith(('表1','表2','表A')):p.paragraph_format.keep_with_next=True
    for table in d.tables:
        table.autofit=True
        for i,row in enumerate(table.rows):
            pr=row._tr.get_or_add_trPr();cant=OxmlElement('w:cantSplit');pr.append(cant)
            if i==0:
                repeat=OxmlElement('w:tblHeader');pr.append(repeat)
            for c in row.cells:
                for p in c.paragraphs:
                    p.paragraph_format.space_after=Pt(3);p.paragraph_format.line_spacing=1.1
                    for r in p.runs:setfont(r,8.5,i==0)
                if i==0:
                    tcpr=c._tc.get_or_add_tcPr();shade=OxmlElement('w:shd');shade.set(qn('w:fill'),'EEF3F6');tcpr.append(shade)
    if '课程论文' in path.name:
        for i in range(min(4,len(d.paragraphs))):d.paragraphs[i].alignment=WD_ALIGN_PARAGRAPH.CENTER
        for r in d.paragraphs[0].runs:setfont(r,18,True)
    d.core_properties.author='詹步超';d.core_properties.subject='集成学习课程作业：真实实验与可复现图表'
    d.save(path)

if __name__=='__main__':
    import sys
    for x in sys.argv[1:]:style(Path(x))
