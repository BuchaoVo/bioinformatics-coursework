#!/usr/bin/env python3
"""Typeset real-data Chinese dual-column journal-style manuscript. No invented performance."""
from pathlib import Path
import sys,json,math
import numpy as np,pandas as pd
from docx import Document
from docx.shared import Pt,Cm,RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH,WD_BREAK,WD_LINE_SPACING
from docx.enum.section import WD_SECTION_START
from docx.enum.table import WD_TABLE_ALIGNMENT,WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
ROOT=Path(__file__).resolve().parents[2]
R=ROOT/'results';FIG=ROOT/'figures';OUT=ROOT/'report'
audit=json.loads((R/'dataset_summary.json').read_text());score=json.loads((R/'locked_test_result.json').read_text());val=pd.read_csv(R/'validation_model_metrics.csv');test=pd.read_csv(R/'locked_test_all_models.csv');cv=pd.read_csv(R/'training_cross_validation.csv'); imp=pd.read_csv(R/'validation_permutation_importance.csv');ab=pd.read_csv(R/'validation_ablation.csv');rank=pd.read_csv(R/'numeric_mannwhitney_fdr.csv');fisher=pd.read_csv(R/'binary_feature_fisher_fdr.csv');groups=pd.read_csv(R/'test_subgroup_descriptive.csv');sig=pd.read_csv(R/'locked_test_mcnemar_holm.csv');
D=Document();sec=D.sections[0];sec.page_height=Cm(29.7);sec.page_width=Cm(21);sec.top_margin=Cm(1.6);sec.bottom_margin=Cm(1.6);sec.left_margin=Cm(1.65);sec.right_margin=Cm(1.65);sec.header_distance=Cm(.6);sec.footer_distance=Cm(.65)
styles=D.styles;normal=styles['Normal'];normal.font.name='Noto Serif CJK SC';normal.font.size=Pt(8.7);normal.font.color.rgb=RGBColor(32,37,43);normal._element.rPr.rFonts.set(qn('w:eastAsia'),'Noto Serif CJK SC');normal.paragraph_format.line_spacing=1.13;normal.paragraph_format.space_after=Pt(2.3);normal.paragraph_format.first_line_indent=Pt(16)
for nm,sz,color,bold in [('Title',16.6,(18,45,66),True),('Heading 1',10.3,(16,51,76),True),('Heading 2',9.4,(27,65,86),True)]:
 s=styles[nm];s.font.name='Noto Sans CJK SC';s._element.rPr.rFonts.set(qn('w:eastAsia'),'Noto Sans CJK SC');s.font.size=Pt(sz);s.font.bold=bold;s.font.color.rgb=RGBColor(*color)
 s.paragraph_format.space_before=Pt(7 if nm!='Title' else 2);s.paragraph_format.space_after=Pt(3);s.paragraph_format.keep_with_next=True;s.paragraph_format.first_line_indent=Pt(0)
footer=sec.footer.paragraphs[0];footer.alignment=WD_ALIGN_PARAGRAPH.CENTER;footer.style='Normal';footer.paragraph_format.first_line_indent=Pt(0)
r=footer.add_run('课程研究稿 · 数据分析日期：2026-10-22  |  第 ');r.font.size=Pt(7.5)
fld=OxmlElement('w:fldSimple');fld.set(qn('w:instr'),'PAGE');footer._p.append(fld);footer.add_run(' 页')

def p(text='',first=True,boldlead=None):
 q=D.add_paragraph();q.paragraph_format.first_line_indent=Pt(16 if first else 0);q.paragraph_format.widow_control=True
 if boldlead and text.startswith(boldlead):q.add_run(boldlead).bold=True;q.add_run(text[len(boldlead):])
 else:q.add_run(text)
 return q

def heading(text,level=1):D.add_heading(text,level=level)

def note(t):
 q=p(t,False);q.paragraph_format.left_indent=Pt(8);q.paragraph_format.right_indent=Pt(8)
 for rr in q.runs:rr.italic=True;rr.font.size=Pt(7.6);rr.font.color.rgb=RGBColor(93,102,109)
 return q

def fig(name,cap,w=7.85):
 fp=FIG/name
 if not fp.exists():return
 q=D.add_paragraph();q.alignment=WD_ALIGN_PARAGRAPH.CENTER;q.paragraph_format.first_line_indent=Pt(0);q.paragraph_format.space_before=Pt(3);q.paragraph_format.space_after=Pt(1);q.paragraph_format.keep_with_next=True
 q.add_run().add_picture(str(fp),width=Cm(w))
 t=D.add_paragraph(cap);t.alignment=WD_ALIGN_PARAGRAPH.CENTER;t.paragraph_format.first_line_indent=Pt(0);t.paragraph_format.space_after=Pt(5);t.paragraph_format.keep_together=True
 for run in t.runs:run.font.name='Noto Sans CJK SC';run.font.size=Pt(7.3);run.bold=True

def table(cols,rows,width=16.9,compact=True):
 t=D.add_table(rows=1,cols=len(cols));t.alignment=WD_TABLE_ALIGNMENT.CENTER;t.style='Light Shading Accent 1';t.autofit=False
 for j,c in enumerate(cols):t.cell(0,j).text=str(c)
 for row in rows:
  cc=t.add_row().cells
  for j,x in enumerate(row):cc[j].text=str(x)
 for i,row in enumerate(t.rows):
  for cc in row.cells:
   cc.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
   for q in cc.paragraphs:
    q.paragraph_format.first_line_indent=Pt(0);q.paragraph_format.space_after=Pt(1);q.paragraph_format.line_spacing=1.03
    for r in q.runs:r.font.name='Noto Sans CJK SC';r.font.size=Pt(6.8 if compact else 7.2);r.bold=i==0
 return t

# --- title in one column ---
q=D.add_paragraph(style='Title');q.alignment=WD_ALIGN_PARAGRAPH.CENTER;q.add_run('基于集成学习与可解释性分析的甲状腺诊断代码识别研究')
q=D.add_paragraph();q.paragraph_format.first_line_indent=Pt(0);q.alignment=WD_ALIGN_PARAGRAPH.CENTER;r=q.add_run('——UCI Thyroid Disease 9 172条历史临床记录的八模型比较');r.font.size=Pt(10.5);r.font.color.rgb=RGBColor(58,75,86)
q=D.add_paragraph();q.alignment=WD_ALIGN_PARAGRAPH.CENTER;q.paragraph_format.first_line_indent=Pt(0);r=q.add_run('作者：________　单位：________　（中文双栏期刊体例课程研究稿）');r.font.size=Pt(8.7)
q=D.add_paragraph();q.paragraph_format.first_line_indent=Pt(0);q.add_run('摘　要').bold=True
q=p(f'目的：研究多种传统与集成机器学习方法对甲状腺历史诊断代码的识别能力，并系统分析数据质量、变量贡献及内部泛化性。方法：基于UCI thyroid0387.data的9 172条记录、29项输入特征，对4条非合理年龄值转为缺失；在严格隔离相同特征组合的前提下按5 504/1 834/1 834划分训练、验证和测试集。比较高斯朴素贝叶斯、逻辑回归、随机森林、RBF-SVM、XGBoost、LightGBM、CatBoost及MLP，采用分组交叉验证、AP作为主指标、验证集阈值选择、Bootstrap置信区间及McNemar-Holm检验，并结合SHAP、置换重要性、差异检验、热力图与降维进行解释。结果：原始代码“-”6 771例，非“-”2 401例（26.18%）。验证集以LightGBM的AP最高（{val.iloc[0].pr_auc_average_precision:.4f}）；其锁定测试集AP={score["pr_auc_average_precision"]:.4f}（95%CI {score["confidence_intervals"]["ap_ci95"][0]:.4f}～{score["confidence_intervals"]["ap_ci95"][1]:.4f}）、ROC-AUC={score["roc_auc"]:.4f}、F1={score["f1"]:.4f}、灵敏度={score["recall_sensitivity"]:.2%}。去除六项实验室数值后验证集AP由{ab.iloc[0].validation_ap:.4f}降至{ab.loc[ab.ablation=="minus_labs","validation_ap"].iloc[0]:.4f}。结论：结构化实验室数据可支持较高的内部诊断代码判别能力，但历史标签、检测选择机制和缺乏患者级独立标识限制了结果向临床迁移。')
q=D.add_paragraph();q.paragraph_format.first_line_indent=Pt(0);q.add_run('关键词：').bold=True;q.add_run('甲状腺疾病；集成学习；PR-AUC；SHAP；生物医学数据可视化；缺失值；多重检验')
q=D.add_paragraph();q.alignment=WD_ALIGN_PARAGRAPH.CENTER;q.paragraph_format.first_line_indent=Pt(0);r=q.add_run('Comparison of Eight Machine-Learning Models for Historical Thyroid Diagnostic Codes');r.italic=True;r.bold=True;r.font.name='Times New Roman';r.font.size=Pt(10)
q=D.add_paragraph();q.paragraph_format.first_line_indent=Pt(0);q.add_run('Abstract: ').bold=True;q.add_run(f'This retrospective benchmarking study analyzed 9,172 UCI thyroid records with 29 candidate features, comparing eight machine-learning classifiers. Four implausible ages were treated as missing; exact feature signatures were segregated across train, validation and held-out test subsets (5,504/1,834/1,834). Average precision (AP) determined model selection on validation only. LightGBM achieved test AP {score["pr_auc_average_precision"]:.4f}, ROC-AUC {score["roc_auc"]:.4f}, F1 {score["f1"]:.4f}, and sensitivity {score["recall_sensitivity"]:.4f}. Laboratory-value ablation reduced validation AP to {ab.loc[ab.ablation=="minus_labs","validation_ap"].iloc[0]:.4f}. Results describe recognition of recorded diagnosis codes, not prospective disease risk or clinical utility. External validation remains necessary.')
q=D.add_paragraph();q.paragraph_format.first_line_indent=Pt(0);q.add_run('Keywords: ').bold=True;q.add_run('Thyroid; LightGBM; CatBoost; Interpretability; Data quality; Average precision')
q=D.add_paragraph();q.paragraph_format.first_line_indent=Pt(0);r=q.add_run('基金项目：无（待按实际情况填写）　　稿件性质：课程研究/方法比较；尚未针对任何具体期刊模板校核。');r.font.size=Pt(7.7)

# --- two-column paper proper ---
s2=D.add_section(WD_SECTION_START.CONTINUOUS);s2.page_width=Cm(21);s2.page_height=Cm(29.7);s2.left_margin=Cm(1.65);s2.right_margin=Cm(1.65);s2.top_margin=Cm(1.6);s2.bottom_margin=Cm(1.6);columns=s2._sectPr.xpath('./w:cols')[0];columns.set(qn('w:num'),'2');columns.set(qn('w:space'),'425');columns.set(qn('w:equalWidth'),'1')
heading('1　引言')
p('甲状腺相关疾病的辅助分析涉及临床症状、治疗背景和多种生化检验。实验室变量之间存在相关性、强偏态和非随机缺失，使简单的条件独立概率模型难以充分描述复杂特征关系。树集成算法能够处理非线性和交互效应，但内部分类精度亦可能受历史诊断流程、检测选择及记录结构影响。')
p('既有120条急性炎症教学记录显示：重复特征模式可能令随机测试的准确率虚高。为避免只关注单个准确率，本研究在UCI甲状腺历史记录中同时检查组间分布、重复模式、概率可靠性、分组交叉验证、特征消融和多模型比较。研究主要回答：（1）哪种模型获得较高阳性平均精度AP？（2）实验室数据、检测指示变量和转诊来源的作用如何？（3）高内部性能是否伴随潜在的解释和推广限制？')
heading('2　材料与方法')
heading('2.1　数据来源与标签定义',2)
p('原始数据为UCI Machine Learning Repository的Thyroid Disease集合中thyroid0387.data版本，1980年代采集，9 172行。每行通过逗号和诊断代码后的方括号记录标识解析为29项预测字段、1项诊断字段及1项记录标识。预测字段覆盖年龄与性别、用药/病史、6项检验是否执行、6项检验数值以及转诊来源。')
p('诊断代码可出现多字母复合标记（例如GK、AK与C|I），故不将原始代码直接转换为互斥20分类。主任务规定原始代码“-”为0，任一非“-”代码为1。此标签仅表示“历史记录是否含附加诊断代码”，不是严格的健康/患病金标准；涉及治疗状态、转诊过程的特征可能存在过程性相关。方括号标识完全排除在特征集合外。')
fig('10_target_and_diagnosis_codes.png','图1　研究样本诊断代码及二分类标签分布')
heading('2.2　数据质控与缺失值策略',2)
p('对全部9 172例保留原始行，不进行按结果变量的删除。对4条不合理年龄（455、65511、65512、65526岁）实施预先定义的1～110岁有效区间检查，超界转换为缺失。对“?”记为NaN；数值变量在训练折内使用中位数填补并添加缺失指示，之后标准化；二元变量用训练折众数填补，性别与转诊来源经训练折众数填补和独热编码。')
p(f'TBG缺失{audit["missing_rate"]["TBG"]:.2%}，T3缺失{audit["missing_rate"]["T3"]:.2%}，TSH缺失{audit["missing_rate"]["TSH"]:.2%}。检验是否执行标记与实验室数值缺失完全对应，应视为采样和临床决策信息而非独立实验室指标。为评价影响，另设删除检测标记的消融组。极端数值仅在小提琴图和箱线图中为可读性采用第99百分位截尾显示，正式模型中不作该截尾。')
fig('11_missingness_patterns.png','图2　实验室数据缺失模式与主要变量缺失率')
heading('2.3　划分、模型与调参协议',2)
p('根据29个预处理前的完整特征组合建立分组键，使用分层分组划分得到训练5 504、验证1 834和锁定测试1 834条，三个子集不共享完全相同的完整特征模式；训练内部使用3折StratifiedGroupKFold比较AP，全部数据变换均限定在各自训练折内拟合。这样的分组不能保证跨折属于不同患者，因为缺乏可验证的患者唯一标识。')
p('高斯朴素贝叶斯使用每类各维条件高斯密度连乘；逻辑回归以正则化logit进行判别；随机森林采用自助采样和特征随机化；RBF-SVM通过核函数寻找非线性分类超平面；XGBoost、LightGBM、CatBoost通过梯度提升树学习特征交互；MLP由64和32个神经元的两层隐含层构成，使用早停防止过拟合。各算法采用固定、预先设定的中等规模参数，并未宣称进行全局最优超参数搜索。')
heading('2.4　指标及统计方法',2)
p('主评价指标为Average Precision（AP）：AP=Σ(Rₙ−Rₙ₋₁)Pₙ，是对Precision-Recall曲线的离散概括；同一实验还计算ROC-AUC、灵敏度、特异度、F1、Balanced Accuracy、Brier分数及Log Loss。选型严格依照验证集AP最大值；在验证集上扫描0.05至0.95的阈值，以最大F1确定最终分类决策阈值。所有锁定测试表现只在模型选定后计算。')
p('对选中模型的测试集AP及ROC-AUC使用650次非参数Bootstrap生成百分位95%置信区间。以高斯朴素贝叶斯为对照的7次配对McNemar精确检验采用Holm逐步校正；每项检验基于两模型正确与否不一致的病例数，不涉及因果推断。对数值变量使用双侧Mann–Whitney U检验和秩二列相关效应量，对二元变量用Fisher精确检验，分别用Benjamini–Hochberg控制假发现率；组间差异为探索性的单变量相关，不是变量诊断意义的独立证明。')
heading('3　实验结果')
heading('3.1　数据分布与测量结构',2)
p(f'非“-”代码2 401例（26.18%），其中“-”代码6 771例（73.82%）。原始完全相同特征模式涉及{audit["exact_raw_duplicate_feature_rows"]}条记录；经签名分组划分，验证和测试集中与训练集相同的完整特征向量均为0条。三组阳性比例接近，目标标签分层成功。')
fig('27_split_target_proportions.png','图3　训练、验证及测试数据的标签比例')
p('热力图发现临床检验指标存在相互相关性及特定诊断代码对应的差异表型。因TSH分布右偏，诊断代码表型图对TSH仅做log(1+x)显示转换，后续以全体样本的中位数及IQR对各诊断代码组的中位数进行标准化，未用于监督模型训练。')
fig('12_clinical_feature_correlations.png','图4　年龄及六项检验数值的Spearman相关矩阵')
fig('15_diagnosis_laboratory_profiles.png','图5　主要诊断代码的实验室组间表型热图')
heading('3.2　组间统计差异与异质性',2)
p('对七项数值指标分别进行Mann–Whitney U检验，TSH组间中位数为2.10对1.30，秩二列相关系数为0.201；T3分别为1.60对2.00，效应量为−0.199。TBG仅有349个非缺失值，其中非“-”代码98个，极端缺失条件下即使差异显著也不宜过度推广。年龄差异效应量很小，显示统计显著不代表实质预测价值。')
fig('13_numeric_effect_sizes.png','图6　连续变量的组间效应量与BH校正后的显著性')
p('Fisher检验显示pregnant、psych和on_thyroxine等二元字段与标签存在统计关联，但这些指标涉及用药、选择检测及患者构成，不能解读为致病效应或因果方向。')
fig('14_binary_feature_associations.png','图7　二元字段与目标标签的单因素关联方向')
p('连续指标的小提琴分布揭示同一类别内部亦存在显著重叠。PCA以标准化并填补后的输入特征在2 400例随机样本上进行非监督二维投影，可检视总体结构，但其分离程度不应被视为监督分类性能。')
fig('16_clinical_feature_pca.png','图8　输入特征主成分二维投影（探索性）')
heading('3.3　八模型判别性能',2)
p(f'验证集的AP最高模型为LightGBM（{val.iloc[0].pr_auc_average_precision:.4f}），随后为随机森林（{val.loc[val.model=="RandomForest","pr_auc_average_precision"].iloc[0]:.4f}）和CatBoost（{val.loc[val.model=="CatBoost","pr_auc_average_precision"].iloc[0]:.4f}）。训练集三折AP平均值CatBoost={cv.loc[cv.model=="CatBoost","cv_ap_mean"].iloc[0]:.4f}，LightGBM={cv.loc[cv.model=="LightGBM","cv_ap_mean"].iloc[0]:.4f}；三折结果仅用于稳定性描述，模型选型仍按照既定验证集主指标完成。')
fig('21_cross_validation_stability.png','图9　训练集分组交叉验证AP均值和标准差')
fig('19_model_metrics_heatmap.png','图10　八模型锁定测试集多项评价指标热图')
p(f'在锁定测试集，LightGBM的AP={score["pr_auc_average_precision"]:.4f}、ROC-AUC={score["roc_auc"]:.4f}、F1={score["f1"]:.4f}，其AP的95% Bootstrap CI为[{score["confidence_intervals"]["ap_ci95"][0]:.4f}，{score["confidence_intervals"]["ap_ci95"][1]:.4f}]。已使用验证集选择的阈值{score["threshold"]:.2f}，得到真正例{score["TP"]}、假阳例{score["FP"]}、真阴例{score["TN"]}、假阴例{score["FN"]}。')
fig('30_all_models_roc_curves.png','图11　八模型测试集ROC曲线（模型选择前独立锁定）')
fig('31_all_models_pr_curves.png','图12　八模型测试集Precision–Recall曲线')
fig('20_model_performance_radar.png','图13　不同模型的多维评价雷达图（原始0～1比例）')
p('图10与图13表明树集成算法在大多数判别及分类指标上优于线性/概率基线。但不同指标回答的问题不同：AP衡量不平衡标签下阳性排序，F1受验证集选定的阈值影响，敏感性偏高通常伴随假阳性增加。因此不能简单以雷达图面积定义单一最优模型。')
heading('3.4　统计比较和概率可靠性',2)
p(f'与GaussianNB比较，LightGBM仅其判断正确的病例数b={int(sig.loc[sig.model=="LightGBM","b_model_only_correct"].iloc[0])}，而仅NB判断正确的病例数c={int(sig.loc[sig.model=="LightGBM","c_NB_only_correct"].iloc[0])}；经Holm校正的精确McNemar P={sig.loc[sig.model=="LightGBM","holm_adjusted_p"].iloc[0]:.2e}，支持两模型内部分类错误发生概率存在差异。该检验并不证明任一方法对新医院或新时期数据具有同样提升。')
fig('05_test_confusion_calibration.png','图14　LightGBM测试集混淆矩阵与概率可靠性曲线')
fig('32_model_calibration_curves.png','图15　主要模型预测概率与观测标签比例关系')
p(f'虽然LightGBM有高ROC-AUC，其Brier分数仍为{score["brier"]:.4f}，需与校准图联合解释。本文绘制的是未经再校准的原生概率，可能存在系统性偏差；对类别权重处理后的概率尤其不能直接作为个人临床风险。未来应使用独立校准集实施Platt缩放或等距回归，再在完全独立测试集验证。')
heading('3.5　特征贡献与消融',2)
p(f'验证集置换重要性显示TSH对AP的平均下降贡献最高（{imp.iloc[0].ap_decrease:.3f}），其次为T3（{imp.iloc[1].ap_decrease:.3f}）、TT4（{imp.iloc[2].ap_decrease:.3f}）与FTI（{imp.iloc[3].ap_decrease:.3f}）。模型解释是关联性的，且在TSH、T3、TT4等相关输入同时存在时，单项重要性会因特征冗余而分摊。')
fig('24_permutation_importance.png','图16　验证集置换重要性与重复实验变异')
fig('09_optional_validation_shap.png','图17　验证集LightGBM平均绝对SHAP值排序')
p(f'移除六项检验数值后，LightGBM验证集AP由{ab.iloc[0].validation_ap:.4f}降为{ab.loc[ab.ablation=="minus_labs","validation_ap"].iloc[0]:.4f}；移除转诊来源后为{ab.loc[ab.ablation=="minus_referral","validation_ap"].iloc[0]:.4f}；移除六项测量标记后为{ab.loc[ab.ablation=="minus_measurement_flags","validation_ap"].iloc[0]:.4f}。实验支持实验室数值提供了主导的内部预测信息，但不能由此断言某一检验变量本身构成因果医学机制。')
fig('23_feature_group_ablation.png','图18　变量组消融的验证集AP对比')
heading('3.6　稳健性与亚组探索',2)
p('记录的不同年龄、性别及转诊来源组在样本量、类别基线发生率和敏感性/特异度上可能有所差异；亚组结果仅作描述性检查，未事先设定临床公平性检验假设，也未进行相应多重校正。小亚组和低阳性样本数量下，单个百分比存在较高抽样波动。')
fig('25_test_age_sex_subgroups.png','图19　LightGBM测试集年龄及性别分层性能')
heading('4　讨论')
p('本研究与简单朴素贝叶斯工作单相比，优势在于：使用较大的原始历史医疗数据，处理了年龄异常编码和严重非随机缺失，严格隔离完全重复的特征向量，通过同一数据协议比较八种不同建模范式，并把模型选择、阈值确定与最终评估分离。树集成算法相对线性和高斯朴素贝叶斯表现较强，与数值检测指标的非线性关系、特征间相关和跨类别交互相符。')
p('需要警惕四项限制。第一，研究终点为原始附加诊断代码，既含疾病，也可能含治疗/检验状态，不能直接等同临床确诊。第二，TSH等检验值可能本身参与当年的诊断过程，因此模型预测部分是在复现诊断规则；并非预测未来事件。第三，测量与转诊机制可能传递医疗流程的信息，TBG的高度缺失亦降低通用性。第四，唯一记录号不能验证患者身份，虽然做了完整特征组合分组，仍可能包含重复患者的不同就诊记录；同时缺少独立医院和时间外验证。')
p('在统计分析中，检验组间差异得到极小P值，并不意味着真实效应很大；例如年龄差异效应量远低于TSH或T3。多模型McNemar-Holm差异很显著，但选定模型相较其他先进树模型的AP差异较小，尚需开展配对Bootstrap和更严格的外部重复实验。模型雷达图与热力图属于多指标综合展示，不取代预注册主指标。')
heading('5　结论与研究展望')
p(f'基于9 172条UCI thyroid0387记录开展的八模型比较表明，LightGBM在预定义验证集上获得最高AP，其冻结测试集AP={score["pr_auc_average_precision"]:.4f}，ROC-AUC={score["roc_auc"]:.4f}，F1={score["f1"]:.4f}。消融实验显示实验室数值是主要判别依据。数据预处理、成组划分、概率校准诊断、特征解释和统计比较构成了较完整的课程研究流程。上述表现仅反映历史数据内部分类，不用于临床部署或个人医疗建议。')
p('后续可将复合诊断代码按官方疾病语义拆解为多标签任务，重点报告macro/micro-F1和稀有亚类AP；进一步使用UCI Diabetes 130-US Hospitals的101 766次住院记录，在患者级划分和30天再入院终点上独立复现；对于MIMIC-IV须按要求完成数据使用培训与访问审批，并考虑时间/机构外验证和模型漂移评估。')
heading('参考文献')
refs=[
'[1] QUINLAN R. Thyroid Disease [DB/OL]. UCI Machine Learning Repository, 1986. DOI:10.24432/C5D010.',
'[2] CHEN T, GUESTRIN C. XGBoost: A Scalable Tree Boosting System[C]//KDD. 2016:785-794.',
'[3] KE G, MENG Q, FINLEY T, et al. LightGBM: A Highly Efficient Gradient Boosting Decision Tree[C]//NeurIPS. 2017.',
'[4] PROKHORENKOVA L, GUSEV G, VOROBEV A, et al. CatBoost: Unbiased Boosting with Categorical Features[C]//NeurIPS. 2018.',
'[5] BREIMAN L. Random Forests[J]. Machine Learning, 2001,45:5-32.',
'[6] CORTES C, VAPNIK V. Support-Vector Networks[J]. Machine Learning, 1995,20:273-297.',
'[7] LUNDBERG S M, LEE S I. A Unified Approach to Interpreting Model Predictions[C]//NeurIPS. 2017.',
'[8] SAITO T, REHMSMEIER M. The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets[J]. PLoS ONE, 2015,10:e0118432.',
'[9] PEDREGOSA F, VAROQUAUX G, GRAMFORT A, et al. Scikit-learn: Machine Learning in Python[J]. JMLR, 2011,12:2825-2830.',
'[10] HOLM S. A simple sequentially rejective multiple test procedure[J]. Scandinavian Journal of Statistics, 1979,6:65-70.',
'[11] BENJAMINI Y, HOCHBERG Y. Controlling the false discovery rate: a practical and powerful approach to multiple testing[J]. JRSS B, 1995,57:289-300.',
'[12] MCHUGH M L. The chi-square test of independence[J]. Biochem Med, 2013,23:143-149.'
]
for ref in refs:
 q=p(ref,False);q.paragraph_format.left_indent=Pt(4);q.paragraph_format.hanging=Pt(4);q.paragraph_format.space_after=Pt(2)

# full width supplemental tables on final page
s3=D.add_section(WD_SECTION_START.NEW_PAGE);s3._sectPr.xpath('./w:cols')[0].set(qn('w:num'),'1');heading('附录A　实验数据表及复现实验记录')
p('表A1列示所有八模型的锁定测试指标（阈值已由验证集独立选定）。该表为完整披露，不用于重新选择最终模型。',False)
table(['模型','AP','ROC-AUC','F1','敏感性','特异度','Brier'],[[x.model,f'{x.pr_auc_average_precision:.4f}',f'{x.roc_auc:.4f}',f'{x.f1:.4f}',f'{x.recall_sensitivity:.4f}',f'{x.specificity:.4f}',f'{x.brier:.4f}'] for x in test.itertuples()])
p('表A2展示训练集交叉验证和验证集选型AP，三折标准差不是置信区间。',False)
q=cv.set_index('model');table(['模型','3折CV AP均值','3折CV标准差','验证AP','验证F1','阈值'],[[x.model,f'{q.loc[x.model,"cv_ap_mean"]:.4f}',f'{q.loc[x.model,"cv_ap_sd"]:.4f}',f'{x.pr_auc_average_precision:.4f}',f'{x.f1:.4f}',f'{x.threshold:.2f}'] for x in val.itertuples()])
p('表A3列示LightGBM的预指定组消融结果，基于验证集AP。',False)
table(['输入变量组','特征数','验证AP','验证ROC-AUC'],[[x.ablation,x.n_features,f'{x.validation_ap:.4f}',f'{x.validation_roc_auc:.4f}'] for x in ab.itertuples()])
p('附录B　实验环境、数据利用与可重复性说明',False)
p('数据来源：UCI Thyroid Disease，thyroid0387.data；有效样本数9 172；诊断字段是原始数据的负号/字母代码。代码使用Python、NumPy、Pandas、scikit-learn、XGBoost、LightGBM、CatBoost、SHAP、Matplotlib和SciPy。随机种子2026；各数值填补和标准化限制在训练折内部。CSV指标和每模型锁定测试概率见results/，33幅高分辨率图见figures/，完整处理程序见code/。')
p('投稿提示：这是一份按照一般中文科技期刊双栏习惯排版的课程论文稿，而非特定CNKI期刊的官方模板；最终投稿仍需选定目标期刊、核验版面尺寸、参考文献GB/T 7714细则、伦理声明、作者署名与数据共享格式。')
D.core_properties.title='基于集成学习与可解释性分析的甲状腺诊断代码识别研究';D.core_properties.subject='UCI thyroid0387实测分析；中文双栏期刊式课程论文';D.core_properties.author='课程研究稿（作者待填）'
target=OUT/'thyroid_model_comparison.docx';D.save(target);print(target)
