from pathlib import Path
import pandas as pd,json,re
ROOT=Path(__file__).resolve().parents[2]
paper=ROOT/'report'/'source'
figs=[
('Fig01_design','图1　数据构成与分析框架。a，训练、验证及锁定测试特征的记录数；有标签集合用阴/阳性堆叠条表示，斜线灰条表示测试标签未读取。WDBC训练/验证为341/114条；CTG为1279/424片段，分别来自209/73个录音。b，共享5折开发集内部CV及固定验证分析；这两种评价使用重叠资料。c，Bagging的并行树聚合示意，随机森林另在每次分裂随机选候选特征。d，Boosting的顺序更新示意，式中η为学习率；c、d为机制图，不是经验测量。'),
('Fig02_tree_bootstrap','图2　WDBC单树复杂度、根节点变化与Bootstrap抽样覆盖。a，固定训练/验证划分下的树深度—AUC；每个深度一次固定拟合，无误差条。b，前10次训练Bootstrap重拟合的深度2树根节点频数及同一特征阈值范围，不同特征的阈值不作跨量纲比较。c、d，n=5和n=341各1000次抽样的独特记录比例分布；虚线为有限n理论期望，实线为经验均值，OOB比例等于1减去独特比例。'),
('Fig03_wdbc_comparison','图3　WDBC公平模型比较。a–c，455条开发记录的共享分层5折CV：每个小点代表一折，大点及横线为均值±样本SD（ddof=1），不是置信区间。d、e，在341条train拟合后，于固定114条validation上的ROC与精确率–召回率曲线；43条阳性，对应PR无信息参考线0.377。SVM使用未校准决策分数，其余曲线使用模型概率；完全生长单树的补充结果见附录表A1。两种评价均属内部分析。'),
('Fig04_aggregation','图4　WDBC树聚合收益及其边界。a、b，Bagging和随机森林在1、10、50、100、300棵树下的固定验证AUC与BA，均为单种子结果，无重复拟合误差条。c，100棵树在114条验证记录上的二元错误两两Pearson相关分布；均值0.270与0.280，树对相互依赖，不能当作独立重复。d，等方差、等两两相关条件下的理论方差曲线，纵轴除以单模型方差σ²；此ρ不同于c的经验错误相关。'),
('Fig05_stability_uncertainty','图5　WDBC两类Bootstrap回答不同问题。a，30次共享训练重抽样下每个固定验证样本的预测概率SD；点为样本，箱体为25%–75%分位数，中线为中位数，须为1.5 IQR范围，菱形为均值，箱体不是CI。b，实心圆为相对完整train模型的平均类别翻转率，空心方块为30次预测中出现两类的样本比例。c、d，原train拟合模型冻结后，对114条validation进行2000次配对记录Bootstrap，点为原始指标差，线为2.5%–97.5%百分位区间；正值表示RF较高。该区间不包括重训练或模型选择不确定性。'),
('Fig06_boosting','图6　WDBC梯度提升的学习率—轮数关系。三列学习率η为0.03、0.1、0.3，均使用深度3树和最多300轮；a–c为log-loss，d–f为ROC-AUC，虚线表示训练，实线表示固定验证。圆点标记所考察范围内的最低验证损失；η=0.03的最小值位于第300轮边界。各纵轴显示范围可不同，应按刻度读数。曲线为单次预设拟合，没有置信带，最小值未反馈更新主比较的100轮参数。'),
('Fig07_feature_importance','图7　WDBC相关特征与重要性。选取MDI前8与置换重要性前3的并集，共9个变量；a，以341条train计算Spearman相关，色标固定−1至1，特征编号供b、c对应。b，固定100树随机森林的归一化训练不纯度重要性MDI。c，固定114条validation上逐特征随机置换30次的AUC下降均值±样本SD，保留负值。b、c单位不同；该特征子集不等于完整置换重要性排名。相关特征可互相替代，置换SD也不是因果效应的CI。'),
('Fig08_ctg_grouped','图8　CTG的录音分组验证与簇重抽样。a–c，1703个开发片段、282个录音的共享分层分组5折CV，小点为折值，大点及线为均值±折间SD。d，以209个训练录音簇进行30次共享有放回重抽样，对424个固定验证片段计算预测概率SD；箱线、点和菱形含义同图5a。e，以73个验证录音簇进行2000次配对Bootstrap，展示固定模型AP差值及百分位95%区间；RF−LR区间跨0。指标按片段计算，但划分及重抽样按录音进行；录音隔离不等同于已证实的孕妇个体独立。'),
('FigS01_adaboost_weights','补图S1　WDBC AdaBoost前10轮的真实样本权重变化。a，每轮当前错分样本的权重总和在更新前后变化，当前SAMME配置中更新后约为0.5。b，8条固定训练记录更新后的权重相对于初始均匀权重1/341的倍数，色标采用对数尺度；×表示该样本在相应轮被误分。示例优先包含首轮误分记录，并非总体随机样本。所有权重来自实际模型且逐轮加权误差已与sklearn存储结果核对。')]
(paper/'figure_captions.md').write_text('# 主图及补图图注\n\n'+'\n\n'.join(c for _,c in figs))
text=(paper/'manuscript_base.md').read_text()
for i,(stem,cap) in enumerate(figs[:8],1):
    text=text.replace(f'@FIG{i:02d}@',f'![]({(ROOT/"figures"/stem).as_posix()}.png){{width=6.75in}}\n\n{cap}')
names={'logistic':'LR','rbf_svm':'RBF-SVM','single_tree':'Tree（d=4）','deep_tree':'Tree（全树）','bagging':'Bagging','random_forest':'RF','adaboost':'AdaBoost','gradient_boosting':'GB'}
order=list(names)
for ds in ['wdbc','ctg']:
    val=pd.read_csv(ROOT/'results'/ds/'validation-metrics.csv').set_index('model')
    cv=pd.read_csv(ROOT/'results'/ds/'cv-summary.csv').set_index('model')
    table='| 模型 | 验证AUC | 验证AP | 验证BA | CV AUC | CV AP | CV BA |\n|:--|--:|--:|--:|:--|:--|:--|\n'
    for m in order:
        row=[names[m]]+[f'{val.loc[m,k]:.4f}' for k in ['roc_auc','average_precision','balanced_accuracy']]+[f'{cv.loc[m,k+"_mean"]:.4f}±{cv.loc[m,k+"_sd"]:.4f}' for k in ['roc_auc','average_precision','balanced_accuracy']]
        table+='| '+' | '.join(row)+' |\n'
    text=text.replace('@TABLE_'+ds.upper()+'@',table)
st='| 数据/模型 | 概率SD均值 | 平均翻转率 | 曾变化比例 |\n|:--|--:|--:|--:|\n'
for ds in ['wdbc','ctg']:
    s=pd.read_csv(ROOT/'results'/ds/'stability-summary.csv').set_index('model')
    for m in ['single_tree','deep_tree','bagging','random_forest','gradient_boosting','logistic']:
        r=s.loc[m]
        st+=f'| {ds.upper()} / {names[m]} | {r.mean_prediction_sd:.4f} | {r.mean_class_flip_vs_full_train:.2%} | {r.fraction_samples_any_class_change:.2%} |\n'
text=text.replace('@TABLE_STABILITY@',st)
p=json.loads((ROOT/'results/wdbc/protocol.json').read_text())
for key in ['python','sklearn','numpy','pandas']:
    text=text.replace('@'+key.upper()+'@',p['versions'][key])
assert not re.search(r'@[A-Z0-9_]+@',text)
text=text.replace(r'\begin{multicols}{2}', '\n```{=latex}\n'+r'\begin{multicols}{2}'+'\n```\n').replace(r'\end{multicols}', '\n```{=latex}\n'+r'\end{multicols}'+'\n```\n')
(paper/'manuscript.md').write_text(text)
(paper/'manuscript_pdf.md').write_text(re.sub(r'(figures/[^\)]+)\.png',r'\1.pdf',text))
word=re.sub(r'\\begin\{center\}.*?\\end\{center\}', '# 生物医学表格数据中集成学习的性能与稳定性研究\n\n——基于决策树、Bootstrap 与双数据集内部验证的比较\n\n作者：詹步超　学号：24520260158428　专业：健康大数据与智能医学\n\n第九讲课程作业 · 集成学习与 Bootstrapping', text, flags=re.S)
word=re.sub(r'\\(?:begin|end)\{multicols\}(?:\{2\})?', '', word).replace('\\newpage','')
word=word.replace('```{=latex}\n\n```','')
(paper/'manuscript_word.md').write_text(word)
# Supplement: a complete A-H response followed by the plotting protocol and a real AdaBoost trace.
sup=(paper/'worksheet_final.md').read_text()+'\n\n\\newpage\n\n'+(paper/'experiment_design.md').read_text()+'\n\n\\newpage\n\n# 补充图：真实AdaBoost权重轨迹\n\n'+f'![]({ROOT}/figures/FigS01_adaboost_weights.png){{width=6.75in}}\n\n'+figs[-1][1]+'\n\n# 主图完整图注\n\n'+'\n\n'.join(c for _,c in figs[:8])
# Enable explicit TeX math delimiters in pandoc reader.
(paper/'supplement.md').write_text(sup)
(paper/'supplement_pdf.md').write_text(sup.replace('FigS01_adaboost_weights.png','FigS01_adaboost_weights.pdf'))
print('Assembled manuscript and supplement')
