# Bioinformatics Course

> 一个学期（约 16 周）的生物信息学课程学习与作业仓库。  
> **Repository visibility:** Public · **Languages:** Python, Shell · **Reports:** Markdown, PDF

## 课程简介

本仓库用于系统管理生物信息学课程的学习记录、实验代码、Jupyter Notebook、数据处理流程、课程报告和参考资料。仓库按周组织，每周作业遵循统一结构，保证代码可复现、报告可追溯、学习过程可持续维护。

> **课程信息待核对：** 当前收到的压缩包名为“健康医疗大数据-实践”，其中可识别的材料包括高维基因表达与降维、统计检验与假设设计、甲状腺数据分析、线性模型与 LDA、集成学习等。它与“生物信息学”课程可能存在交叉，但不能仅凭文件名认定为完整的 16 周课程大纲。请在 `course-info/` 中补入正式课程大纲和作业清单后，再确认周次映射。

## Repository Map

```text
.
├── README.md
├── course-info/              # 课程大纲、教学日历、作业清单
├── notes/                    # 课程笔记与知识点整理
│   ├── concepts/              # 生物信息学核心概念
│   ├── tools/                 # Linux、Shell、Python、常用工具
│   └── literature/            # 论文阅读记录
├── assignments/              # 每周作业（Week 01–16）
│   ├── week-01-topic/
│   │   ├── README.md
│   │   ├── code/
│   │   ├── notebooks/
│   │   ├── report/
│   │   ├── figures/
│   │   ├── results/
│   │   └── data/              # 仅放可公开、可再分发的小型数据或说明
│   └── ...
├── projects/                 # 综合实践/课程项目
├── resources/                # 教程、软件、数据库和参考文献索引
├── templates/                # 作业、报告、Notebook 模板
├── environment/              # 环境依赖与配置
└── .github/workflows/        # 可选：Notebook/Markdown 检查
```

## 课程内容索引

请根据正式教学日历调整周次和主题。下表仅作为初始占位，不代表教师实际授课顺序。

| 周次 | 主题 | 主要产出 |
|---|---|---|
| 01 | 生物信息学导论与研究问题 | 学习笔记、环境检查 |
| 02 | 生物医学数据类型与数据质量 | 数据字典、质量审计 |
| 03 | Linux / Shell 与可复现工作流 | Shell 脚本 |
| 04 | Python 数据处理与可视化 | Notebook |
| 05 | 统计推断与假设检验 | 分析报告 |
| 06 | 多重检验与 FDR | 模拟/真实数据分析 |
| 07 | 高维数据处理与降维 | PCA 等分析 |
| 08 | 线性模型与分类方法 | 模型比较报告 |
| 09 | 集成学习与 Bootstrap | 实验代码、报告 |
| 10–16 | 待依据课程大纲补充 | 每周作业与报告 |

## 每周作业交付规范

每个作业目录至少包含：

- `README.md`：研究问题、任务要求、数据来源、运行方式、主要结果和限制。
- `code/`：可复用的 `.py` 与 `.sh` 脚本。
- `notebooks/`：探索、分析与可视化 Notebook。
- `report/`：最终报告的 `.md` 与 `.pdf`。
- `figures/`：报告引用图表。
- `results/`：必要的表格、模型指标和日志。
- `data/README.md`：数据来源、许可、字段、获取方法；默认不提交原始数据。

## Reproducibility

建议每次作业记录 Python 版本、依赖、随机种子、数据来源与校验值。优先使用相对路径，避免写入本机绝对路径。

```bash
git clone https://github.com/<YOUR_GITHUB_USERNAME>/bioinformatics-course.git
cd bioinformatics-course
python -m venv .venv
# Linux / macOS
source .venv/bin/activate
python -m pip install -r environment/requirements.txt
```

## Public Repository & Data Policy

本仓库设置为 Public。公开前请逐项确认：

1. 不提交患者级可识别信息、姓名学号、账号凭据、访问令牌或私人路径。
2. 不默认上传原始医学数据、受限数据、教师答案或未经授权的第三方材料。
3. 在 `data/README.md` 中记录数据来源、DOI/URL、许可、版本、下载日期和 SHA-256。
4. 大文件优先提供下载脚本或来源链接；确有必要时再评估 Git LFS。
5. Notebook 输出中检查是否残留患者记录、个人信息、绝对路径或访问密钥。

## License

暂不自动添加开源许可证。课程报告、课件、第三方数据和代码可能分别受不同许可约束。确认权利归属后，再为原创代码和原创文字选择合适许可证。
