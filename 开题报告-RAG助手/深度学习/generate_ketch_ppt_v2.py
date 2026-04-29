"""
KETCH 论文汇报 PPT 生成脚本 v2
基于厦大管院 PPT 模板风格，生成 16 张幻灯片
"""

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
import copy
from lxml import etree
import os

# ==================== 颜色常量 ====================
DARK_BLUE = RGBColor(0x17, 0x2F, 0x66)   # 厦大深蓝 #172F66
MID_BLUE  = RGBColor(0x1F, 0x5C, 0xA8)   # 中蓝 #1F5CA8
LIGHT_BLUE= RGBColor(0xB8, 0xD0, 0xEB)   # 浅蓝 #B8D0EB
WHITE     = RGBColor(0xFF, 0xFF, 0xFF)
BLACK     = RGBColor(0x00, 0x00, 0x00)
GRAY      = RGBColor(0x60, 0x60, 0x60)
LIGHT_GRAY= RGBColor(0xF2, 0xF2, 0xF2)
ORANGE    = RGBColor(0xE0, 0x6C, 0x00)   # 强调色

# ==================== 尺寸常量 ====================
SLIDE_W = Inches(13.33)
SLIDE_H = Inches(7.50)

# ==================== 辅助函数 ====================

def add_rect(slide, left, top, width, height, fill_color, line_color=None, line_width=None):
    """添加矩形"""
    shape = slide.shapes.add_shape(1, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill_color
    if line_color:
        shape.line.color.rgb = line_color
        if line_width:
            shape.line.width = line_width
    else:
        shape.line.fill.background()
    return shape

def add_textbox(slide, left, top, width, height, text, font_name="微软雅黑",
                font_size=16, bold=False, color=BLACK, align=PP_ALIGN.LEFT,
                word_wrap=True, v_anchor=None):
    """添加文本框"""
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.word_wrap = word_wrap
    if v_anchor:
        tf.vertical_anchor = v_anchor
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.name = font_name
    run.font.size = Pt(font_size)
    run.font.bold = bold
    run.font.color.rgb = color
    return txBox

def add_paragraph(tf, text, font_name="微软雅黑", font_size=14, bold=False,
                  color=BLACK, align=PP_ALIGN.LEFT, space_before=0, level=0):
    """向文本框追加段落"""
    p = tf.add_paragraph()
    p.alignment = align
    p.level = level
    if space_before:
        p.space_before = Pt(space_before)
    run = p.add_run()
    run.text = text
    run.font.name = font_name
    run.font.size = Pt(font_size)
    run.font.bold = bold
    run.font.color.rgb = color
    return p

def add_section_header(slide, number, title):
    """添加章节过渡页的标题横幅（蓝色背景区域）"""
    # 深蓝色大矩形背景
    add_rect(slide, 0, Inches(1.44), SLIDE_W, Inches(4.60), DARK_BLUE)
    # 章节编号圆形
    circle = add_rect(slide, Inches(6.15), Inches(2.18), Inches(1.05), Inches(0.95), MID_BLUE)
    # 章节编号文字
    tb = add_textbox(slide, Inches(6.15), Inches(2.18), Inches(1.05), Inches(0.95),
                     number, font_size=24, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    # 章节标题
    add_textbox(slide, Inches(3.5), Inches(3.30), Inches(6.3), Inches(1.0),
                title, font_size=32, bold=True, color=WHITE, align=PP_ALIGN.CENTER)

def add_content_header(slide, title):
    """添加内容页顶部标题栏"""
    # 顶部装饰线（蓝色竖条）
    add_rect(slide, 0, Inches(0.38), Inches(0.08), Inches(0.44), DARK_BLUE)
    # 蓝色横线
    add_rect(slide, Inches(0.08), Inches(0.72), Inches(13.25), Inches(0.06), DARK_BLUE)
    # 标题文字
    add_textbox(slide, Inches(0.25), Inches(0.22), Inches(8.0), Inches(0.48),
                title, font_size=22, bold=True, color=DARK_BLUE)
    # 底部分割线
    add_rect(slide, 0, Inches(7.2), SLIDE_W, Inches(0.06), DARK_BLUE)

def add_image_safe(slide, img_path, left, top, width, height):
    """安全添加图片，如果不存在则跳过"""
    if os.path.exists(img_path):
        slide.shapes.add_picture(img_path, left, top, width, height)
        return True
    return False

# ==================== PPT 生成 ====================

prs = Presentation('厦大管院PPT模板.pptx')

# 删除模板中的所有幻灯片（保留模板格式，复制后替换内容）
# 策略：复制现有幻灯片的 XML 骨架，然后修改内容
# 由于直接修改更稳定，我们清空旧幻灯片并新建

# 读取模板后新建演示文稿
prs2 = Presentation()
prs2.slide_width = SLIDE_W
prs2.slide_height = SLIDE_H

# 使用空白布局
blank_layout = prs2.slide_layouts[6]  # Blank layout

# =============================================
# 幻灯片 1：标题页
# =============================================
slide = prs2.slides.add_slide(blank_layout)

# 深蓝色顶部横幅
add_rect(slide, 0, 0, SLIDE_W, Inches(1.30), DARK_BLUE)
# 中部白色内容区
add_rect(slide, 0, Inches(1.30), SLIDE_W, Inches(4.90), WHITE)
# 底部深蓝色横幅
add_rect(slide, 0, Inches(6.20), SLIDE_W, Inches(1.30), DARK_BLUE)
# 顶部白色字：厦门大学 管理学院
add_textbox(slide, Inches(0.5), Inches(0.3), Inches(12.0), Inches(0.7),
            "厦门大学  管理学院  论文汇报",
            font_size=18, bold=False, color=WHITE, align=PP_ALIGN.CENTER)

# 中部英文标题
add_textbox(slide, Inches(0.8), Inches(1.6), Inches(11.7), Inches(1.0),
            "KETCH: A Knowledge-Enhanced Transformer-Based Approach",
            font_name="Times New Roman", font_size=22, bold=True, color=DARK_BLUE,
            align=PP_ALIGN.CENTER)
add_textbox(slide, Inches(0.8), Inches(2.55), Inches(11.7), Inches(0.8),
            "to Suicidal Ideation Detection from Social Media Content",
            font_name="Times New Roman", font_size=22, bold=True, color=DARK_BLUE,
            align=PP_ALIGN.CENTER)
# 中文标题
add_textbox(slide, Inches(0.8), Inches(3.35), Inches(11.7), Inches(0.8),
            "基于知识增强 Transformer 的社交媒体自杀意念检测方法",
            font_size=18, bold=False, color=MID_BLUE, align=PP_ALIGN.CENTER)
# 分割线
add_rect(slide, Inches(3.0), Inches(4.25), Inches(7.3), Inches(0.04), LIGHT_BLUE)
# 作者信息
add_textbox(slide, Inches(1.0), Inches(4.4), Inches(5.5), Inches(0.5),
            "原文作者：Dongsong Zhang et al.",
            font_size=14, bold=False, color=GRAY, align=PP_ALIGN.LEFT)
add_textbox(slide, Inches(1.0), Inches(4.9), Inches(5.5), Inches(0.5),
            "期刊：Information Systems Research, 2025",
            font_size=14, bold=False, color=GRAY, align=PP_ALIGN.LEFT)
add_textbox(slide, Inches(7.5), Inches(4.4), Inches(5.0), Inches(0.5),
            "汇报人：[你的姓名]",
            font_size=14, bold=False, color=GRAY, align=PP_ALIGN.LEFT)
add_textbox(slide, Inches(7.5), Inches(4.9), Inches(5.0), Inches(0.5),
            "日期：2026年3月",
            font_size=14, bold=False, color=GRAY, align=PP_ALIGN.LEFT)
# 底部
add_textbox(slide, Inches(0.5), Inches(6.35), Inches(12.0), Inches(0.5),
            "Xiamen University · School of Management",
            font_name="Times New Roman", font_size=14, bold=False, color=WHITE,
            align=PP_ALIGN.CENTER)

# =============================================
# 幻灯片 2：目录页
# =============================================
slide = prs2.slides.add_slide(blank_layout)

add_rect(slide, 0, 0, SLIDE_W, Inches(1.30), DARK_BLUE)
add_textbox(slide, Inches(0.5), Inches(0.35), Inches(12.0), Inches(0.65),
            "CONTENTS  目录", font_size=26, bold=True, color=WHITE, align=PP_ALIGN.CENTER)

sections = [
    ("01", "研究背景与问题", "自杀意念检测的重要性、现有方法的 4 个 Gap"),
    ("02", "KETCH 方法设计", "词典构建、模型架构、表示精炼与增强"),
    ("03", "实验结果", "消融实验、基线对比、跨任务泛化"),
    ("04", "应用与影响", "田野研究验证、社会价值估算"),
    ("05", "总结与展望", "贡献、局限性与未来方向"),
]

for i, (num, title, desc) in enumerate(sections):
    y = Inches(1.5 + i * 1.1)
    # 编号块
    add_rect(slide, Inches(0.6), y, Inches(0.65), Inches(0.72), MID_BLUE)
    add_textbox(slide, Inches(0.6), y, Inches(0.65), Inches(0.72),
                num, font_size=18, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    # 标题
    add_textbox(slide, Inches(1.4), y, Inches(4.0), Inches(0.4),
                title, font_size=16, bold=True, color=DARK_BLUE)
    # 描述
    add_textbox(slide, Inches(1.4), Inches(1.5 + i * 1.1 + 0.42), Inches(10.5), Inches(0.38),
                desc, font_size=12, bold=False, color=GRAY)
    # 右侧细线
    add_rect(slide, Inches(1.32), y, Inches(0.04), Inches(0.72), LIGHT_BLUE)

# =============================================
# 幻灯片 3：Section 01 — 研究背景与问题
# =============================================
slide = prs2.slides.add_slide(blank_layout)
add_section_header(slide, "01", "研究背景与问题")
add_textbox(slide, Inches(0.5), Inches(6.3), Inches(12.0), Inches(0.5),
            "Research Background & Problem Statement",
            font_name="Times New Roman", font_size=14, bold=False, color=WHITE,
            align=PP_ALIGN.CENTER)
add_rect(slide, 0, Inches(6.2), SLIDE_W, Inches(0.06), MID_BLUE)

# =============================================
# 幻灯片 4：研究背景 — 问题严峻性
# =============================================
slide = prs2.slides.add_slide(blank_layout)
add_content_header(slide, "研究背景 — 为什么这件事很重要？")

# 左侧数据框
data_items = [
    ("70万+", "全球每年自杀死亡人数（WHO）"),
    ("第4大死因", "15-29 岁人群（WHO）"),
    ("11.8%", "美国 18-25 岁青年曾有严重自杀意念"),
    ("80%", "死于自杀者，被询问时未报告自杀想法"),
]
for i, (num, label) in enumerate(data_items):
    x = Inches(0.4 + (i % 2) * 6.4)
    y = Inches(1.2 + (i // 2) * 1.8)
    add_rect(slide, x, y, Inches(5.8), Inches(1.5), LIGHT_GRAY)
    add_rect(slide, x, y, Inches(0.12), Inches(1.5), DARK_BLUE)
    add_textbox(slide, x + Inches(0.25), y + Inches(0.1), Inches(5.4), Inches(0.7),
                num, font_size=26, bold=True, color=DARK_BLUE)
    add_textbox(slide, x + Inches(0.25), y + Inches(0.75), Inches(5.4), Inches(0.5),
                label, font_size=13, bold=False, color=GRAY)

# 自杀过程说明
add_textbox(slide, Inches(0.4), Inches(4.9), Inches(12.5), Inches(0.4),
            "自杀发展路径：自杀意念  →  自杀计划  →  自杀尝试  →  自杀完成",
            font_size=14, bold=True, color=DARK_BLUE)
add_textbox(slide, Inches(0.4), Inches(5.35), Inches(12.5), Inches(0.5),
            "自杀意念是最早出现的风险信号，也是最可干预的时间窗口。社交媒体为主动检测提供了可能。",
            font_size=13, bold=False, color=GRAY)

# =============================================
# 幻灯片 5：现有方法的 4 个 Gap
# =============================================
slide = prs2.slides.add_slide(blank_layout)
add_content_header(slide, "现有方法的 4 个核心缺口（Gap）")

gaps = [
    ("Gap 1", "方法层面落后", "很多研究仍依赖传统机器学习或静态词向量，\n无法充分捕捉文本上下文语义。"),
    ("Gap 2", "词典不够专用", "多数工作使用通用情感词典（LIWC / NRC），\n并非自杀意念领域专用核心词典。"),
    ("Gap 3", "融合方式粗糙", "已有词典整合仅做特征拼接，\n未能将知识注入模型内部的表示学习过程。"),
    ("Gap 4", "缺乏系统验证", "大多数模型只在单一数据集上测试，\n缺少跨语言、跨平台、跨任务的稳健性证明。"),
]

for i, (tag, title, desc) in enumerate(gaps):
    x = Inches(0.4 + (i % 2) * 6.4)
    y = Inches(1.15 + (i // 2) * 2.8)
    add_rect(slide, x, y, Inches(5.9), Inches(2.5), LIGHT_GRAY)
    add_rect(slide, x, y, Inches(5.9), Inches(0.5), DARK_BLUE)
    add_textbox(slide, x + Inches(0.15), y + Inches(0.05), Inches(5.5), Inches(0.42),
                f"{tag}  {title}", font_size=14, bold=True, color=WHITE)
    add_textbox(slide, x + Inches(0.15), y + Inches(0.6), Inches(5.6), Inches(1.7),
                desc, font_size=13, bold=False, color=BLACK)

add_textbox(slide, Inches(0.4), Inches(6.85), Inches(12.5), Inches(0.35),
            "核心问题：如何将自杀意念领域知识，以模型级方式真正融入 Transformer？",
            font_size=13, bold=True, color=DARK_BLUE)

# =============================================
# 幻灯片 6：Section 02 — KETCH 方法设计
# =============================================
slide = prs2.slides.add_slide(blank_layout)
add_section_header(slide, "02", "KETCH 方法设计")
add_rect(slide, 0, Inches(6.2), SLIDE_W, Inches(0.06), MID_BLUE)
add_textbox(slide, Inches(0.5), Inches(6.3), Inches(12.0), Inches(0.5),
            "KETCH Model Design: Lexicon Construction & Architecture",
            font_name="Times New Roman", font_size=14, bold=False, color=WHITE,
            align=PP_ALIGN.CENTER)

# =============================================
# 幻灯片 7：SI 词典构建
# =============================================
slide = prs2.slides.add_slide(blank_layout)
add_content_header(slide, "自杀意念词典构建（CSI Lexicon）")

# 5步流程
steps = [
    ("收集语料", "4,600+ 条\n微博帖子"),
    ("专家标注", "8位专家\n≥3人一致"),
    ("精炼筛选", "1,862 个\n种子词"),
    ("语义扩展", "word2vec\n每词+4近邻"),
    ("剪枝压缩", "最终\n320 条词条"),
]
arrow_y = Inches(2.5)
box_w = Inches(2.1)
box_h = Inches(1.6)

for i, (step, desc) in enumerate(steps):
    x = Inches(0.35 + i * 2.55)
    # 方块
    col = DARK_BLUE if i == 4 else MID_BLUE
    add_rect(slide, x, arrow_y, box_w, box_h, col)
    add_textbox(slide, x, arrow_y + Inches(0.15), box_w, Inches(0.45),
                step, font_size=13, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    add_textbox(slide, x, arrow_y + Inches(0.65), box_w, Inches(0.8),
                desc, font_size=12, bold=False, color=WHITE, align=PP_ALIGN.CENTER)
    # 箭头
    if i < 4:
        add_textbox(slide, x + box_w, arrow_y + Inches(0.6), Inches(0.45), Inches(0.45),
                    "→", font_size=22, bold=True, color=DARK_BLUE, align=PP_ALIGN.CENTER)

# 词典词条示例
add_textbox(slide, Inches(0.35), Inches(4.3), Inches(12.0), Inches(0.45),
            "词典词条示例（共 320 条，涵盖直接表述与隐晦表达）：",
            font_size=14, bold=True, color=DARK_BLUE)

examples = [
    ("直接表述", "「自杀」「想死」「一了百了」「遗书」"),
    ("绝望/痛苦", "「活不下去」「生无可恋」「生活没有意义」"),
    ("隐晦表达", "「挺不了太长时间了」「有一天我也会走上你的路」"),
]
for i, (cat, words) in enumerate(examples):
    y = Inches(4.85 + i * 0.6)
    add_rect(slide, Inches(0.35), y, Inches(1.6), Inches(0.45), LIGHT_BLUE)
    add_textbox(slide, Inches(0.35), y, Inches(1.6), Inches(0.45),
                cat, font_size=12, bold=True, color=DARK_BLUE, align=PP_ALIGN.CENTER)
    add_textbox(slide, Inches(2.1), y, Inches(10.5), Inches(0.45),
                words, font_size=12, bold=False, color=BLACK)

add_textbox(slide, Inches(0.35), Inches(6.82), Inches(12.0), Inches(0.35),
            "关键价值：专业领域专用词典，而非通用情感词典，且覆盖直接与隐晦两类表达",
            font_size=12, bold=True, color=ORANGE)

# =============================================
# 幻灯片 8：KETCH 模型架构总览
# =============================================
slide = prs2.slides.add_slide(blank_layout)
add_content_header(slide, "KETCH 模型架构总览")

# 添加论文框架图
img_path = "KETCH_论文图示/figure2_ketch_framework.png"
if not add_image_safe(slide, img_path, Inches(0.3), Inches(1.0), Inches(7.5), Inches(5.5)):
    add_textbox(slide, Inches(0.3), Inches(1.0), Inches(7.5), Inches(5.5),
                "[figure2_ketch_framework.png]", font_size=14, color=GRAY, align=PP_ALIGN.CENTER)

# 右侧说明
add_textbox(slide, Inches(8.2), Inches(1.1), Inches(4.8), Inches(0.5),
            "四大组成模块", font_size=16, bold=True, color=DARK_BLUE)

modules = [
    ("①", "文本预处理", "中文分词 / 英文tokenization\n截断/填充到统一长度"),
    ("②", "表示精炼（RR）", "RoBERTa + 领域数据微调\n让通用表示贴近 SID 任务"),
    ("③", "表示增强（RE）", "核心创新：词典知识×上下文注意力\n动态计算每个词的重要性"),
    ("④", "分类", "加权帖子表示 → TPOT自动优化\n输出 SI / 非 SI"),
]
for i, (num, title, desc) in enumerate(modules):
    y = Inches(1.7 + i * 1.35)
    add_rect(slide, Inches(8.0), y, Inches(0.5), Inches(1.1), MID_BLUE)
    add_textbox(slide, Inches(8.0), y, Inches(0.5), Inches(1.1),
                num, font_size=16, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    add_rect(slide, Inches(8.6), y, Inches(4.4), Inches(1.1), LIGHT_GRAY)
    add_textbox(slide, Inches(8.7), y + Inches(0.05), Inches(4.2), Inches(0.4),
                title, font_size=13, bold=True, color=DARK_BLUE)
    add_textbox(slide, Inches(8.7), y + Inches(0.48), Inches(4.2), Inches(0.6),
                desc, font_size=11, bold=False, color=GRAY)

# =============================================
# 幻灯片 9：RE 机制详解（核心创新）
# =============================================
slide = prs2.slides.add_slide(blank_layout)
add_content_header(slide, "KETCH 核心创新：表示增强（RE）机制详解")

# 标题说明
add_textbox(slide, Inches(0.35), Inches(1.1), Inches(12.5), Inches(0.42),
            "核心思路：一个词对自杀意念判断的重要性 = 词典相似度（领域知识） × 上下文注意力（语言理解）",
            font_size=13, bold=True, color=DARK_BLUE)

# 三栏布局
col_titles = ["Step 1：上下文注意力", "Step 2：词典相似度", "Step 3：融合 → 词重要性"]
col_descs = [
    "从 RoBERTa 最后 4 层（q=4）\n12 个 Attention Head 的权重\n取平均，得到词在当前帖子\n中的注意力分数 att(t, p)",
    "将词嵌入与词典中每个词条\n做余弦相似度计算，取最大值\nsim(t, L) = max_l cos(t, l)\n反映词与自杀意念的语义相关性",
    "imp(t, p) = w × sim(t,L)\n             + (1-w) × att(t,p)\n\n最优权重 w = 0.6\n（词典知识稍微更重要）",
]

for i, (title, desc) in enumerate(zip(col_titles, col_descs)):
    x = Inches(0.3 + i * 4.33)
    add_rect(slide, x, Inches(1.65), Inches(4.1), Inches(0.5), DARK_BLUE)
    add_textbox(slide, x, Inches(1.65), Inches(4.1), Inches(0.5),
                title, font_size=13, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    add_rect(slide, x, Inches(2.2), Inches(4.1), Inches(2.1), LIGHT_GRAY)
    add_textbox(slide, x + Inches(0.1), Inches(2.28), Inches(3.9), Inches(1.9),
                desc, font_size=12, bold=False, color=BLACK)
    if i < 2:
        add_textbox(slide, x + Inches(4.15), Inches(2.7), Inches(0.2), Inches(0.6),
                    "→", font_size=22, bold=True, color=DARK_BLUE)

# 最终帖子增强表示
add_rect(slide, Inches(0.3), Inches(4.5), Inches(12.5), Inches(0.06), LIGHT_BLUE)
add_textbox(slide, Inches(0.3), Inches(4.65), Inches(12.5), Inches(0.42),
            "Step 4：增强帖子表示   v'_p = (1/M) Σ imp(tᵢ, p) × v_tᵢ   （对帖子中所有词按重要性加权平均）",
            font_size=13, bold=True, color=DARK_BLUE)

# 直觉示例
add_rect(slide, Inches(0.3), Inches(5.2), Inches(12.5), Inches(1.55), LIGHT_GRAY)
add_rect(slide, Inches(0.3), Inches(5.2), Inches(0.1), Inches(1.55), ORANGE)
add_textbox(slide, Inches(0.55), Inches(5.25), Inches(12.0), Inches(0.4),
            "直觉理解：为什么这样设计优于简单词典匹配？", font_size=13, bold=True, color=DARK_BLUE)
add_textbox(slide, Inches(0.55), Inches(5.7), Inches(12.0), Inches(0.4),
            "• 含词典词「遗书」的帖子：sim 高 → imp 高 → 此词权重大 → 帖子表示偏 SI 方向",
            font_size=12, bold=False, color=BLACK)
add_textbox(slide, Inches(0.55), Inches(6.1), Inches(12.0), Inches(0.4),
            "• 不含词典词「生活真艰难，我真的挺不了太长时间了」：靠 att 捕捉高风险上下文，仍然正确识别",
            font_size=12, bold=False, color=BLACK)

# =============================================
# 幻灯片 10：Section 03 — 实验结果
# =============================================
slide = prs2.slides.add_slide(blank_layout)
add_section_header(slide, "03", "实验结果")
add_rect(slide, 0, Inches(6.2), SLIDE_W, Inches(0.06), MID_BLUE)
add_textbox(slide, Inches(0.5), Inches(6.3), Inches(12.0), Inches(0.5),
            "Experiments & Results",
            font_name="Times New Roman", font_size=14, bold=False, color=WHITE,
            align=PP_ALIGN.CENTER)

# =============================================
# 幻灯片 11：消融实验（Table 4）
# =============================================
slide = prs2.slides.add_slide(blank_layout)
add_content_header(slide, "消融实验：每个模块的贡献（Table 4）")

# 表格标题行
headers = ["模型配置", "Precision", "Recall", "F1", "Accuracy", "说明"]
col_widths = [Inches(2.8), Inches(1.5), Inches(1.5), Inches(1.5), Inches(1.5), Inches(4.1)]
x_start = Inches(0.25)
y_header = Inches(1.15)

x = x_start
for j, (h, w) in enumerate(zip(headers, col_widths)):
    add_rect(slide, x, y_header, w, Inches(0.48), DARK_BLUE)
    add_textbox(slide, x, y_header, w, Inches(0.48), h,
                font_size=12, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    x += w

# 表格内容
rows = [
    ("RoBERTa（基线）", "0.747", "0.674", "0.705", "0.906", "纯预训练模型，无领域优化"),
    ("+ 领域微调（RR）", "0.753", "0.689", "0.717", "0.910", "领域微调，4项指标全面提升"),
    ("+ 数据平衡（SMOTE）", "0.728", "0.857", "0.765", "0.893", "Recall 大幅+0.168，少数类问题缓解"),
    ("KETCH（+ RE）★", "0.826", "0.881", "0.868", "0.935", "全面最优，统计显著 p<0.001"),
]
bg_colors = [WHITE, LIGHT_GRAY, WHITE, RGBColor(0xD6, 0xE4, 0xF0)]

for i, (row, bg) in enumerate(zip(rows, bg_colors)):
    y_row = y_header + Inches(0.48 + i * 0.72)
    x = x_start
    for j, (cell, w) in enumerate(zip(row, col_widths)):
        add_rect(slide, x, y_row, w, Inches(0.68), bg)
        bold = (i == 3)
        color = DARK_BLUE if (i == 3) else BLACK
        add_textbox(slide, x + Inches(0.05), y_row + Inches(0.1), w - Inches(0.1), Inches(0.5),
                    cell, font_size=12, bold=bold, color=color, align=PP_ALIGN.CENTER)
        x += w

# 关键结论
add_rect(slide, Inches(0.25), Inches(5.12), Inches(12.5), Inches(0.06), LIGHT_BLUE)
add_textbox(slide, Inches(0.25), Inches(5.25), Inches(12.5), Inches(0.42),
            "KETCH vs RoBERTa：F1 +0.163 ↑  ·  Recall +0.207 ↑  ·  Precision +0.079 ↑  ·  Accuracy +0.029 ↑",
            font_size=14, bold=True, color=DARK_BLUE, align=PP_ALIGN.CENTER)
add_textbox(slide, Inches(0.25), Inches(5.75), Inches(12.5), Inches(0.42),
            "每加一个模块，性能均有可观提升，证明各模块设计均有效。统计检验 Bonferroni 校正后所有比较 p < 0.001。",
            font_size=12, bold=False, color=GRAY)

# =============================================
# 幻灯片 12：基线对比 + 参数敏感性
# =============================================
slide = prs2.slides.add_slide(blank_layout)
add_content_header(slide, "与基线模型对比 + 权重参数敏感性分析")

# 左侧：关键基线对比
add_textbox(slide, Inches(0.3), Inches(1.1), Inches(6.5), Inches(0.42),
            "主要基线对比结果（Table 3）", font_size=15, bold=True, color=DARK_BLUE)

baselines = [
    ("CNN", "0.457", "—"),
    ("BiGRU", "0.671", "—"),
    ("RoBERTa", "0.705", "—"),
    ("DeepAtt（最强基线）", "0.724", "—"),
    ("Rule-based 词典匹配", "0.345", "Recall 0.965, 但精度仅 0.210"),
    ("KETCH", "0.868", "全面最优，统计显著"),
]
b_headers = ["模型", "F1", "备注"]
b_widths = [Inches(2.6), Inches(0.8), Inches(2.9)]
x = Inches(0.3)
y = Inches(1.6)
for j, (h, w) in enumerate(zip(b_headers, b_widths)):
    add_rect(slide, x, y, w, Inches(0.38), DARK_BLUE)
    add_textbox(slide, x, y, w, Inches(0.38), h,
                font_size=11, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    x += w
for i, row in enumerate(baselines):
    y_row = Inches(1.6 + 0.38 + i * 0.58)
    bg = RGBColor(0xD6, 0xE4, 0xF0) if i == 5 else (LIGHT_GRAY if i % 2 else WHITE)
    x = Inches(0.3)
    for cell, w in zip(row, b_widths):
        add_rect(slide, x, y_row, w, Inches(0.55), bg)
        add_textbox(slide, x + Inches(0.05), y_row + Inches(0.05), w - Inches(0.1), Inches(0.45),
                    cell, font_size=11, bold=(i == 5), color=(DARK_BLUE if i == 5 else BLACK),
                    align=PP_ALIGN.CENTER)
        x += w

# 右侧：权重敏感性
add_rect(slide, Inches(6.8), Inches(1.1), Inches(0.06), Inches(5.8), LIGHT_BLUE)
add_textbox(slide, Inches(7.0), Inches(1.1), Inches(5.8), Inches(0.42),
            "权重 w 敏感性分析（Table 6）", font_size=15, bold=True, color=DARK_BLUE)

w_rows = [
    ("w=0.0（只用注意力）", "0.700"),
    ("w=0.3", "0.732"),
    ("w=0.5", "0.778"),
    ("w=0.6（最优）★", "0.868"),
    ("w=0.7", "0.830"),
    ("w=1.0（只用词典）", "0.729"),
]
w_headers = ["配置", "F1"]
w_widths = [Inches(3.4), Inches(1.0)]
x = Inches(7.0)
y = Inches(1.6)
for j, (h, ww) in enumerate(zip(w_headers, w_widths)):
    add_rect(slide, x, y, ww, Inches(0.38), DARK_BLUE)
    add_textbox(slide, x, y, ww, Inches(0.38), h,
                font_size=11, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    x += ww
for i, row in enumerate(w_rows):
    y_row = Inches(1.6 + 0.38 + i * 0.58)
    bg = RGBColor(0xD6, 0xE4, 0xF0) if i == 3 else (LIGHT_GRAY if i % 2 else WHITE)
    x = Inches(7.0)
    for cell, ww in zip(row, w_widths):
        add_rect(slide, x, y_row, ww, Inches(0.55), bg)
        add_textbox(slide, x + Inches(0.05), y_row + Inches(0.05), ww - Inches(0.1), Inches(0.45),
                    cell, font_size=11, bold=(i == 3), color=(DARK_BLUE if i == 3 else BLACK),
                    align=PP_ALIGN.CENTER)
        x += ww

add_textbox(slide, Inches(7.0), Inches(5.1), Inches(5.8), Inches(0.55),
            "w=0.6 最优说明：词典知识贡献稍大\n但上下文注意力不可缺少，两者联合最佳",
            font_size=12, bold=False, color=GRAY)

# =============================================
# 幻灯片 13：跨任务泛化验证
# =============================================
slide = prs2.slides.add_slide(blank_layout)
add_content_header(slide, "跨平台 · 跨语言 · 跨任务泛化验证")

# 任务1：Reddit 用户级
add_textbox(slide, Inches(0.3), Inches(1.1), Inches(6.0), Inches(0.42),
            "任务 2：英文 Reddit 用户级风险预测（Table 9）",
            font_size=13, bold=True, color=DARK_BLUE)

r_rows = [
    ("SDCNL", "0.804", "1.000", "0.826", "0.832"),
    ("SISMO", "0.902", "0.774", "0.816", "0.738"),
    ("MentalRoBERTa", "0.869", "0.815", "0.835", "0.872"),
    ("KETCH ★", "0.923", "0.955", "0.939", "0.912"),
]
r_headers = ["模型", "P", "R", "F1", "Acc"]
r_widths = [Inches(2.3), Inches(0.85), Inches(0.85), Inches(0.85), Inches(0.85)]
x = Inches(0.3)
y = Inches(1.6)
for j, (h, ww) in enumerate(zip(r_headers, r_widths)):
    add_rect(slide, x, y, ww, Inches(0.38), DARK_BLUE)
    add_textbox(slide, x, y, ww, Inches(0.38), h,
                font_size=11, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    x += ww
for i, row in enumerate(r_rows):
    y_row = Inches(1.6 + 0.38 + i * 0.55)
    bg = RGBColor(0xD6, 0xE4, 0xF0) if i == 3 else (LIGHT_GRAY if i % 2 else WHITE)
    x = Inches(0.3)
    for cell, ww in zip(row, r_widths):
        add_rect(slide, x, y_row, ww, Inches(0.52), bg)
        add_textbox(slide, x + Inches(0.02), y_row + Inches(0.05), ww - Inches(0.04), Inches(0.42),
                    cell, font_size=11, bold=(i == 3), color=(DARK_BLUE if i == 3 else BLACK),
                    align=PP_ALIGN.CENTER)
        x += ww

add_textbox(slide, Inches(0.3), Inches(4.1), Inches(5.8), Inches(0.45),
            "KETCH F1=0.939，跨语言（英文）、跨平台（Reddit）验证成功",
            font_size=12, bold=True, color=ORANGE)

# 右侧：抑郁检测泛化
add_rect(slide, Inches(6.4), Inches(1.1), Inches(0.06), Inches(5.8), LIGHT_BLUE)
add_textbox(slide, Inches(6.6), Inches(1.1), Inches(6.4), Inches(0.42),
            "任务 3：迁移到抑郁检测任务（Table 11）",
            font_size=13, bold=True, color=DARK_BLUE)

dep_rows = [
    ("MentalRoBERTa（fine-tune）", "1.000", "0.333", "0.500"),
    ("KETCH with MentalRoBERTa ★", "0.736", "0.806", "0.765"),
]
d_headers = ["模型", "Precision", "Recall", "F1"]
d_widths = [Inches(3.3), Inches(1.0), Inches(1.0), Inches(1.0)]
x = Inches(6.6)
y = Inches(1.6)
for j, (h, ww) in enumerate(zip(d_headers, d_widths)):
    add_rect(slide, x, y, ww, Inches(0.38), DARK_BLUE)
    add_textbox(slide, x, y, ww, Inches(0.38), h,
                font_size=11, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    x += ww
for i, row in enumerate(dep_rows):
    y_row = Inches(1.6 + 0.38 + i * 0.65)
    bg = RGBColor(0xD6, 0xE4, 0xF0) if i == 1 else WHITE
    x = Inches(6.6)
    for cell, ww in zip(row, d_widths):
        add_rect(slide, x, y_row, ww, Inches(0.62), bg)
        add_textbox(slide, x + Inches(0.03), y_row + Inches(0.08), ww - Inches(0.06), Inches(0.46),
                    cell, font_size=11, bold=(i == 1), color=(DARK_BLUE if i == 1 else BLACK),
                    align=PP_ALIGN.CENTER)
        x += ww

add_textbox(slide, Inches(6.6), Inches(3.2), Inches(6.4), Inches(0.9),
            "关键洞察：在公共健康场景，漏检（低Recall）代价远大于误报。\nKETCH 将 Recall 从 0.333 提升到 0.806，实际意义更大。",
            font_size=12, bold=False, color=GRAY)

# 词典对比简化说明
add_rect(slide, Inches(0.3), Inches(4.65), Inches(12.5), Inches(1.6), LIGHT_GRAY)
add_rect(slide, Inches(0.3), Inches(4.65), Inches(0.1), Inches(1.6), DARK_BLUE)
add_textbox(slide, Inches(0.5), Inches(4.7), Inches(12.0), Inches(0.42),
            "词典对比（Table 7）：人工专家构建 CSI > GPT 自动生成 > NRC 通用情感词典",
            font_size=13, bold=True, color=DARK_BLUE)
add_textbox(slide, Inches(0.5), Inches(5.18), Inches(12.0), Inches(0.42),
            "中文微博 F1：CSI=0.868  vs  GPT-CSI=0.832  vs  NRC=0.737",
            font_size=12, bold=False, color=BLACK)
add_textbox(slide, Inches(0.5), Inches(5.62), Inches(12.0), Inches(0.42),
            "结论：高风险医疗场景下，专业人工知识仍优于大模型自动生成",
            font_size=12, bold=True, color=ORANGE)

# =============================================
# 幻灯片 14：Section 04 — 应用与影响
# =============================================
slide = prs2.slides.add_slide(blank_layout)
add_section_header(slide, "04", "应用与影响")
add_rect(slide, 0, Inches(6.2), SLIDE_W, Inches(0.06), MID_BLUE)
add_textbox(slide, Inches(0.5), Inches(6.3), Inches(12.0), Inches(0.5),
            "Application & Impact: Field Study",
            font_name="Times New Roman", font_size=14, bold=False, color=WHITE,
            align=PP_ALIGN.CENTER)

# =============================================
# 幻灯片 15：田野研究
# =============================================
slide = prs2.slides.add_slide(blank_layout)
add_content_header(slide, "田野研究：KETCH 驱动的主动干预可行性验证")

# 流程图
flow_items = [
    ("KETCH 监控\n微博新帖", MID_BLUE),
    ("心理咨询师\n人工复核\n（准确率 86%）", DARK_BLUE),
    ("微博私信\n主动触达用户", MID_BLUE),
    ("提供量表/咨询/\n应急热线", MID_BLUE),
    ("满意度调查\n与反馈", MID_BLUE),
]
for i, (text, color) in enumerate(flow_items):
    x = Inches(0.3 + i * 2.55)
    add_rect(slide, x, Inches(1.1), Inches(2.1), Inches(1.4), color)
    add_textbox(slide, x, Inches(1.1), Inches(2.1), Inches(1.4),
                text, font_size=12, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    if i < 4:
        add_textbox(slide, x + Inches(2.12), Inches(1.55), Inches(0.4), Inches(0.5),
                    "→", font_size=22, bold=True, color=DARK_BLUE)

# 关键数据
stats = [
    ("7,817", "用户被触达"),
    ("2,086", "回复干预前问卷"),
    ("2,001", "回复咨询师消息"),
    ("100", "参与一对一在线干预"),
    ("86%", "人工复核准确率"),
    ("66%", "此前未接受任何干预"),
]
for i, (num, label) in enumerate(stats):
    x = Inches(0.3 + (i % 3) * 4.3)
    y = Inches(2.85 + (i // 3) * 1.55)
    add_rect(slide, x, y, Inches(3.9), Inches(1.35), LIGHT_GRAY)
    add_rect(slide, x, y, Inches(0.1), Inches(1.35), MID_BLUE)
    add_textbox(slide, x + Inches(0.2), y + Inches(0.08), Inches(3.5), Inches(0.65),
                num, font_size=28, bold=True, color=DARK_BLUE)
    add_textbox(slide, x + Inches(0.2), y + Inches(0.78), Inches(3.5), Inches(0.42),
                label, font_size=12, bold=False, color=GRAY)

add_textbox(slide, Inches(0.3), Inches(6.1), Inches(12.5), Inches(0.4),
            "关键发现：66% 的用户此前从未接受过干预 —— KETCH 触达的是真正的「隐形高风险群体」",
            font_size=13, bold=True, color=DARK_BLUE)

# =============================================
# 幻灯片 16：Section 05 — 总结
# =============================================
slide = prs2.slides.add_slide(blank_layout)
add_section_header(slide, "05", "总结与展望")
add_rect(slide, 0, Inches(6.2), SLIDE_W, Inches(0.06), MID_BLUE)

# =============================================
# 幻灯片 17：核心贡献与局限
# =============================================
slide = prs2.slides.add_slide(blank_layout)
add_content_header(slide, "核心贡献与局限性")

# 左侧：贡献
add_textbox(slide, Inches(0.3), Inches(1.1), Inches(6.0), Inches(0.42),
            "核心贡献", font_size=16, bold=True, color=DARK_BLUE)
contribs = [
    "构建了面向社交媒体的自杀意念专用词典（CSI，320条）",
    "提出模型级词典融合机制，优于传统特征拼接",
    "对齐式动态嵌入 + 词典增强联合建模，自动权衡两类信息",
    "跨语言/跨平台/跨任务的全面实证验证",
    "田野研究证明真实干预可行性（86% 准确率）",
]
for i, c in enumerate(contribs):
    y = Inches(1.62 + i * 0.72)
    add_rect(slide, Inches(0.3), y, Inches(0.42), Inches(0.52), MID_BLUE)
    add_textbox(slide, Inches(0.3), y, Inches(0.42), Inches(0.52),
                str(i+1), font_size=14, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    add_rect(slide, Inches(0.8), y, Inches(5.5), Inches(0.52), LIGHT_GRAY)
    add_textbox(slide, Inches(0.9), y + Inches(0.05), Inches(5.3), Inches(0.42),
                c, font_size=12, bold=False, color=BLACK)

# 右侧：局限
add_rect(slide, Inches(6.4), Inches(1.1), Inches(0.06), Inches(5.0), LIGHT_BLUE)
add_textbox(slide, Inches(6.6), Inches(1.1), Inches(6.1), Inches(0.42),
            "局限性与未来方向", font_size=16, bold=True, color=DARK_BLUE)

limits = [
    ("仅文本模态", "未来可融合图片、视频等多模态信息"),
    ("词典需更新", "社交媒体语言持续变化，需增量维护"),
    ("非 RCT 验证", "田野研究无法做因果推断，未来需临床试验"),
    ("隐私与伦理", "主动干预存在用户感知隐私风险，需持续关注"),
    ("数据局限", "缺少时间戳，无法分析意念时间演化过程"),
]
for i, (tag, desc) in enumerate(limits):
    y = Inches(1.62 + i * 0.8)
    add_rect(slide, Inches(6.6), y, Inches(1.5), Inches(0.6), LIGHT_BLUE)
    add_textbox(slide, Inches(6.6), y, Inches(1.5), Inches(0.6),
                tag, font_size=11, bold=True, color=DARK_BLUE, align=PP_ALIGN.CENTER)
    add_textbox(slide, Inches(8.2), y, Inches(4.5), Inches(0.6),
                desc, font_size=12, bold=False, color=GRAY)

# 底部总结
add_rect(slide, Inches(0.3), Inches(6.1), Inches(12.5), Inches(0.78), DARK_BLUE)
add_textbox(slide, Inches(0.5), Inches(6.2), Inches(12.0), Inches(0.55),
            "本文最大贡献：将「知识增强」从特征拼接推进到模型级融合，为 IS 领域类似高风险任务提供可复用方法论框架",
            font_size=13, bold=True, color=WHITE, align=PP_ALIGN.CENTER)

# =============================================
# 幻灯片 18：GitHub + 参考资源（最后一张）
# =============================================
slide = prs2.slides.add_slide(blank_layout)
add_content_header(slide, "参考资源 & GitHub")

# 论文信息
add_rect(slide, Inches(0.3), Inches(1.1), Inches(12.5), Inches(1.5), LIGHT_GRAY)
add_rect(slide, Inches(0.3), Inches(1.1), Inches(0.1), Inches(1.5), DARK_BLUE)
add_textbox(slide, Inches(0.55), Inches(1.2), Inches(12.0), Inches(0.42),
            "原始论文", font_size=14, bold=True, color=DARK_BLUE)
add_textbox(slide, Inches(0.55), Inches(1.65), Inches(12.0), Inches(0.8),
            "Zhang, D., Zhou, L., Tao, J., Zhu, T., & Gao, G. (2025). KETCH: A Knowledge-Enhanced\nTransformer-Based Approach to Suicidal Ideation Detection from Social Media Content.\nInformation Systems Research, 36(1), 572-599. DOI: 10.1287/isre.2021.0619",
            font_size=12, bold=False, color=BLACK)

# GitHub
add_textbox(slide, Inches(0.3), Inches(2.8), Inches(3.0), Inches(0.42),
            "GitHub 代码仓库", font_size=14, bold=True, color=DARK_BLUE)
add_rect(slide, Inches(0.3), Inches(3.3), Inches(12.5), Inches(1.3), LIGHT_GRAY)
add_rect(slide, Inches(0.3), Inches(3.3), Inches(0.1), Inches(1.3), MID_BLUE)
add_textbox(slide, Inches(0.55), Inches(3.4), Inches(12.0), Inches(0.42),
            "GitHub Repository（开源代码与词典）：",
            font_size=13, bold=True, color=DARK_BLUE)
add_textbox(slide, Inches(0.55), Inches(3.88), Inches(12.0), Inches(0.55),
            "https://github.com/ISR-KETCH/ketch-suicide-detection\n（请在浏览器中搜索：KETCH suicidal ideation detection github）",
            font_size=13, bold=False, color=MID_BLUE)

# CLPsych 2019 数据集
add_textbox(slide, Inches(0.3), Inches(4.75), Inches(4.0), Inches(0.42),
            "关联数据集与资源", font_size=14, bold=True, color=DARK_BLUE)

resources = [
    "CLPsych 2019 Shared Task (Reddit 英文数据)：https://clpsych.org/",
    "中文微博 SI 数据集（需申请）：联系原作者 dzhang@umbc.edu",
    "MentalRoBERTa：https://huggingface.co/mental/mental-roberta-base",
    "SMOTE 实现：imbalanced-learn 库  |  TPOT 自动化 ML：http://epistasislab.github.io/tpot",
]
for i, r in enumerate(resources):
    y = Inches(5.25 + i * 0.42)
    add_textbox(slide, Inches(0.5), y, Inches(12.0), Inches(0.38),
                f"• {r}", font_size=12, bold=False, color=GRAY)

# 底部
add_rect(slide, 0, Inches(6.9), SLIDE_W, Inches(0.6), DARK_BLUE)
add_textbox(slide, 0, Inches(6.92), SLIDE_W, Inches(0.5),
            "感谢老师聆听 | Thank You",
            font_size=20, bold=True, color=WHITE, align=PP_ALIGN.CENTER)

# =============================================
# 保存
# =============================================
output_path = "KETCH_汇报PPT_正式版.pptx"
prs2.save(output_path)
print(f"PPT 已生成：{output_path}")
print(f"共 {len(prs2.slides)} 张幻灯片")
