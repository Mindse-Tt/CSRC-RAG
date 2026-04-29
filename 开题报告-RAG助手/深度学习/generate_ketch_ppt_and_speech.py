from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_AUTO_SHAPE_TYPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Cm as PCm
from pptx.util import Pt as PPt


BASE = Path("/Users/mindset/Desktop/课程日常/深度学习")
FIG_DIR = BASE / "KETCH_论文图示"
PPT_OUT = BASE / "KETCH_论文汇报PPT.pptx"
DOC_OUT = BASE / "KETCH_五分钟发言稿.docx"


COLORS = {
    "navy": RGBColor(22, 47, 88),
    "blue": RGBColor(43, 99, 184),
    "light_blue": RGBColor(231, 239, 251),
    "teal": RGBColor(38, 136, 130),
    "light_teal": RGBColor(230, 245, 242),
    "orange": RGBColor(208, 120, 55),
    "light_orange": RGBColor(250, 239, 227),
    "gray": RGBColor(84, 95, 112),
    "light_gray": RGBColor(243, 245, 248),
    "dark": RGBColor(38, 43, 51),
    "white": RGBColor(255, 255, 255),
}


def set_docx_font(run, east_asia="宋体", latin="Times New Roman", size=None, bold=None, italic=None):
    r = run._element
    r_pr = r.get_or_add_rPr()
    r_fonts = r_pr.rFonts
    if r_fonts is None:
        r_fonts = OxmlElement("w:rFonts")
        r_pr.append(r_fonts)
    r_fonts.set(qn("w:eastAsia"), east_asia)
    r_fonts.set(qn("w:ascii"), latin)
    r_fonts.set(qn("w:hAnsi"), latin)
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if italic is not None:
        run.italic = italic


def add_doc_paragraph(doc, text, style=None, size=12, bold=False, italic=False, align=None):
    p = doc.add_paragraph(style=style)
    if align is not None:
        p.alignment = align
    run = p.add_run(text)
    set_docx_font(run, size=size, bold=bold, italic=italic)
    return p


def build_speech_doc():
    doc = Document()
    section = doc.sections[0]
    section.top_margin = Cm(2.3)
    section.bottom_margin = Cm(2.3)
    section.left_margin = Cm(2.5)
    section.right_margin = Cm(2.5)

    styles = doc.styles
    styles["Normal"].font.name = "宋体"
    styles["Normal"].font.size = Pt(12)
    styles["Normal"]._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
    for style_name in ["Title", "Heading 1", "Heading 2"]:
        styles[style_name].font.name = "黑体"
        styles[style_name]._element.rPr.rFonts.set(qn("w:eastAsia"), "黑体")

    add_doc_paragraph(doc, "KETCH 论文五分钟发言稿", style="Title", size=18, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER)
    add_doc_paragraph(
        doc,
        "论文：KETCH: A Knowledge-Enhanced Transformer-Based Approach to Suicidal Ideation Detection from Social Media Content",
        size=10.5,
        italic=True,
        align=WD_ALIGN_PARAGRAPH.CENTER,
    )
    add_doc_paragraph(
        doc,
        "用途：课堂汇报 / 五分钟口头发言 / 面向非专业听众",
        size=10.5,
        align=WD_ALIGN_PARAGRAPH.CENTER,
    )

    add_doc_paragraph(doc, "一、五分钟正式发言稿", style="Heading 1", size=15, bold=True)

    speech = [
        "大家好，今天我想介绍一篇关于社交媒体自杀意念检测的论文，名字叫 KETCH。",
        "如果用一句很白话的话概括，这篇文章想做的事情是：让电脑更早地从社交媒体文字里，看出一个人是不是可能出现了自杀意念，从而尽早提醒和干预。",
        "这个问题为什么重要？因为很多真正有危险的人，并不会主动去找医生、心理咨询师或者家人求助。但他们可能会在微博、Reddit 这样的平台上发一些很痛苦、很绝望的话。所以作者认为，社交媒体可以成为一个早期预警窗口。",
        "但只靠人工去盯这些帖子并不现实，因为帖子数量太大、更新太快。所以这篇论文就在做一件事：设计一个更聪明的 AI 模型，自动识别哪些帖子可能在表达自杀想法。",
        "作者认为，以前的方法有两个常见问题。第一，很多模型只会看字面。比如一个帖子里出现“死”“结束”“离开”这些词，它就容易判成高风险，但现实里不是所有带这些词的句子都真的在表达自杀意念。第二，很多模型又只靠统计规律，不懂这个领域的专业知识。比如一个人没直接写“我想死”，但写“我撑不下去了”或者“有一天我也会走上你的路”，其实也可能很危险，普通模型不一定能抓住这种含蓄表达。",
        "所以作者的核心想法是：把专业词典知识和上下文理解能力结合起来。",
        "这篇文章提出了一个模型，名字叫 KETCH。我们可以把它理解成一个有“两只眼睛”的模型。一只眼睛看专业词典，也就是作者专门构建的自杀意念相关表达词典，里面不是普通情感词，而是和自杀风险更相关的词和表达。另一只眼睛看上下文，它不是只看某个词出现没出现，而是看这个词在整句话里到底重不重要、到底是什么意思。",
        "所以 KETCH 不是简单地问：这句话里有没有危险词？它更像是在问：这句话里哪些词最重要？这些词和自杀风险词典有多像？它们在当前语境下是不是在表达真实的危险信号？这就是这篇论文最核心的创新。",
        "举个很简单的例子，如果有人写“我真的活不下去了”或者“我好像走到头了”，这些话虽然没有直接写“自杀”两个字，但上下文已经很危险。反过来，如果有人写“疼死我了”或者“累死了”，这里虽然有“死”，但很多时候并不是自杀意念。所以这个模型厉害的地方就在于，它不是死板地抓关键词，而是同时看关键词和上下文。",
        "在数据上，作者用了两类数据：一类是中文的新浪微博数据，一类是英文的 Reddit 数据。微博那部分规模比较大，抓了很多用户和帖子，再请精神健康相关专家去标注，判断哪些帖子带有自杀意念。另外他们还做了一个中文自杀意念词典，也是请专家参与整理的。这说明这篇论文不是只在模型结构上做文章，它连数据和词典都认真搭了。",
        "结果方面，KETCH 比很多常见方法都更好，包括 CNN、LSTM、BERT、RoBERTa 这些基线模型。更重要的不是它分数高一点，而是它证明了一件事：在这种高风险文本任务里，把领域知识真正融进模型，是有明显帮助的。",
        "作者还进一步做了两件事来证明它不是只会做一道题。第一，它不只在中文微博上测，还在英文 Reddit 上测。第二，它不只做单条帖子判断，还做用户整体风险判断。结果都不错，说明这个方法有一定泛化能力。",
        "这篇论文还有一个很现实的地方，它不只是停留在实验室里跑分。作者还做了一个实际场景的尝试：当模型发现疑似高风险帖子后，再由专业人员人工复核，然后主动联系用户，看看能不能提供帮助。也就是说，这篇论文的目标不是单纯做一个分类器，而是想走向“检测、预警、干预”的完整流程。",
        "如果让我最后总结成五点，我会这样说：第一，这篇论文做的是社交媒体自杀意念检测。第二，它想解决的问题是高风险人群不会主动求助，但会在网上表达痛苦。第三，它的方法叫 KETCH，核心思想是词典知识和上下文理解一起用。第四，它比很多常见深度学习模型效果更好。第五，它的最终目标不是替代医生，而是做更早的风险发现和辅助干预。",
        "所以你可以把 KETCH 理解成一个更懂心理风险语言的文本预警系统。它不是在说“我能准确诊断一个人会不会自杀”，而是在说“我可以比普通模型更早、更稳地发现一些危险信号，帮助专业人员更快介入”。这就是这篇论文最值得关注的地方。谢谢大家。",
    ]
    for para in speech:
        add_doc_paragraph(doc, para, size=12)

    add_doc_paragraph(doc, "二、30 秒结尾总结", style="Heading 1", size=15, bold=True)
    summary = (
        "这篇论文研究的是如何通过社交媒体文本识别自杀意念。作者提出了 KETCH，把 Transformer 的上下文理解能力和专家构建的自杀意念词典结合起来。"
        "实验表明，它在中文微博、英文 Reddit、用户级风险预测和抑郁检测上都优于很多基线方法。"
        "所以它的核心价值在于证明了：在高风险心理健康任务里，领域知识增强的 Transformer 是有效的。"
    )
    add_doc_paragraph(doc, summary, size=12)

    add_doc_paragraph(doc, "三、提词关键词", style="Heading 1", size=15, bold=True)
    keywords = [
        "背景：高风险人群不主动求助，但会在社交媒体表达痛苦",
        "难点：只看关键词会误判，只靠统计规律又抓不住含蓄表达",
        "方法：KETCH = 专业词典知识 + 上下文理解",
        "数据：微博 + Reddit，专家标注，专家参与构建词典",
        "结果：优于 CNN/LSTM/BERT/RoBERTa，且有跨语言与用户级泛化",
        "意义：从“分类”走向“检测 -> 预警 -> 干预”",
    ]
    for item in keywords:
        p = doc.add_paragraph(style="List Bullet")
        run = p.add_run(item)
        set_docx_font(run, size=12)

    doc.save(DOC_OUT)


def set_slide_bg(slide, color):
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = color


def add_textbox(slide, left, top, width, height, text, font_size=20, bold=False, color=None, font_name="Microsoft YaHei", align=PP_ALIGN.LEFT):
    tx = slide.shapes.add_textbox(left, top, width, height)
    tf = tx.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.TOP
    p = tf.paragraphs[0]
    p.text = text
    p.alignment = align
    run = p.runs[0]
    run.font.name = font_name
    run.font.size = PPt(font_size)
    run.font.bold = bold
    run.font.color.rgb = color or COLORS["dark"]
    return tx


def add_bullets(slide, left, top, width, height, bullets, font_size=22, color=None, level0_bold=False):
    tx = slide.shapes.add_textbox(left, top, width, height)
    tf = tx.text_frame
    tf.word_wrap = True
    tf.clear()
    for i, bullet in enumerate(bullets):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = bullet
        p.level = 0
        p.space_after = PPt(7)
        p.alignment = PP_ALIGN.LEFT
        if p.runs:
            run = p.runs[0]
        else:
            run = p.add_run()
        run.font.name = "Microsoft YaHei"
        run.font.size = PPt(font_size)
        run.font.color.rgb = color or COLORS["dark"]
        run.font.bold = level0_bold
    return tx


def add_header_band(slide, title, subtitle=None):
    band = slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.RECTANGLE, 0, 0, PCm(33.87), PCm(2.0))
    band.fill.solid()
    band.fill.fore_color.rgb = COLORS["navy"]
    band.line.fill.background()
    add_textbox(slide, PCm(0.9), PCm(0.35), PCm(20), PCm(1.0), title, font_size=24, bold=True, color=COLORS["white"])
    if subtitle:
        add_textbox(slide, PCm(21.3), PCm(0.45), PCm(11.5), PCm(0.8), subtitle, font_size=10, color=RGBColor(224, 231, 242), align=PP_ALIGN.RIGHT)


def add_card(slide, left, top, width, height, title, bullets, fill_color, title_color=COLORS["navy"]):
    shape = slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.ROUNDED_RECTANGLE, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill_color
    shape.line.color.rgb = RGBColor(210, 219, 232)
    add_textbox(slide, left + PCm(0.4), top + PCm(0.25), width - PCm(0.8), PCm(0.9), title, font_size=18, bold=True, color=title_color)
    add_bullets(slide, left + PCm(0.4), top + PCm(1.1), width - PCm(0.8), height - PCm(1.3), bullets, font_size=15)


def build_ppt():
    prs = Presentation()
    prs.slide_width = PCm(33.867)
    prs.slide_height = PCm(19.05)
    blank = prs.slide_layouts[6]

    # Slide 1
    slide = prs.slides.add_slide(blank)
    set_slide_bg(slide, COLORS["white"])
    banner = slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.RECTANGLE, 0, 0, PCm(33.867), PCm(4.3))
    banner.fill.solid()
    banner.fill.fore_color.rgb = COLORS["navy"]
    banner.line.fill.background()
    add_textbox(slide, PCm(1.2), PCm(0.8), PCm(24.5), PCm(1.8), "KETCH：知识增强 Transformer 的\n社交媒体自杀意念检测方法", font_size=26, bold=True, color=COLORS["white"])
    add_textbox(slide, PCm(1.2), PCm(3.0), PCm(20), PCm(1.0), "Information Systems Research, 2025", font_size=14, color=RGBColor(217, 226, 239))
    add_textbox(slide, PCm(1.2), PCm(5.4), PCm(18), PCm(1.6), "一句话概括：\n让电脑更早地从社交媒体文字里，看出一个人是不是可能出现了“自杀意念”，从而尽早提醒和干预。", font_size=20, bold=True, color=COLORS["navy"])
    add_textbox(slide, PCm(22.0), PCm(5.3), PCm(10.4), PCm(6.2), "关键词\n社交媒体\n自杀意念检测\n知识增强\nTransformer\n早期预警", font_size=18, color=COLORS["gray"], align=PP_ALIGN.CENTER)

    # Slide 2
    slide = prs.slides.add_slide(blank)
    set_slide_bg(slide, COLORS["white"])
    add_header_band(slide, "1. 为什么这个问题重要", "Background")
    add_card(
        slide,
        PCm(0.9),
        PCm(3.0),
        PCm(14.9),
        PCm(11.8),
        "现实痛点",
        [
            "很多真正有危险的人不会主动去找医生、咨询师或家人求助",
            "但他们可能会在微博、Reddit 等平台发出痛苦和绝望表达",
            "社交媒体因此可以成为“早期预警窗口”",
        ],
        COLORS["light_blue"],
    )
    add_card(
        slide,
        PCm(16.4),
        PCm(3.0),
        PCm(16.1),
        PCm(11.8),
        "为什么需要 AI",
        [
            "人工盯帖不现实：帖子太多、更新太快、语言表达很隐晦",
            "目标不是替代医生，而是更早发现风险、辅助专业人员介入",
            "论文最终目标：从“文本分类”走向“检测 -> 预警 -> 干预”",
        ],
        COLORS["light_teal"],
        title_color=COLORS["teal"],
    )

    # Slide 3
    slide = prs.slides.add_slide(blank)
    set_slide_bg(slide, COLORS["white"])
    add_header_band(slide, "2. 以前的方法难在哪里", "Problem")
    add_card(
        slide,
        PCm(1.2),
        PCm(3.1),
        PCm(14.8),
        PCm(11.0),
        "难点一：很多模型只会“看字面”",
        [
            "帖子里出现“死”“结束”“离开”就容易被判成高风险",
            "但现实里不是所有带这些词的句子都在表达自杀意念",
            "例子：'疼死我了'、'累死了' 往往不是自杀风险",
        ],
        COLORS["light_orange"],
        title_color=COLORS["orange"],
    )
    add_card(
        slide,
        PCm(17.0),
        PCm(3.1),
        PCm(15.3),
        PCm(11.0),
        "难点二：只靠统计规律，不懂领域知识",
        [
            "有些危险表达并不会直接出现“我想死”",
            "例子：'我撑不下去了'、'我也会走上你的路'",
            "普通模型不一定能抓住这种含蓄但高风险的表达",
        ],
        COLORS["light_blue"],
    )
    add_textbox(slide, PCm(1.2), PCm(15.1), PCm(30), PCm(1), "所以作者的核心想法是：把“专业词典知识” + “上下文理解能力”结合起来。", font_size=20, bold=True, color=COLORS["navy"], align=PP_ALIGN.CENTER)

    # Slide 4
    slide = prs.slides.add_slide(blank)
    set_slide_bg(slide, COLORS["white"])
    add_header_band(slide, "3. KETCH 的核心想法", "Core Idea")
    add_textbox(slide, PCm(1.2), PCm(2.6), PCm(31), PCm(1.2), "可以把 KETCH 理解成一个有“两只眼睛”的模型", font_size=24, bold=True, color=COLORS["navy"], align=PP_ALIGN.CENTER)
    add_card(
        slide,
        PCm(2.1),
        PCm(5.0),
        PCm(12.8),
        PCm(7.5),
        "第一只眼睛：看专业词典",
        [
            "作者专门构建了“自杀意念相关表达词典”",
            "词典里不是普通情感词，而是和自杀风险更相关的词和表达",
        ],
        COLORS["light_blue"],
    )
    add_card(
        slide,
        PCm(18.2),
        PCm(5.0),
        PCm(12.8),
        PCm(7.5),
        "第二只眼睛：看上下文",
        [
            "不只是看某个词是否出现",
            "还要看这个词在整句话里重不重要、到底是什么意思",
        ],
        COLORS["light_teal"],
        title_color=COLORS["teal"],
    )
    add_textbox(slide, PCm(2.1), PCm(13.5), PCm(28.9), PCm(2.2), "它不是简单地问“这句话里有没有危险词”，\n而是在问“哪些词最重要，它们和自杀风险词典有多像，它们在当前语境下是不是危险信号”。", font_size=18, color=COLORS["dark"], align=PP_ALIGN.CENTER)

    # Slide 5
    slide = prs.slides.add_slide(blank)
    set_slide_bg(slide, COLORS["white"])
    add_header_band(slide, "4. 模型流程怎么走", "Method")
    add_bullets(
        slide,
        PCm(1.0),
        PCm(3.0),
        PCm(10.0),
        PCm(6.5),
        [
            "1. 文本预处理",
            "2. 表示精炼（RoBERTa 微调）",
            "3. 表示增强（词典相似度 + 注意力）",
            "4. 分类输出",
        ],
        font_size=20,
    )
    slide.shapes.add_picture(str(FIG_DIR / "figure2_ketch_framework.png"), PCm(12.0), PCm(3.0), width=PCm(20.2))
    add_textbox(slide, PCm(1.0), PCm(10.6), PCm(31), PCm(3.2), "最关键的是第 3 步：模型不是只看有没有危险词，而是综合“词和词典有多像”以及“这个词在句子里有多重要”。", font_size=18, color=COLORS["gray"])
    add_textbox(slide, PCm(12.0), PCm(14.2), PCm(20.0), PCm(1.0), "图：KETCH 总体框架（原论文 Figure 2）", font_size=10, color=COLORS["gray"], align=PP_ALIGN.CENTER)

    # Slide 6
    slide = prs.slides.add_slide(blank)
    set_slide_bg(slide, COLORS["white"])
    add_header_band(slide, "5. 数据和词典从哪里来", "Data")
    add_card(
        slide,
        PCm(0.9),
        PCm(2.7),
        PCm(14.5),
        PCm(8.5),
        "数据来源",
        [
            "中文：新浪微博",
            "英文：Reddit SuicideWatch",
            "微博规模较大：29,388 个用户，104,219 条帖子，清洗后 99,030 条",
            "专家参与标注哪些帖子表达了自杀意念",
        ],
        COLORS["light_blue"],
    )
    add_card(
        slide,
        PCm(0.9),
        PCm(11.6),
        PCm(14.5),
        PCm(4.3),
        "词典构建",
        [
            "从 4,600 多条微博中抽取 seed terms",
            "多轮专家筛选，最后形成 320 个中文 SI 词条",
        ],
        COLORS["light_orange"],
        title_color=COLORS["orange"],
    )
    slide.shapes.add_picture(str(FIG_DIR / "figurec2_lexicon_distribution.png"), PCm(16.2), PCm(3.0), width=PCm(15.9))
    add_textbox(slide, PCm(16.2), PCm(12.2), PCm(15.9), PCm(2.8), "图：词典词项在帖子中的分布。大多数帖子只包含少量词典词项，说明仅靠关键词命中并不够，还要结合上下文。", font_size=14, color=COLORS["gray"])

    # Slide 7
    slide = prs.slides.add_slide(blank)
    set_slide_bg(slide, COLORS["white"])
    add_header_band(slide, "6. 结果怎么样", "Results")
    add_textbox(slide, PCm(1.0), PCm(2.8), PCm(31), PCm(1.0), "关键结果：KETCH 在中文微博、英文 Reddit、用户级任务上都优于常见基线。", font_size=20, bold=True, color=COLORS["navy"])
    table = slide.shapes.add_table(4, 5, PCm(1.2), PCm(4.2), PCm(17.0), PCm(5.0)).table
    headers = ["任务/模型", "Precision", "Recall", "F1", "Accuracy"]
    rows = [
        ["微博 RoBERTa", "0.747", "0.674", "0.705", "0.906"],
        ["微博 KETCH", "0.826", "0.881", "0.868", "0.935"],
        ["Reddit 用户级 KETCH", "0.923", "0.955", "0.939", "0.912"],
    ]
    for j, h in enumerate(headers):
        cell = table.cell(0, j)
        cell.text = h
        cell.fill.solid()
        cell.fill.fore_color.rgb = COLORS["navy"]
        for p in cell.text_frame.paragraphs:
            for run in p.runs:
                run.font.name = "Microsoft YaHei"
                run.font.size = PPt(14)
                run.font.bold = True
                run.font.color.rgb = COLORS["white"]
                p.alignment = PP_ALIGN.CENTER
    for i, row in enumerate(rows, start=1):
        for j, val in enumerate(row):
            cell = table.cell(i, j)
            cell.text = val
            cell.fill.solid()
            cell.fill.fore_color.rgb = COLORS["light_gray"] if i % 2 == 1 else COLORS["white"]
            for p in cell.text_frame.paragraphs:
                for run in p.runs:
                    run.font.name = "Microsoft YaHei"
                    run.font.size = PPt(13)
                    run.font.bold = (j == 0 or "KETCH" in row[0])
                    run.font.color.rgb = COLORS["dark"]
                    p.alignment = PP_ALIGN.CENTER
    slide.shapes.add_picture(str(FIG_DIR / "figure3_rep_enhancement.png"), PCm(19.3), PCm(4.0), width=PCm(13.0))
    add_textbox(slide, PCm(19.3), PCm(13.7), PCm(13.0), PCm(2.0), "图：增强表示学习流程（原论文 Figure 3）。最关键的提升来自“词典知识 + 上下文”的联合建模。", font_size=13, color=COLORS["gray"])

    # Slide 8
    slide = prs.slides.add_slide(blank)
    set_slide_bg(slide, COLORS["white"])
    add_header_band(slide, "7. 这篇论文为什么有现实意义", "Practice")
    add_card(
        slide,
        PCm(1.0),
        PCm(3.0),
        PCm(16.0),
        PCm(9.8),
        "它不只是实验室跑分",
        [
            "作者做了一个实际场景尝试：模型先找出疑似高风险帖子",
            "再由专业人员人工复核",
            "确认后主动联系用户，尝试提供帮助",
            "目标从“分类”走向“检测 -> 预警 -> 干预”",
        ],
        COLORS["light_teal"],
        title_color=COLORS["teal"],
    )
    slide.shapes.add_picture(str(FIG_DIR / "figured1_field_study_flow.png"), PCm(19.5), PCm(3.3), width=PCm(10.5))
    add_textbox(slide, PCm(18.8), PCm(11.6), PCm(12.0), PCm(2.2), "图：田野研究中的干预流程（原论文 Figure D.1）", font_size=13, color=COLORS["gray"], align=PP_ALIGN.CENTER)
    add_textbox(slide, PCm(1.0), PCm(13.6), PCm(31.0), PCm(1.2), "所以这篇论文的价值，不只是“模型更准”，而是它证明了这个方向具有实际应用潜力。", font_size=18, bold=True, color=COLORS["navy"], align=PP_ALIGN.CENTER)

    # Slide 9
    slide = prs.slides.add_slide(blank)
    set_slide_bg(slide, COLORS["white"])
    add_header_band(slide, "8. 局限性和我的评价", "Reflection")
    add_card(
        slide,
        PCm(1.0),
        PCm(3.0),
        PCm(14.8),
        PCm(10.8),
        "论文局限",
        [
            "主要还是文本，没有做多模态",
            "词典需要不断更新，社交媒体语言变化很快",
            "田野研究更多证明“可行性”，不是严格临床因果结论",
            "不同人群、不同平台上的公平性问题仍值得继续研究",
        ],
        COLORS["light_orange"],
        title_color=COLORS["orange"],
    )
    add_card(
        slide,
        PCm(17.0),
        PCm(3.0),
        PCm(15.3),
        PCm(10.8),
        "我的评价",
        [
            "它不是只换了一个大模型，而是认真解决了“领域知识怎么进模型”",
            "数据、词典、实验和实际干预尝试都做得比较完整",
            "最值得记住的一点：高风险任务里，领域知识增强的 Transformer 确实有明显帮助",
        ],
        COLORS["light_blue"],
    )

    # Slide 10
    slide = prs.slides.add_slide(blank)
    set_slide_bg(slide, COLORS["navy"])
    add_textbox(slide, PCm(1.4), PCm(1.2), PCm(30), PCm(1.2), "9. 最后记住这 5 点", font_size=28, bold=True, color=COLORS["white"], align=PP_ALIGN.CENTER)
    end_points = [
        "1. 这篇论文做的是“社交媒体自杀意念检测”",
        "2. 它要解决的是：高风险人群不会主动求助，但会在网上表达痛苦",
        "3. KETCH 的核心思想是“词典知识 + 上下文理解”一起用",
        "4. 它比很多常见深度学习模型效果更好，而且有跨语言与用户级泛化",
        "5. 它的目标不是替代医生，而是帮助更早发现风险、辅助干预",
    ]
    add_bullets(slide, PCm(3.4), PCm(4.0), PCm(27), PCm(9.0), end_points, font_size=20, color=COLORS["white"])
    add_textbox(slide, PCm(8.8), PCm(15.0), PCm(16), PCm(1.0), "谢谢大家  |  Q&A", font_size=24, bold=True, color=RGBColor(239, 244, 255), align=PP_ALIGN.CENTER)

    prs.save(PPT_OUT)


if __name__ == "__main__":
    build_speech_doc()
    build_ppt()
    print(PPT_OUT)
    print(DOC_OUT)
