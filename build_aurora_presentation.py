from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_AUTO_SIZE, PP_ALIGN
from pptx.oxml.xmlchemy import OxmlElement
from pptx.util import Inches, Pt


ROOT = Path(__file__).resolve().parent
TEMPLATE = ROOT / "Supplied files" / "Project_Presentation_Template.pptx"
OUT_DIR = ROOT / "presentations"
OUT_DIR.mkdir(exist_ok=True)
OUTPUT = OUT_DIR / "Aurora_Stays_RAG_Project_Presentation_Template_Style.pptx"


def remove_shape(shape):
    shape._element.getparent().remove(shape._element)


NAVY = RGBColor(18, 43, 70)
BLUE = RGBColor(0, 120, 166)
TEAL = RGBColor(0, 156, 166)
WHITE = RGBColor(255, 255, 255)
CHARCOAL = RGBColor(42, 45, 50)
GREEN = RGBColor(31, 119, 82)
RED = RGBColor(166, 42, 42)


def set_shape_text(shape, text, font_size=None, bold=None, color=None):
    shape.text = text
    if not shape.has_text_frame:
        return
    for paragraph in shape.text_frame.paragraphs:
        for run in paragraph.runs:
            run.font.name = "Aptos Display" if bold else "Aptos"
            if font_size:
                run.font.size = Pt(font_size)
            if bold is not None:
                run.font.bold = bold
            if color is not None:
                run.font.color.rgb = color


def find_title_shape(slide):
    text_shapes = [
        shape for shape in slide.shapes
        if hasattr(shape, "text_frame") and shape.text.strip()
    ]
    return min(text_shapes, key=lambda shape: shape.top) if text_shapes else None


def send_to_back(shape):
    tree = shape._element.getparent()
    tree.remove(shape._element)
    tree.insert(2, shape._element)


def add_header_band(slide):
    # Remove the template's small circular decoration; the redesigned layout
    # uses a restrained executive header instead.
    for shape in list(slide.shapes):
        if (
            hasattr(shape, "text_frame")
            and not shape.text.strip()
            and shape.width <= Inches(0.5)
            and shape.height <= Inches(0.5)
        ):
            remove_shape(shape)

    header = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(10), Inches(0.92)
    )
    header.fill.solid()
    header.fill.fore_color.rgb = NAVY
    header.line.fill.background()
    send_to_back(header)

    rule = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(0), Inches(0.92), Inches(10), Inches(0.05)
    )
    rule.fill.solid()
    rule.fill.fore_color.rgb = TEAL
    rule.line.fill.background()
    send_to_back(rule)


def add_title_background(slide):
    background = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(10), Inches(5.625)
    )
    background.fill.solid()
    background.fill.fore_color.rgb = NAVY
    background.line.fill.background()
    send_to_back(background)

    accent = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(1.78), Inches(3.12), Inches(2.35), Inches(0.07)
    )
    accent.fill.solid()
    accent.fill.fore_color.rgb = TEAL
    accent.line.fill.background()


def add_body(slide, lines, font_size=17, mono=False, body_top=1.45, add_header_line=False):
    # Remove template instruction text while preserving the template background.
    title_shape = find_title_shape(slide)
    for shape in list(slide.shapes):
        if shape._element is not title_shape._element and hasattr(shape, "text_frame") and shape.text.strip():
            remove_shape(shape)

    accent = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE,
        Inches(0.68), Inches(1.24), Inches(0.06), Inches(3.98),
    )
    accent.fill.solid()
    accent.fill.fore_color.rgb = TEAL
    accent.line.fill.background()

    box = slide.shapes.add_textbox(
        Inches(0.98), Inches(1.27), Inches(8.28), Inches(3.94)
    )
    tf = box.text_frame
    tf.clear()
    tf.word_wrap = True
    tf.auto_size = MSO_AUTO_SIZE.TEXT_TO_FIT_SHAPE
    tf.margin_left = Inches(0.06)
    tf.margin_right = Inches(0.06)
    tf.margin_top = Inches(0.04)
    tf.margin_bottom = Inches(0.04)

    for index, line in enumerate(lines):
        paragraph = tf.paragraphs[0] if index == 0 else tf.add_paragraph()
        is_bullet = line.startswith("• ")
        is_indented = line.startswith("  ")
        display_line = line[2:] if is_bullet else line.lstrip() if is_indented else line
        paragraph.text = display_line
        paragraph.level = 1 if is_indented else 0
        paragraph.space_before = Pt(0)
        paragraph.space_after = Pt(3 if is_bullet else 5)
        paragraph.line_spacing = 1.02 if is_bullet else 1.05
        paragraph.font.name = "Aptos"
        paragraph.font.size = Pt(font_size)
        paragraph.font.color.rgb = CHARCOAL
        if is_bullet:
            p_pr = paragraph._p.get_or_add_pPr()
            bullet = OxmlElement("a:buChar")
            bullet.set("char", "•")
            p_pr.insert(0, bullet)
            # Explicit DrawingML margins create a true hanging indent for
            # wrapped bullets; relying on the inherited theme indent causes
            # continuation lines to hang beneath the bullet marker.
            p_pr.set("marL", str(Inches(0.30)))
            p_pr.set("indent", str(Inches(-0.18)))
        elif is_indented:
            p_pr = paragraph._p.get_or_add_pPr()
            p_pr.set("marL", str(Inches(0.30)))
            p_pr.set("indent", str(Inches(0)))
        else:
            p_pr = paragraph._p.get_or_add_pPr()
            p_pr.set("marL", str(Inches(0)))
            p_pr.set("indent", str(Inches(0)))
        if not is_bullet and not is_indented and (display_line.endswith(":" ) or display_line in {
            "Problem",
            "Solution approach",
            "Relevant evidence",
            "Example focused chunk",
            "Held-out DeepEval comparison (10 cases)",
        }):
            for run in paragraph.runs:
                run.font.bold = True
                run.font.color.rgb = NAVY
        if display_line.startswith("Selected:"):
            for run in paragraph.runs:
                run.font.bold = True
                run.font.color.rgb = GREEN
        if display_line.startswith("Only remaining failure:"):
            for run in paragraph.runs:
                run.font.bold = True
                run.font.color.rgb = RED

    return box


def set_title(slide, title):
    title_shape = find_title_shape(slide)
    if title_shape is None:
        title_shape = next(
            shape for shape in slide.shapes
            if hasattr(shape, "text_frame") and shape.text.strip()
        )
    set_shape_text(title_shape, title, font_size=22, bold=True, color=WHITE)


# ---------------------------------------------------------------------------
# Template-preserving overrides
#
# The supplied deck uses a white canvas, blue Nunito/Montserrat typography,
# and small blue circular bullets.  These replacements intentionally keep the
# original slide geometry and do not add a new header/background system.
# ---------------------------------------------------------------------------
TEMPLATE_BLUE = RGBColor(14, 57, 169)
TEMPLATE_BULLET = RGBColor(9, 68, 161)
TEMPLATE_BLACK = RGBColor(0, 0, 0)


def set_shape_text(shape, text, font_name=None, font_size=None, bold=None, color=None):
    shape.text = text
    if not shape.has_text_frame:
        return
    for paragraph in shape.text_frame.paragraphs:
        for run in paragraph.runs:
            if font_name:
                run.font.name = font_name
            if font_size:
                run.font.size = Pt(font_size)
            if bold is not None:
                run.font.bold = bold
            if color is not None:
                run.font.color.rgb = color


def add_header_band(slide):
    # Deliberately empty: the original template has no header band.
    return None


def add_title_background(slide):
    # Deliberately empty: preserve the template's original white title slide.
    return None


def _normalise_text(value):
    return (
        value.replace("â€¢", "•")
        .replace("â€“", "–")
        .replace("â€”", "—")
        .replace("â€™", "’")
        .replace("â€œ", "“")
        .replace("â€", "”")
    )


def _add_template_row(slide, top, text, font_size, heading=False, color=None):
    text = _normalise_text(text)
    box = slide.shapes.add_textbox(Inches(0.76), Inches(top), Inches(7.36), Inches(0.55))
    box.text = text
    box.text_frame.word_wrap = True
    box.text_frame.auto_size = MSO_AUTO_SIZE.TEXT_TO_FIT_SHAPE
    box.text_frame.margin_left = Inches(0.09)
    box.text_frame.margin_right = Inches(0.04)
    box.text_frame.margin_top = Inches(0.02)
    box.text_frame.margin_bottom = Inches(0.02)
    for paragraph in box.text_frame.paragraphs:
        paragraph.space_before = Pt(0)
        paragraph.space_after = Pt(0)
        paragraph.line_spacing = 1.0
        for run in paragraph.runs:
            run.font.name = "Montserrat"
            run.font.size = Pt(font_size)
            run.font.bold = heading
            run.font.color.rgb = color or (TEMPLATE_BLUE if heading else TEMPLATE_BLACK)
    if not heading:
        dot = slide.shapes.add_shape(
            MSO_SHAPE.OVAL, Inches(0.44), Inches(top + 0.11), Inches(0.21), Inches(0.21)
        )
        dot.fill.solid()
        dot.fill.fore_color.rgb = TEMPLATE_BULLET
        dot.line.fill.background()
    return box


def add_template_divider(slide, top=1.10, left=0.55, width=8.90):
    rule = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(0.035)
    )
    rule.fill.solid()
    rule.fill.fore_color.rgb = TEMPLATE_BLUE
    rule.line.fill.background()


def add_body(slide, lines, font_size=17, mono=False, body_top=1.45, add_header_line=False):
    title_shape = find_title_shape(slide)
    for shape in list(slide.shapes):
        if shape._element is not title_shape._element:
            remove_shape(shape)

    content = []
    for raw in lines:
        raw = _normalise_text(raw)
        is_bullet = raw.startswith("• ")
        is_indented = raw.startswith("  ")
        text = raw[2:] if is_bullet else raw.lstrip() if is_indented else raw
        heading = (
            not is_bullet
            and not is_indented
            and (text.endswith(":") or text in {
                "Problem", "Solution approach", "Relevant evidence", "Example focused chunk",
                "Held-out DeepEval comparison (10 cases)",
                "Production checklist", "Known limitation",
            })
        )
        content.append((text, is_bullet or is_indented, heading))

    # Keep body typography consistent across the deck.  The text boxes use
    # fit-to-shape for the few dense slides, while all body rows retain the
    # same font, line spacing, and bullet treatment.
    chars_per_line = 62
    body_size = 13.5
    line_height = 0.22
    if add_header_line:
        add_template_divider(slide)
    top = body_top
    for text, is_bullet, heading in content:
        estimated_lines = max(1, (len(text) + chars_per_line - 1) // chars_per_line)
        row_height = max(0.28, estimated_lines * line_height + 0.05)
        if top + row_height > 5.12:
            row_height = max(0.20, 5.12 - top)
        box = _add_template_row(
            slide, top, text, body_size,
            heading=heading and not is_bullet,
            color=TEMPLATE_BLUE if heading else None,
        )
        box.height = Inches(row_height)
        if text.startswith("Selected:"):
            for paragraph in box.text_frame.paragraphs:
                for run in paragraph.runs:
                    run.font.bold = True
                    run.font.color.rgb = GREEN
        elif text.startswith("Only remaining failure:"):
            for paragraph in box.text_frame.paragraphs:
                for run in paragraph.runs:
                    run.font.bold = True
                    run.font.color.rgb = RED
        top += row_height + 0.035


def set_title(slide, title):
    title_shape = find_title_shape(slide)
    set_shape_text(title_shape, title, font_name="Montserrat", font_size=22, bold=True, color=TEMPLATE_BLUE)


prs = Presentation(str(TEMPLATE))

# Populate the complete template through the appendix. The final template
# slide is an unused placeholder and is removed.
while len(prs.slides) > 31:
    slide_id = prs.slides._sldIdLst[-1]
    rel_id = slide_id.rId
    prs.part.drop_rel(rel_id)
    prs.slides._sldIdLst.remove(slide_id)

# Title slide.
add_title_background(prs.slides[0])
title_text_shapes = sorted(
    [shape for shape in prs.slides[0].shapes if hasattr(shape, "text_frame") and shape.text.strip()],
    key=lambda shape: shape.top,
)
set_shape_text(title_text_shapes[0], "Aurora Stays Q&A", font_name="Nunito", font_size=41, bold=True, color=TEMPLATE_BLUE)
title_text_shapes[0].left = Inches(1.78)
title_text_shapes[0].width = Inches(6.8)
set_shape_text(title_text_shapes[1], "Retrieval-Augmented Generation (RAG) POC", font_name="Nunito", font_size=23, color=TEMPLATE_BLUE)
title_text_shapes[1].left = Inches(1.78)
title_text_shapes[1].width = Inches(7.6)
set_shape_text(title_text_shapes[2], "14 October 2026", font_name="Nunito", font_size=22, color=TEMPLATE_BLUE)
title_text_shapes[2].left = Inches(1.78)
title_text_shapes[2].width = Inches(4.2)

# Agenda without the excluded deployment section.
agenda_slide = prs.slides[1]
add_header_band(agenda_slide)
agenda_text_shapes = sorted(
    [shape for shape in agenda_slide.shapes if hasattr(shape, "text_frame") and shape.text.strip()],
    key=lambda shape: shape.top,
)
set_shape_text(agenda_text_shapes[0], "Contents / Agenda", font_name="Montserrat", font_size=22, bold=True, color=TEMPLATE_BLUE)
agenda = [
    "Executive Summary",
    "Business Problem and Solution Approach",
    "Data Understanding",
    "Retrieval-Augmented Generation System",
    "Evaluation",
    "Deployment, Inference and Handoff",
]
for shape, text in zip(agenda_text_shapes[1:], agenda):
    set_shape_text(shape, text, font_name="Montserrat", font_size=14, color=TEMPLATE_BLACK)
add_template_divider(agenda_slide, top=1.02)

slides = {
    3: ("Executive Summary", [
        "• Built a source-grounded RAG assistant for Aurora Stays hotel-policy questions.",
        "• Four policy PDFs are converted to structured Markdown, chunked, embedded, and indexed in FAISS.",
        "• Selected baseline: k=5, MMR lambda=0.50, fetch_k=20, CrossEncoder reranking.",
        "• Held-out result: 9/10 cases passed; composite score 0.941.",
        "• Average metrics: Faithfulness 0.993 | Recall 0.977 | Precision 0.808 | Relevancy 0.939.",
        "• Main remaining issue: the Accessible King benchmark question is underspecified; a focused clarification experiment raised precision to 0.961.",
    ], 15.5),
    4: ("Business Problem Overview and Solution Approach", [
        "Problem",
        "• Staff handle about 1,200 guest queries per week; roughly 40% require multiple policy documents.",
        "• Manual lookup takes 8–12 minutes and creates a 15–20% risk of incomplete or incorrect answers.",
        "• Property-specific fees, room features, cancellation rules, and accessibility guidance are easy to mix up.",
        "Solution approach",
        "• Extract policy PDFs with Docling; preserve headings and tables as Markdown.",
        "• Use token-aware chunking, domain embeddings, FAISS retrieval, MMR, and CrossEncoder reranking.",
        "• Generate concise answers with GPT-4o-mini using a strict retrieved-context-only prompt.",
    ], 15),
    5: ("Data Understanding", [
        "• Knowledge base: four selectable-text PDF policy guides.",
        "• Coverage: room/property information; booking and cancellation; refunds and additional charges; conduct and accessibility.",
        "• Source structure includes headings, policy sections, Markdown tables, and a six-property matrix.",
        "• Properties represented: ASH-ATL, ASH-BOS, ASH-CHI, ASH-DEN, ASH-MIA, and ASH-SEA.",
        "• OCR is disabled because the supplied PDFs contain selectable text.",
        "• Chunks are enriched with source, section, property-code, and topic metadata for focused retrieval.",
    ], 16),
    6: ("RAG - Chunking - Embedding Model", [
        "• Embedding model: HarishMaths/Hotel-Policy-Embedding.",
        "• Tokenizer maximum length: 128 tokens.",
        "• Embeddings run on CPU with normalized vectors and progress rendering disabled.",
        "• The same model is used when building and loading the FAISS index.",
        "• Domain-specific embeddings improve matching for hotel-policy terms such as late checkout, pet fees, and room features.",
    ], 17),
    7: ("RAG - Chunking - Chunker", [
        "• Chunker: RecursiveCharacterTextSplitter.from_huggingface_tokenizer.",
        "• Chunk size: 512 tokens; overlap: 64 tokens.",
        "• Split priority: paragraph, line, heading, sentence, word, then character boundaries.",
        "• Atomic property rows are added for property-matrix entries so a single property and fee can be retrieved precisely.",
        "• Metadata is added after splitting to preserve section, topic, and property identity.",
    ], 16),
    8: ("RAG - Chunking - Chunk Documents", [
        "• Total chunks created: 43, including atomic property rows.",
        "Example focused chunk",
        "  ## Property-specific matrix | Property: ASH-MIA",
        "  ASH-MIA Brickell | Parking: Valet, $55 per night | Pets: $100 per stay; 2 pets max",
        "• One-property rows reduce contamination from unrelated properties and broad policy sections.",
        "• Each chunk carries source, section, property_codes, and topics metadata.",
    ], 15.5, True),
    9: ("RAG - Vector Database", [
        "• Vector database: FAISS with 43 normalized document vectors.",
        "• Index is persisted for the backend under backend/faiss_index.",
        "• Final retrieval setting: k=5 chunks.",
        "• Candidate search uses MMR lambda=0.50 and fetch_k=20.",
        "• CrossEncoder ms-marco-MiniLM-L-6-v2 reranks candidates; topic and property metadata provide additional boosts.",
    ], 16),
    10: ("RAG - Vector Database - Retrieval Function", [
        "Query: What are the parking fees at the Atlanta and Miami hotels?",
        "• Property routing identifies ASH-ATL and ASH-MIA; topic routing identifies parking.",
        "• Candidate chunks are retrieved with MMR, then reranked for direct relevance.",
        "Relevant evidence",
        "  ASH-ATL Midtown: self parking, $28 per night.",
        "  ASH-MIA Brickell: valet, $55 per night.",
        "• The final context contains the two requested property rows instead of unrelated policy sections.",
    ], 15.5),
    11: ("RAG - Generation System - Load the LLM", [
        "• Generator: OpenAI-compatible ChatOpenAI wrapper.",
        "• Model: gpt-4o-mini.",
        "• temperature=0.0 | top_p=1.0 | max_tokens=512.",
        "• API key and base URL are loaded from environment/configuration rather than hard-coded.",
        "• Deterministic settings favor repeatable, concise policy answers for evaluation and operations.",
    ], 17),
    12: ("RAG - Generation System - Generation Prompt", [
        "Prompt inputs: user query + retrieved policy information.",
        "Core system prompt: You are the Aurora Stays policy assistant. Use only the retrieved policy context to answer.",
        "• Use only information explicitly supported by retrieved policy text.",
        "• Include relevant conditions, exclusions, limits, exceptions, and coverage details.",
        "• Focus on the question; omit unrelated content and do not invent facts.",
        "• If evidence is insufficient, return: The available policy documents do not contain enough information to answer this question.",
        "• Additional guards focus Advance Purchase, Accessible King, and late-checkout questions on their relevant evidence.",
    ], 15),
    13: ("RAG - Generation System - RAG Function", [
        "Pipeline: query -> retrieve top-k chunks -> generate grounded answer -> return answer and evidence.",
        "Example query: What is the cancellation policy for a flexible reservation?",
        "Example answer: Flexible rates can normally be canceled by 6:00 PM local property time one day before arrival without a room charge. Later cancellation or no-show may be charged one night plus tax; the confirmation controls if it differs.",
        "• The response endpoint can return the answer alone or the answer plus relevant chunks for verification.",
    ], 15.5),
    14: ("Evaluation - Load the Evaluation Data", [
        "• Development evaluation file: data/evaluation.csv.",
        "• 10 records with question, expected answer, and ground-truth relevant context.",
        "Sample question: What time can I check in at ASH-CHI Loop, and when do I need to be out?",
        "Expected answer: Standard check-in is 3:00 PM and standard check-out is 12:00 PM local property time; early arrival and late departure depend on availability and approval.",
        "• Multi-context records use <> separators so relevant policy sections can be evaluated together.",
    ], 15),
    15: ("Evaluation - Define the Evaluation Model", [
        "• Judge model: gpt-4o through the DeepEval OpenAIModel wrapper.",
        "• The generator remains gpt-4o-mini with temperature=0.0, top_p=1.0, and max_tokens=512.",
        "• The stronger judge model scores factual support, retrieval coverage, retrieval focus, and answer relevance.",
        "• Evaluations run synchronously so each test case has a stable, auditable metric result.",
    ], 16),
    16: ("Evaluation - Define the Evaluation Metrics", [
        "Faithfulness >= 0.85: every important answer claim is supported by retrieved context.",
        "Contextual Recall >= 0.80: retrieved context contains information needed for the expected answer.",
        "Contextual Precision >= 0.75: relevant context is prioritized over unrelated content.",
        "Answer Relevancy >= 0.75: answer directly addresses the user question.",
        "Composite = 0.35F + 0.25Recall + 0.20Precision + 0.20Relevancy.",
        "A case passes only when all four metric thresholds are met.",
    ], 15.5),
    17: ("Evaluation - Build and Run the Evaluation", [
        "• Retrieval tuning matrix selected chunk_size=512, overlap=64, MMR lambda=0.50, and fetch_k=20.",
        "• Dev proxy coverage: 0.836; held-out test proxy coverage: 0.850.",
        "• Full DeepEval is the acceptance test; proxy coverage is used only to narrow candidates.",
        "• The evaluation creates LLMTestCase objects with input, expected output, actual output, retrieval context, and reference context.",
        "• Output artifacts: deepeval_options_summary.csv and deepeval_options_cases.csv.",
        "• Cases are classified as AUTO ANSWER, ASSISTED REVIEW, or MANUAL LOOKUP from the composite score.",
        "• Result: baseline k=5 is the selected configuration by highest composite score.",
    ], 14.5),
    18: ("Evaluation - Load the Test Data", [
        "• Held-out file: data/test.xlsx.",
        "• 10 unseen questions covering cancellation, discounts, room occupancy, accessibility, parking, breakfast, rate changes, checkout, pets, and incidental holds.",
        "Sample question: How many people can stay in a Double Queen room?",
        "Expected answer: Up to 4 guests, subject to fire-code and property occupancy limits; recheck the room and rate if the group size changes.",
        "• The held-out set is used only for final comparison and selection of the baseline.",
    ], 15),
    19: ("Evaluation - Build and Run the Evaluation", [
        "Held-out case-level output — selected baseline k=5",
        "Case 01 | OTA cancellation/refund | PASS | Composite .944",
        "Case 02 | Member discount/cancellation | PASS | Composite .956",
        "Case 03 | Double Queen occupancy | PASS | Composite .988",
        "Case 04 | Accessible King inclusions | FAIL | Precision .451",
        "Case 05 | Atlanta/Miami parking | PASS | Composite .975",
        "Case 06 | Miami breakfast | PASS | Composite .963",
        "Case 07 | Advance Purchase change | PASS | Composite .916",
        "Case 08 | Denver late checkout | PASS | Composite .939",
        "Case 09 | ASH-CHI pet fee | PASS | Composite .974",
        "Case 10 | Card-hold difference | PASS | Composite .934",
        "Selected baseline result: 9/10 passed; weighted composite .941.",
        "The focused clarification experiment raised the failed case's Precision to .961 and passed all four metrics.",
    ], 14),
    20: ("Deployment - Backend - Serialize the Vector Database", [
        "Backend artifact: backend/faiss_index/index.faiss.",
        "Metadata and document store: backend/faiss_index/index.pkl.",
        "The persisted FAISS index contains 43 normalized vectors from the policy knowledge base.",
        "The backend loads the trusted local artifact at startup; the Docker image copies the index into /app/faiss_index.",
        "The artifact must be rebuilt whenever source documents, chunking, or embeddings change.",
    ], 15),
    21: ("Deployment - Backend - Define the Flask Application", [
        "Service entry point: backend/app.py; production command uses Gunicorn and app:rag_api.",
        "GET / returns a lightweight readiness message.",
        "POST /v1/relevant_chunks returns the top-k retrieved policy chunks.",
        "POST /v1/answer_with_relevant_chunks returns the generated answer and evidence chunks.",
        "POST /v1/answer returns the generated answer only.",
        "Defaults: k=5, gpt-4o-mini, temperature=0.0, top_p=1.0, max_tokens=512.",
        "Container binding: 0.0.0.0:7860; local smoke tests use port 5000.",
    ], 13.5),
    22: ("Deployment - Backend - Define the Dependencies", [
        "Pinned backend dependencies are declared in backend/requirements.txt.",
        "Flask 3.1.3 and Gunicorn 26.2.0 provide the API service runtime.",
        "LangChain HuggingFace, LangChain Community, and LangChain OpenAI provide retrieval and generation integration.",
        "sentence-transformers 6.1.0 and faiss-cpu 1.15.0 provide embeddings and vector search.",
        "The Docker build installs CPU-only PyTorch before installing the requirements file.",
        "The embedding model is cached during image build so runtime can use offline mode.",
    ], 13.5),
    23: ("Deployment - Backend - Define the Dockerfile", [
        "Base image: python:3.13.15-slim.",
        "Copies the backend application, FAISS artifact, and requirements into /app.",
        "Installs CPU PyTorch and the pinned backend dependencies without pip cache.",
        "Downloads HarishMaths/Hotel-Policy-Embedding at build time and sets HF_HOME=/app/huggingface.",
        "TRANSFORMERS_OFFLINE=1 and HF_HUB_OFFLINE=1 prevent runtime model downloads.",
        "Starts one Gunicorn worker on port 7860 with a 120-second request timeout.",
    ], 13.5),
    24: ("Deployment - Frontend - Define the Streamlit App", [
        "Frontend entry point: frontend/app.py; it provides the Aurora Stays question-and-answer interface.",
        "Sidebar controls: k from 1 to 10, temperature from 0.0 to 1.0, and max_tokens from 64 to 2048.",
        "Users can select Answer only or Answer + Relevant Chunks.",
        "The backend address is supplied through BACKEND_URL; local default is http://localhost:5000.",
        "The interface displays the answer, retrieved evidence, and a clear API connectivity error when the backend is unavailable.",
        "Validated locally with Streamlit on port 8501 and the Flask API on port 5000.",
    ], 13.5),
    25: ("Deployment - Frontend - Define the Dockerfile", [
        "Base image: python:3.13.15-slim.",
        "Copies the Streamlit application and installs requests 2.34.2 and streamlit 1.64.0.",
        "Binds Streamlit to 0.0.0.0:8501 for Codespaces or container access.",
        "XSRF protection is disabled in the supplied deployment command to simplify external access.",
        "Set BACKEND_URL at runtime so the frontend can call the backend service or forwarded port.",
    ], 14),
    26: ("Deployment - Frontend - Push the Files to GitHub", [
        "Repository: github.com/raovutukoor1/aurora-stay-qna.",
        "Latest published commit: 3b34c59 — removed the handoff-owner line from slide 31.",
        "Key files: backend/app.py, backend/rag_utils.py, frontend/app.py, Dockerfiles, and requirements files.",
        "Evaluation files: notebooks/Aurora_Stays_RAG_POC_Local_final_executed.ipynb, reports/evaluation_results.md, and deepeval_options_cases.csv.",
        "Presentation files: build_aurora_presentation.py, the template-style PPTX, and the generated PDF packs.",
        "Credentials remain excluded: config/config.json, .env, and supplied token/config files are not tracked.",
    ], 13.5),
    27: ("Deployment - Inference - Define the Forwarded URL", [
        "In GitHub Codespaces, open the Ports panel and forward the Streamlit port 8501.",
        "The public URL is generated per Codespace session; it is not hard-coded in the repository.",
        "Set BACKEND_URL to the backend service URL or its forwarded port before starting Streamlit.",
        "Validated local endpoints: frontend http://localhost:8501 and backend http://127.0.0.1:5000.",
        "Production smoke tests used the parking-fee query and the ASH-CHI Loop pet-charge query.",
        "API keys belong in Codespaces Secrets or a private local config file, never in the forwarded URL or source code.",
    ], 14),
    28: ("Deployment - Inference - Relevant Chunks", [
        "Endpoint: POST {BACKEND_URL}/v1/relevant_chunks.",
        'Payload: {"query": "What are the parking fees at the Atlanta and Miami hotels?", "k": 5}.',
        "Response fields: query, k, and relevant_chunks.",
        "Response excerpt: relevant_chunks include ASH-ATL Midtown self parking at $28 per night and ASH-MIA Brickell valet at $55 per night.",
        "This endpoint is useful for checking retrieval quality before evaluating the generated answer.",
    ], 14),
    29: ("Deployment - Inference - Answer with Relevant Chunks", [
        "Endpoint: POST {BACKEND_URL}/v1/answer_with_relevant_chunks.",
        'Payload includes query, k=5, temperature=0.0, and max_tokens=512.',
        "Response fields include answer, relevant_chunks, model_name, and the generation parameters used.",
        "Answer excerpt: Atlanta Midtown self parking is $28 per night; Miami Brickell uses valet parking at $55 per night.",
        "The response also returns the supporting chunks for auditability.",
        "Use POST /v1/answer when only the final answer is required; use this endpoint for auditability.",
    ], 14),
    30: ("APPENDIX", [
        "Production checklist",
        "â€¢ Create config/config.json locally or set Codespaces Secrets for OPENAI_API_KEY and OPENAI_BASE_URL.",
        "â€¢ Start the backend first, confirm the readiness response, then start Streamlit on port 8501.",
        "â€¢ Run a property-specific query and an evidence-display query before wider use.",
        "Known limitation",
        "â€¢ The Accessible King case has Contextual Precision=.451 because the benchmark question is underspecified for the auto-answer strategy.",
        "â€¢ The focused clarification experiment reached Precision=.961; the benchmark question and retrieval baseline remain unchanged.",
    ], 13.5),
    31: ("Recommendations and Success Measures", [
        "Recommendation 1: pilot the k=5 baseline with evidence display enabled for support staff.",
        "Recommendation 2: route ambiguous or insufficient-context questions to assisted review rather than forcing an auto-answer.",
        "Recommendation 3: keep the benchmark question unchanged, but improve production question clarification and prompt handling.",
        "Track success through held-out pass rate, Contextual Precision, escalation rate, answer latency, and evidence-click usage.",
        "Before broader rollout, capture the live Codespaces URL and screenshots, then repeat the two production smoke tests.",
        "Decision: proceed with a controlled pilot; review the single known Accessible King failure before wider release.",
    ], 13.5),
}

for slide_number, (title, lines, size, *mono) in slides.items():
    slide = prs.slides[slide_number - 1]
    add_header_band(slide)
    # The template's APPENDIX title is intentionally positioned low on the
    # slide; move it into the common title position before body cleanup so it
    # is preserved as the title shape without crowding the appendix content.
    if slide_number == 30:
        set_title(slide, title)
        appendix_title = find_title_shape(slide)
        appendix_title.top = Inches(0.32)
    add_body(
        slide,
        lines,
        font_size=size,
        mono=bool(mono),
        body_top=1.58 if slide_number in {15, 18} else 1.45,
        add_header_line=True,
    )
    if slide_number != 30:
        set_title(slide, title)

prs.save(str(OUTPUT))
print(f"Saved {OUTPUT}")
print(f"Slides: {len(prs.slides)}")
