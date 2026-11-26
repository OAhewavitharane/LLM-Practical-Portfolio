import re
import unicodedata
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def clean_text(text):
    # the guide's fonts break "fi", "ft" and "ti" (found this in the Lab 5 tests)
    text = unicodedata.normalize("NFKC", text)  # turns the ligature character "ﬁ" into "fi"
    text = re.sub(r"/f_(\w)", r"f\1", text)      # "/f_ive" -> "five"
    text = text.replace("So�ware", "Software")
    text = text.replace("�", "ti")          # every other broken one in this PDF was "ti"
    return text


def load_pages(pdf_path):
    pages = []
    for page_no, page in enumerate(PdfReader(pdf_path).pages, start=1):
        text = page.extract_text()
        if text:
            pages.append((page_no, clean_text(text)))
    return pages


# method 1 (baseline): same as ingest.py, the whole text cut every 1000 characters
def fixed_chunks(pages):
    text = "\n".join(text for page_no, text in pages)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return [{"text": piece} for piece in splitter.split_text(text)]


def is_heading(line):
    line = line.strip()
    if re.match(r"^MSC\d{4}:", line):
        return True
    title = re.sub(r"\d+$", "", line)  # page numbers get stuck to titles, e.g. "COURSE DESCRIPTIONS05"
    return len(title) >= 6 and title.isupper() and not any(ch.isdigit() for ch in title)


# method 2: one chunk per section of the guide, cut at its headings
def section_chunks(pages, max_chars=2500):
    sections = []
    title, body, start_page = "", [], 1
    for page_no, text in pages:
        for line in text.splitlines():
            if is_heading(line):
                if body:
                    sections.append((title, "\n".join(body), start_page))
                    title, body, start_page = line.strip(), [], page_no
                else:
                    # two heading lines in a row are one title split over two lines
                    title = (title + " " + line.strip()).strip()
            elif line.strip():
                body.append(line)
    if body:
        sections.append((title, "\n".join(body), start_page))

    # only very long sections get cut, and every piece keeps its section title
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = []
    for title, body, page_no in sections:
        if len(body) <= max_chars:
            chunks.append({"text": title + "\n" + body, "page": page_no})
        else:
            for piece in splitter.split_text(body):
                chunks.append({"text": title + "\n" + piece, "page": page_no})
    return chunks


# method 3: small child chunks are searched, but the whole page (the parent) is sent to the model
def child_chunks(pages):
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    children = []
    for page_no, text in pages:
        for piece in splitter.split_text(text):
            children.append({"text": piece, "page": page_no})
    return children
