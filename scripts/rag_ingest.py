#!/usr/bin/env python3
"""
rag_ingest.py

Chunk a document, embed each chunk with BGE-M3, and upsert into Qdrant.
Standard library only.

Chunking strategy (tuned for Chinese documents):
  1. Split on Markdown headings first, so a chunk never straddles two
     sections. The heading path travels with the chunk as metadata AND is
     prepended to the embedded text — section titles carry a lot of the
     signal for questions like "X 的核心模組有哪些".
  2. Inside a section, pack whole paragraphs up to --chunk-size characters.
     Paragraphs are never split mid-way unless one alone exceeds the size,
     in which case it is split on sentence boundaries (。！？；).
  3. Consecutive chunks within a section carry --overlap characters of the
     previous chunk's tail (trimmed to a sentence boundary) so a definition
     split across a boundary is still retrievable from either side.

Chunk ids are uuid5(source + chunk_index), so re-ingesting the same
document overwrites its chunks rather than creating duplicates.

Usage:
  python3 rag_ingest.py --input psf-eim.md --collection psf_eim_kb
  python3 rag_ingest.py --input psf-eim.md --dry-run     # preview only
"""
import argparse
import json
import os
import re
import statistics
import sys
import uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rag_common import (  # noqa: E402
    QDRANT_URL,
    die,
    embed_batch,
    ensure_collection,
    http_json,
    log,
)

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？；!?;])")
TABLE_LINE_RE = re.compile(r"^\s*\|.*\|\s*$")
TABLE_SEPARATOR_RE = re.compile(r"^\s*\|[\s:|-]+\|\s*$")

# Meeting records run as a time-ordered sequence of discussion points. Packing
# purely by character count merges two unrelated agenda items into one chunk,
# so these markers force a chunk boundary regardless of how much room is left.
MEETING_BOUNDARY_RE = re.compile(
    r"^\s*(?:"
    r"\d{1,2}[:：]\d{2}"                     # 10:30 / 10：30
    r"|[（(]?[一二三四五六七八九十\d]+[）).、]"   # (一) / 1. / 3、
    r"|議題|議程|討論事項|決議|結論|行動項目|待辦|追蹤事項"
    r"|Action\s*Item|Decision|Agenda"
    r")"
)

# Per-document-type presets. Chunk size and overlap are what actually differ;
# table atomicity is unconditional because splitting a table mid-row is never
# right, whatever the document type.
DOC_TYPE_PRESETS = {
    # Sectioned planning/introduction docs — the PSF EIM baseline, validated
    # in docs/rag-findings.md.
    "structured": {"chunk_size": 400, "overlap": 80, "hard_boundary": None},
    # Meeting minutes: short conversational paragraphs, agenda-ordered. Smaller
    # chunks keep one discussion point per chunk; the boundary regex stops two
    # agenda items being merged just because they fit.
    "meeting": {"chunk_size": 250, "overlap": 40, "hard_boundary": MEETING_BOUNDARY_RE},
    # Due-diligence / investment reports: tables and figures need their
    # surrounding narrative, so allow a bit more room per chunk.
    "report": {"chunk_size": 500, "overlap": 80, "hard_boundary": None},
}
DEFAULT_DOC_TYPE = "structured"


def is_table_block(text: str) -> bool:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    return len(lines) >= 2 and all(TABLE_LINE_RE.match(ln) for ln in lines)


def split_table_block(block: str, chunk_size: int):
    """
    Split an over-long markdown table by rows, never mid-row, repeating the
    header in every part. Without the repeat, a continuation chunk holds bare
    numbers whose columns are defined in a chunk the retriever may never
    return — "0.8 億" with no way to tell which year it belongs to.
    """
    lines = [ln for ln in block.splitlines() if ln.strip()]
    header = lines[:1]
    if len(lines) > 1 and TABLE_SEPARATOR_RE.match(lines[1]):
        header = lines[:2]
    body = lines[len(header):]
    if not body:
        return [block]

    header_text = "\n".join(header)
    parts, buf = [], []
    for row in body:
        candidate = "\n".join(header + buf + [row])
        if buf and len(candidate) > chunk_size:
            parts.append("\n".join(header + buf))
            buf = [row]
        else:
            buf.append(row)
    if buf:
        parts.append("\n".join(header + buf))
    # A single row longer than chunk_size still beats splitting it in half.
    return parts or [header_text]
# Stable namespace so ids are reproducible across runs and machines.
ID_NAMESPACE = uuid.UUID("6f3d0c9e-1b6a-5c7e-9f2d-8a4b1c0d5e73")


def split_sentences(paragraph: str):
    parts = [s for s in SENTENCE_SPLIT_RE.split(paragraph) if s.strip()]
    return parts or [paragraph]


def tail_for_overlap(text: str, overlap: int) -> str:
    """Last `overlap` chars, trimmed forward to the next sentence boundary."""
    if overlap <= 0 or not text:
        return ""
    tail = text[-overlap:]
    for i, ch in enumerate(tail):
        if ch in "。！？；!?;\n" and i < len(tail) - 1:
            return tail[i + 1:].strip()
    return tail.strip()


def parse_sections(text: str, default_title: str):
    """-> [(heading_path, [paragraph, ...]), ...] preserving document order."""
    sections = []
    heading_stack = []
    current_paras = []
    current_path = default_title

    def flush():
        if any(p.strip() for p in current_paras):
            sections.append((current_path, [p for p in current_paras if p.strip()]))

    for raw_block in re.split(r"\n\s*\n", text):
        block = raw_block.strip("\n")
        if not block.strip():
            continue
        lines = block.split("\n")
        # A block may start with a heading line followed by body lines.
        buffered = []
        for line in lines:
            m = HEADING_RE.match(line)
            if m:
                if buffered:
                    current_paras.append("\n".join(buffered))
                    buffered = []
                flush()
                current_paras = []
                level, title = len(m.group(1)), m.group(2)
                heading_stack[:] = heading_stack[: level - 1]
                while len(heading_stack) < level - 1:
                    heading_stack.append("")
                heading_stack.append(title)
                current_path = " > ".join(h for h in heading_stack if h)
            else:
                buffered.append(line)
        if buffered:
            current_paras.append("\n".join(buffered))
    flush()
    return sections


def chunk_section(paragraphs, chunk_size, overlap, min_chunk, hard_boundary=None):
    """
    Pack paragraphs into chunk-sized pieces with sentence-safe overlap.

    Pieces are (text, force_new_chunk). force_new_chunk is set when a
    paragraph opens a new agenda item (hard_boundary) or is a continuation
    part of a split table — both are cases where merging with what came
    before would put two unrelated things in one chunk.
    """
    pieces = []
    for para in paragraphs:
        opens_item = bool(hard_boundary and hard_boundary.match(para.lstrip()))

        if is_table_block(para):
            parts = split_table_block(para, chunk_size)
            for i, part in enumerate(parts):
                # Every part after the first starts a new chunk, so a table
                # never spans a boundary without carrying its header.
                pieces.append((part, opens_item if i == 0 else True))
            continue

        if len(para) <= chunk_size:
            pieces.append((para, opens_item))
            continue

        buf, first = "", True
        for sent in split_sentences(para):
            while len(sent) > chunk_size:  # pathological single sentence
                pieces.append((sent[:chunk_size], opens_item and first))
                first = False
                sent = sent[chunk_size:]
            if len(buf) + len(sent) <= chunk_size:
                buf += sent
            else:
                if buf:
                    pieces.append((buf, opens_item and first))
                    first = False
                buf = sent
        if buf:
            pieces.append((buf, opens_item and first))

    chunks, buf = [], ""
    buf_forced = False   # did the current buffer start at a hard boundary?
    for text, force_new in pieces:
        if buf and force_new:
            chunks.append(buf)
            buf, buf_forced = text, True
            continue
        candidate = (buf + "\n" + text) if buf else text
        if len(candidate) <= chunk_size or not buf:
            if not buf:
                buf_forced = force_new
            buf = candidate
        else:
            chunks.append(buf)
            carry = tail_for_overlap(buf, overlap)
            buf = (carry + "\n" + text) if carry else text
            buf_forced = False
    if buf:
        # Fold a runt tail back into the previous chunk instead of storing it
        # alone — a 20-character orphan chunk retrieves badly. Two exemptions:
        # a table part (folding re-creates the split-table problem) and a
        # piece that deliberately opened a new agenda item (folding a short
        # 決議 back into the preceding 議題 is exactly the merge the hard
        # boundary exists to prevent).
        if (chunks and len(buf) < min_chunk
                and not is_table_block(buf) and not buf_forced):
            chunks[-1] = chunks[-1] + "\n" + buf
        else:
            chunks.append(buf)
    return chunks


def build_chunks(text, source, chunk_size, overlap, min_chunk, extra_metadata,
                 doc_type=DEFAULT_DOC_TYPE, project=None, hard_boundary=None):
    default_title = os.path.splitext(os.path.basename(source))[0]
    sections = parse_sections(text, default_title)
    if not sections:
        die(f"從 {source} 解析不到任何內容,請確認檔案不是空的。")

    records, index = [], 0
    for heading_path, paragraphs in sections:
        for body in chunk_section(paragraphs, chunk_size, overlap, min_chunk,
                                  hard_boundary):
            embedded_text = f"【{heading_path}】\n{body}" if heading_path else body
            records.append(
                {
                    # Namespaced by project so two projects can hold
                    # same-named files without their chunk ids colliding.
                    "id": str(uuid.uuid5(
                        ID_NAMESPACE, f"{project or ''}::{source}::{index}")),
                    "text": embedded_text,
                    "payload": {
                        "text": embedded_text,
                        "body": body,
                        # --- provenance: also the fields Phase 2 will filter
                        # on for Identity/Permission isolation, recorded now so
                        # the corpus doesn't need re-chunking and re-ingesting
                        # when that lands ---
                        "source": source,
                        "doc_type": doc_type,
                        "project": project or "",
                        # --- structure ---
                        "heading_path": heading_path,
                        "chunk_index": index,
                        "char_count": len(body),
                        # "contains a table row", not "is entirely a table" —
                        # a chunk of prose plus its table should still be
                        # findable as tabular content.
                        "has_table": any(TABLE_LINE_RE.match(ln)
                                         for ln in body.splitlines()),
                        **extra_metadata,
                    },
                }
            )
            index += 1
    return records


def report_chunks(records):
    sizes = [r["payload"]["char_count"] for r in records]
    log(f"切分完成:共 {len(records)} 個 chunk")
    log(
        f"  字數分布: 最小 {min(sizes)} / 中位數 {int(statistics.median(sizes))} "
        f"/ 平均 {int(statistics.mean(sizes))} / 最大 {max(sizes)}"
    )
    headings = []
    for r in records:
        hp = r["payload"]["heading_path"]
        if hp not in headings:
            headings.append(hp)
    log(f"  涵蓋 {len(headings)} 個章節")


def preview(records, limit):
    print("\n=== Chunk 預覽 ===")
    for r in records[:limit]:
        p = r["payload"]
        print(f"\n[#{p['chunk_index']}] {p['heading_path']}  ({p['char_count']} 字)")
        body = p["body"]
        print(f"  {body[:160]}{'…' if len(body) > 160 else ''}")
    if len(records) > limit:
        print(f"\n... 其餘 {len(records) - limit} 個 chunk 未顯示"
              f"(用 --preview-count 調整)")
    print("==================\n")


def upsert(collection, records, batch_size):
    texts = [r["text"] for r in records]
    vectors = embed_batch(texts)
    points = [
        {"id": r["id"], "vector": v, "payload": r["payload"]}
        for r, v in zip(records, vectors)
    ]
    written = 0
    for start in range(0, len(points), batch_size):
        batch = points[start:start + batch_size]
        status, resp = http_json(
            "PUT",
            f"{QDRANT_URL}/collections/{collection}/points?wait=true",
            {"points": batch},
        )
        if status not in (200, 201):
            die(f"寫入 Qdrant 失敗 (HTTP {status}): {resp}")
        written += len(batch)
        log(f"已寫入 {written}/{len(points)} 個 chunk 到 '{collection}'。")
    return written


def main():
    ap = argparse.ArgumentParser(description="Chunk + embed + upsert a document into Qdrant.")
    ap.add_argument("--input", required=True, help="來源文件路徑 (.md / .txt)")
    ap.add_argument("--collection", default="psf_eim_kb", help="Qdrant collection 名稱")
    ap.add_argument("--source", default=None, help="metadata 中的來源名稱(預設用檔名)")
    ap.add_argument("--doc-type", default=DEFAULT_DOC_TYPE, choices=sorted(DOC_TYPE_PRESETS),
                    help="文件類型，決定 chunk 策略預設值："
                         "structured=有章節的規劃/介紹文件 (400/80)、"
                         "meeting=會議記錄，較小 chunk 且議程標記強制斷開 (250/40)、"
                         "report=投資評估/盡職調查，容納表格與前後敘述 (500/80)。"
                         "表格永不從列中間切開，與類型無關")
    ap.add_argument("--project", default=None,
                    help="所屬專案/案子名稱。寫入 metadata，供跨文件混淆測試與"
                         "Phase 2 權限隔離過濾使用")
    ap.add_argument("--chunk-size", type=int, default=None,
                    help="每個 chunk 目標字數。未指定時依 --doc-type 取預設")
    ap.add_argument("--overlap", type=int, default=None,
                    help="chunk 之間重疊字數。未指定時依 --doc-type 取預設")
    ap.add_argument("--min-chunk", type=int, default=100, help="小於此字數的尾段併回前一個 chunk")
    ap.add_argument("--metadata", default="{}", help='額外 metadata JSON,例如 \'{"department":"PSF"}\'')
    ap.add_argument("--batch-size", type=int, default=32, help="每次寫入 Qdrant 的點數")
    ap.add_argument("--dry-run", action="store_true", help="只切分並預覽,不 embedding、不寫入")
    ap.add_argument("--preview-count", type=int, default=5, help="預覽顯示幾個 chunk")
    args = ap.parse_args()

    if not os.path.isfile(args.input):
        die(f"找不到輸入檔案: {args.input}")

    preset = DOC_TYPE_PRESETS[args.doc_type]
    chunk_size = args.chunk_size if args.chunk_size is not None else preset["chunk_size"]
    overlap = args.overlap if args.overlap is not None else preset["overlap"]
    hard_boundary = preset["hard_boundary"]
    if overlap >= chunk_size:
        die(f"--overlap ({overlap}) 必須小於 --chunk-size ({chunk_size})。")

    try:
        extra_metadata = json.loads(args.metadata)
        if not isinstance(extra_metadata, dict):
            raise ValueError("must be a JSON object")
    except (json.JSONDecodeError, ValueError) as e:
        die(f"--metadata 不是合法的 JSON 物件: {e}")

    with open(args.input, "r", encoding="utf-8") as f:
        text = f.read()

    source = args.source or os.path.basename(args.input)
    log(f"讀取 {args.input}(共 {len(text)} 字元)")
    log(f"文件類型 {args.doc_type}"
        f"{'(專案: ' + args.project + ')' if args.project else ''}"
        f"，chunk_size={chunk_size}, overlap={overlap}"
        f"{'，議程標記強制斷開' if hard_boundary else ''}")

    records = build_chunks(
        text, source, chunk_size, overlap, args.min_chunk, extra_metadata,
        doc_type=args.doc_type, project=args.project, hard_boundary=hard_boundary
    )
    report_chunks(records)
    preview(records, args.preview_count)

    if args.dry_run:
        log("--dry-run:未執行 embedding,也沒有寫入 Qdrant。")
        return

    ensure_collection(args.collection)
    written = upsert(args.collection, records, args.batch_size)
    log(f"完成:{written} 個 chunk 已寫入 collection '{args.collection}'(來源: {source})。")


if __name__ == "__main__":
    main()
