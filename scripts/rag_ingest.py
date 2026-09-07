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


def chunk_section(paragraphs, chunk_size, overlap, min_chunk):
    """Pack paragraphs into chunk-sized pieces with sentence-safe overlap."""
    pieces = []
    for para in paragraphs:
        if len(para) <= chunk_size:
            pieces.append(para)
            continue
        buf = ""
        for sent in split_sentences(para):
            while len(sent) > chunk_size:  # pathological single sentence
                pieces.append(sent[:chunk_size])
                sent = sent[chunk_size:]
            if len(buf) + len(sent) <= chunk_size:
                buf += sent
            else:
                if buf:
                    pieces.append(buf)
                buf = sent
        if buf:
            pieces.append(buf)

    chunks, buf = [], ""
    for piece in pieces:
        candidate = (buf + "\n" + piece) if buf else piece
        if len(candidate) <= chunk_size or not buf:
            buf = candidate
        else:
            chunks.append(buf)
            carry = tail_for_overlap(buf, overlap)
            buf = (carry + "\n" + piece) if carry else piece
    if buf:
        # Fold a runt tail back into the previous chunk instead of storing it
        # alone — a 20-character orphan chunk retrieves badly.
        if chunks and len(buf) < min_chunk:
            chunks[-1] = chunks[-1] + "\n" + buf
        else:
            chunks.append(buf)
    return chunks


def build_chunks(text, source, chunk_size, overlap, min_chunk, extra_metadata):
    default_title = os.path.splitext(os.path.basename(source))[0]
    sections = parse_sections(text, default_title)
    if not sections:
        die(f"從 {source} 解析不到任何內容,請確認檔案不是空的。")

    records, index = [], 0
    for heading_path, paragraphs in sections:
        for body in chunk_section(paragraphs, chunk_size, overlap, min_chunk):
            embedded_text = f"【{heading_path}】\n{body}" if heading_path else body
            records.append(
                {
                    "id": str(uuid.uuid5(ID_NAMESPACE, f"{source}::{index}")),
                    "text": embedded_text,
                    "payload": {
                        "text": embedded_text,
                        "body": body,
                        "source": source,
                        "heading_path": heading_path,
                        "chunk_index": index,
                        "char_count": len(body),
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
    ap.add_argument("--chunk-size", type=int, default=400, help="每個 chunk 目標字數 (預設 400)")
    ap.add_argument("--overlap", type=int, default=80, help="chunk 之間重疊字數 (預設 80)")
    ap.add_argument("--min-chunk", type=int, default=100, help="小於此字數的尾段併回前一個 chunk")
    ap.add_argument("--metadata", default="{}", help='額外 metadata JSON,例如 \'{"department":"PSF"}\'')
    ap.add_argument("--batch-size", type=int, default=32, help="每次寫入 Qdrant 的點數")
    ap.add_argument("--dry-run", action="store_true", help="只切分並預覽,不 embedding、不寫入")
    ap.add_argument("--preview-count", type=int, default=5, help="預覽顯示幾個 chunk")
    args = ap.parse_args()

    if not os.path.isfile(args.input):
        die(f"找不到輸入檔案: {args.input}")
    if args.overlap >= args.chunk_size:
        die(f"--overlap ({args.overlap}) 必須小於 --chunk-size ({args.chunk_size})。")

    try:
        extra_metadata = json.loads(args.metadata)
        if not isinstance(extra_metadata, dict):
            raise ValueError("must be a JSON object")
    except (json.JSONDecodeError, ValueError) as e:
        die(f"--metadata 不是合法的 JSON 物件: {e}")

    with open(args.input, "r", encoding="utf-8") as f:
        text = f.read()

    source = args.source or os.path.basename(args.input)
    log(f"讀取 {args.input}(共 {len(text)} 字元),chunk_size={args.chunk_size}, overlap={args.overlap}")

    records = build_chunks(
        text, source, args.chunk_size, args.overlap, args.min_chunk, extra_metadata
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
