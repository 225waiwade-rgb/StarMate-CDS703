import json
from pathlib import Path
from pypdf import PdfReader


def recursive_split(
    text: str,
    chunk_size: int = 800,
    chunk_overlap: int = 80,
    separators: tuple = ("\n\n", "\n", ". ", " ", ""),
) -> list[str]:
    if len(text) <= chunk_size:
        return [text.strip()] if text.strip() else []

    sep = separators[0]
    next_seps = separators[1:] if len(separators) > 1 else ("",)
    parts = text.split(sep) if sep else list(text)

    chunks, current = [], ""
    for part in parts:
        piece = (part + sep) if sep else part
        if len(current) + len(piece) <= chunk_size:
            current += piece
        else:
            if current.strip():
                chunks.append(current.strip())
            if len(piece) > chunk_size and len(separators) > 1:
                chunks.extend(recursive_split(piece, chunk_size, chunk_overlap, next_seps))
                current = ""
            else:
                overlap = current[-chunk_overlap:] if chunk_overlap > 0 else ""
                current = overlap + piece
    if current.strip():
        chunks.append(current.strip())
    return chunks


def ingest_pdfs(
    pdf_dir: str = "data/raw_pdfs",
    output_path: str = "data/chunks.json",
    chunk_size: int = 800,
    chunk_overlap: int = 80,
) -> list[dict]:
    chunks = []
    empty_files = []
    pdf_files = sorted(Path(pdf_dir).rglob("*.pdf"))

    for pdf_path in pdf_files:
        course_name = pdf_path.parent.name
        reader = PdfReader(pdf_path)
        file_chunks = 0
        for page_idx, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if not text:
                continue
            pieces = recursive_split(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            for chunk_idx, piece in enumerate(pieces, start=1):
                safe_id = f"{course_name}_{pdf_path.stem}_p{page_idx}_c{chunk_idx}".replace(" ", "_")
                chunks.append(
                    {
                        "chunk_id": safe_id,
                        "course": course_name,
                        "source_file": pdf_path.name,
                        "page_number": page_idx,
                        "text": piece,
                    }
                )
                file_chunks += 1
        if file_chunks == 0:
            empty_files.append(str(pdf_path))

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    if empty_files:
        print("Warning: No extractable text found in the following PDFs:", empty_files)
    return chunks


if __name__ == "__main__":
    all_chunks = ingest_pdfs()
    print(f"Successfully parsed PDFs into {len(all_chunks)} chunks. Saved to data/chunks.json")
    courses = {}
    for c in all_chunks:
        courses[c["course"]] = courses.get(c["course"], 0) + 1
    print("Chunk distribution by course:", courses)