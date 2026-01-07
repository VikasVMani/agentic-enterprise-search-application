import os, uuid
import pdfplumber

def extract_pdf_pagewise(pdf_path: str):
    """
    Extract text page-wise from a PDF using pdfplumber.
    Each page = one chunk.
    """
    document_name = os.path.basename(pdf_path)
    records = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_no, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()

            if not text or not text.strip():
                continue

            records.append({
                "document_name": document_name,
                "page_no": page_no,
                "text": text.strip()
            })

    return records


PARTITION_MAP = {'International Program License Agreement.pdf':"International_Program_License_Agreement", 
 'IBM Standard Terms and Conditions.pdf':'IBM_Standard_Terms_and_Conditions', 
 'International Agreement for Acquisition of Software Maintenance.pdf':'International_Agreement_for_Acquisition_of_Software_Maintenance', 
 'IBM PurchaseTerms.pdf':'IBM_PurchaseTerms'
 }


def prepare_chunks(data_dir):

    prepared = []
    all_chunks = []
    for file in os.listdir(data_dir):
        raw_chunks = extract_pdf_pagewise(data_dir+'/'+file)
        all_chunks += raw_chunks
    
    # print(all_chunks)
    # print(len(all_chunks))

    for c in all_chunks:
        prepared.append({
            "id": str(uuid.uuid4()),
            "document_name": c["document_name"],
            "page_no": c["page_no"],
            "text": c["text"],
            "partition": PARTITION_MAP.get(
                c["document_name"], "general"
            )
        })

    return prepared