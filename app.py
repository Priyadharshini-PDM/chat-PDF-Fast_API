from fastapi import FastAPI,UploadFile, File, Query,Body
from fastapi.responses import JSONResponse, RedirectResponse,ORJSONResponse,StreamingResponse
from typing import List
from pydantic import BaseModel
from langchain.text_splitter import CharacterTextSplitter
from PyPDF2 import PdfReader


app = FastAPI()

class ItemList(BaseModel):
    text: List[List[str]]

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks=text_splitter.split_text(text)
    # chunks = text_splitter.split_text(text)
    # print(chunks)
    return chunks

@app.get("/get_pdf_text/")
def read_root():
    return {"Hello": "World"}

@app.post("/get_pdf_text/")
def get_pdf_text(files: list[UploadFile]):
    text = ''
    for pdf in files:
        pdf_reader = PdfReader(pdf.file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    text_chunks = get_text_chunks(text)
    print(text_chunks)

    return StreamingResponse(text_chunks, media_type="text/event-stream")