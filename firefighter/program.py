import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
import queue
import torch
import lancedb
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import warnings
import time
from typing import Tuple, List
import fitz  # PyMuPDF for PDF reading
from sentence_transformers import SentenceTransformer
import os

warnings.filterwarnings("ignore")

# Determine device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def remove_last_word_if(text, word):
    words = text.strip().rstrip('.!?').split()
    if words and words[-1].lower() == word.lower():
        words.pop()
    return ' '.join(words)

def ingest_pdf_to_lancedb(pdf_path: str, db_path: str, table_name: str):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at {pdf_path}")
    
    print("Ingesting PDF to LanceDB... This may take a minute.")
    doc = fitz.open(pdf_path)
    all_text = []
    for page in doc:
        text = page.get_text().strip()  # type: ignore[attr-defined]
        if text:
            all_text.append(text)
    doc.close()

    chunk_size = 500
    overlap = 100
    chunks = []
    for text in all_text:
        words = text.split()
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            if chunk:
                chunks.append(chunk)

    print(f"Total chunks created: {len(chunks)}")

    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedder.encode(chunks, show_progress_bar=True)

    db = lancedb.connect(db_path)

    if table_name in db.table_names():
        table = db.open_table(table_name)
        print("Table exists, appending new records...")
    else:
        table = db.create_table(
            table_name,
            data=[{
                "vector": embeddings[0],
                "text": chunks[0],
                "metadata": {"source": "Firefighter I Textbook"}
            }],
            mode="overwrite"
        )
        embeddings = embeddings[1:]
        chunks = chunks[1:]

    records = [
        {
            "vector": emb,
            "text": chunk,
            "metadata": {"source": "Firefighter I Textbook"}
        }
        for emb, chunk in zip(embeddings, chunks)
    ]
    table.add(records)
    print("PDF ingestion completed and stored in LanceDB.")
    return table

def init_db():
    db_path = "C:/Users/Judah/Documents/Projects/Firefighter AI/data/lancedb"
    table_name = "firefighter_textbook"
    db = lancedb.connect(db_path)

    if table_name in db.table_names():
        return db.open_table(table_name)
    else:
        pdf_path = "C:/Users/Judah/Documents/Projects/Firefighter AI/data/book/firefighter_txtbook.pdf"
        return ingest_pdf_to_lancedb(pdf_path, db_path, table_name)

def get_context(query: str, table, num_results: int = 5) -> Tuple[str, List[str]]:
    actual = table.search(query).limit(num_results).to_list()
    contexts = []
    urls = []
    for record in actual:
        text = record.get('text', '')
        if not isinstance(text, str):
            text = str(text)
        contexts.append(text)
        metadata = record.get('metadata', {})
        url = metadata.get('url', '')
        urls.append(url)
    return "\n\n".join(contexts), urls

# Load Flan-T5 model and tokenizer
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base").to(device)
    return model, tokenizer

# Use Flan-T5 to generate response
def generate_words_fast(messages):
    combined_prompt = ""
    for msg in messages:
        combined_prompt += msg.get("content", "") + "\n"

    inputs = tokenizer(
        combined_prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to(device)

    outputs = res_model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=128,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        num_beams=1,
        early_stopping=True,
    )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text.strip()

# Initialize LanceDB and Flan-T5
table = init_db()
res_model, tokenizer = load_model()

# Whisper voice model setup
model_size = "medium"
model1 = WhisperModel(model_size, compute_type="int8", device='cuda')
model3 = WhisperModel(model_size, compute_type="int8", device='cuda')

samplerate = 16000
blocksize = 4000
audio_queue1 = queue.Queue()
audio_queue3 = queue.Queue()

def callback1(indata, frames, time, status):
    audio_queue1.put(indata.copy())

def callback3(indata, frames, time, status):
    audio_queue3.put(indata.copy())

stream1 = sd.InputStream(samplerate=samplerate, channels=1, callback=callback1, blocksize=blocksize)
stream3 = sd.InputStream(samplerate=samplerate, channels=1, callback=callback3, blocksize=blocksize)

stream1.start()

rec_int = False
start_word = 'oscar'
stop_word = 'cancel'
end_word = 'go'

print("Listening with Whisper... (Ctrl+C to stop)")

try:
    buffer1 = np.empty((0,), dtype=np.float32)
    buffer3 = np.empty((0,), dtype=np.float32)
    while True:
        block1 = audio_queue1.get()
        block1 = block1.flatten()
        buffer1 = np.concatenate((buffer1, block1))

        if rec_int:
            try:
                while True:
                    block3 = audio_queue3.get_nowait()
                    block3 = block3.flatten()
                    buffer3 = np.concatenate((buffer3, block3))
            except queue.Empty:
                pass

        seconds = 2
        if len(buffer1) >= samplerate * seconds:
            segment1 = buffer1[:samplerate * seconds]
            buffer1 = buffer1[samplerate * seconds:]

            segments1, _ = model1.transcribe(segment1, language="en")
            segments1 = list(segments1)
            if segments1:
                txt = segments1[-1].text.strip().lower()

                if not rec_int and start_word in txt:
                    stream3.start()
                    buffer3 = np.empty((0,), dtype=np.float32)
                    rec_int = True
                    print('Oscar Recognized\n')

                elif rec_int and stop_word in txt:
                    print('Cancelled')
                    rec_int = False
                    stream3.stop()

                elif rec_int and end_word in txt:
                    print('\nGO Recognized')

                    try:
                        while True:
                            block3 = audio_queue3.get_nowait()
                            block3 = block3.flatten()
                            buffer3 = np.concatenate((buffer3, block3))
                    except queue.Empty:
                        pass

                    stream3.stop()

                    segment3 = buffer3
                    segments3, _ = model3.transcribe(segment3, language="en")
                    segments3 = list(segments3)
                    if segments3:
                        prompt = ' '.join(seg.text.strip() for seg in segments3)
                        prompt = remove_last_word_if(prompt, "go")
                        print(prompt)

                        start_time = time.time()
                        context, urls = get_context(prompt, table)
                        for url in urls[:3]:
                            if url:
                                print(url)

                        system_prompt = f"""You are a helpful Firefighter I exam study guide named {start_word}, giving short two sentence answers to questions. \
Use the context below to help answer only with verified information from the Firefighter I textbook:
{context}"""
                        messages = [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt},
                        ]

                        response = generate_words_fast(messages)
                        print(response)
                        rec_int = False
                        elapsed_time = time.time() - start_time
                        print(f"Generated in {elapsed_time:.4f} seconds")

except KeyboardInterrupt:
    print("\nExiting...")

except Exception as e:
    print(f"Error occurred: {e}")

finally:
    if stream1.active:
        stream1.stop()
    stream1.close()

    if stream3.active:
        stream3.stop()
    stream3.close()
