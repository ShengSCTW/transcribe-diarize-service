from fastapi import FastAPI, File, UploadFile
from whisperx import load_audio, transcribe
from pyannote.audio import Pipeline

app = FastAPI()
diar = Pipeline.from_pretrained("pyannote/speaker-diarization")

@app.post("/process")
async def process(file: UploadFile = File(...)):
    tmp = f"/tmp/{file.filename}"
    with open(tmp, "wb") as f: f.write(await file.read())

    audio = load_audio(tmp)
    res = transcribe("large-v3", audio)
    text = res["text"]

    diarization = []
    for turn, _, spk in diar(tmp).itertracks(yield_label=True):
        diarization.append({
          "start": turn.start, "end": turn.end, "speaker": spk
        })

    return {"text": text, "diarization": diarization}
