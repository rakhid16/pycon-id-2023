import numpy as np
import torch

from fastapi import FastAPI, File, UploadFile
from joblib import load
from pydantic import BaseModel
from skimage import io, transform

app = FastAPI()


iris_clf = load("models/iris.pkl")
image_clf = load("models/image.pkl")


class Data(BaseModel):
    item_id: int
    name: str = None


class IrisInput(BaseModel):
    panjang_kelopak: int
    lebar_kelopak: int
    panjang_mahkota: int
    lebar_mahkota: int


@app.post("/new_items")
def read_item(data: Data):
    return {"new added items": data}


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, name: str = None):
    return {"item_id": item_id, "name": name}


@app.post("/iris")
def read_item(data: IrisInput):
    hasil_prediksi = iris_clf.predict(
        [
            [
                data.panjang_kelopak,
                data.lebar_kelopak,
                data.panjang_mahkota,
                data.lebar_mahkota,
            ]
        ]
    )

    return {"Prediction results": hasil_prediksi[0]}


@app.post("/cat_dog")
def read_item(file: UploadFile = File(...)):
    img = io.imread(file.file)
    resized_img = transform.resize(img, (100, 100))

    outputs = image_clf(torch.tensor([np.transpose(resized_img).astype(np.float32)]))
    predicted = torch.max(outputs, 1)[1][0]

    return {"Prediction results": "kocheng" if int(predicted) == 0 else "anjing"}
