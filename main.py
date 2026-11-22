from typing import Optional

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
    name: Optional[str] = None


class IrisInput(BaseModel):
    panjang_kelopak: float
    lebar_kelopak: float
    panjang_mahkota: float
    lebar_mahkota: float


@app.post("/new_items")
def create_item(data: Data):
    return {"new added items": data}


@app.get("/")
def get_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def get_item(item_id: int, name: Optional[str] = None):
    return {"item_id": item_id, "name": name}


@app.post("/iris")
def predict_iris(data: IrisInput):
    prediction = iris_clf.predict(
        [
            [
                data.panjang_kelopak,
                data.lebar_kelopak,
                data.panjang_mahkota,
                data.lebar_mahkota,
            ]
        ]
    )

    return {"Prediction results": prediction[0]}


@app.post("/cat_dog")
def predict_cat_or_dog(file: UploadFile = File(...)):
    img = io.imread(file.file)
    resized_img = transform.resize(img, (100, 100))

    image = torch.tensor([np.transpose(resized_img).astype(np.float32)])
    outputs = image_clf(image)
    predicted = torch.max(outputs, 1)[1][0]

    return {"Prediction results": "kocheng" if int(predicted) == 0 else "anjing"}
