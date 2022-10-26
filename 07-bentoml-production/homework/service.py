import bentoml

import numpy as np

from bentoml.io import JSON, NumpyNdarray

from pydantic import BaseModel

class UserProfile(BaseModel):
    name: str
    age: int
    country: str
    rating: float

# model_ref = bentoml.sklearn.get("mlzoomcamp_homework:qtzdz3slg6mwwdu5") # Model 1
model_ref = bentoml.sklearn.get("mlzoomcamp_homework:jsi67fslz6txydu5") # Model 2

model_runner = model_ref.to_runner()

svc = bentoml.Service("mlzoomcamp_homework_classifier", runners=[model_runner])

@svc.api(input=NumpyNdarray(shape=(-1,4), dtype=np.float32, enforce_dtype=True, enforce_shape=True), output=JSON())
async def classify(vector):
    prediction = await model_runner.predict.async_run(vector)
    print(prediction)