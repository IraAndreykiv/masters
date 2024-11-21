from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
import numpy as np
import joblib


# Створюємо FastAPI додаток
app = FastAPI()

# Завантажуємо модель
model = joblib.load('titanic_model.pkl')

# Клас для валідації вхідних даних
class PassengerData(BaseModel):
    Pclass: int
    Sex: str
    Age: float
    SibSp: int
    Parch: int
    Fare: float
    Embarked: str

    @validator('Sex')
    def check_sex(cls, value):
        if value not in ['male', 'female']:
            raise ValueError('Sex must be either "male" or "female".')
        return value

    @validator('Embarked')
    def check_embarked(cls, value):
        if value not in ['C', 'Q', 'S']:
            raise ValueError('Embarked must be either "C", "Q", or "S".')
        return value

# Функція для попередньої обробки даних
def preprocess_data(data: PassengerData):
    sex = 0 if data.Sex == 'male' else 1
    embarked = 0 if data.Embarked == 'C' else 1 if data.Embarked == 'Q' else 2
    return np.array([[data.Pclass, sex, data.Age, data.SibSp, data.Parch, data.Fare, embarked]])

# Ендпоінт для передбачення
@app.post("/predict")
def predict_survival(passenger: PassengerData):
    input_data = preprocess_data(passenger)
    try:
        prediction = model.predict(input_data)
        return {"Survived": int(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Додайте базовий ендпоінт
@app.get("/")
def read_root():
    return {"message": "Welcome to the Titanic Survival Prediction API!"}

# Запуск сервера
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
