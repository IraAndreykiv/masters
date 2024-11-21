import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from pyswarm import pso
import joblib
from sklearn.metrics import accuracy_score
import time

# Завантаження даних
train_data = pd.read_csv('train.csv')

# Обробка відсутніх даних
imputer = SimpleImputer(strategy='mean')
train_data['Age'] = imputer.fit_transform(train_data[['Age']])
train_data['Embarked'].fillna(train_data['Embarked'].mode()[0], inplace=True)

# Кодування категоріальних змінних
le = LabelEncoder()
train_data['Sex'] = le.fit_transform(train_data['Sex'])
train_data['Embarked'] = le.fit_transform(train_data['Embarked'])

# Вибір основних ознак
X = train_data.drop(columns=['Survived', 'PassengerId', 'Name', 'Ticket', 'Cabin'])
y = train_data['Survived']

# Поділ даних на тренувальні та валідаційні набори
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Функція для оптимізації гіперпараметрів
def objective_function(params):
    n_estimators = int(params[0])
    max_depth = int(params[1])
    learning_rate = params[2]

    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    return -accuracy_score(y_val, y_pred)  # Мінімізуємо негативну точність

# Межі для параметрів 
bounds = [(50, 200), (2, 10), (0.01, 0.3)]

# Запуск таймера
start_time = time.time()

# Запуск PSO для пошуку найкращих гіперпараметрів
best_params, best_score = pso(objective_function, lb=[50, 2, 0.01], ub=[200, 10, 0.3], swarmsize=10, maxiter=10)

# Вимірювання часу
end_time = time.time()
elapsed_time = end_time - start_time

print(f"Найкращі параметри: {best_params}")
print(f"Найкраща точність: {-best_score}")
print(f"Час виконання: {elapsed_time:.2f} секунд")  # Виводимо час виконання

# Навчання моделі з найкращими параметрами
final_model = XGBClassifier(
    n_estimators=int(best_params[0]),
    max_depth=int(best_params[1]),
    learning_rate=best_params[2],
    random_state=42
)
final_model.fit(X_train, y_train)

# Збереження моделі
joblib.dump(final_model, 'titanic_model.pkl')
