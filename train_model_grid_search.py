import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
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

# Налаштування параметрів для Grid Search
param_grid = {
    'n_estimators': [50, 100, 150, 200],
    'max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
    'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3]
}

# Створення моделі
model = XGBClassifier(random_state=42)

# Запуск таймера перед Grid Search
start_time = time.time()

# Виконання Grid Search для оптимізації гіперпараметрів
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=5)
grid_search.fit(X_train, y_train)

# Вимірювання часу виконання
end_time = time.time()
execution_time = end_time - start_time  # Час виконання в секундах

# Виведення найкращих параметрів та їх точності
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print(f"Найкращі параметри: {best_params}")
print(f"Найкраща точність на тренувальному наборі: {best_score}")
print(f"Час виконання Grid Search: {execution_time:.2f} секунд")  # Виведення часу виконання


# Навчання моделі з найкращими параметрами на всіх тренувальних даних
final_model = XGBClassifier(
    n_estimators=best_params['n_estimators'],
    max_depth=best_params['max_depth'],
    learning_rate=best_params['learning_rate'],
    random_state=42
)
final_model.fit(X_train, y_train)

# Оцінка моделі на валідаційному наборі
y_pred = final_model.predict(X_val)
validation_accuracy = accuracy_score(y_val, y_pred)
print(f"Точність на валідаційному наборі: {validation_accuracy}")

# Збереження моделі
joblib.dump(final_model, 'titanic_model_2.pkl')
