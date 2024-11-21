import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
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


# Алгоритм штучних бджіл (ABC)
def abc_algorithm(objective, bounds, n_bees=20, n_iterations=50):
    # Ініціалізація
    n_params = len(bounds)
    best_solution = None
    best_score = float('inf')

    # Генерація початкової популяції
    solutions = np.random.rand(n_bees, n_params)
    for i in range(n_params):
        solutions[:, i] = bounds[i][0] + solutions[:, i] * (bounds[i][1] - bounds[i][0])

    # Основний цикл ABC
    for iteration in range(n_iterations):
        for i in range(n_bees):
            # Генерація нового кандидата
            new_solution = solutions[i] + np.random.normal(0, 0.1, n_params)
            for j in range(n_params):
                new_solution[j] = np.clip(new_solution[j], bounds[j][0], bounds[j][1])

            # Оцінка нової кандидати
            score = objective(new_solution)

            # Порівняння з поточним кандидатом
            if score < best_score:
                best_score = score
                best_solution = new_solution
                solutions[i] = new_solution

    return best_solution, best_score

# Межі для параметрів
bounds = [(50, 200), (2, 10), (0.01, 0.3)]

# Запуск таймера
start_time = time.time()

# Запуск ABC для пошуку найкращих гіперпараметрів
best_params, best_score = abc_algorithm(objective_function, bounds, n_bees=20, n_iterations=50)

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
joblib.dump(final_model, 'titanic_model_1.pkl')
