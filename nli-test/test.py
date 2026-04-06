import json
from nli_model import ContradictionDetector

# Список тестов на противоречие (Premise vs Hypothesis)
CONTRADICTION_TESTS = [
    {
        "id": 1,
        "premise": "Я учусь только в БГУИР",
        "hypothesis": "Я учусь только в БГУ",
        "note": "Разные университеты"
    },
    {
        "id": 2,
        "premise": "Компания уволила 50 сотрудников в марте",
        "hypothesis": "Компания не увольняла сотрудников в марте",
        "note": "Прямое отрицание факта"
    },
    {
        "id": 3,
        "premise": "Москва — столица России",
        "hypothesis": "Столица России — Санкт-Петербург",
        "note": "Фактологическое противоречие"
    },
    {
        "id": 4,
        "premise": "Все сотрудники отдела работают удалённо",
        "hypothesis": "Некоторые сотрудники отдела работают из офиса",
        "note": "Кванторное противоречие (все ↔ некоторые)"
    },
    {
        "id": 5,
        "premise": "Документ подписан директором и главным бухгалтером",
        "hypothesis": "Документ подписан только директором",
        "note": "Исключение второго подписанта"
    },
    {
        "id": 6,
        "premise": "Проект стартовал 1 января 2024 года",
        "hypothesis": "Проект стартовал 15 марта 2024 года",
        "note": "Конфликт дат"
    },
    {
        "id": 7,
        "premise": "Цена товара составляет 1000 рублей",
        "hypothesis": "Товар стоит 500 рублей",
        "note": "Числовое противоречие"
    },
    {
        "id": 8,
        "premise": "Иванов — единственный автор статьи",
        "hypothesis": "Статью написали Иванов и Петров",
        "note": "Противоречие по авторству"
    },
    {
        "id": 9,
        "premise": "Система работает на Linux",
        "hypothesis": "Система работает на Windows",
        "note": "Взаимоисключающие ОС"
    },
    {
        "id": 10,
        "premise": "Клиент отменил заказ",
        "hypothesis": "Клиент подтвердил заказ",
        "note": "Противоположные действия"
    },
    {
        "id": 11,
        "premise": "Свет в комнате включен.",
        "hypothesis": "В комнате темно.",
        "note": "Противоречие состояния освещения"
    },
    {
        "id": 12,
        "premise": "Контейнер1 пуст.",
        "hypothesis": "В контейнере1 лежит яблоко.",
        "note": "Противоречие наполненности"
    },
    {
        "id": 13,
        "premise": "Мария — единственный ребенок в семье.",
        "hypothesis": "У Марии есть старший брат.",
        "note": "Противоречие семейного статуса"
    },
    {
        "id": 14,
        "premise": "Поезд отправился ровно в 12:00.",
        "hypothesis": "Поезд все еще стоял на платформе в 12:05.",
        "note": "Временное противоречие"
    },
    {
        "id": 15,
        "premise": "Дверь заперта изнутри.",
        "hypothesis": "Кто-то вошел через эту дверь, не открывая её.",
        "note": "Физическое противоречие"
    },
    {
        "id": 16,
        "premise": "Все билеты на концерт распроданы.",
        "hypothesis": "Я купил билет в кассе сегодня утром.",
        "note": "Противоречие доступности"
    },
    {
        "id": 17,
        "premise": "Число 7 является нечетным.",
        "hypothesis": "Число 7 делится на 2 без остатка.",
        "note": "Математическое противоречие"
    },
    {
        "id": 18,
        "premise": "Он молчал всю встречу.",
        "hypothesis": "Он высказал свое мнение вслух во время встречи.",
        "note": "Противоречие действия"
    },
    {
        "id": 19,
        "premise": "Эта картина написана маслом.",
        "hypothesis": "Эта картина выполнена акварелью.",
        "note": "Противоречие материала"
    },
    {
        "id": 20,
        "premise": "Машина стоит в гараже.",
        "hypothesis": "Машина едет по шоссе.",
        "note": "Противоречие местоположения"
    },
    {
        "id": 21,
        "premise": "Никто не знает ответа на этот вопрос.",
        "hypothesis": "Иван знает правильный ответ.",
        "note": "Кванторное противоречие"
    },
    {
        "id": 22,
        "premise": "Температура воды 0 градусов Цельсия, и она замерзла.",
        "hypothesis": "Вода находится в жидком состоянии.",
        "note": "Противоречие агрегатного состояния"
    },
    {
        "id": 23,
        "premise": "Завтра будет вторник.",
        "hypothesis": "Завтра будет пятница.",
        "note": "Календарное противоречие"
    },
    {
        "id": 24,
        "premise": "Собака спит на коврике.",
        "hypothesis": "Собака бегает по саду.",
        "note": "Противоречие активности"
    },
    {
        "id": 25,
        "premise": "Этот документ является оригиналом.",
        "hypothesis": "Это ксерокопия документа.",
        "note": "Противоречие типа документа"
    },
    {
        "id": 26,
        "premise": "The box is completely empty.",
        "hypothesis": "There is a book inside the box.",
        "note": "Contradiction of contents"
    },
    {
        "id": 27,
        "premise": "John is taller than Mike.",
        "hypothesis": "Mike is taller than John.",
        "note": "Logical contradiction"
    },
    {
        "id": 28,
        "premise": "The store is open 24/7.",
        "hypothesis": "The store is closed right now.",
        "note": "Contradiction of status"
    },
    {
        "id": 29,
        "premise": "She has never been to Paris.",
        "hypothesis": "She visited Paris last summer.",
        "note": "Factual contradiction"
    },
    {
        "id": 30,
        "premise": "All cats are mammals.",
        "hypothesis": "Some cats are reptiles.",
        "note": "Taxonomic contradiction"
    },
    {
        "id": 31,
        "premise": "The glass is full of water.",
        "hypothesis": "The glass contains no liquid.",
        "note": "Contradiction of volume"
    },
    {
        "id": 32,
        "premise": "He failed the exam.",
        "hypothesis": "He passed the exam with high marks.",
        "note": "Contradiction of outcome"
    },
    {
        "id": 33,
        "premise": "It is raining heavily outside.",
        "hypothesis": "The ground is completely dry.",
        "note": "Physical consequence"
    },
    {
        "id": 34,
        "premise": "This is a digital photograph.",
        "hypothesis": "This is a hand-painted oil canvas.",
        "note": "Medium contradiction"
    },
    {
        "id": 35,
        "premise": "The meeting was cancelled.",
        "hypothesis": "We discussed the project in the meeting today.",
        "note": "Event occurrence"
    },
    {
        "id": 36,
        "premise": "Alice is an only child.",
        "hypothesis": "Alice has a younger sister.",
        "note": "Family structure"
    },
    {
        "id": 37,
        "premise": "The car is parked in the garage.",
        "hypothesis": "The car is driving on the highway.",
        "note": "Location/State"
    },
    {
        "id": 38,
        "premise": "No one entered the room.",
        "hypothesis": "Bob walked into the room.",
        "note": "Quantifier contradiction"
    },
    {
        "id": 39,
        "premise": "The light is off.",
        "hypothesis": "The bulb is emitting light.",
        "note": "State contradiction"
    },
    {
        "id": 40,
        "premise": "Today is Monday.",
        "hypothesis": "Today is Wednesday.",
        "note": "Temporal contradiction"
    },
    {
        "id": 41,
        "premise": "The password is case-sensitive.",
        "hypothesis": "Capitalization does not matter for the password.",
        "note": "Rule contradiction"
    },
    {
        "id": 42,
        "premise": "John is asleep.",
        "hypothesis": "John is watching a movie.",
        "note": "Activity contradiction"
    },
    {
        "id": 43,
        "premise": "The bottle is sealed.",
        "hypothesis": "The cap is off the bottle.",
        "note": "Physical state"
    },
    {
        "id": 44,
        "premise": "I have zero dollars in my wallet.",
        "hypothesis": "I have a ten-dollar bill in my wallet.",
        "note": "Numerical contradiction"
    },
    {
        "id": 45,
        "premise": "The software is free to use.",
        "hypothesis": "You must pay a subscription fee to use the software.",
        "note": "Cost contradiction"
    },
    {
        "id": 46,
        "premise": "The bird is flying south.",
        "hypothesis": "The bird is stationary on a branch.",
        "note": "Motion contradiction"
    },
    {
        "id": 47,
        "premise": "This fruit is an apple.",
        "hypothesis": "This fruit is a banana.",
        "note": "Identity contradiction"
    },
    {
        "id": 48,
        "premise": "The door is locked.",
        "hypothesis": "The door is wide open.",
        "note": "State contradiction"
    },
    {
        "id": 49,
        "premise": "Everyone agreed with the proposal.",
        "hypothesis": "Sarah voted against the proposal.",
        "note": "Quantifier contradiction"
    },
    {
        "id": 50,
        "premise": "The battery is dead.",
        "hypothesis": "The device is powered on and working.",
        "note": "Functional contradiction"
    }
]


def print_table(results: list, threshold: float = 0.5):
    """Вывод таблицы результатов тестирования."""
    header = f"{'ID':<4} {'Res':<5} {'Conf':<7} {'Label':<15} {'Note'}"
    separator = "─" * 80
    
    print("\n" + separator)
    print(header)
    print(separator)
    
    detected_count = 0
    confidences = []
    
    for res in results:
        is_det = res["detected"]
        conf = res["confidence"]
        label = res.get("predicted_label", "N/A")
        
        if is_det:
            detected_count += 1
            status_mark = "✅"
        else:
            status_mark = "❌"
            
        confidences.append(conf)
        
        # Цветовая маркировка уверенности (просто текстом)
        conf_str = f"{conf:.2f}"
        
        print(f"{res['id']:<4} {status_mark:<5} {conf_str:<7} {label:<15} {res['note']}")
    
    print(separator)
    
    total = len(results)
    accuracy = (detected_count / total * 100) if total > 0 else 0
    
    print(f"\n🎯 Результат: {detected_count}/{total} противоречий найдено ({accuracy:.1f}%)")
    
    if confidences:
        avg_conf = sum(confidences) / len(confidences)
        min_conf = min(confidences)
        max_conf = max(confidences)
        print(f"📊 Статистика уверенности (Contradiction Prob):")
        print(f"   • Средняя: {avg_conf:.3f}")
        print(f"   • Мин: {min_conf:.3f} | Макс: {max_conf:.3f}")
    print()


def main():
    print("🚀 Инициализация детектора противоречий...")
    try:
        detector = ContradictionDetector()
    except Exception as e:
        print(f"❌ Ошибка загрузки модели: {e}")
        return

    print(f"🧪 Запуск {len(CONTRADICTION_TESTS)} тестов...\n")
    
    results = []
    for test in CONTRADICTION_TESTS:
        # Вызов метода detect
        detection_result = detector.detect(test["premise"], test["hypothesis"])
        
        results.append({
            "id": test["id"],
            "detected": detection_result["is_contradiction"],
            "confidence": detection_result["contradiction_probability"],
            "predicted_label": detection_result["predicted_label"],
            "note": test["note"]
        })
    
    print_table(results)
    
    # Сохранение результатов в файл
    output_file = "contradiction_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"💾 Результаты сохранены в {output_file}")


if __name__ == "__main__":
    main()