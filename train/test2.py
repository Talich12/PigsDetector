import os
from ultralytics import YOLO
import cv2

def process_images(input_folder, output_folder):
    # Создаем папку для результатов, если ее нет
    os.makedirs(output_folder, exist_ok=True)
    
    # Загружаем модель
    model = YOLO('best.pt')
    
    # Получаем список изображений
    image_files = [f for f in os.listdir(input_folder) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    
    if not image_files:
        print(f"В папке {input_folder} не найдено изображений!")
        return
    
    print(f"Найдено {len(image_files)} изображений для обработки...")
    
    # Обрабатываем каждое изображение
    for i, filename in enumerate(image_files, 1):
        try:
            # Загружаем изображение
            img_path = os.path.join(input_folder, filename)
            image = cv2.imread(img_path)
            
            if image is None:
                print(f"Ошибка загрузки: {filename}")
                continue
            
            # Детекция объектов
            results = model.predict(image, conf=0.5)
            
            # Обрабатываем результаты
            for r in results:
                # Рисуем bounding boxes
                result_img = r.plot()
                
                # Сохраняем результат
                output_path = os.path.join(output_folder, f"result_{filename}")
                cv2.imwrite(output_path, result_img)
                
                # Выводим статистику
                print(f"\n[{i}/{len(image_files)}] {filename}:")
                print(f"Размер: {image.shape[1]}x{image.shape[0]}")
                print("Обнаружены объекты:")
                for box in r.boxes:
                    print(f"- {model.names[int(box.cls)]}: уверенность {box.conf.item():.2f}")
            
            print(f"Успешно обработано: {filename}")
            
        except Exception as e:
            print(f"Ошибка при обработке {filename}: {str(e)}")
    
    print("\nОбработка завершена!")

if __name__ == "__main__":
    # Укажите ваши пути
    input_folder = "train/test_shots/shots1"  # Папка с исходными изображениями
    output_folder = "train/results"    # Папка для сохранения результатов
    
    process_images(input_folder, output_folder)