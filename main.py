from ultralytics import YOLO
import cv2
import os

# Eğitilmiş YOLOv8 modelinizi yükleyin
model = YOLO("")  # Kendi eğitilmiş model dosyanızı kullanın(best.pt)

# Video üzerinde tahmin yapacak fonksiyon
def predict_video(input_video_path, output_video_path):
    if not os.path.exists(input_video_path):
        print(f"Error: The file '{input_video_path}' does not exist.")
        return

    # Video dosyasını aç
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: Failed to open the video '{input_video_path}'.")
        return

    # Video özelliklerini alın
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # VideoWriter ile yeni video dosyasını oluştur
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 video formatı
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # YOLOv8 ile tahmin yapın
        results = model(frame)

        # Sonuçları işleyin ve görüntüye çizin
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = box.conf[0]
                class_id = int(box.cls[0])
                label = model.names[class_id]

                # Kutuları ve etiketleri çiz
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Kırmızı renk (BGR formatında)
                cv2.putText(frame, f"{label}: {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)  # Kırmızı renk (BGR formatında)

        # İşlenmiş kareyi yeni videoya yaz
        out.write(frame)

    # Video ve VideoWriter nesnelerini serbest bırak
    cap.release()
    out.release()
    print(f"Output saved to {output_video_path}")

# Klasördeki tüm videoları işlemek için fonksiyon
def process_videos_in_folder(input_folder, output_folder):
    # Çıkış klasörünü oluştur
    os.makedirs(output_folder, exist_ok=True)

    # Klasördeki her bir video dosyasını işleyin
    for filename in os.listdir(input_folder):
        if filename.endswith(".mp4") or filename.endswith(".avi") or filename.endswith(".mov") or filename.endswith(".mkv"):
            input_video_path = os.path.join(input_folder, filename)
            output_video_path = os.path.join(output_folder, f"output_{filename}")  # Yeni dosya adı
            # Tahmin yap ve çıktıyı kaydet
            predict_video(input_video_path, output_video_path)

    print("All videos processed.")

# Klasör yolları
input_folder = ""  # Giriş video dosyalarının bulunduğu klasör
output_folder = ""  # Çıkış video dosyalarının kaydedileceği klasör

# Klasördeki tüm videoları işleyin
process_videos_in_folder(input_folder, output_folder)
