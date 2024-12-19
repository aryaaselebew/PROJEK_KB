import cv2
import os
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Fungsi untuk membaca dataset wajah
def load_faces_and_labels(dataset_path):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    face_data = []
    labels = []

    print("[INFO] Membaca dataset wajah...")
    for person_name in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, person_name)
        if not os.path.isdir(person_path):
            continue

        # Baca semua gambar dalam folder
        for image_name in os.listdir(person_path):
            image_path = os.path.join(person_path, image_name)
            gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if gray_image is None:
                continue

            # Deteksi wajah
            faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            for (x, y, w, h) in faces:
                face = gray_image[y:y+h, x:x+w]  # Potong wajah
                face_resized = cv2.resize(face, (100, 100))  # Resize wajah agar konsisten
                face_data.append(face_resized.flatten())  # Ubah ke bentuk 1D array
                labels.append(person_name)

    return np.array(face_data), np.array(labels)

# Path ke dataset wajah
dataset_path = "dataset_webcam"

# Load dataset wajah
faces, labels = load_faces_and_labels(dataset_path)
if len(faces) == 0:
    print("[ERROR] Dataset kosong atau wajah tidak terdeteksi!")
    exit()

# Encode label menjadi angka
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Bagi data menjadi train-test
X_train, X_test, y_train, y_test = train_test_split(faces, encoded_labels, test_size=0.2, random_state=42)

# Latih model SVM
print("[INFO] Melatih model SVM untuk Face Recognition...")
svm_model = SVC(kernel='linear', probability=True)
svm_model.fit(X_train, y_train)

# Evaluasi akurasi model
y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"[INFO] Akurasi model: {accuracy * 100:.2f}%")

# Inisialisasi webcam
print("[INFO] Memulai Face Recognition...")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("[ERROR] Tidak dapat membaca dari webcam")
        break

    # Membalik frame horizontal agar tidak mirror
    frame = cv2.flip(frame, 1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces_detected = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Deteksi wajah dan prediksi
    for (x, y, w, h) in faces_detected:
        face = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (100, 100)).flatten()  # Resize wajah

        # Prediksi menggunakan SVM
        label_idx = svm_model.predict([face_resized])[0]
        confidence = max(svm_model.predict_proba([face_resized])[0]) * 100

        if confidence > 50:
            name = label_encoder.inverse_transform([label_idx])[0]
            box_color = (0, 255, 0)  # Hijau untuk wajah dikenali
            text_color = (0, 255, 0)
        else:
            name = "Unknown"
            box_color = (0, 0, 255)  # Merah untuk wajah tidak dikenal
            text_color = (0, 0, 255)

        # Tampilkan hasil
        cv2.rectangle(frame, (x, y), (x+w, y+h), box_color, 2)
        cv2.putText(frame, f"{name} ({confidence:.2f}%)", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)

    # Tampilkan video
    cv2.imshow("Face Recognition - Tekan 'q' untuk keluar", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Tutup proses
video_capture.release()
cv2.destroyAllWindows()
print("[INFO] Selesai Face Recognition.")
