import cv2
from deepface import DeepFace
import time
import threading
import queue

result_queue = queue.Queue()

def analyze_face_thread(face_img):
    try:
        analyze = DeepFace.analyze(
            face_img,
            actions=['emotion', 'age', 'gender', 'race'],
            enforce_detection=False
        )
        result_queue.put(analyze)
    except Exception as e:
        print(f'Erro na thread de análise: {e}')
        result_queue.put(None)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

video = cv2.VideoCapture(0, cv2.CAP_DSHOW)

video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not video.isOpened():
    raise IOError("Nao foi possivel abrir a webcam")

last_analysis_time = 0
analysis_interval = 1.5

current_analysis = None
analysis_in_progress = False

emotion_map = {
    'angry': 'raiva',
    'disgust': 'nojo',
    'fear': 'medo',
    'happy': 'feliz',
    'sad': 'triste',
    'surprise': 'surpresa',
    'neutral': 'neutro'
}

race_map = {
    'asian': 'asiatico',
    'indian': 'indiano',
    'black': 'negro',
    'white': 'branco',
    'middle eastern': 'oriente medio',
    'latino hispanic': 'latino'
}

frame_count = 0
fps_start_time = time.time()
fps = 0

while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break

    current_time = time.time()

    frame_count += 1
    if current_time - fps_start_time >= 1.0:
        fps = frame_count
        frame_count = 0
        fps_start_time = current_time

    scale_factor = 0.5
    frame_small = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)

    gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if not result_queue.empty():
        current_analysis = result_queue.get()
        analysis_in_progress = False

    for x_small, y_small, w_small, h_small in faces:
        x = int(x_small / scale_factor)
        y = int(y_small / scale_factor)
        w = int(w_small / scale_factor)
        h = int(h_small / scale_factor)

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if (current_time - last_analysis_time >= analysis_interval) and not analysis_in_progress:
            try:
                face_roi = frame[y:y + h, x:x + w].copy()

                if face_roi.size > 0:
                    analysis_thread = threading.Thread(
                        target=analyze_face_thread,
                        args=(face_roi,)
                    )
                    analysis_thread.daemon = True
                    analysis_thread.start()

                    analysis_in_progress = True
                    last_analysis_time = current_time
            except Exception as e:
                print(f'Erro ao iniciar análise: {e}')

        if current_analysis:
            analyze = current_analysis
            if isinstance(analyze, list):
                analyze = analyze[0]

            emotion = analyze['dominant_emotion']
            age = analyze['age']

            if emotion in emotion_map:
                emotion = emotion_map[emotion]

            gender_dict = analyze['gender']
            if isinstance(gender_dict, dict):
                gender = max(gender_dict, key=gender_dict.get)
                gender_confidence = gender_dict[gender]

                gender_display = f"{'Homem' if gender == 'Man' else 'Mulher'} ({gender_confidence:.1f}%)"
            else:
                gender_display = 'Homem' if gender_dict == 'Man' else 'Mulher'

            race = analyze['dominant_race']
            race_percentage = analyze['race'][race]

            if race in race_map:
                race = race_map[race]

            text_color = (0, 255, 0)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2

            texts = [
                f"Emocao: {emotion}",
                f"Idade: {int(age)}",
                f"Genero: {gender_display}",
                f"Raca: {race} ({race_percentage:.1f}%)"
            ]

            for i, text in enumerate(texts):
                y_pos = y + h + 20 + (i * 25)

                cv2.putText(frame, text, (x, y_pos),
                            font, font_scale, text_color, thickness)

    cv2.putText(frame, f"FPS: {fps}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Analise Emocoes', frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
