from fer import FER
import cv2

# Start the capture of the video stream
cap = cv2.VideoCapture(0)

frame_width = cap.get(3)  # float `width`
frame_height = cap.get(4)  # float `height`

# Initialize the FER
emo_detector = FER(mtcnn=True)

text_color = (50, 209, 89)
face_box_color = (255, 255, 0)

red_color = (0,0,255)
green_color = (51, 255, 51)
white_color = (255, 255, 255)
black_color = (0, 0, 0)
yellow_color = (255, 255, 51)
grey_color = (128, 128, 128)
blue_color = (204, 0, 0)
teal_color = (0, 204, 204)
pink_color=(255, 0, 255)
orange_color = (255, 166, 66)
platinum_color = (178, 190, 181)

angry_val = 0
angry_per = 0

disgust_val = 0
disgust_per = 0

fear_val = 0
fear_per = 0

happy_val = 0
happy_per = 0

sad_val = 0
sad_per = 0

surprised_val = 0
surprised_per = 0

neutral_val = 0
neutral_per = 0

current_frame_val = 1

graph_start_x = 100

while True:
    # get a frame
    ret, frame = cap.read()

    captured_emotions = emo_detector.detect_emotions(frame)
    print("Captured Emotion ", captured_emotions)
    dominant_emotion, emotion_score = emo_detector.top_emotion(frame)
    print("Dominant Emotion ", dominant_emotion, emotion_score)

    if captured_emotions:
        current_frame_val = current_frame_val + 1
        bounding_box = captured_emotions[0]["box"]
        x1 = bounding_box[0]
        y1 = bounding_box[1]
        width = bounding_box[2]
        height = bounding_box[3]
        x2 = x1 + width
        y2 = y1 + height

        if dominant_emotion == 'angry':
            angry_val = angry_val + 1
            face_box_color = red_color
        if dominant_emotion == 'disgust':
            disgust_val = disgust_val + 1
            face_box_color = grey_color
        if dominant_emotion == 'fear':
            fear_val = fear_val + 1
            face_box_color = blue_color
        if dominant_emotion == 'happy':
            happy_val = happy_val + 1
            face_box_color = green_color
        if dominant_emotion == 'sad':
            sad_val = sad_val + 1
            face_box_color = orange_color
        if dominant_emotion == 'surprise':
            surprised_val = surprised_val + 1
            face_box_color = pink_color
        if dominant_emotion == 'neutral':
            neutral_val = neutral_val + 1;
            face_box_color = white_color

        # Calculte the mean of the dominant emotion per frame
        angry_per = int(round((angry_val / current_frame_val) * 100, 1)) * 2
        disgust_per = int(round((disgust_val / current_frame_val) * 100, 1)) * 2
        fear_per = int(round((fear_val / current_frame_val) * 100, 1)) * 2
        happy_per = int(round((happy_val / current_frame_val) * 100, 1)) * 2
        sad_per = int(round((sad_val / current_frame_val) * 100, 1)) * 2
        surprised_per = int(round((surprised_val / current_frame_val) * 100, 1)) * 2
        neutral_per = int(round((neutral_val / current_frame_val) * 100, 0)) * 2

        cv2.rectangle(frame, (x1, y1), (x2, y2), face_box_color, 2, )
        cv2.putText(frame, (dominant_emotion.upper() + "-" + str(round(emotion_score * 100,1)) + "%"), (x1, y2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    face_box_color, 2)

    #background for the graph
    cv2.rectangle(frame, (0, 5), (300, 270), platinum_color, -1)


    cv2.putText(frame, "CUMULATIVE SENTIMENTS", (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, black_color, 1, cv2.CV_AA)

    cv2.putText(frame, "Neutral", (5, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, black_color, 0, cv2.CV_AA)
    cv2.rectangle(frame, (graph_start_x, 50), (neutral_per + graph_start_x, 70), white_color, -1)

    cv2.putText(frame, "Happy", (5, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, black_color, 1, cv2.CV_AA)
    cv2.rectangle(frame, (graph_start_x, 80), (happy_per + graph_start_x, 100), green_color, -1)

    cv2.putText(frame, "Surprised", (5, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.5, black_color, 1, cv2.CV_AA)
    cv2.rectangle(frame, (graph_start_x, 110), (surprised_per + graph_start_x, 130), pink_color, -1)

    cv2.putText(frame, "Angry", (5, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.5, black_color, 1, cv2.CV_AA)
    cv2.rectangle(frame, (graph_start_x, 140), (angry_per + graph_start_x, 160), red_color, -1)

    cv2.putText(frame, "Sad", (5, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.5, black_color, 1, cv2.CV_AA)
    cv2.rectangle(frame, (graph_start_x, 170), (sad_per + graph_start_x, 190), orange_color, -1)

    cv2.putText(frame, "Fear", (5, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.5, black_color, 1, cv2.CV_AA)
    cv2.rectangle(frame, (graph_start_x, 200), (fear_per + graph_start_x, 220), blue_color, -1)

    cv2.putText(frame, "Disgust", (5, 245), cv2.FONT_HERSHEY_SIMPLEX, 0.5, black_color, 1, cv2.CV_AA)
    cv2.rectangle(frame, (graph_start_x, 230), (disgust_per + graph_start_x, 250), grey_color, -1)

    happy_q = round(((neutral_val + happy_val + surprised_val) / current_frame_val) * 100, 1)
    sad_q = round(((fear_val + sad_val + angry_val + disgust_val) / current_frame_val) * 100, 1)

    cv2.rectangle(frame, (int(frame_width)-320, 12), (int(frame_width)-50, 35), platinum_color, -1)

    cv2.putText(frame, "INTERVENTION INDICATOR", (int(frame_width)-300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, black_color, 1, cv2.CV_AA)
    if happy_q >= 50:
        cv2.circle(frame, (int(frame_width)-80, 25), 8, green_color, -1)
    else:
        cv2.circle(frame, (int(frame_width)-80, 25), 8, red_color, -1)

    cv2.imshow("Facial Expression Analysis", frame)

    k = cv2.waitKey(30) & 0xff
    if k == 27:  # press 'ESC' to quit
        break
cap.release()
cv2.destroyAllWindows()
