import os
import pickle
import cv2
import mediapipe as mp
import numpy as np
import time
from gtts import gTTS

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(1)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'}

sentence = ""
last_prediction_time = time.time()
last_prediction = ""
color_change_time = time.time()
space_pressed_time = 0

while True:

    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()

    H, W, _ = frame.shape

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)


    # Calculate the width and height of the text box
    (text_width, text_height) = cv2.getTextSize(f"Current sentence: {sentence}", cv2.FONT_HERSHEY_SIMPLEX, 2, 2)[0]

    # Set the text start position
    text_x, text_y = 10, 50

    # Draw the rectangle background
    cv2.rectangle(frame, (text_x - 5, text_y - text_height - 5), (text_x + text_width + 5, text_y + 5), (255, 255, 255), -1)

    # Display the current sentence on the window
    cv2.putText(frame, f"Current sentence: {sentence}", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2, cv2.LINE_AA)
    

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10

        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        try:
            prediction = model.predict([np.asarray(data_aux)])
        except ValueError:
            # print("Try again. More than 42 features detected.")
            continue

        predicted_character = prediction[0]

        color = (0, 0, 0)  # default color for reading (black)
        if time.time() - last_prediction_time > 2:
            if predicted_character == last_prediction:
                sentence += predicted_character
                color = (0, 255, 0)  # color for confirmed letter (green)
                color_change_time = time.time() + 1  # set the time when the color should change back to black
            last_prediction = predicted_character
            last_prediction_time = time.time()

        if time.time() < color_change_time:
            color = (0, 255, 0)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 3,
                    cv2.LINE_AA)

        

    cv2.imshow('frame', frame)
    key = cv2.waitKey(1)

    if key == ord(' '):  # if space is pressed
        sentence += ' '
    elif key == ord('\r'):  # if enter is pressed
        tts = gTTS(text=sentence, lang='en')  # create gTTS object
        tts.save("sentence.mp3")  # save the speech audio into a file
        os.system("afplay sentence.mp3")  # play the audio file
        sentence = ""
        cv2.putText(frame, f"Current sentence: {sentence}", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2, cv2.LINE_AA) 

cap.release()
cv2.destroyAllWindows()