import os
import time

import cv2


class EyesDetection:
    FACE_CASCADE = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    EYE_CASCADE = cv2.CascadeClassifier('haarcascade_eye.xml')
    VIDEO_DEVICE = 0
    SECONDS_OF_INATTENTION = 1

    def do(self):
        cap = cv2.VideoCapture(self.VIDEO_DEVICE)
        time_face = None
        eyes_detected = True

        while True:
            ret, img = cap.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.FACE_CASCADE.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                roi_gray = gray[y:y + h, x:x + w]
                roi_color = img[y:y + h, x:x + w]

                eyes = self.EYE_CASCADE.detectMultiScale(roi_gray, 1.3, 30)
                if len(eyes) != 0:
                    time_face = time.time()
                    eyes_detected = True
                elif time_face and int(time.time() - time_face) == self.SECONDS_OF_INATTENTION:
                    eyes_detected = False
                if not eyes_detected:
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(img, 'Focus on the road!!', (10, 500), font, 4, (0, 0, 255), 2, cv2.LINE_AA)
                    os.system('spd-say "Focus on the road"')
                    time.sleep(1)
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

            cv2.imshow('img', img)
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    detection = EyesDetection()
    detection.do()
