import cv2

class VideoCamera():
    def __init__(self):
        from classification import ImageClassifier
        
        self.video = cv2.VideoCapture(0)
        self.classifier = ImageClassifier('wide_resnet')

    def __del__(self):
        self.video.release()   

    def get_frame(self):
        ret, frame = self.video.read()

        # DO WHAT YOU WANT WITH TENSORFLOW / KERAS AND OPENCV
        jpeg = self.classifier.classify_image(frame)
        ret, jpeg = cv2.imencode('.jpg',jpeg)

        return jpeg.tobytes()