import cv2
from classification import ImageClassifier

class VideoCamera():
    def __init__(self,model_name):
        
        assert(model_name=="vgg" or model_name=="wide_resnet")
        
        self.video = cv2.VideoCapture(0)
        self.classifier = ImageClassifier(model_name)

    def __del__(self):
        self.video.release()   

    def get_frame(self):
        ret, frame = self.video.read()

        # Perform bounding box detection and classification
        jpeg = self.classifier.classify_image(frame)
        ret, jpeg = cv2.imencode('.jpg',jpeg)

        return jpeg.tobytes()