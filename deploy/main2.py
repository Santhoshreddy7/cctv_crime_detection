# from kivy.logger import Logger
import logging
# Logger.setLevel(logging.TRACE)
import kivy
import cv2
from kivy.app import App
from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout
import time
import stamptime2
Builder.load_string('''
<CameraClick>:
    orientation: 'vertical'
    Camera:
        id: camera
        resolution: (640, 480)
        play: True
    ToggleButton:
        text: 'Start'
        on_press: camera.play = camera.play
        size_hint_y: None
        height: '48dp'
    Button:
        text: 'Stop'
        size_hint_y: None
        height: '48dp'
        on_press: root.capture()
    Button:
        id: btnExit
        text: 'Exit'
        size_hint_y: None
        height: '48dp'
        on_press: app.stop() 
''')


class CameraClick(BoxLayout):
    def capture(self):
        '''
        Function to capture the images and give them the names
        according to their captured time and date.
        '''


class Detect(kivy.app.APP):
    def process_vedio(self):
        video_path = cv2.VideoCapture("test2.mp4")
        video_features = stamptime2.extract_features(video_path)
        predicted_class = stamptime2.predict_output("yolov3_custom_train_1000.weights",video_features, activation="findObjects")
        self.root.ids["lables"].text ="predicted_class:" + predicted_class



class TestCamera(App):

    def build(self):
        return CameraClick()


TestCamera().run()