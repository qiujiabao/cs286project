import speech_recognition as sr

class Speech2Text():
    def __init__(self):
        recognizer = sr.Recognizer()

    def stt(file):
        temp = sr.AudioFile(file)
        with temp as source:
            audio = recognizer.record(source)
        return recognizer.recognize_google(audio)
