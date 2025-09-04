import os
import time
import sys
from queue import Queue
import base64
import asyncio
import  signal
import pyaudio
from audio.wakeup import Porcupine
import speech_recognition as sr
import struct


class AerisEars2:

    def __init__(self):
        # audio settings
        self.chunk_size = 1024
        self.sample_format = pyaudio.paInt16
        self.channels = 1
        self.sample_rate = 44100 # frequency

        self.wakeword = Porcupine(sensitivity=0.25, callback=self._on_wakeword)
        
        # oggetto di PyAudio
        self.audio = pyaudio.PyAudio()
        self.audio_stream = self.audio.open(
            rate=self.sample_rate,
            channels=self.channels,
            input=True,
            format=self.sample_format,
            output_device_index=1,
            frames_per_buffer=self.wakeword.porcupine.frame_length,
            stream_callback=self._wakeword_detection
        )
        self.audio_stream.start_stream()
        
        self.recorder = sr.Recognizer()
        self.microphone = sr.Microphone(device_index=1, chunk_size=self.chunk_size, sample_rate=self.sample_rate)
        # signal handler 
        signal.signal(signal.SIGINT, self.signal_handler)

    """ Funzione che permette una gracefull degradation
        attraverso l'handling dei segnali. """
    def signal_handler(self, sig, frame):
        self.stop_recording()
        sys.exit(0)

    """ Funzione che lista i dispositivi audio connessi alla Pi. 
        Utilizzata solo per riconoscere il nome del dispositivo audio connesso. """
    def list_audio_dev(self):
        print("Available audio devices: ")
        for i in range(self.audio.get_device_count()):
            print(f"{i} : {self.audio.get_device_info_by_index(i)}")
            
    """ Funzione che richiama il listen_loop dell'oggetto Porcupine e richiama
        la callback associata all'oggetto se la wakeword è rilevata."""
    def _wakeword_detection(self, in_data, frame_count, time_info, status):
        if status:
            print(f"[ERROR] Audio callback status: {status}")
        print("In attesa della parola 'Hey Aeris'")
        pcm = struct.unpack_from("h" * self.wakeword.porcupine.frame_length, in_data)
        keyword_index = self.wakeword.porcupine.process(pcm)
        if keyword_index >= 0:
            print("[INFO] Wake word detected")
            self.wakeword.callback()
        return (in_data, pyaudio.paContinue)
        
    """ Funzione che registra l'audio e lo passa al modello di trascrizione."""
    def _on_wakeword(self):
        audio = self.record_audio()
        if not audio:
            return
        transcript = self._transcribe_audio(audio)
        if not transcript:
            return
        self.stop_recording()
        return transcript.strip()
        
    """ Funzione che chiama il modello API di azure per la trascrizione dell'audio registrato.
        La trascrizione avviene usando la lingua italiana."""
    def _transcribe_audio(self, audio):
        try:
            return self.recorder.recognize_bing(audio, language="it-IT", key=os.getenv("AZURE_API_KEY"))
        except sr.UnknownValueError:
            return None
        except sr.RequestError as e:
            print(f"Speech recognition error: {e}")
            return None
        
    """Funzione che permette al microfono di registrare l'audio e restituire AudioData. """
    def record_audio(self):
        with self.microphone as source:
            self.recorder.adjust_for_ambient_noise(source)
            try:
                return self.recorder.listen(source, timeout=5, phrase_time_limit=10)
            except sr.WaitTimeoutError:
                return None
    

    """ Metodo che ferma gli oggetti audio stream adibiti alla trascrizione..."""
    def stop_recording(self):
        if self.audio_stream:
            self.audio_stream.stop_stream()
            self.audio_stream.close()
        