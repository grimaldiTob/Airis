import pvporcupine
import pyaudio
import struct
import threading
from typing import Callable, Optional
import os
import librosa
import time
import numpy as np

class Porcupine:
    def __init__(self,
                 sensitivity : float = 0.5,
                 callback: Callable = None):
        self.access_key = os.getenv("PICOVOICE_ACCESS_KEY")
        self.keyword = "/home/aeris/aeris/audio/eris.ppn"
        self.sensitivity = sensitivity
        self.callback = callback
        
        self.porcupine = None
        self.audio_stream = None
        self.pa = None
        self.is_listening = False
        self.thread = None
        
    def start(self, timeout: int):
        if self.is_listening:
            return False
        
        try:
            keyword_path = self.keyword
            
            # definisci l'ggetto porcupine di rilevazione della parola
            self.porcupine = pvporcupine.create(
                access_key=self.access_key,
                keyword_paths=[keyword_path],
                sensitivities=[self.sensitivity] # sensitività di rilevamento della parola
            )
            
            self.pa = pyaudio.PyAudio()
            self.audio_stream = self.pa.open(
                rate=44100,
                channels=1,
                format=pyaudio.paInt16,
                input=True,
                frames_per_buffer=self.porcupine.frame_length,
                input_device_index=0
            )
            
            # avvia thread di ascolto per la parola
            self.is_listening = True
            self.thread = threading.Thread(target=self.listen_loop, args=(timeout, ), daemon=True)
            self.thread.start()
            print(f"In ascolto per la keyword \"{self.keyword}\"")
            
            try:
                while self.is_listening:
                    time.sleep(1)
            except KeyboardInterrupt:
                self.stop()
            
        except Exception as e:
            print(f"Errore all'avvio: {e}")
            return False
        
    def stop(self):
        if self.thread:
            self.thread.join()
            
        if self.audio_stream:
            self.audio_stream.close()
        
        if self.pa:
            self.pa.terminate()
            
        if self.porcupine:
            self.porcupine.delete()
            
    def listen_loop(self, timeout: int):
        print("Ascoltando...")
        start_time = time.time()
        while self.is_listening and (time.time() - start_time) < timeout:
            try:
                pcm_bytes = self.audio_stream.read(
                    self.porcupine.frame_length,
                    exception_on_overflow=False
                )
                pcm = np.frombuffer(pcm_bytes, dtype=np.int16)
                pcm = librosa.resample(pcm.astype(np.float32), orig_sr=44100, target_sr=16000)
                pcm = pcm.astype(np.int16) # chiedi spiegazioni su questo
                                
                keyboard_index = self.porcupine.process(pcm)
                
                if keyboard_index >= 0:
                    if self.callback:
                        self.callback()
                    self.is_listening = False
                    return
            except Exception as e:
                print(f"Errore nel loop di ascolto: {e}")
        self.is_listening = False
        print(f"Timeout raggiunto ({timeout}s). Ascolto terminato.")
                
    