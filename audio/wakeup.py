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
        
        # buffer
        self.resample_buffer = np.array([], dtype=np.int16)
        
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
            print(f"In ascolto per la keyword \"Hey Eris\"")
            
            try:
                while self.is_listening:
                    time.sleep(1)
                time.sleep(1)
                self.stop()
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
                """
                In questo caso leggiamo uno stream audio ad una frequenza di 44.1kHz
                ma porcupine opera ad una frequenza di 16kHz ad un frame_rate di 512...
                Si esegue la proporzione x * (16000/44100) = 512 e ricaviamo:
                            x ~= 1411
                frame rate necessario per far funzionare correttamente porcupine
                """   
                pcm_bytes = self.audio_stream.read(
                    1411,
                    exception_on_overflow=False
                )
                pcm = np.frombuffer(pcm_bytes, dtype=np.int16)
                pcm = librosa.resample(pcm.astype(np.float32), orig_sr=44100, target_sr=16000)
                pcm = np.clip(pcm, -32767, 32767).astype(np.int16)
                # normalizzi un valore a 16 bit [-32767, 32767] dopo la conversione 
                # float32 ---> int16
                                             
                keyboard_index = self.porcupine.process(pcm)
                
                if keyboard_index >= 0:
                    self.audio_stream.close()
                    self.pa.terminate()
                    if self.callback:
                        self.callback()
                    self.is_listening = False
                    return
            except Exception as e:
                print(f"Errore nel loop di ascolto: {e}")
                return
        self.is_listening = False
        print(f"Timeout raggiunto ({timeout}s). Ascolto terminato.")
        return False
                
    