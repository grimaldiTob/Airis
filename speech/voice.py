from piper import PiperVoice
import wave
import pygame
import numpy as np
import io

class AerisVoice:
    def __init__(self, voice: str = "/home/aeris/aeris/speech/it_IT-paola-medium.onnx"):
        
        # inizializza Piper
        self.voice = voice
        
        self.tts = PiperVoice.load(self.voice)
        
        pygame.mixer.init()
        
    """ Metodo che prende in input il testo restituito dal modello ia e una frequenza di campionamento.
        Successivamente riproduce l'audio tramite gli altoparlanti del dispotivo"""
    def play_audio(self, text: str, sample_rate: int):
        try:
            if not text:
                raise ValueError("Non c'è risposta dal modello")

            audio_buffer = io.BytesIO() # permette di creare un buffer in memoria RAM
            
            """ Sezione di codice che scrive nell'audio buffer i bytes presenti
                nell'array audio dopo che è stato normalizzato."""
            with wave.open(audio_buffer, 'wb') as wav_buffer: # wb = scrittura binaria
                wav_buffer.setnchannels(1)
                wav_buffer.setsampwidth(2)
                wav_buffer.setframerate(sample_rate)
                
                self.tts.synthesize_wav(text, wav_buffer)      
            
            audio_buffer.seek(0) # punta all'inizio dell'audio buffer dopo aver scritto fino alla fine
            
            pygame.mixer.music.load(audio_buffer)
            
            pygame.mixer.music.play()
            
            # aspetta 100 millisecondi che pygame abbia finito di riprodurre l'audio
            while pygame.mixer.music.get_busy():
                pygame.time.wait(100)
        except Exception as e:
            print(f"Errore durante riproduzione audio: {e}")
            
    """ Funzione usata per modificare la voce del modello. """
    def set_voice(self, voice: str):
        self.voice = voice
        self.tts = PiperVoice.load(self.voice)
        
