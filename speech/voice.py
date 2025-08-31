from kittentts import KittenTTS
import wave
import pygame
import numpy as np
import io

# available_voices : [  'expr-voice-2-m', 'expr-voice-2-f', 'expr-voice-3-m', 'expr-voice-3-f',  'expr-voice-4-m', 'expr-voice-4-f', 'expr-voice-5-m', 'expr-voice-5-f' ]

class AerisVoice:
    def __init__(self, model : str = "", voice: str = None):
        self.model = model
        
        # inizializza Kitten
        self.voice = voice
        
        self.tts = KittenTTS(self.model, self.voice)
        
    """ Funzione che usa KittenTTS per generare un array di numpy che rappresenta
        la traccia audio della trascrizione."""
    def generate_speech(self, text: str):
        audio = self.tts.generate(text=text, voice=self.voice)
        return audio
    
    """ Metodo che prende in input un array di byte e una frequenza di campionamento.
        Successivamente riproduce l'audio tramite gli altoparlanti del dispotivo"""
    def play_audio(self, audio: np.ndarray, sample_rate: int):
        try:
            if len(audio) == 0:
                raise ValueError("Array vuoto")
            
            # Solitamente i dati audio ricevuti da metodi di ML sono con valori float tra -1.0 e 1
            if audio.dtype != np.int16:
                audio = (audio * 32767).astype(np.int16) # normalizzazione dei dati audio

            audio_buffer = io.BytesIO() # permette di creare un buffer in memoria RAM
            
            
            """ Sezione di codice che scrive nell'audio buffer i bytes presenti
                nell'array audio dopo che è stato normalizzato."""
            with wave.open(audio_buffer, 'wb') as wav_buffer: # wb = scrittura binaria
                wav_buffer.setnchannels(2)
                wav_buffer.setsampwidth(2)
                wav_buffer.setframerate(sample_rate)
                
                wav_buffer.writeframes(audio.tobytes())        
            
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
        self.tts = KittenTTS(self.model, self.voice)
        
