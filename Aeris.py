from audio.wakeup import Porcupine
from audio.audio_local import AerisEars
from model.response_gen import AerisMind
import time


class Aeris:
    def __init__(self, gpt_model="gpt-4.1-mini"):
        
        # Inizializza i componenti
        self.ears = None
        self.wakeword = None
        self.mind = AerisMind(model=gpt_model)
        
        self.is_processing = False
        self.transcription_complete = False
        self.current_transcription = "" # prenderà audio di proceess audio queue e chiamerà create_response
    
    """ Funzione chiamata quando una wakeword viene detectata da Porcupine..."""    
    def on_wake_word_detected(self):
        if not self.ears:
            self.ears = AerisEars()
        
        self.transcription_complete = False
        self.current_transcription = ""
        
        self.ears.process_audio_queue = self.custom_process_queue
        
        self.ears.start_recording(device_index=0)
        
    """ Funzione che sostituisce process_audio_queue presente nel
        gruppo audio. Permette di unire le trascrizioni generate dal
        modello whisper e ricavare una stringa che diventerà il prompt."""    
    def custom_process_queue(self):
        transcribed_parts = []
        
        while self.ears.is_recording or not self.ears.audio_queue.empty():
            try:
                if not self.ears.audio_queue.empty():
                    audio_data = self.ears.audio_queue.get(timeout=1)
                    
                    # Resampling dell'audio da 44.1kHz a 16kHz
                    import librosa
                    audio_resampled = librosa.resample(audio_data, orig_sr=44100, target_sr=16000)
                    transcript = self.ears.transcribe_audio(audio_resampled)
                    
                    if transcript.strip():
                        transcribed_parts.append(transcript)
                    
                    self.ears.audio_queue.task_done()
                else:
                    time.sleep(0.1)
            except Exception as e:
                print(f"Errore nel processamento: {e}")
                break
        
        # Processa eventuali elementi rimanenti nella coda
        while not self.ears.audio_queue.empty():
            try:
                audio_data = self.ears.audio_queue.get_nowait()
                import librosa
                audio_resampled = librosa.resample(audio_data, orig_sr=44100, target_sr=16000)
                transcript = self.ears.transcribe_audio(audio_resampled)
                if transcript and transcript.strip():
                    transcribed_parts.append(transcript)
            except:
                break
        
        if transcribed_parts:
            self.current_transcription = " ".join(transcribed_parts).strip()
            self.generate_response()
        else:
            print("Nessuna trascrizione ottenuta.")
        
        self.transcription_complete = True
        
    """ Funzione che richiama il modello di OpenAi per la generazione
        della risposta dato il prompt passato."""
    def generate_response(self):
        if self.current_transcription.strip():
            response = self.mind.create_response(self.current_transcription)
            print(f"{response}")
            
                
    """ Funzione che inizializza l'agente Porcupine per la rilevazione della wakeword.
         Nel momento in cui la parola viene rilevata il loop continua e viene chiamata
         on_wake_word_detected() come callback. """
    def start_listening_cycle(self, timeout: int = 10):
        self.wakeword = Porcupine(self.on_wake_word_detected)
        
        self.wakeword.start(timeout)
        
        while not self.transcription_complete:
            time.sleep(0.5)
            
        return True
        
    """ Funzione che fa partire il loop di ascolto per la wakeword."""
    def run_continous(self, timeout: int = 10):
        try:
            cycle_count = 0
            while True:
                cycle_count += 1
                self.start_listening_cycle(timeout)
        except KeyboardInterrupt:
            print("Uscita.")
        finally:
            self.cleanup()
            
    """Funzione che chiama la chiusura dei microfoni."""
    def cleanup(self):
        if self.ears:
            self.ears.stop_recording()
        if self.wakeword:
            self.wakeword.stop()
        time.sleep(1)
        print("Sistema terminato")       
        
        
def main():
    try:
        system = Aeris()
        system.run_continous(15)
    except Exception as e:
        print(f"Errore nell'avvio: {e}")
            
            
if __name__ == "__main__":
    main()