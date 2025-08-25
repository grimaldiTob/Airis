import pyaudio
import threading
import time
import numpy as np
from queue import Queue
import signal
import sys
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import warnings

class AerisEars:
    def __init__(self, model="openai/whisper-small"):
      self.model_name = model

        # audio settings
      self.chunk_size = 1024
      self.sample_format = pyaudio.paInt16
      self.channels = 1
      self.sample_rate = 16000 # frequency
      self.record_seconds = 3

      self.audio_queue = Queue(maxsize=5)
      self.is_recording = False
      
      # Oggetto PyAudio
      self.audio = pyaudio.Pyaudio()

      #Model Componenets
      self.processor = None
      self.model = None

      signal.signal(signal.SIGINT, self.signal_handler)
      self.load_model()

    """ Gestisce l'uscita gracefull del programma. """
    def signal_handler(self, sig, frame):
      self.stop_recording()
      sys.exit(0)


    """ Funzione che carica il modello."""
    def load_model(self):
      try:
        self.processor = WhisperProcessor.from_pretrained(self.model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(
          self.model_name,
          torch_dtype=torch.float32, # forza il modello ad usare float a 32 bit 
          low_cpu_usage=True
        ) # RICONTROLLA
        
        self.model.to("cpu")
        self.model.eval()
        
      except Exception as e:
        print(f"Error loading the model: {e}")
        sys.exit(1)
        
    """ Funzione che lista i dispositivi audio connessi alla Pi. 
        Utilizzata solo per riconoscere il nome del dispositivo audio connesso. """
    def list_audio_dev(self):
        print("Available audio devices: ")
        for i in range(self.audio.get_device_count()):
            print(f"{i} : {self.audio.get_device_info_by_index(i)}")
            
            
    """ Funzione che legge i byte dal microfono e li mette in una coda di audio dopo
        aver effettuato operazioni di conversione dati e normalizzazione."""
    def record_audio_chunk(self, device_index):
      try:
        config = {
          'format' : self.sample_format,
          'channels' : self.channels,
          'rate' : self.sample_rate,
          'frames_per_buffer' : self.chunk_size,
          'input' : True
        }
        if device_index is not None:
          config['input_device_index'] = device_index
        
        stream = self.audio.open(**config)
        
        while self.is_recording:
          frames = [] # pezzi di audio grezzi letti dal microfono
          
          """
          n. di campioni al secondo / n.di campioni in un frame * secondi di registrazione
          indica il numero di chunk da registrare in n secondi di registrazione
          """
          for _ in range(int(self.sample_rate / self.chunk_size * self.record_seconds)):
            if not self.is_recording:
              break
            # legge chunk size campioni di audio dalla stream 
            frames.append(stream.read(self.chunk_size, exception_on_overflow=False))
          
          if frames and self.is_recording:
            # converte i bytes in un array di interi a 16 bit unendo i bytes grezzi presenti in frames
            audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)
            
            # converte in float e normalizza in un range compreso tra [-1; 1]
            audio_data = audio_data.astype(np.float32) / 32768.0
            
            try:
              self.audio_queue.put_nowait(audio_data)
            except:
              pass
            
        stream.stop_stream()
        stream.close()
      except Exception as e:
        print(f"Recording error: {e}")
        
    """ Funzione che si occupa della trascrizione dei dati audio passati presenti nella
        audio_queue. Riprendere e verificare questi concetti!!!"""
    def transcribe_audio(self, audio_data):
      try:
        # se i dati audio sono inferiori ad una data cifra in valore assoluto
        if np.max(np.abs(audio_data)) < 0.01:
          return None

        inputs = self.processor(audio_data, sampling_rate=self.sample_rate, return_tensors="pt")
        with torch.no_grad():
          forced_decoder_ids = self.processor.get_decoder_prompt_ids(
            language="italian", task="transcribe"
          )
          predicted_ids = self.model.generate(
            inputs["input_features"],
            forced_decoder_ids=forced_decoder_ids,
            max_new_tokens=448,
            do_sample=False,
            num_beams=1
          )
          
          transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
          return transcription.strip()
      except Exception as e:
        print(f"Transcription error: {e}")
        
        
    """ Funzione che verifica la presenza di byte all'interno della audio_queue
        e chiama transcript_audio. """
    def process_audio_queue(self):
      while self.is_recording or not self.audio_queue.empty():
        try:
          if not self.audio_queue.empty():
            audio_data = self.audio_queue.get(timeout=1)
            transcript = self.transcribe_audio(audio_data)
            
            if transcript:
              print(f"Transcript: {transcript}")
              
            self.audio_queue.task_done()
          else:
            time.sleep(0.1)
        except Exception as e:
          print(f"Processing error: {e}")
          break
    
    
    """ Funzione che inizializza l'audio thread per catturare l'audio del microfono
         e il thread che processa la coda audio."""  
    def start_recording(self, device_index):
      # se stai già registrando o non c'è modello salta 
      if self.is_recording or self.model is None:
        return
      
      self.is_recording = True
      
      self.audio_thread = threading.Thread(
        target=self.record_audio_chunk,
        args=(device_index, ),
        deamon=True
      )
      self.processing_thread = threading.Thread(
        target=self.process_audio_queue,
        deamon=True
      )
      self.audio_thread.start()
      self.processing_thread.start()
      
      try:
        while self.is_recording:
          time.sleep(1)
      except KeyboardInterrupt:
        self.stop_recording()
        
    """ Funzione che ferma la registrazione, in particolare controlla che ci sia l'attributo
        audio_thread nell'oggetto e che sia attivo e svuota la audio queue per trascrivere
        gli ultimi byte rimasti. Infine termina l'oggetto audio."""
    def stop_recording(self):
      if not self.is_recording:
        return
      
      self.is_recording = False
      
      if hasattr(self, 'audio_thread') and self.audio_thread.is_alive():
        # aspetta finché il thread non termina
        self.audio_thread.join(timeout=3)
        
      while not self.audio_queue.empty():
        try:
          audio_data = self.audio_queue.get_nowait()
          transcript = self.transcribe_audio(audio_data)
          if transcript and transcript.strip():
            print(f"Transcript: {transcript}")
        except:
          break
      self.audio.terminate()
      print("Stopped")
      
        

      
