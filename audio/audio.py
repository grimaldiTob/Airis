import json
import websockets
import os
import time
import sys
from queue import Queue
import asyncio
import  signal
import pyaudio
import threading

class STTWebSocket:

    def __init__(self):
        self.api_key = os.getenv("OPEN_AI_KEY")
        self.websocket_url = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-12-17"

        # audio settings
        self.chunk_size = 1024
        self.sample_format = pyaudio.paInt16
        self.channels = 1
        self.sample_rate = 24000 # frequency for OpenAi realtime API

        # Threading and control
        self.audio_queue = Queue()
        self.is_recording = False
        self.websocket = None
        self.audio_thread = None

        # oggetto di PyAudio
        self.audio = pyaudio.PyAudio()

        # signal handler 
        signal.signal(signal.SIGINT, self.signal_handler)

    """ Funzione che permette una gracefull degradation
        attraverso l'handling dei segnali. """
    def signal_handler(self):
        self.stop_record()
        sys.exit(0)

    """ Funzione che lista i dispositivi audio connessi alla Pi. 
        Utilizzata solo per riconoscere il nome del dispositivo audio connesso. """
    def list_audio_dev(self):
        print("Available audio devices: ")
        for i in range(self.audio.get_device_count()):
            print(f"{i} : {self.audio.get_device_info_by_index(i).get('name')}")

    """ FUnzione che gestisce la connessione asincrona del websocket all'Api di OpenAI.
        Si definiscono prima gli header specificando la chiave API. 
        """
    async def websocket_handler(self):
        headers = {
            "Authorization":f"Bearer {self.api_key}",
            "OpenAI-Beta": "realtime=v1"
        }

        try:
            async with websockets.connect(
                self.websocket_url,
                additional_headers=headers,
                ping_interval=20,
                ping_timeout=10
            ) as websocket:
                self.websocket = websocket

                # TODO

        except websockets.exceptions.InvalidStatus as e:
            print(f"Connessione fallita al Websocket: {e}")
        except Exception as e:
            print(f"Errore generico: {e}") 


    """ Funzione che verifica parametri self.is_recording e self.api_key...
        Successivamente crea il thread audio che si occupa di catturare la traccia
        audio registrata dal microfono.
        
        Parallelamente viene creato il websocket che si occuperà di inviare le richieste
        all'API di OpenAI"""
    def start_recording(self, device_index=None):
        if self.is_recording:
            return
        
        if not self.api_key:
            print("Generic error to the API key")
            return
        
        self.is_recording = True

        # Thread settings 
        self.audio_thread = threading.Thread(
            target=self.capture_audio, # RICORDA DI IMPLEMENTARE
            args=(device_index, ),
            daemon=True
        )
        self.audio_thread.start()

        try:
            asyncio.run(self.websocket_handler())
        except KeyboardInterrupt:
            self.stop_recording()

    def capture_audio(self, device_index=None):
        try:
            stream_config = {
                'format' : self.sample_format,
                'channels' : self.channels,
                'rate' : self.sample_rate,
                'frames_per_buffer' : self.chunk_size,
                'input' : True
            }

            if device_index is not None:
                stream_config['input_device_index'] = device_index # DA VERIFICARE
            
            stream = self.audio.open(**stream_config) # DA VERIFICARE

            while self.is_recording:
                try:
                    # leggi byte audio e restituiscili all'oggetto PyAudio
                    audio_data = stream.read(self.chunk_size, exception_on_overflow=False)
                    self.audio.put(audio_data)
                except Exception as e:
                    print(f"Errore generico: {e}")
                    break
            
            stream.stop_stream()
            stream.close()
        except Exception as e:
            print(f"Errore generico: {e}")

    """ Funzione che ferma la registrazione dal dispositivo audio iniziata da PyAudio
        terminando il thread di riferimento se ancora attivo. """
    def stop_recording(self):
        if not self.is_recording:
            return

        self.is_recording = False

        if self.audio_thread.is_alive():
            self.audio_thread.join(timeout=2)
        
        # Termina PyAudio
        self.audio.terminate()

def main():
    stt = STTWebSocket()

    stt.list_audio_dev()

if __name__ == "__main__":
    main()