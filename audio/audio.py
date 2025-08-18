import json
import websockets
import os
import time
import sys
from queue import Queue
import asyncio
import  signal
import pyaudio

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


        except websockets.exceptions.InvalidStatus as e:
            print(f"Connessione fallita al Websocket: {e}")
        except Exception as e:
            print(f"Errore generico: {e}") 

def main():
    stt = STTWebSocket()

    stt.list_audio_dev()

if __name__ == "__main__":
    main()