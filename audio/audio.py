import json
import websockets
import os
import time
import sys
from queue import Queue
import base64
import asyncio
import  signal
import pyaudio
import threading

class STTWebSocket:

    def __init__(self):
        self.api_key = os.getenv("OPEN_AI_KEY")
        self.websocket_url = "wss://api.openai.com/v1/realtime?model=gpt-4o-mini-realtime-preview-2024-12-17"

        # audio settings
        self.chunk_size = 1024
        self.sample_format = pyaudio.paInt16
        self.channels = 1
        self.sample_rate = 44100 # frequency

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
    def signal_handler(self, sig, frame):
        self.stop_recording()
        sys.exit(0)

    """ Funzione che lista i dispositivi audio connessi alla Pi. 
        Utilizzata solo per riconoscere il nome del dispositivo audio connesso. """
    def list_audio_dev(self):
        print("Available audio devices: ")
        for i in range(self.audio.get_device_count()):
            print(f"{i} : {self.audio.get_device_info_by_index(i)}")

    """ Funzione che inizializza la sessione websocket.
        Specifica inoltre i parametri su cui si terrà la conversazione. 
        (Studia meglio il funzionamento del dizionario definito in questa funzione)"""
    async def initialize_session(self):
        session_config = {
            "type" : "session.update",
            "session" : {
                "modalities": ["text", "audio"],
                "instructions": "Sei Aeris e per adesso devi solo trascrivere testo senza rispondermi."
                "and responding in very concise way.",
                "voice": "alloy",
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "input_audio_transcription": {
                    "model": "whisper-1",
                    "language": "it"
                },
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.5,
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": 1000
                }
            } # Ipotesi Tools cerca sulle docs
        }

        await self.websocket.send(json.dumps(session_config))

    """ Funzione che invia i dati audio al WebSocket.
        Nel while loop si verifica che la coda di audio non sia vuota. A quel punto si converte
        il dato audio in base64 e si invia il messaggio codificato al websocket. """
    async def send_audio_data(self):
        while self.is_recording or not self.audio_queue.empty():
            try:
                if not self.audio_queue.empty():
                    audio_data = self.audio_queue.get(timeout=0.1) #blocca se non c'è un oggetto in quel timeout

                    # converte dati audio in base64
                    audio_base64 = base64.b64encode(audio_data).decode('utf-8')

                    message = {
                        "type" : "input_audio_buffer.append",
                        "audio" : audio_base64
                    }

                    await self.websocket.send(json.dumps(message))
                
                else:
                    await asyncio.sleep(0.1)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                print(f"Generic error: {e}")
                break

    """ Metodo con funzione principalmente di debugging (verrà cambiato in futuro).
        Utile per vedere le risposte ottenute dal websocket in base a ciò che si riceve."""
    async def process_websocket(self, data):
        message_type = data.get('type')

        if message_type == 'conversation.item.input_audio_transcription.completed':
            transcript = data.get('transcript', '').strip()
            if transcript:
                print(f"Transcript: {transcript}")
        elif message_type == 'input_audio_buffer.speech_started':
            print("Speech detected.")
        elif message_type == 'input_audio_buffer.speech_stopped':
            print("Speech stopped.")
        elif message_type == 'session.created':
            print("Session created.")
        

    """ Funzione che gestisce i messaggi ricevuti dal websocket. 
        Richiama process_websocket() per il processamento degli stessi. !!!"""
    async def handle_websocket_message(self):
        try:
            async for message in self.websocket:
                try:
                    data = json.loads(message)
                    await self.process_websocket(data)
                except json.JSONDecodeError as e:
                    print(f"Errore nel parsing del messaggio ricevuto dal websocket: {e}")
                except Exception as e:
                    print(f"Generic error: {e}")
        except websockets.exceptions.ConnectionClosed:
            print("Connessione con il websocket chiusa")
        except Exception as e:
            print(f"Websocket generic error: {e}")

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

                await self.initialize_session()

                audio_task = asyncio.create_task(self.send_audio_data())

                await self.handle_websocket_message()

                await audio_task # attendi che send_audio_data finisca 


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
            target=self.capture_audio,
            args=(device_index, ),
            daemon=True
        )
        self.audio_thread.start()

        try:
            asyncio.run(self.websocket_handler())
        except KeyboardInterrupt:
            self.stop_recording()

    """ Funzione di registrazione dello stream audio dal microfono.
        Funzione che passa i parametri di configurazione all'oggetto PyAudio.
        Legge inoltre i byte tramite un oggetto Stream. Infine la Stream viene
        fermata e chiusa."""
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
                stream_config['input_device_index'] = device_index
            
            stream = self.audio.open(**stream_config) # Spacchetta il dizionario e passa gli argomenti con nome a una funzione

            while self.is_recording:
                try:
                    # leggi byte audio e restituiscili all'oggetto PyAudio
                    audio_data = stream.read(self.chunk_size, exception_on_overflow=False)
                    self.audio_queue.put(audio_data)
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
            self.audio_thread.join(timeout=2) # aspetta timeout e termina il thread

        print("Stop Recording...")
        
        # Termina PyAudio
        self.audio.terminate()

def main():
    stt = STTWebSocket()

    #stt.list_audio_dev()

    print(os.getenv("OPEN_AI_KEY"))


    stt.start_recording(device_index=0)

if __name__ == "__main__":
    main()