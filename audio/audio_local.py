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
        self.model = model

          # audio settings
        self.chunk_size = 1024
        self.sample_format = pyaudio.paInt16
        self.channels = 1
        self.sample_rate = 44100 # frequency
        self.record_seconds = 3

        self.audio_queue = Queue(maxsize=5)
        self.is_recording = False
        
        # Oggetto PyAudio
        self.audio = pyaudio.Pyaudio()
