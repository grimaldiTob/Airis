from openai import OpenAI
from audio.audio_local import AerisEars
import os

class AerisMind:
    def __init__(self, model="gpt-4.1-mini"):
        self.model = model
        self.api_key = os.getenv("OPENAI_API_KEY")
        
        # array di pezzi di trascrizione ottenuti dal modello whisper di trascrizione
        self.input_transcription = []
        
        # instanzia il client di OpenAI
        self.client = OpenAI(api_key=self.api_key)
        self.istructions = "Sei un assistente AI di nome Aeris. Rispondi in maniera simpatica e concisa senza usare emoji."
        self.response = None
        
    def create_response(self, prompt: str):
        
        try:
            response = self.client.responses.create(
                model=self.model,
                input=[
                    {
                        "role": "developer",
                        "content": self.istructions
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_output_tokens=400,
                text={
                    "verbosity": "medium"
                }
            )
            
            self.response = response.output_text
            
            return self.response
            
        except Exception as e:
            print(f"Errore nella creazione della risposta: {e}")
            return        
        
        
        