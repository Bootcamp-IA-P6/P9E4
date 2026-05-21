import os
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

def get_client():
    url = os.getenv('SUPABASE_URL')
    key = os.getenv('SUPABASE_KEY')
    return create_client(url, key)

def save_prediction(text: str, text_clean: str, prediction: int,
                    probability: float, source: str = 'manual',
                    video_url: str = None) -> str:
    client = get_client()
    data = {
        'text':        text,
        'text_clean':  text_clean,
        'prediction':  prediction,
        'probability': probability,
        'source':      source,
        'video_url':   video_url
    }
    result = client.table('predictions').insert(data).execute()
    return result.data[0]['id']

def save_feedback(prediction_id: str, is_correct: bool, comment: str = None):
    client = get_client()
    data = {
        'prediction_id': prediction_id,
        'is_correct':    is_correct,
        'comment':       comment
    }
    client.table('feedback').insert(data).execute()

def get_predictions(limit: int = 100):
    client = get_client()
    result = client.table('predictions')\
                .select('*')\
                .order('created_at', desc=True)\
                .limit(limit)\
                .execute()
    return result.data

def get_feedback():
    client = get_client()
    result = client.table('feedback')\
                .select('*, predictions(*)')\
                .order('created_at', desc=True)\
                .execute()
    return result.data