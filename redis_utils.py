# redis_utils.py
import redis
import os
import pickle
import json
import io
from dotenv import load_dotenv

load_dotenv()

REDIS_HOST = os.getenv("REDIS_HOST")
REDIS_PORT = int(os.getenv("REDIS_PORT"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")

r = redis.StrictRedis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    password=REDIS_PASSWORD,
    # decode_responses=True  # changed to True for human-readable keys
)

def cache_file(key: str, filepath: str):
    with open(filepath, "rb") as f:
        r.set(key, f.read())

def get_file_from_cache(key: str) -> io.BytesIO:
    data = r.get(key)
    return io.BytesIO(data) if data else None

def cache_json(key: str, obj: dict):
    r.set(key, json.dumps(obj))

def get_json(key: str) -> dict:
    data = r.get(key)
    return json.loads(data) if data else None

def cache_pickle(key: str, obj: any):
    r.set(key, pickle.dumps(obj))

def get_pickle(key: str):
    data = r.get(key)
    return pickle.loads(data) if data else None
