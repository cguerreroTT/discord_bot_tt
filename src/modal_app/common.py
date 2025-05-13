import pathlib
import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from modal import App, Image, Secret, Volume
import sqlite3
import sqlite_vec
import struct
from typing import List
from openai import OpenAI


# AI CONSTANTS
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "decide_approach",
            "description": "Decide whether to use similarity search (RAG) or to run a SQL query over the DB.",
            "parameters": {
                "type": "object",
                "properties": {
                    "approach": {
                        "type": "string",
                        "enum": ["rag", "sql"],
                        "description": "Which approach to use? 'rag' or 'sql'."
                    },
                    "sql_query": {
                        "type": "string",
                        "description": "If 'sql' approach, the SQL to be executed."
                    }
                },
                "required": ["approach"]
            }
        },
    },
]

#Discord CONSTANTS
DEFAULT_LIMIT = 100

# DB CONSTANTS
DB_FILENAME = "discord_messages.db"
VOLUME_DIR = "/cache-vol"
DB_PATH = pathlib.Path(VOLUME_DIR, DB_FILENAME)

volume = Volume.from_name("sqlite-db-vol", create_if_missing=True)
image = Image.debian_slim().pip_install_from_pyproject("pyproject.toml")

secrets = Secret.from_dotenv()

app = App(name="starter_template", secrets=[secrets], image=image)

load_dotenv()

# OpenAI
GPT_MODEL = "gpt-4.1"
client = OpenAI(api_key=os.environ["OPENAI_API_KEY_TT"])

# Create a FastAPI instance here so it can be shared across modules
fastapi_app = FastAPI()

# Configure CORS
allowed_origins_str = os.environ["ALLOW_ORIGINS"]
allowed_origins_list = [origin.strip() for origin in allowed_origins_str.split(",")]
fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins_list,
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


def serialize(vector: List[float]) -> bytes:
    """Serializes a list of floats into a compact 'raw bytes' format."""
    return struct.pack(f"{len(vector)}f", *vector)

def get_db_conn(db_path):
    conn = sqlite3.connect(DB_PATH)
    sqlite_vec.load(conn)
    return conn
