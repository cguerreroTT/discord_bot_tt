import sqlite3
import requests
import os
from .common import DB_PATH, DEFAULT_LIMIT, volume, get_db_conn, serialize
from typing import Dict
from openai import OpenAI


DISCORD_BASE_URL = "https://discord.com/api/v10"

def fetch_and_store_channel_messages(channel_id: str, channel_name: str, headers: Dict, limit: int = DEFAULT_LIMIT) -> None:
    """Fetch up to `limit` messages from the given channel and store in SQLite."""

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    url = f"{DISCORD_BASE_URL}/channels/{channel_id}/messages?limit={limit}"
    resp = requests.get(url, headers=headers)
    if resp.status_code == 403:
            print(f"[403 Forbidden] Skipping channel {channel_id} — no permission.")
            return

    resp.raise_for_status()
    messages = resp.json()

    conn = get_db_conn(DB_PATH)
    cursor = conn.cursor()

    # insert metadata for the channel
    cursor.execute("""
        INSERT OR REPLACE INTO channel_metadata (channel_id, channel_name)
        VALUES (?, ?)
    """, (channel_id, channel_name))

    for msg in messages:
            message_id = msg["id"]
            author_id = msg["author"]["id"]
            content = msg["content"]
            timestamp = msg["timestamp"]  # e.g. '2025-01-01T00:00:00.000000+00:00'


            # Insert or ignore if row already exists
            cursor.execute(
                """
                INSERT OR IGNORE INTO discord_messages (id, channel_id, author_id, content, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (message_id, channel_id, author_id, content, timestamp),
            )
            embedding = client.embeddings.create(model="text-embedding-ada-002", input=content).data[0].embedding
            cursor.execute(
                "SELECT id FROM vec_discord_messages WHERE id = ?", (message_id,)
            )
            row = cursor.fetchone()

            if row is None:
                # Row does not exist, so insert
                cursor.execute(
                    """
                    INSERT INTO vec_discord_messages (id, embedding)
                    VALUES (?, ?)
                    """,
                    (
                        message_id,
                        serialize(embedding),
                    ),
                )
            else:
                # Row exists, so update
                cursor.execute(
                    """
                    UPDATE vec_discord_messages
                    SET embedding = ?
                    WHERE id = ?
                    """,
                    (serialize(embedding), message_id),
                )


    conn.commit()
    conn.close()
    # volume.commit()

def scrape_discord_server(guild_id: str, headers: Dict, limit: int = DEFAULT_LIMIT) -> None:
    """
    Fetch all channels from the given Discord server (guild),
    filter for text channels, and then fetch recent messages
    for each one, storing them in the database.
    """
    # 1) Get all channels
    channels_url = f"{DISCORD_BASE_URL}/guilds/{guild_id}/channels"
    response = requests.get(channels_url, headers=headers)
    response.raise_for_status()
    channels = response.json()

    # 2) Iterate over channels and scrape if it's a text channel
    for channel in channels:
        # Discord 'type=0' => GUILD_TEXT (i.e. text channel)
        if channel.get("type") == 0:
            channel_id = channel["id"]
            channel_name = channel["name"]
            print(f"Scraping channel: {channel_name} (ID: {channel_id})")
            fetch_and_store_channel_messages(channel_id, channel_name, headers, limit=limit)

    print(f"Done scraping up to {limit} messages from all text channels in guild {guild_id}.")
