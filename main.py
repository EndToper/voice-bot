import discord
from discord.ext import commands
from discord.ext.voice_recv import VoiceRecvClient, AudioSink
import whisper
from transformers import pipeline
import asyncio
import wave
import os
from datetime import datetime
from security import token

# ====== НАСТРОЙКИ ======
TOKEN = token
GUILD_ID = 1333447583947034705

# ====== НАЧАЛО БОТА ======
intents = discord.Intents.default()
intents.message_content = True
intents.voice_states = True
intents.guilds = True
intents.members = True

bot = commands.Bot(command_prefix="!", intents=intents)
tree = bot.tree  # для slash-команд

# ====== МОДЕЛИ ======
whisper_model = whisper.load_model("base")
summarizer = pipeline("summarization", model="t5-small", tokenizer="t5-small")

# ====== СИНК ДЛЯ ЗАПИСИ ======
class RecordingSink(AudioSink):
    def __init__(self):
        self.audio_data = {}

    def write(self, user, data):
        if user.id not in self.audio_data:
            self.audio_data[user.id] = {"name": user.display_name, "frames": []}
        self.audio_data[user.id]["frames"].append(data)

    def save_to_wav(self):
        os.makedirs("recordings", exist_ok=True)
        files = []
        for uid, info in self.audio_data.items():
            frames = b"".join(info["frames"])
            filename = f"recordings/{info['name']}_{uid}_{datetime.now().strftime('%H%M%S')}.wav"
            with wave.open(filename, "wb") as wf:
                wf.setnchannels(2)
                wf.setsampwidth(2)
                wf.setframerate(48000)
                wf.writeframes(frames)
            files.append((info['name'], filename))
        return files

# ====== ОБРАБОТКА РАСПОЗНАВАНИЯ ======
async def process_audio_and_respond(interaction, files):
    full_text = ""

    for username, filename in files:
        await interaction.followup.send(f"🔍 Распознаю голос {username}...")
        result = whisper_model.transcribe(filename)
        text = result["text"].strip()
        full_text += f"{username}: {text}\n"
        os.remove(filename)

    if not full_text.strip():
        await interaction.followup.send("⚠️ Пусто. Никто ничего не сказал.")
        return

    await interaction.followup.send(f"📜 Расшифровка:\n```{full_text[:1900]}```")

    try:
        summary_input = "summarize: " + full_text
        summary = summarizer(summary_input, max_length=100, min_length=30, do_sample=False)[0]['summary_text']
        await interaction.followup.send(f"🧠 Сводка беседы:\n```{summary}```")
    except Exception as e:
        await interaction.followup.send(f"❌ Ошибка при создании сводки: {str(e)}")

# ====== /join ======
@tree.command(name="join", description="Подключить бота к голосовому каналу", guild=discord.Object(id=GUILD_ID))
async def join(interaction: discord.Interaction):
    if interaction.user.voice:
        channel = interaction.user.voice.channel
        await channel.connect(cls=VoiceRecvClient)
        await interaction.response.send_message("✅ Подключился к голосовому каналу.")
    else:
        await interaction.response.send_message("❌ Ты не в голосовом канале.")

# ====== /record ======
@tree.command(name="record", description="Записать голосовой чат и сделать краткий итог", guild=discord.Object(id=GUILD_ID))
async def record(interaction: discord.Interaction, duration: int = 10):
    vc = interaction.guild.voice_client
    if not vc or not isinstance(vc, VoiceRecvClient):
        await interaction.response.send_message("❌ Сначала вызови /join.")
        return

    sink = RecordingSink()
    vc.listen(sink)
    await interaction.response.send_message(f"🎧 Запись началась на {duration} секунд...")
    await asyncio.sleep(duration)
    vc.stop_listening()
    files = sink.save_to_wav()
    await process_audio_and_respond(interaction, files)

# ====== /leave ======
@tree.command(name="leave", description="Отключить бота от голосового канала", guild=discord.Object(id=GUILD_ID))
async def leave(interaction: discord.Interaction):
    if interaction.guild.voice_client:
        await interaction.guild.voice_client.disconnect()
        await interaction.response.send_message("👋 Отключился.")
    else:
        await interaction.response.send_message("❌ Я не подключён.")

# ====== СИНХРОНИЗАЦИЯ / КОМАНД ======
@bot.event
async def on_ready():
    print(f"✅ Бот запущен: {bot.user}")
    await tree.sync(guild=discord.Object(id=GUILD_ID))

# ====== ЗАПУСК ======
if __name__ == "__main__":
    bot.run(TOKEN)
