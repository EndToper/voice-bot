import discord
from discord.ext import commands
import asyncio
import wave
import os
from datetime import datetime
import whisper
from transformers import pipeline

import security

TOKEN = security.token
GUILD_ID = 1333447583947034705

intents = discord.Intents.default()
intents.message_content = True
intents.voice_states = True
intents.guilds = True
intents.members = True

bot = commands.Bot(command_prefix="!", intents=intents)

whisper_model = whisper.load_model("base")
summarizer = pipeline("summarization", model="t5-small")

@bot.event
async def on_ready():
    print(f"Бот запущен как {bot.user}")

@bot.command(name="join")
async def join(ctx):
    if ctx.author.voice:
        channel = ctx.author.voice.channel
        await channel.connect()
        await ctx.send("✅ Подключился к голосовому каналу.")
    else:
        await ctx.send("❌ Вы не находитесь в голосовом канале.")

@bot.command(name="record")
async def record(ctx, duration: int = 10):
    vc = ctx.voice_client
    if not vc:
        await ctx.send("❌ Бот не подключен к голосовому каналу.")
        return

    audio_sink = discord.sinks.WaveSink()
    vc.start_recording(
        audio_sink,
        once_done,
        ctx.channel
    )
    await ctx.send(f"🎧 Начинаю запись на {duration} секунд...")
    await asyncio.sleep(duration)
    vc.stop_recording()

async def once_done(sink: discord.sinks.WaveSink, channel: discord.TextChannel):
    recorded_files = []

    for user_id, audio in sink.audio_data.items():
        try:
            # Получаем имя пользователя
            user = await channel.guild.fetch_member(user_id)
            username = user.display_name if user else f"user_{user_id}"
        except Exception:
            username = f"user_{user_id}"

        # Создаём путь и имя файла
        filename = f"{username}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
        filepath = os.path.join("recordings", filename)

        # Убеждаемся, что папка существует
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        try:
            # Сохраняем WAV-файл с высоким качеством
            with wave.open(filepath, "wb") as f:
                f.setnchannels(2)          # Стерео
                f.setsampwidth(2)          # 16 бит = 2 байта
                f.setframerate(48000)      # 48kHz — стандарт Discord
                f.writeframes(audio.file.getvalue())

            recorded_files.append((user_id, filepath))
            print(f"✅ Сохранено аудио: {filepath}")
        except Exception as e:
            await channel.send(f"❌ Ошибка при сохранении аудио от {username}: {e}")

    if recorded_files:
        await channel.send("📥 Обработка аудиофайлов...")
        await process_audio_and_respond(channel, recorded_files)
    else:
        await channel.send("⚠️ Нет записанных аудиофайлов.")


async def process_audio_and_respond(channel, files):
    full_text = ""

    for user_id, filepath in files:
        username = await bot.fetch_user(user_id)
        await channel.send(f"🔍 Распознаю голос {username}...")
        result = whisper_model.transcribe(filepath)
        text = result["text"].strip()
        full_text += f"{username}: {text}\n"
        os.remove(filepath)

    if not full_text.strip():
        await channel.send("⚠️ Никто ничего не сказал.")
        return

    await channel.send(f"📜 Расшифровка:\n```{full_text[:1900]}```")

    try:
        summary_input = "summarize: " + full_text
        summary = summarizer(summary_input, max_length=100, min_length=30, do_sample=False)[0]["summary_text"]
        await channel.send(f"🧠 Сводка:\n```{summary}```")
    except Exception as e:
        await channel.send(f"❌ Ошибка суммирования: {str(e)}")

@bot.command(name="leave")
async def leave(ctx):
    if ctx.voice_client:
        await ctx.voice_client.disconnect()
        await ctx.send("👋 Отключился от голосового канала.")
    else:
        await ctx.send("❌ Бот не подключен к голосовому каналу.")

bot.run(TOKEN)
