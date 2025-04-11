import discord
from discord.sinks import WaveSink
import asyncio
import wave
import os
from datetime import datetime
import whisper
from transformers import pipeline
import torch
import json

import security

TOKEN = security.token

class VoiceRecorderCog(discord.Cog):
    def __init__(self, bot: discord.Bot):
        self.bot = bot

        # Определяем устройство для вычислений
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Предзагружаем модель по умолчанию и организуем кэш моделей
        self.default_model_name = "medium"
        self.loaded_whisper_models = {}
        self.loaded_whisper_models[self.default_model_name] = whisper.load_model(self.default_model_name, device=self.device)

        # Загрузчик для суммаризации (если потребуется)
        self.summarizer = pipeline("summarization", model="t5-small", device=0 if self.device=="cuda" else -1)

        # Файл для сохранения настроек серверов и загрузка настроек
        self.settings_file = "server_settings.json"
        self.server_settings = self.load_settings()  # Формат: { "<guild_id>": {"save_folder": ..., "model_name": ...} }

        # Эфемерное состояние для работы с непрерывной записью (только в оперативной памяти)
        self.continuous_recording = {}      # { "<guild_id>": bool }
        self.recording_loop_tasks = {}      # { "<guild_id>": asyncio.Task }
        self.transcript_paths = {}          # { "<guild_id>": transcript_path }

    def load_settings(self) -> dict:
        if os.path.exists(self.settings_file):
            try:
                with open(self.settings_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return {}
        else:
            return {}

    def save_settings(self):
        with open(self.settings_file, "w", encoding="utf-8") as f:
            json.dump(self.server_settings, f, ensure_ascii=False, indent=4)

    @discord.Cog.listener()
    async def on_ready(self):
        print(f"✅ Logged in as {self.bot.user}")

    @discord.slash_command(name="join", description="Присоединиться к голосовому каналу")
    async def join(self, ctx: discord.ApplicationContext):
        if ctx.author.voice:
            await ctx.author.voice.channel.connect()
            await ctx.respond("✅ Подключено к голосовому каналу.")
        else:
            await ctx.respond("❌ Вы должны находиться в голосовом канале.")

    @discord.slash_command(name="leave", description="Покинуть голосовой канал")
    async def leave(self, ctx: discord.ApplicationContext):
        if ctx.guild.voice_client:
            await ctx.guild.voice_client.disconnect()
            await ctx.respond("👋 Отключено.")
        else:
            await ctx.respond("❌ Я не подключён к голосовому каналу.")

    @discord.slash_command(name="record_once", description="Записать аудио на заданное количество секунд")
    async def record_once(self, ctx: discord.ApplicationContext, duration: int):
        vc = ctx.guild.voice_client
        if not vc:
            await ctx.respond("❌ Бот не находится в голосовом канале. Используйте /join.")
            return

        await ctx.respond(f"🎙 Идёт запись в течение {duration} секунд...")
        sink = WaveSink()
        vc.start_recording(sink, self.on_recording_complete_once, ctx.channel)
        await asyncio.sleep(duration)
        if vc.recording:
            vc.stop_recording()

    async def on_recording_complete_once(self, sink: WaveSink, channel: discord.TextChannel):
        guild_id = str(channel.guild.id)
        # Получение настроек: если для сервера их нет, используем значения по умолчанию
        settings = self.server_settings.get(guild_id, {"save_folder": "transcripts", "model_name": self.default_model_name})
        save_folder = settings.get("save_folder", "transcripts")
        os.makedirs(save_folder, exist_ok=True)
        transcript_file = os.path.join(save_folder, f"recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

        for user_id, audio in sink.audio_data.items():
            try:
                filename = f"temp_{user_id}.wav"
                with wave.open(filename, "wb") as f:
                    f.setnchannels(2)
                    f.setsampwidth(2)
                    f.setframerate(48000)
                    f.writeframes(audio.file.getvalue())

                model_name = settings.get("model_name", self.default_model_name)
                model = self.loaded_whisper_models.get(model_name)
                if model is None:
                    model = await asyncio.to_thread(whisper.load_model, model_name, device=self.device)
                    self.loaded_whisper_models[model_name] = model

                result = await asyncio.to_thread(model.transcribe, filename)
                os.remove(filename)

                user = await self.bot.fetch_user(user_id)
                timestamp = datetime.now().strftime("%H:%M:%S")
                text = f"[{timestamp}] {user.display_name}: {result['text'].strip()}\n"

                with open(transcript_file, "a", encoding="utf-8") as f:
                    f.write(text)

                print(f"📝 {text.strip()}")
            except Exception as e:
                await channel.send(f"❌ Ошибка при дешифровке: {e}")

        await channel.send(f"✅ Запись завершена. Транскрипт сохранён в `{transcript_file}`")

    @discord.slash_command(name="record_continuous", description="Начать непрерывную запись (30-секундными отрезками)")
    async def record_continuous(self, ctx: discord.ApplicationContext):
        guild_id = str(ctx.guild.id)
        vc = ctx.guild.voice_client
        if not vc:
            await ctx.respond("❌ Бот не находится в голосовом канале. Используйте /join.")
            return

        if self.continuous_recording.get(guild_id, False):
            await ctx.respond("⚠️ Запись уже идёт. Используйте /stop_recording чтобы остановить.")
            return

        settings = self.server_settings.get(guild_id, {"save_folder": "transcripts", "model_name": self.default_model_name})
        save_folder = settings.get("save_folder", "transcripts")
        os.makedirs(save_folder, exist_ok=True)
        self.transcript_paths[guild_id] = os.path.join(save_folder, f"transcript_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        self.continuous_recording[guild_id] = True

        await ctx.respond(f"🎙 Запись начата. Текст будет сохранён в `{self.transcript_paths[guild_id]}`")
        self.recording_loop_tasks[guild_id] = asyncio.create_task(self.recording_loop(ctx, vc))

    @discord.slash_command(name="stop_recording", description="Остановить непрерывную запись")
    async def stop_recording(self, ctx: discord.ApplicationContext):
        guild_id = str(ctx.guild.id)
        if not self.continuous_recording.get(guild_id, False):
            await ctx.respond("ℹ️ Запись не активна.")
            return

        self.continuous_recording[guild_id] = False
        await ctx.respond("🛑 Запись остановлена.")

    async def recording_loop(self, ctx: discord.ApplicationContext, vc: discord.VoiceClient):
        guild_id = str(ctx.guild.id)
        while self.continuous_recording.get(guild_id, False) and vc and vc.is_connected():
            if not vc.channel.members or len([m for m in vc.channel.members if not m.bot]) == 0:
                await ctx.followup.send("👥 Все покинули голосовой канал. Остановка записи.")
                break

            sink = WaveSink()
            vc.start_recording(sink, self.on_recording_complete, ctx.channel)
            await asyncio.sleep(30)
            if vc.recording:
                vc.stop_recording()
            await asyncio.sleep(1)

        self.continuous_recording[guild_id] = False
        await ctx.followup.send("✅ Непрерывная запись завершена.")

    async def on_recording_complete(self, sink: WaveSink, channel: discord.TextChannel):
        guild_id = str(channel.guild.id)
        settings = self.server_settings.get(guild_id, {"save_folder": "transcripts", "model_name": self.default_model_name})
        for user_id, audio in sink.audio_data.items():
            try:
                filename = f"temp_{user_id}.wav"
                with wave.open(filename, "wb") as f:
                    f.setnchannels(2)
                    f.setsampwidth(2)
                    f.setframerate(48000)
                    f.writeframes(audio.file.getvalue())

                model_name = settings.get("model_name", self.default_model_name)
                model = self.loaded_whisper_models.get(model_name)
                if model is None:
                    model = await asyncio.to_thread(whisper.load_model, model_name, device=self.device)
                    self.loaded_whisper_models[model_name] = model

                result = await asyncio.to_thread(model.transcribe, filename)
                os.remove(filename)

                user = await self.bot.fetch_user(user_id)
                timestamp = datetime.now().strftime("%H:%M:%S")
                text = f"[{timestamp}] {user.display_name}: {result['text'].strip()}\n"

                transcript_path = self.transcript_paths.get(guild_id, "transcript_default.txt")
                with open(transcript_path, "a", encoding="utf-8") as f:
                    f.write(text)

                print(f"📝 {text.strip()}")
            except Exception as e:
                await channel.send(f"❌ Ошибка при дешифровке: {e}")

    @discord.slash_command(name="set_save_folder", description="Установить папку для сохранения транскриптов для этого сервера")
    async def set_save_folder(self, ctx: discord.ApplicationContext, folder: str):
        guild_id = str(ctx.guild.id)
        settings = self.server_settings.get(guild_id, {"save_folder": "transcripts", "model_name": self.default_model_name})
        settings["save_folder"] = folder
        self.server_settings[guild_id] = settings
        self.save_settings()
        await ctx.respond(f"✅ Папка для сохранения установлена в `{folder}` для этого сервера.")

    @discord.slash_command(name="set_transcription_model", description="Установить модель для дешифровки (транскрипции) для этого сервера")
    async def set_transcription_model(self, ctx: discord.ApplicationContext, model_name: str):
        guild_id = str(ctx.guild.id)
        settings = self.server_settings.get(guild_id, {"save_folder": "transcripts", "model_name": self.default_model_name})
        settings["model_name"] = model_name
        self.server_settings[guild_id] = settings
        self.save_settings()

        if model_name not in self.loaded_whisper_models:
            await ctx.respond(f"Загружаю модель `{model_name}`... Это может занять некоторое время.")
            model = await asyncio.to_thread(whisper.load_model, model_name, device=self.device)
            self.loaded_whisper_models[model_name] = model
            await ctx.followup.send(f"✅ Модель для транскрипции установлена в `{model_name}` для этого сервера.")
        else:
            await ctx.respond(f"✅ Модель для транскрипции установлена в `{model_name}` для этого сервера.")

# Создаём инстанс бота и регистрируем Cog
intents = discord.Intents.all()
bot = discord.Bot(intents=intents)
bot.add_cog(VoiceRecorderCog(bot))

bot.run(TOKEN)
