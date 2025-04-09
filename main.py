import discord
from discord.sinks import WaveSink
import asyncio
import wave
import os
from datetime import datetime
import whisper
from transformers import pipeline
import torch

import security

TOKEN = security.token

intents = discord.Intents.all()
bot = discord.Bot(intents=intents)

# Load models
device = "cuda" if torch.cuda.is_available() else "cpu"
whisper_model = whisper.load_model("medium", device=device)
summarizer = pipeline("summarization", model="t5-small", device=0 if device == "cuda" else -1)

# Globals
continuous_recording = False
recording_loop_task = None
transcript_path = None


@bot.event
async def on_ready():
    print(f"✅ Logged in as {bot.user}")


@bot.slash_command(name="join", description="Join your voice channel")
async def join(ctx: discord.ApplicationContext):
    if ctx.author.voice:
        await ctx.author.voice.channel.connect()
        await ctx.respond("✅ Connected to voice channel.")
    else:
        await ctx.respond("❌ You must be in a voice channel.")


@bot.slash_command(name="leave", description="Leave the voice channel")
async def leave(ctx: discord.ApplicationContext):
    if ctx.guild.voice_client:
        await ctx.guild.voice_client.disconnect()
        await ctx.respond("👋 Disconnected.")
    else:
        await ctx.respond("❌ I'm not connected to a voice channel.")


@bot.slash_command(name="record_once", description="Record for a specific number of seconds")
async def record_once(ctx: discord.ApplicationContext, duration: int):
    vc = ctx.guild.voice_client
    if not vc:
        await ctx.respond("❌ Bot is not in a voice channel. Use /join.")
        return

    await ctx.respond(f"🎙 Recording for {duration} seconds...")
    sink = WaveSink()
    vc.start_recording(sink, on_recording_complete_once, ctx.channel)
    await asyncio.sleep(duration)
    if vc.recording:
        vc.stop_recording()


async def on_recording_complete_once(sink: WaveSink, channel: discord.TextChannel):
    os.makedirs("transcripts", exist_ok=True)
    temp_path = os.path.join("transcripts", f"recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

    for user_id, audio in sink.audio_data.items():
        try:
            filename = f"temp_{user_id}.wav"
            with wave.open(filename, "wb") as f:
                f.setnchannels(2)
                f.setsampwidth(2)
                f.setframerate(48000)
                f.writeframes(audio.file.getvalue())

            result = whisper_model.transcribe(filename)
            os.remove(filename)

            user = await bot.fetch_user(user_id)
            timestamp = datetime.now().strftime("%H:%M:%S")
            text = f"[{timestamp}] {user.display_name}: {result['text'].strip()}\n"

            with open(temp_path, "a", encoding="utf-8") as f:
                f.write(text)

            print(f"📝 {text.strip()}")

        except Exception as e:
            await channel.send(f"❌ Error during transcription: {e}")

    await channel.send(f"✅ Recording complete. Transcript saved to `{temp_path}`")


@bot.slash_command(name="record_continuous", description="Start continuous 30s chunk recording")
async def record_continuous(ctx: discord.ApplicationContext):
    global continuous_recording, recording_loop_task, transcript_path

    vc = ctx.guild.voice_client
    if not vc:
        await ctx.respond("❌ Bot is not in a voice channel. Use /join.")
        return

    if continuous_recording:
        await ctx.respond("⚠️ Already recording. Use /stop_recording.")
        return

    os.makedirs("transcripts", exist_ok=True)
    transcript_path = os.path.join("transcripts", f"transcript_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

    continuous_recording = True
    await ctx.respond(f"🎙 Recording started. Text saved to `{transcript_path}`")
    recording_loop_task = asyncio.create_task(recording_loop(ctx, vc))


@bot.slash_command(name="stop_recording", description="Stop continuous recording")
async def stop_recording(ctx: discord.ApplicationContext):
    global continuous_recording

    if not continuous_recording:
        await ctx.respond("ℹ️ Not currently recording.")
        return

    continuous_recording = False
    await ctx.respond("🛑 Recording stopped.")


async def recording_loop(ctx: discord.ApplicationContext, vc: discord.VoiceClient):
    global continuous_recording, transcript_path

    while continuous_recording and vc and vc.is_connected():
        if not vc.channel.members or len([m for m in vc.channel.members if not m.bot]) == 0:
            await ctx.followup.send("👥 Everyone left the voice channel. Stopping.")
            break

        sink = WaveSink()
        vc.start_recording(sink, on_recording_complete, ctx.channel)
        await asyncio.sleep(30)
        if vc.recording:
            vc.stop_recording()
        await asyncio.sleep(1)

    continuous_recording = False
    await ctx.followup.send("✅ Continuous transcription ended.")


async def on_recording_complete(sink: WaveSink, channel: discord.TextChannel):
    global transcript_path

    for user_id, audio in sink.audio_data.items():
        try:
            filename = f"temp_{user_id}.wav"
            with wave.open(filename, "wb") as f:
                f.setnchannels(2)
                f.setsampwidth(2)
                f.setframerate(48000)
                f.writeframes(audio.file.getvalue())

            result = whisper_model.transcribe(filename)
            os.remove(filename)

            user = await bot.fetch_user(user_id)
            timestamp = datetime.now().strftime("%H:%M:%S")
            text = f"[{timestamp}] {user.display_name}: {result['text'].strip()}\n"

            with open(transcript_path, "a", encoding="utf-8") as f:
                f.write(text)

            print(f"📝 {text.strip()}")

        except Exception as e:
            await channel.send(f"❌ Error during transcription: {e}")


bot.run(TOKEN)
