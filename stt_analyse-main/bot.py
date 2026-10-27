import asyncio
import logging
import mimetypes
import os
import re
import tempfile
import time
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Optional

import psycopg2
import requests
from dotenv import load_dotenv
from openai import OpenAI
from protocol_docx import write_protocol_docx
from telegram import Message, Update
from telegram.constants import ChatAction
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

ASSEMBLYAI_UPLOAD_URL = "https://api.assemblyai.com/v2/upload"
ASSEMBLYAI_TRANSCRIPT_URL = "https://api.assemblyai.com/v2/transcript"
MESSAGE_CHUNK_SIZE = 3900
DB_CONNECTION_RETRIES = 30
DB_CONNECTION_RETRY_DELAY_SECONDS = 2
DATA_FOLDER_POLL_SECONDS = 15
SUPPORTED_EXTENSIONS = {".mp3", ".mp4", ".mkv"}
PROCESS_LOCK = asyncio.Lock()

mimetypes.add_type("video/mp4", ".mp4")
mimetypes.add_type("video/x-matroska", ".mkv")
mimetypes.add_type("audio/mpeg", ".mp3")


class AssemblyAIError(RuntimeError):
    """Raised when AssemblyAI returns an error state."""


@dataclass
class Settings:
    telegram_bot_token: str
    assemblyai_api_key: str
    openai_api_key: str
    database_url: str
    openai_model: str = "gpt-4.1-mini"
    openai_base_url: str = "https://api.openai.com/v1"
    data_dir: str = "data"

    @classmethod
    def from_env(cls) -> "Settings":
        telegram_bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
        assemblyai_api_key = os.getenv("ASSEMBLYAI_API_KEY", "").strip()
        openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()
        database_url = os.getenv("DATABASE_URL", "").strip()
        openai_model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini").strip() or "gpt-4.1-mini"
        openai_base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").strip() or "https://api.openai.com/v1"
        data_dir = os.getenv("DATA_DIR", "data").strip() or "data"

        missing = []
        if not telegram_bot_token:
            missing.append("TELEGRAM_BOT_TOKEN")
        if not assemblyai_api_key:
            missing.append("ASSEMBLYAI_API_KEY")
        if not openai_api_key:
            missing.append("OPENAI_API_KEY")
        if not database_url:
            missing.append("DATABASE_URL")
        if missing:
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")

        return cls(
            telegram_bot_token=telegram_bot_token,
            assemblyai_api_key=assemblyai_api_key,
            openai_api_key=openai_api_key,
            database_url=database_url,
            openai_model=openai_model,
            openai_base_url=openai_base_url,
            data_dir=data_dir,
        )


def resolve_data_dir(settings: Settings) -> Path:
    path = Path(settings.data_dir)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent / path
    path.mkdir(parents=True, exist_ok=True)
    return path


def init_database(database_url: str) -> None:
    with psycopg2.connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS video_audits (
                    id BIGSERIAL PRIMARY KEY,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    chat_id BIGINT NOT NULL,
                    user_id BIGINT NULL,
                    username TEXT NULL,
                    file_id TEXT NOT NULL,
                    file_unique_id TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    transcript TEXT NULL,
                    analysis TEXT NULL,
                    status TEXT NOT NULL,
                    error_message TEXT NULL
                );
                """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_video_audits_chat_id
                ON video_audits (chat_id);
                """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_video_audits_created_at
                ON video_audits (created_at DESC);
                """
            )
        connection.commit()


def init_database_with_retry(database_url: str) -> None:
    for attempt in range(1, DB_CONNECTION_RETRIES + 1):
        try:
            init_database(database_url)
            logging.info("Postgres is ready. Database schema initialized.")
            return
        except Exception as error:
            logging.warning(
                "Postgres is not ready yet (attempt %s/%s): %s",
                attempt,
                DB_CONNECTION_RETRIES,
                error,
            )
            time.sleep(DB_CONNECTION_RETRY_DELAY_SECONDS)

    raise RuntimeError("Could not initialize Postgres schema after multiple attempts.")


def save_audit_result(
    database_url: str,
    chat_id: int,
    user_id: Optional[int],
    username: Optional[str],
    file_id: str,
    file_unique_id: str,
    filename: str,
    transcript: Optional[str],
    analysis: Optional[str],
    status: str,
    error_message: Optional[str],
) -> None:
    with psycopg2.connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO video_audits (
                    chat_id,
                    user_id,
                    username,
                    file_id,
                    file_unique_id,
                    filename,
                    transcript,
                    analysis,
                    status,
                    error_message
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
                """,
                (
                    chat_id,
                    user_id,
                    username,
                    file_id,
                    file_unique_id,
                    filename,
                    transcript,
                    analysis,
                    status,
                    error_message,
                ),
            )
        connection.commit()


def load_protocol_prompt(prompt_path: Path) -> str:
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file was not found: {prompt_path}")

    content = prompt_path.read_text(encoding="utf-8").strip()
    if not content:
        raise ValueError(f"Prompt file is empty: {prompt_path}")
    return content


def assemblyai_request(method: str, url: str, **kwargs) -> requests.Response:
    last_error: Optional[Exception] = None
    for attempt in range(1, 4):
        try:
            return requests.request(method, url, **kwargs)
        except requests.RequestException as error:
            last_error = error
            logging.warning(
                "AssemblyAI %s failed (attempt %s/3): %s",
                method.upper(),
                attempt,
                type(error).__name__,
            )
            if attempt < 3:
                time.sleep(2)
    assert last_error is not None
    raise last_error


def raise_for_assemblyai(response: requests.Response, action: str) -> None:
    if response.status_code < 400:
        return
    detail = (response.text or "").strip().replace("\n", " ")
    raise AssemblyAIError(f"{action}: {response.status_code} {detail[:500]}")


def upload_to_assemblyai(file_path: Path, api_key: str) -> str:
    headers = {"authorization": api_key}
    with file_path.open("rb") as audio_file:
        response = assemblyai_request(
            "POST",
            ASSEMBLYAI_UPLOAD_URL,
            headers=headers,
            data=audio_file,
            timeout=1200,
        )
    raise_for_assemblyai(response, "AssemblyAI upload failed")

    upload_url = response.json().get("upload_url")
    if not upload_url:
        raise AssemblyAIError("AssemblyAI did not return upload_url.")
    return upload_url


def format_timestamp(milliseconds: int) -> str:
    total_seconds = max(milliseconds, 0) // 1000
    minutes, seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def format_transcript(payload: dict) -> str:
    utterances = payload.get("utterances") or []
    lines: list[str] = []
    for item in utterances:
        text = (item.get("text") or "").strip()
        if not text:
            continue
        speaker = item.get("speaker") or "?"
        start = item.get("start")
        stamp = format_timestamp(start) if isinstance(start, int) else ""
        prefix = f"[{stamp}] " if stamp else ""
        lines.append(f"{prefix}Спикер {speaker}: {text}")
    if lines:
        return "\n".join(lines)

    text = (payload.get("text") or "").strip()
    if not text:
        raise AssemblyAIError("AssemblyAI returned empty transcript.")
    return text


def create_assemblyai_transcript(upload_url: str, api_key: str) -> str:
    headers = {"authorization": api_key, "content-type": "application/json"}
    payload = {
        "audio_url": upload_url,
        "speech_models": ["universal-3-5-pro", "universal-2"],
        "speaker_labels": True,
        "language_code": "ru",
        "punctuate": True,
        "format_text": True,
    }
    response = assemblyai_request(
        "POST",
        ASSEMBLYAI_TRANSCRIPT_URL,
        headers=headers,
        json=payload,
        timeout=60,
    )
    raise_for_assemblyai(response, "AssemblyAI transcript create failed")

    transcript_id = response.json().get("id")
    if not transcript_id:
        raise AssemblyAIError("AssemblyAI did not return transcript id.")
    return transcript_id


def poll_assemblyai_transcript(transcript_id: str, api_key: str, timeout_seconds: int = 3600) -> str:
    headers = {"authorization": api_key}
    deadline = time.monotonic() + timeout_seconds
    endpoint = f"{ASSEMBLYAI_TRANSCRIPT_URL}/{transcript_id}"

    while time.monotonic() < deadline:
        response = assemblyai_request("GET", endpoint, headers=headers, timeout=60)
        raise_for_assemblyai(response, "AssemblyAI transcript poll failed")
        payload = response.json()
        status = payload.get("status")

        if status == "completed":
            logging.info("AssemblyAI model used: %s", payload.get("speech_model_used"))
            return format_transcript(payload)

        if status == "error":
            raise AssemblyAIError(payload.get("error") or "Unknown AssemblyAI error.")

        time.sleep(3)

    raise TimeoutError("Timed out while waiting for AssemblyAI transcription.")


def transcribe_video(file_path: Path, api_key: str) -> str:
    upload_url = upload_to_assemblyai(file_path=file_path, api_key=api_key)
    transcript_id = create_assemblyai_transcript(upload_url=upload_url, api_key=api_key)
    return poll_assemblyai_transcript(transcript_id=transcript_id, api_key=api_key)


def analyze_with_openai(transcript: str, protocol_prompt: str, settings: Settings, source_name: str) -> str:
    client = OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url, timeout=180)
    prompt = (
        "Ты составляешь протокол встречи по правилам из архитектурного документа. "
        "Следуй порядку разделов. Не добавляй фактов, которых нет в транскрипте.\n\n"
        f"{protocol_prompt}\n\n"
        f"Имя файла: {source_name}\n\n"
        "Ниже транскрипт встречи. Метки «Спикер A», «Спикер B» и далее — разные люди.\n"
        "<transcript>\n"
        f"{transcript}\n"
        "</transcript>"
    )
    response = client.responses.create(model=settings.openai_model, input=prompt)
    result = (response.output_text or "").strip()
    if not result:
        raise RuntimeError("OpenAI returned empty protocol.")
    return result


def build_protocol(file_path: Path, settings: Settings, protocol_prompt: str) -> tuple[str, str]:
    transcript = transcribe_video(file_path=file_path, api_key=settings.assemblyai_api_key)
    protocol = analyze_with_openai(
        transcript=transcript,
        protocol_prompt=protocol_prompt,
        settings=settings,
        source_name=file_path.name,
    )
    return transcript, protocol


def normalize_extension(suffix: str, mime: str) -> str:
    extension = suffix.lower()
    if extension == ".mpga":
        extension = ".mp3"
    if extension in SUPPORTED_EXTENSIONS:
        return extension
    guessed = (mimetypes.guess_extension(mime or "") or "").lower()
    if guessed == ".mpga":
        guessed = ".mp3"
    if guessed in SUPPORTED_EXTENSIONS:
        return guessed
    return ""


def safe_media_filename(original_name: str, file_unique_id: str, extension: str) -> str:
    stem = Path(original_name).stem if original_name else ""
    safe_stem = re.sub(r"[^\w.\-]", "_", stem, flags=re.UNICODE).strip("._") if stem else ""
    if not safe_stem:
        safe_stem = file_unique_id
    return f"{safe_stem}{extension}"


def extract_media_file_data(message: Message) -> Optional[tuple[str, str, str]]:
    if message.video:
        extension = normalize_extension("", message.video.mime_type or "video/mp4") or ".mp4"
        if extension not in SUPPORTED_EXTENSIONS:
            return None
        filename = safe_media_filename("", message.video.file_unique_id, extension)
        return message.video.file_id, message.video.file_unique_id, filename

    if message.audio:
        original_name = message.audio.file_name or ""
        extension = normalize_extension(Path(original_name).suffix, message.audio.mime_type or "audio/mpeg")
        if extension not in SUPPORTED_EXTENSIONS:
            return None
        filename = safe_media_filename(original_name, message.audio.file_unique_id, extension)
        return message.audio.file_id, message.audio.file_unique_id, filename

    if message.document:
        original_name = message.document.file_name or ""
        extension = normalize_extension(Path(original_name).suffix, message.document.mime_type or "")
        if extension not in SUPPORTED_EXTENSIONS:
            return None
        filename = safe_media_filename(original_name, message.document.file_unique_id, extension)
        return message.document.file_id, message.document.file_unique_id, filename

    return None


def protocol_sidecar_paths(media_path: Path) -> tuple[Path, Path, Path]:
    transcript_path = media_path.with_name(media_path.name + ".transcript.txt")
    protocol_path = media_path.with_name(media_path.name + ".protocol.md")
    error_path = media_path.with_name(media_path.name + ".error.txt")
    return transcript_path, protocol_path, error_path


def local_file_identity(path: Path) -> tuple[str, str]:
    stat = path.stat()
    unique = sha256(f"{path.name}:{stat.st_size}:{stat.st_mtime_ns}".encode("utf-8")).hexdigest()
    return f"local:{path.name}", unique


def list_data_media(data_dir: Path) -> list[Path]:
    files = [
        path
        for path in data_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return sorted(files, key=lambda item: item.name.lower())


async def file_size_is_stable(path: Path) -> bool:
    if not path.exists() or not path.is_file():
        return False
    size_before = path.stat().st_size
    if size_before <= 0:
        return False
    await asyncio.sleep(2)
    if not path.exists():
        return False
    return path.stat().st_size == size_before


def process_data_file(media_path: Path, settings: Settings, protocol_prompt: str) -> None:
    transcript_path, protocol_path, error_path = protocol_sidecar_paths(media_path)
    file_id, file_unique_id = local_file_identity(media_path)
    transcript: Optional[str] = None
    protocol: Optional[str] = None
    logging.info("Data folder: start %s", media_path.name)
    try:
        transcript, protocol = build_protocol(media_path, settings, protocol_prompt)
        transcript_path.write_text(transcript, encoding="utf-8")
        protocol_path.write_text(protocol, encoding="utf-8")
        docx_path = protocol_path.with_suffix(".docx")
        write_protocol_docx(protocol, docx_path)
        if error_path.exists():
            error_path.unlink()
        logging.info("Data folder: protocol saved %s and %s", protocol_path.name, docx_path.name)
    except Exception as error:
        logging.exception("Data folder: failed %s", media_path.name)
        error_path.write_text(str(error), encoding="utf-8")
        try:
            save_audit_result(
                settings.database_url,
                0,
                None,
                "data-folder",
                file_id,
                file_unique_id,
                media_path.name,
                transcript,
                protocol,
                "failed",
                str(error),
            )
        except Exception as db_error:
            logging.exception("Data folder: failed to save error row: %s", db_error)
        return

    try:
        save_audit_result(
            settings.database_url,
            0,
            None,
            "data-folder",
            file_id,
            file_unique_id,
            media_path.name,
            transcript,
            protocol,
            "success",
            None,
        )
    except Exception as db_error:
        logging.exception("Data folder: protocol is on disk, database row was not saved: %s", db_error)


async def watch_data_folder(application: Application) -> None:
    settings: Settings = application.bot_data["settings"]
    protocol_prompt: str = application.bot_data["protocol_prompt"]
    data_dir = resolve_data_dir(settings)
    logging.info("Watching data folder: %s", data_dir)
    in_progress: set[str] = set()

    while True:
        try:
            for media_path in list_data_media(data_dir):
                _, protocol_path, error_path = protocol_sidecar_paths(media_path)
                key = str(media_path.resolve())
                if protocol_path.exists():
                    docx_path = protocol_path.with_suffix(".docx")
                    if not docx_path.exists():
                        logging.info("Data folder: writing docx for %s", protocol_path.name)
                        await asyncio.to_thread(
                            write_protocol_docx,
                            protocol_path.read_text(encoding="utf-8"),
                            docx_path,
                        )
                    continue
                if key in in_progress or error_path.exists():
                    continue
                if not await file_size_is_stable(media_path):
                    continue
                in_progress.add(key)
                try:
                    async with PROCESS_LOCK:
                        await asyncio.to_thread(process_data_file, media_path, settings, protocol_prompt)
                finally:
                    in_progress.discard(key)
        except Exception as error:
            logging.exception("Data folder scan failed: %s", error)
        await asyncio.sleep(DATA_FOLDER_POLL_SECONDS)


def split_for_telegram(text: str, chunk_size: int = MESSAGE_CHUNK_SIZE) -> list[str]:
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    remaining = text

    while remaining:
        if len(remaining) <= chunk_size:
            chunks.append(remaining)
            break

        split_at = remaining.rfind("\n", 0, chunk_size)
        if split_at == -1:
            split_at = chunk_size

        chunks.append(remaining[:split_at].rstrip())
        remaining = remaining[split_at:].lstrip()

    return chunks


async def start_handler(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return

    await update.message.reply_text(
        "Отправьте запись встречи файлом mp3, mp4 или mkv.\n"
        "Я разделю говорящих через AssemblyAI и пришлю протокол по правилам из prompt.md.\n"
        "Тот же разбор запускается сам, если положить файл в папку data."
    )


async def help_handler(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return

    await update.message.reply_text(
        "Как использовать:\n"
        "1) Отправьте mp3, mp4 или mkv либо положите файл в папку data.\n"
        "2) Дождитесь транскрибации и протокола.\n"
        "3) В чат придёт протокол текстом и файлом docx, транскрипт — отдельным файлом. Для папки data рядом с записью появятся .transcript.txt, .protocol.md и .protocol.docx."
    )


async def handle_media(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return

    message = update.message
    file_data = extract_media_file_data(message)
    if not file_data:
        await message.reply_text("Не вижу подходящий файл. Отправьте mp3, mp4 или mkv.")
        return

    file_id, file_unique_id, filename = file_data
    settings: Settings = context.application.bot_data["settings"]
    protocol_prompt: str = context.application.bot_data["protocol_prompt"]
    user_id = message.from_user.id if message.from_user else None
    username = message.from_user.username if message.from_user else None
    transcript: Optional[str] = None
    analysis: Optional[str] = None

    progress_message = await message.reply_text("Файл получен. Скачиваю...")
    await context.bot.send_chat_action(chat_id=message.chat_id, action=ChatAction.TYPING)

    try:
        async with PROCESS_LOCK:
            with tempfile.TemporaryDirectory(prefix="tg_meeting_") as tmp_dir:
                local_path = Path(tmp_dir) / filename
                telegram_file = await context.bot.get_file(file_id)
                await telegram_file.download_to_drive(custom_path=str(local_path))

                await progress_message.edit_text("Файл скачан. Транскрибирую через AssemblyAI и собираю протокол...")
                transcript, analysis = await asyncio.to_thread(
                    build_protocol,
                    local_path,
                    settings,
                    protocol_prompt,
                )
                transcript_file = Path(tmp_dir) / f"{local_path.stem}.transcript.txt"
                protocol_docx = Path(tmp_dir) / f"{local_path.stem}.protocol.docx"
                transcript_file.write_text(transcript, encoding="utf-8")
                await asyncio.to_thread(write_protocol_docx, analysis, protocol_docx)

                try:
                    await asyncio.to_thread(
                        save_audit_result,
                        settings.database_url,
                        message.chat_id,
                        user_id,
                        username,
                        file_id,
                        file_unique_id,
                        filename,
                        transcript,
                        analysis,
                        "success",
                        None,
                    )
                except Exception as db_error:
                    logging.exception("Failed to save successful protocol to Postgres: %s", db_error)

                await progress_message.edit_text("Протокол готов. Отправляю результат...")
                await message.reply_text("Транскрипт и протокол в docx приложены файлами. Ниже тот же протокол текстом.")
                with transcript_file.open("rb") as transcript_handle:
                    await message.reply_document(document=transcript_handle, filename=transcript_file.name)
                with protocol_docx.open("rb") as protocol_handle:
                    await message.reply_document(document=protocol_handle, filename=protocol_docx.name)
                for chunk in split_for_telegram(analysis):
                    await message.reply_text(chunk)

                await progress_message.delete()

    except Exception as error:
        logging.exception("Failed to process media file: %s", error)

        try:
            await asyncio.to_thread(
                save_audit_result,
                settings.database_url,
                message.chat_id,
                user_id,
                username,
                file_id,
                file_unique_id,
                filename,
                transcript,
                analysis,
                "failed",
                str(error),
            )
        except Exception as db_error:
            logging.exception("Failed to save failed audit to Postgres: %s", db_error)

        await progress_message.edit_text(
            "Не получилось обработать файл. Проверьте ключи и попробуйте снова."
        )


async def handle_unsupported_message(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return
    await update.message.reply_text("Отправьте mp3, mp4 или mkv, чтобы я собрал протокол встречи.")


async def on_startup(application: Application) -> None:
    application.bot_data["data_watch_task"] = asyncio.create_task(watch_data_folder(application))


def build_application(settings: Settings, protocol_prompt: str) -> Application:
    app = (
        Application.builder()
        .token(settings.telegram_bot_token)
        .post_init(on_startup)
        .build()
    )
    app.bot_data["settings"] = settings
    app.bot_data["protocol_prompt"] = protocol_prompt

    app.add_handler(CommandHandler("start", start_handler))
    app.add_handler(CommandHandler("help", help_handler))
    app.add_handler(
        MessageHandler(
            filters.VIDEO | filters.AUDIO | filters.Document.ALL,
            handle_media,
        )
    )
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_unsupported_message))

    return app


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    load_dotenv()

    settings = Settings.from_env()
    resolve_data_dir(settings)
    init_database_with_retry(settings.database_url)
    prompt_path = Path(__file__).with_name("prompt.md")
    protocol_prompt = load_protocol_prompt(prompt_path)

    app = build_application(settings=settings, protocol_prompt=protocol_prompt)
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
