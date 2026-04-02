from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from app.settings import settings


SESSION_ID_REGEX = re.compile(r'^(\d{2}-\d{2}-\d{4})-session-(\d{3})$')
DATE_KEY_REGEX = re.compile(r'^(\d{2})-(\d{2})-(\d{4})$')
SESSION_FILE_REGEX = re.compile(r'^session_(\d{3})\.json$')


class SessionService:
    def __init__(self, chat_log_root: Path) -> None:
        self._chat_log_root = chat_log_root

    @staticmethod
    def today_date_key() -> str:
        return datetime.now().strftime('%d-%m-%Y')

    @staticmethod
    def parse_date_key(date_key: str) -> dict[str, str] | None:
        match = DATE_KEY_REGEX.fullmatch(date_key)
        if not match:
            return None
        return {
            'day': match.group(1),
            'month': match.group(2),
            'year': match.group(3),
        }

    @staticmethod
    def parse_session_id(session_id: str) -> dict[str, str | int] | None:
        match = SESSION_ID_REGEX.fullmatch(session_id)
        if not match:
            return None
        return {
            'date_key': match.group(1),
            'sequence': int(match.group(2)),
        }

    @staticmethod
    def build_session_file_name(sequence: int) -> str:
        return f'session_{sequence:03d}.json'

    @staticmethod
    def build_session_id(date_key: str, sequence: int) -> str:
        return f'{date_key}-session-{sequence:03d}'

    def is_valid_date_key(self, date_key: str) -> bool:
        return self.parse_date_key(date_key) is not None

    def date_folder_path(self, date_key: str) -> Path:
        parsed = self.parse_date_key(date_key)
        if not parsed:
            raise ValueError(f'Invalid date format: {date_key}')
        return self._chat_log_root / f"{parsed['day']}-{parsed['month']}-{parsed['year']}"

    def session_file_path_from_id(self, session_id: str) -> Path | None:
        parsed = self.parse_session_id(session_id)
        if not parsed:
            return None
        return self.date_folder_path(str(parsed['date_key'])) / self.build_session_file_name(int(parsed['sequence']))

    def _read_json_file(self, file_path: Path) -> dict[str, Any]:
        return json.loads(file_path.read_text(encoding='utf-8'))

    def _write_json_file(self, file_path: Path, payload: dict[str, Any], exclusive: bool = False) -> None:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        mode = 'x' if exclusive else 'w'
        with file_path.open(mode, encoding='utf-8') as stream:
            json.dump(payload, stream, indent=2, ensure_ascii=False)

    def session_files_for_date(self, date_key: str) -> list[str]:
        folder_path = self.date_folder_path(date_key)
        if not folder_path.exists():
            return []

        files = [name for name in [entry.name for entry in folder_path.iterdir()] if SESSION_FILE_REGEX.fullmatch(name)]
        files.sort(key=lambda name: int(SESSION_FILE_REGEX.fullmatch(name).group(1)))
        return files

    def next_session_sequence(self, date_key: str) -> int:
        files = self.session_files_for_date(date_key)
        if not files:
            return 1
        match = SESSION_FILE_REGEX.fullmatch(files[-1])
        if not match:
            return 1
        return int(match.group(1)) + 1

    def create_session_file(self, base_session: dict[str, Any]) -> dict[str, Any]:
        date_key = self.today_date_key()
        folder_path = self.date_folder_path(date_key)
        folder_path.mkdir(parents=True, exist_ok=True)

        for _ in range(10):
            sequence = self.next_session_sequence(date_key)
            file_name = self.build_session_file_name(sequence)
            file_path = folder_path / file_name
            session_id = self.build_session_id(date_key, sequence)

            now_iso = datetime.now().isoformat()
            session = {
                **base_session,
                'id': session_id,
                'date_folder': date_key,
                'sequence': sequence,
                'file_name': file_name,
                'created_at': now_iso,
                'updated_at': now_iso,
            }

            try:
                self._write_json_file(file_path, session, exclusive=True)
                return session
            except FileExistsError:
                continue

        raise RuntimeError('Unable to allocate a new session file. Please retry.')

    def read_session_by_id(self, session_id: str) -> dict[str, Any] | None:
        file_path = self.session_file_path_from_id(session_id)
        if not file_path or not file_path.exists():
            return None
        try:
            return self._read_json_file(file_path)
        except Exception:
            return None

    def save_session_by_id(self, session: dict[str, Any]) -> None:
        session_id = str(session.get('id', ''))
        file_path = self.session_file_path_from_id(session_id)
        if not file_path:
            raise ValueError('Session id format is invalid.')
        self._write_json_file(file_path, session)

    def list_sessions_by_date(self, date_key: str) -> list[dict[str, Any]]:
        files = self.session_files_for_date(date_key)
        folder_path = self.date_folder_path(date_key)

        sessions: list[dict[str, Any]] = []
        for file_name in files:
            try:
                session = self._read_json_file(folder_path / file_name)
                sessions.append(
                    {
                        'session_id': session.get('id', ''),
                        'sequence': int(session.get('sequence', 0)),
                        'file_name': session.get('file_name', file_name),
                        'status': session.get('status', 'pending'),
                        'model': session.get('model', 'mistral'),
                        'query': session.get('query', ''),
                        'created_at': session.get('created_at', ''),
                        'updated_at': session.get('updated_at', ''),
                        'history_count': len(session.get('history', [])) if isinstance(session.get('history'), list) else 0,
                    }
                )
            except Exception:
                continue

        sessions.sort(key=lambda item: int(item.get('sequence', 0)), reverse=True)
        return sessions

    @staticmethod
    def build_prompt_from_feedback(query: str, feedback: str | None = None) -> str:
        if not feedback:
            return query
        return '\n'.join(
            [
                f'Cau hoi goc: {query}',
                f'Nguoi dung khong dong y voi phan hoi truoc va de xuat: {feedback}',
                'Hay tra loi lai sat voi du lieu retrieve hon.',
            ]
        )


session_service = SessionService(settings.chat_log_root)
