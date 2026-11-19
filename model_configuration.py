"""Configuracion local aprobada del modelo de produccion."""

import hashlib
import hmac
import json
import math
import os
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from dotenv import dotenv_values


@dataclass(frozen=True)
class ModelConfiguration:
    pattern_weight: float = 0.30
    n_estimators: int = 100
    max_depth: int = 16
    min_samples_leaf: int = 1
    min_samples_split: int = 2

    def validate(self):
        if (
            isinstance(self.pattern_weight, bool)
            or not isinstance(self.pattern_weight, (int, float))
            or not math.isfinite(self.pattern_weight)
            or not 0 <= self.pattern_weight <= 0.80
        ):
            raise ValueError('El peso del patron debe estar entre 0 y 80%.')
        limits = {
            'n_estimators': (10, 1000),
            'max_depth': (1, 64),
            'min_samples_leaf': (1, 100),
            'min_samples_split': (2, 200),
        }
        for field, (minimum, maximum) in limits.items():
            value = getattr(self, field)
            if type(value) is not int or not minimum <= value <= maximum:
                raise ValueError(
                    f'{field} debe ser entero entre {minimum} y {maximum}.'
                )

    def model_params(self):
        self.validate()
        return {
            key: value for key, value in asdict(self).items()
            if key != 'pattern_weight'
        }


def access_database_path():
    value = os.getenv('ACCESS_DB_PATH', '').strip()
    if not value:
        for env_path in (Path(__file__).with_name('.env'), Path.cwd() / '.env'):
            value = (dotenv_values(env_path).get('ACCESS_DB_PATH') or '').strip()
            if value:
                break
    path = Path(value).expanduser() if value else Path('data') / 'control_acceso.db'
    return path if path.is_absolute() else Path(__file__).parent / path


def password_matches(password, encoded):
    parts = str(encoded).split('$')
    if len(parts) != 4 or parts[0] != 'pbkdf2_sha256':
        return False
    try:
        iterations = int(parts[1])
        salt = bytes.fromhex(parts[2])
        expected = bytes.fromhex(parts[3])
        if iterations <= 0:
            return False
        actual = hashlib.pbkdf2_hmac(
            'sha256', str(password).encode('utf-8'), salt, iterations
        )
    except ValueError:
        return False
    return hmac.compare_digest(actual, expected)


def load_model_configuration():
    path = access_database_path()
    if not path.is_file():
        return None
    with sqlite3.connect(path.as_uri() + '?mode=ro', uri=True, timeout=5) as conn:
        exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' "
            "AND name='model_configuration_history'"
        ).fetchone()
        if not exists:
            return None
        row = conn.execute(
            'SELECT configuration FROM model_configuration_history '
            'ORDER BY id DESC LIMIT 1'
        ).fetchone()
    if row is None:
        return None
    payload = json.loads(row[0])
    if not isinstance(payload, dict) or set(payload) != set(asdict(ModelConfiguration())):
        raise ValueError('La configuracion aprobada esta incompleta o es incompatible.')
    config = ModelConfiguration(**payload)
    config.validate()
    return config


def model_configuration_signature():
    config = load_model_configuration()
    return json.dumps(asdict(config), sort_keys=True) if config else 'default'


def save_model_configuration(config, username, password):
    config.validate()
    if str(username).strip().casefold() != 'admin' or not password:
        raise PermissionError('Solo un administrador autenticado puede aplicar cambios.')
    path = access_database_path()
    if not path.is_file():
        raise ValueError('Inicia ProyAst y configura primero el usuario Admin.')
    with sqlite3.connect(path.as_uri() + '?mode=rw', uri=True, timeout=5) as conn:
        conn.execute('BEGIN IMMEDIATE')
        row = conn.execute(
            'SELECT clave_hash FROM usuarios '
            'WHERE usuario = ? COLLATE NOCASE AND activo = 1',
            (str(username).strip(),),
        ).fetchone()
        if row is None or not password_matches(password, row[0]):
            raise PermissionError('Credenciales de administrador invalidas o inactivas.')
        conn.execute(
            'CREATE TABLE IF NOT EXISTS model_configuration_history ('
            'id INTEGER PRIMARY KEY AUTOINCREMENT, configuration TEXT NOT NULL, '
            'approved_by TEXT NOT NULL, approved_at TEXT NOT NULL)'
        )
        conn.execute(
            'INSERT INTO model_configuration_history '
            '(configuration, approved_by, approved_at) VALUES (?, ?, ?)',
            (
                json.dumps(asdict(config), sort_keys=True, allow_nan=False),
                str(username).strip(),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
