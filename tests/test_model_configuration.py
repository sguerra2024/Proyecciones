import hashlib
import ast
import json
import sqlite3
import subprocess
import sys
import types
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import Graf_evaluacion as graf
import model_configuration as settings
import projection_core as core


@pytest.fixture
def access_db(tmp_path, monkeypatch):
    path = tmp_path / 'access.db'
    monkeypatch.setenv('ACCESS_DB_PATH', str(path))
    salt = b'fixture-salt'
    encoded = 'pbkdf2_sha256$200000$' + salt.hex() + '$' + hashlib.pbkdf2_hmac(
        'sha256', b'test-password', salt, 200_000
    ).hex()
    with sqlite3.connect(path) as conn:
        conn.execute(
            'CREATE TABLE usuarios (usuario TEXT COLLATE NOCASE, '
            'clave_hash TEXT, activo INTEGER)'
        )
        conn.executemany(
            'INSERT INTO usuarios VALUES (?, ?, ?)',
            [('Admin', encoded, 1), ('Tecnico', encoded, 1)],
        )
    return path


def _approve(config=None):
    settings.save_model_configuration(
        config or settings.ModelConfiguration(), 'Admin', 'test-password'
    )


def test_sin_aprobacion_conserva_modelo_actual(access_db):
    assert settings.load_model_configuration() is None
    assert core.build_production_model().get_params() == core.RandomForestRegressor(
        **core.MODEL_PARAMS
    ).get_params()


def test_configuracion_se_persiste_y_cambia_modelo_y_peso_global(access_db):
    config = settings.ModelConfiguration(
        pattern_weight=0.6, n_estimators=20, max_depth=5,
        min_samples_leaf=2, min_samples_split=4,
    )
    original = settings.model_configuration_signature()
    _approve(config)

    assert settings.load_model_configuration() == config
    assert settings.model_configuration_signature() != original
    model = core.build_production_model()
    for key, value in config.model_params().items():
        assert model.get_params()[key] == value
    assert model.random_state == 42
    assert model.max_features == 'sqrt'
    assert core.MODEL_PARAMS['n_estimators'] == 100
    result = core.calibrate_pattern_target_weight(
        pd.DataFrame(), pd.DataFrame(), '001rosa', {}
    )
    assert result['status'] == 'global_aprobado'
    assert result['weight'] == 0.6
    assert result['mape'] is None
    assert result['validation_weeks'] == 0


@pytest.mark.parametrize(
    ('username', 'password'),
    [('Tecnico', 'test-password'), ('Admin', 'wrong'), ('Admin', '')],
)
def test_solo_admin_autenticado_puede_guardar(access_db, username, password):
    with pytest.raises(PermissionError):
        settings.save_model_configuration(
            settings.ModelConfiguration(), username, password
        )
    assert settings.load_model_configuration() is None


def test_admin_inactivo_no_puede_aplicar(access_db):
    with sqlite3.connect(access_db) as conn:
        conn.execute('UPDATE usuarios SET activo=0 WHERE usuario="Admin"')
    with pytest.raises(PermissionError):
        _approve()
    assert settings.load_model_configuration() is None


@pytest.mark.parametrize(
    'kwargs',
    [
        {'pattern_weight': -0.01}, {'pattern_weight': 0.81},
        {'pattern_weight': float('nan')}, {'pattern_weight': True},
        {'n_estimators': 9}, {'n_estimators': 1001}, {'n_estimators': 10.5},
        {'max_depth': 0}, {'max_depth': 65},
        {'min_samples_leaf': 0}, {'min_samples_leaf': 101},
        {'min_samples_split': 1}, {'min_samples_split': 201},
    ],
)
def test_parametros_invalidos_no_modifican_aprobacion(access_db, kwargs):
    _approve()
    before = settings.load_model_configuration()
    with pytest.raises(ValueError):
        _approve(settings.ModelConfiguration(**kwargs))
    assert settings.load_model_configuration() == before


def test_historial_auditable_y_ultima_aprobacion_activa(access_db):
    _approve()
    _approve(settings.ModelConfiguration(pattern_weight=0, n_estimators=10))
    assert settings.load_model_configuration().pattern_weight == 0
    with sqlite3.connect(access_db) as conn:
        rows = conn.execute(
            'SELECT configuration, approved_by, approved_at '
            'FROM model_configuration_history ORDER BY id'
        ).fetchall()
    assert len(rows) == 2
    assert rows[-1][1] == 'Admin'
    assert rows[-1][2]
    assert 'password' not in ''.join(str(row) for row in rows)


def test_no_crea_base_de_acceso_sin_proyast(tmp_path, monkeypatch):
    path = tmp_path / 'not-created.db'
    monkeypatch.setenv('ACCESS_DB_PATH', str(path))
    assert settings.load_model_configuration() is None
    with pytest.raises(ValueError, match='ProyAst'):
        _approve()
    assert not path.exists()


def test_configuracion_corrupta_no_se_ignora(access_db):
    _approve()
    payload = asdict(settings.ModelConfiguration())
    payload['pattern_weight'] = 9
    with sqlite3.connect(access_db) as conn:
        conn.execute(
            'UPDATE model_configuration_history SET configuration=?',
            (json.dumps(payload),),
        )
    with pytest.raises(ValueError, match='peso'):
        core.build_production_model()


def test_configuracion_incompleta_no_rellena_valores_silenciosamente(access_db):
    _approve()
    with sqlite3.connect(access_db) as conn:
        conn.execute(
            'UPDATE model_configuration_history SET configuration=?',
            (json.dumps({'pattern_weight': 0.6}),),
        )
    with pytest.raises(ValueError, match='incompleta'):
        settings.load_model_configuration()


def test_proceso_nuevo_recibe_configuracion_aprobada(access_db):
    _approve(settings.ModelConfiguration(pattern_weight=0.5, n_estimators=20))
    result = subprocess.run(
        [
            sys.executable, '-c',
            'import json; from dataclasses import asdict; '
            'from model_configuration import load_model_configuration; '
            'print(json.dumps(asdict(load_model_configuration())))',
        ],
        cwd=Path(core.__file__).parent, check=True, capture_output=True,
        text=True, timeout=30,
    )
    config = json.loads(result.stdout)
    assert config['pattern_weight'] == 0.5
    assert config['n_estimators'] == 20


def test_fit_real_usa_parametros_aprobados(access_db):
    _approve(settings.ModelConfiguration(n_estimators=10, max_depth=3))
    model = core.fit_production_model(
        pd.DataFrame({'x': np.arange(10)}), np.arange(10)
    )
    assert len(model.estimators_) == 10
    assert all(tree.get_depth() <= 3 for tree in model.estimators_)


def test_cache_peso_streamlit_recibe_firma_de_configuracion(access_db):
    source = Path(core.__file__).with_name('ProyAst.py').read_text(
        encoding='utf-8'
    )
    tree = ast.parse(source)
    wrapper = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef)
        and node.name == 'calibrar_peso_objetivo_patron'
    )
    calls = []
    namespace = {
        'model_configuration_signature': settings.model_configuration_signature,
        '_calibrar_peso_objetivo_patron_cache': lambda *args: calls.append(args),
    }
    exec(compile(ast.Module(body=[wrapper], type_ignores=[]), '<wrapper>', 'exec'), namespace)
    calibrate = namespace['calibrar_peso_objetivo_patron']
    calibrate(None, None, 'case', {})
    _approve(settings.ModelConfiguration(n_estimators=200))
    calibrate(None, None, 'case', {})
    assert calls[0][-1] == 'default'
    assert calls[1][-1] != calls[0][-1]


def test_entrenamiento_api_aplica_peso_global_y_parametros(access_db, monkeypatch):
    config = settings.ModelConfiguration(pattern_weight=0.6, n_estimators=10, max_depth=3)
    _approve(config)
    rows = []
    for case, multiplier in [('001rosa', 1), ('002rosa', 2)]:
        for week in range(1, 17):
            rows.append({
                'Bloque&Varid': case, 'Anio': 2026, 'Semana': week,
                'Produccion': float((100 + week * 5) * multiplier),
                'Tallos/m2': float(10 + week),
                'm2Variedad': 100.0,
            })
    data = pd.DataFrame(rows)
    monkeypatch.setattr(core, 'load_recent_media_adjustments', lambda: {})
    monkeypatch.setattr(
        core, '_apply_difference_factor',
        lambda values, *args: (values, 1.0, 0, 'pending'),
    )
    monkeypatch.setattr(
        core, 'apply_overestimation_buffer',
        lambda *args: (args[4], np.zeros(len(args[4])), 0.0, 0.0),
    )
    fitted = []
    real_fit = core.fit_production_model

    def fit(features, target):
        model = real_fit(features, target)
        fitted.append((model, target))
        return model

    monkeypatch.setattr(core, 'fit_production_model', fit)
    result = core.train_projection_model(data, '001rosa', include_chart_df=False)
    assert result['pattern_target_calibration']['weight'] == 0.6
    assert result['pattern_target_calibration']['status'] == 'global_aprobado'
    assert result['pattern_target_calibration']['mape'] is None
    assert fitted[0][0].n_estimators == 10
    assert fitted[0][0].max_depth == 3


@pytest.mark.parametrize('tema', ['Peso del patrón', 'Lean Production'])
def test_triz_solo_recibe_prompt_publico_y_cierra_recursos(monkeypatch, tema):
    calls = []

    class FakeTRIZ:
        def __init__(self, **kwargs):
            assert kwargs == {'ruta_conocimiento': ':memory:'}

        def consultar(self, question, contexto_adicional):
            calls.append((question, contexto_adicional))
            return 'Idea para experimentar'

        def cerrar(self):
            calls.append('closed')

    monkeypatch.setitem(
        sys.modules, 'agente_triz', types.SimpleNamespace(AgenteTRIZ=FakeTRIZ)
    )
    agente = graf.AgenteEvaluacionTRIZ(
        [123.456, -987.654], 'dato-secreto', ('30', '40')
    )
    assert agente.responder(tema) == 'Idea para experimentar'
    text = str(calls)
    assert '123.456' not in text
    assert '987.654' not in text
    assert 'dato-secreto' not in text
    assert calls[0][0] == graf.TEMAS_TRIZ[tema]
    assert calls[-1] == 'closed'


def test_triz_respuesta_vacia_es_error_y_cierra_recursos(monkeypatch):
    closed = []

    class FakeTRIZ:
        def __init__(self, **kwargs):
            pass

        def consultar(self, *args, **kwargs):
            return ''

        def cerrar(self):
            closed.append(True)

    monkeypatch.setitem(
        sys.modules, 'agente_triz', types.SimpleNamespace(AgenteTRIZ=FakeTRIZ)
    )
    with pytest.raises(RuntimeError, match='no devolvió'):
        graf.consultar_triz_general('Validación de ideas')
    assert closed == [True]
