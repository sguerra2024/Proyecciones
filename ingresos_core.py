"""Calculo de ingresos futuros cruzando la proyeccion de tallos con el catalogo
comercial de precios por variedad y longitud.

Modelo:

    ingreso[variedad, semana] = SUMA sobre longitudes de
        tallos_proyectados[variedad, semana]
        * fraccion[variedad, longitud]              <- % del catalogo
        * precio[variedad, longitud, mercado]       <- precio del catalogo

Una sola fuente de datos. El catalogo que manda el area comercial trae, para
cada variedad y cada longitud, el porcentaje de tallos que cae en esa longitud
y el precio de esa longitud. El precio depende de la variedad, no solo del
largo: un 60cm de una variedad premium no vale lo mismo que un 60cm comun.

Se aceptan los dos formatos en que suele llegar ese archivo (ver
formato_catalogo):

- largo: una fila por variedad+longitud, con columnas de % y precio.
- ancho: una fila por variedad y una columna por longitud, marcando cuales
  columnas son precio y cuales son %.

Cada carga se acumula con su fecha. Eso mantiene dos propiedades:

- El precio de una longitud se arrastra desde la ultima carga que lo trajo y se
  reporta su antiguedad, porque una semana futura no tiene precio propio.
- El % se toma como una foto completa de la ultima carga de esa variedad, no
  arrastrada longitud por longitud. Asi una longitud que sale del mix desaparece
  en vez de quedar viva para siempre, y el ingreso de una semana pasada se puede
  recalcular con el mix que estaba vigente entonces.

Regla de diseno: aqui se hace toda la aritmetica. El LLM solo recibe el
resultado ya calculado (ver resumen_para_llm) para priorizar y explicar, nunca
para multiplicar.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# Escenarios de precio para semanas futuras. El precio de una semana futura no
# existe todavia: se arrastra el ultimo conocido y se le aplica esta variacion.
ESCENARIOS_POR_DEFECTO: dict[str, float] = {
    'pesimista': -0.10,
    'base': 0.0,
    'optimista': 0.10,
}

# Tolerancia al validar que los % de una variedad sumen 100.
TOLERANCIA_MIX_PCT = 0.5

# Dias tras los cuales un precio arrastrado se marca como desactualizado.
DIAS_PRECIO_ANTIGUO = 7

CARPETA_DATOS_MERCADO = 'datos_mercado'
ARCHIVO_CATALOGO = 'catalogo_precios_historico.csv'

# Errores medidos del modelo contra la produccion real, una fila por caso
# evaluado. Es la base empirica de la probabilidad de cumplimiento.
CARPETA_EVALUACION = 'Evaluacion'
ARCHIVO_ERRORES_MODELO = 'errores_evaluacion_modelo.csv'

# Un error de +1 significaria un estimado de cero: se recorta antes de invertir.
_ERROR_MAXIMO = 0.99

# Bandas con que se reporta el cumplimiento, las mismas de Graf_evaluacion.py.
TOLERANCIAS_CUMPLIMIENTO: tuple[float, ...] = (
    0.05, 0.10, 0.15, 0.20, 0.30, 0.50)

# Niveles a los que se reporta el ingreso comprometible.
NIVELES_CONFIANZA: tuple[float, ...] = (
    0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99)

COLUMNAS_CATALOGO = ['fecha', 'variedad', 'grado',
                     'mercado', 'pct', 'fraccion', 'precio']

# Patrones por concepto, en rondas de mayor a menor especificidad. Se evalua
# toda la primera ronda antes de pasar a la siguiente: sin esto,
# "Variedad_proyectada" se confundiria con la columna de tallos por contener
# "proyec".
PATRONES_COLUMNA: dict[str, tuple[str, ...]] = {
    'variedad': (r'^variedad', r'variedad|varid', r'producto'),
    'grado': (r'^longitud|^largo|^grado', r'longitud|largo', r'grado|grade|calibre'),
    'pct': (r'^pct$|^porcentaje$|%', r'pct|porcentaje|porc|particip|mix|share|proporcion'),
    'fecha': (r'^fecha', r'fecha|date|dia'),
    'precio': (r'^precio', r'precio|price|unitario', r'valor|usd'),
    'mercado': (r'^mercado', r'mercado|market|cliente|destino|canal'),
    'semana': (r'anio_semana|ano_semana', r'^semana$|^periodo', r'semana|periodo'),
    'tallos': (r'estimado_modelo', r'^estimado|^tallos|tallos', r'proyec|produccion'),
    # Produccion medida, para comparar contra el estimado y medir el error del modelo.
    'real': (r'produccion_real', r'^real$|_real$', r'^produccion$'),
}

# Una longitud es un numero de 2 o 3 digitos suelto: "60", "60cm", "Precio 60 cm".
# El lookaround evita que un ano ("Precio 2026") se lea como longitud.
_RE_LONGITUD = re.compile(r'(?<!\d)(\d{2,3})(?!\d)')

# Encabezados que son identificadores de la fila, nunca una longitud.
_RE_COLUMNA_ID = re.compile(
    r'variedad|varid|producto|mercado|market|cliente|destino|canal|fecha|date',
    flags=re.IGNORECASE,
)

# Marca que distingue, en formato ancho, la columna de % de la de precio.
_RE_MARCADOR_PCT = re.compile(
    r'%|pct|porcentaje|porc|particip|mix|share|proporcion',
    flags=re.IGNORECASE,
)


def detectar_columna(df: pd.DataFrame, clave: str,
                     requerido: bool = True,
                     excluir: set[str] | None = None) -> str | None:
    """Encuentra la columna que corresponde a un concepto del modelo.

    El catalogo lo mantiene el area comercial y los encabezados cambian, por eso
    se detecta por patron en vez de exigir un nombre exacto. El mapeo se puede
    sobreescribir a mano desde la UI.
    """
    descartadas = excluir or set()
    candidatas = [c for c in df.columns if c not in descartadas]
    for patron in PATRONES_COLUMNA[clave]:
        for columna in candidatas:
            if re.search(patron, str(columna), flags=re.IGNORECASE):
                return columna
    if requerido:
        disponibles = ', '.join(str(c) for c in df.columns)
        raise ValueError(
            f'No se encontro la columna de {clave} en el archivo. '
            f'Columnas disponibles: {disponibles}'
        )
    return None


def _resolver(df: pd.DataFrame, mapeo: dict[str, str] | None, clave: str,
              requerido: bool = True,
              excluir: set[str] | None = None) -> str | None:
    if mapeo and mapeo.get(clave):
        columna = mapeo[clave]
        if columna not in df.columns:
            raise ValueError(
                f'La columna indicada para {clave} ("{columna}") no existe en el archivo.'
            )
        return columna
    return detectar_columna(df, clave, requerido=requerido, excluir=excluir)


def _texto(serie: pd.Series) -> pd.Series:
    return serie.astype(str).str.strip().str.upper()


def _normalizar_grado(serie: pd.Series) -> pd.Series:
    """Unifica la longitud a "60CM".

    Comercial escribe la misma longitud de varias formas ("60", "60 cm",
    "60CM"). Sin unificar, la misma longitud aparece como tres grados distintos y
    el cruce con la proyeccion se parte.
    """
    texto = _texto(serie)
    numero = texto.str.extract(_RE_LONGITUD, expand=False)
    return numero.where(numero.isna(), numero + 'CM').fillna(texto)


def _ultimo_valido(serie: pd.Series) -> float:
    limpia = serie.dropna()
    return float(limpia.iloc[-1]) if not limpia.empty else np.nan


def _longitud_de_columna(nombre) -> str | None:
    texto = str(nombre)
    if _RE_COLUMNA_ID.search(texto):
        return None
    encontrado = _RE_LONGITUD.search(texto)
    return encontrado.group(1) if encontrado else None


def _grupos_longitud(df: pd.DataFrame) -> dict[str, dict[str, object]]:
    """Agrupa las columnas de formato ancho por longitud.

    Devuelve {longitud: {'precio': columna, 'pct': columna}}.
    """
    grupos: dict[str, dict[str, object]] = {}
    for columna in df.columns:
        longitud = _longitud_de_columna(columna)
        if longitud is None:
            continue
        clase = 'pct' if _RE_MARCADOR_PCT.search(str(columna)) else 'precio'
        grupos.setdefault(longitud, {})[clase] = columna
    return grupos


def formato_catalogo(df: pd.DataFrame) -> str:
    """"ancho" si las longitudes estan en los encabezados, "largo" si van en filas."""
    if df is None or df.empty:
        return 'largo'
    tiene_columna_longitud = detectar_columna(
        df, 'grado', requerido=False) is not None
    if not tiene_columna_longitud and _grupos_longitud(df):
        return 'ancho'
    return 'largo'


def _canonico_desde_ancho(df: pd.DataFrame,
                          mapeo: dict[str, str] | None) -> pd.DataFrame:
    grupos = _grupos_longitud(df)
    if not any('precio' in grupo for grupo in grupos.values()):
        raise ValueError(
            'El catalogo ancho no trae precios por longitud. Se espera una '
            'columna por longitud (por ejemplo "Precio 60").'
        )
    if not any('pct' in grupo for grupo in grupos.values()):
        raise ValueError(
            'El catalogo ancho no trae el porcentaje por longitud. Junto a cada '
            'columna de precio agrega una de % (por ejemplo "Precio 60" y "% 60"), '
            'porque sin el % no se sabe cuantos tallos caen en esa longitud.'
        )

    columnas_valor = {
        columna for grupo in grupos.values() for columna in grupo.values()}
    identificadores = [c for c in df.columns if c not in columnas_valor]
    base = df[identificadores]
    col_variedad = _resolver(base, mapeo, 'variedad')
    col_fecha = _resolver(base, mapeo, 'fecha', requerido=False,
                          excluir={col_variedad})
    col_mercado = _resolver(base, mapeo, 'mercado', requerido=False,
                            excluir={col_variedad, col_fecha} - {None})

    piezas = []
    for longitud, grupo in grupos.items():
        pieza = pd.DataFrame({
            'variedad': base[col_variedad],
            'grado': f'{longitud}CM',
            'pct': (pd.to_numeric(df[grupo['pct']], errors='coerce')
                    if 'pct' in grupo else np.nan),
            'precio': (pd.to_numeric(df[grupo['precio']], errors='coerce')
                       if 'precio' in grupo else np.nan),
        })
        pieza['fecha'] = base[col_fecha] if col_fecha else pd.NaT
        pieza['mercado'] = base[col_mercado] if col_mercado else ''
        piezas.append(pieza)
    return pd.concat(piezas, ignore_index=True)


def _canonico_desde_largo(df: pd.DataFrame,
                          mapeo: dict[str, str] | None) -> pd.DataFrame:
    col_variedad = _resolver(df, mapeo, 'variedad')
    usadas = {col_variedad}
    col_grado = _resolver(df, mapeo, 'grado', excluir=usadas)
    usadas.add(col_grado)
    col_pct = _resolver(df, mapeo, 'pct', requerido=False, excluir=usadas)
    if col_pct is not None:
        usadas.add(col_pct)
    col_precio = _resolver(df, mapeo, 'precio', excluir=usadas)
    usadas.add(col_precio)
    col_fecha = _resolver(df, mapeo, 'fecha', requerido=False, excluir=usadas)
    usadas.add(col_fecha)
    col_mercado = _resolver(df, mapeo, 'mercado',
                            requerido=False, excluir=usadas)

    return pd.DataFrame({
        'variedad': df[col_variedad],
        'grado': df[col_grado],
        'pct': (
            pd.to_numeric(df[col_pct], errors='coerce')
            if col_pct is not None
            else np.nan
        ),
        'precio': pd.to_numeric(df[col_precio], errors='coerce'),
        'fecha': df[col_fecha] if col_fecha else pd.NaT,
        'mercado': df[col_mercado] if col_mercado else '',
    })


def normalizar_catalogo(df: pd.DataFrame,
                        mapeo: dict[str, str] | None = None,
                        normalizar: bool = False,
                        fecha_defecto=None) -> tuple[pd.DataFrame, dict[str, float]]:
    """Convierte el archivo del area comercial al catalogo canonico.

    Devuelve el catalogo (fecha, variedad, grado, mercado, pct, fraccion,
    precio) y las variedades cuyos % no suman 100 en su ultima carga. Por
    defecto NO se normaliza: si una variedad suma 97%, ese 3% queda sin valorar
    y se reporta, porque normalizar en silencio esconde un archivo mal armado.
    """
    if df is None or df.empty:
        raise ValueError(
            'El catalogo de precios por variedad y longitud esta vacio.')

    if formato_catalogo(df) == 'ancho':
        crudo = _canonico_desde_ancho(df, mapeo)
    else:
        crudo = _canonico_desde_largo(df, mapeo)

    defecto = (pd.Timestamp(fecha_defecto).normalize() if fecha_defecto is not None
               else pd.Timestamp.today().normalize())
    catalogo = pd.DataFrame({
        'fecha': pd.to_datetime(crudo['fecha'], errors='coerce').fillna(defecto),
        'variedad': _texto(crudo['variedad']),
        'grado': _normalizar_grado(crudo['grado']),
        'mercado': _texto(crudo['mercado']).replace(
            ['', 'NAN', 'NONE'], 'UNICO'),
        'pct': crudo['pct'],
        'precio': crudo['precio'],
    })
    catalogo = catalogo[~catalogo['variedad'].isin(['', 'NAN'])
                        & ~catalogo['grado'].isin(['', 'NAN'])]
    if catalogo.empty:
        raise ValueError(
            'El catalogo no tiene filas utilizables: revisa las columnas de '
            'variedad y longitud.'
        )

    # Si el archivo no trae %, inferirlo para no bloquear la carga:
    # - 1 longitud: 100%
    # - varias longitudes: reparto equitativo por longitud
    faltantes_pct = catalogo['pct'].isna()
    if faltantes_pct.any():
        grupos_pct = catalogo.groupby(['fecha', 'variedad'], dropna=False)
        for (fecha, variedad), grupo in grupos_pct:
            idx_faltantes = grupo.index[grupo['pct'].isna()]
            if len(idx_faltantes) == 0:
                continue

            grados_con_pct = grupo.loc[grupo['pct'].notna(), 'grado']
            if not grados_con_pct.empty:
                continue

            longitudes = grupo['grado'].dropna().astype(str).nunique()
            if longitudes <= 1:
                catalogo.loc[idx_faltantes, 'pct'] = 100.0
            else:
                pct_equilibrado = 100.0 / float(longitudes)
                grados_faltantes = (
                    grupo.loc[grupo['pct'].isna(), 'grado']
                    .dropna()
                    .astype(str)
                    .unique()
                    .tolist()
                )
                for grado in grados_faltantes:
                    idx_grado = grupo.index[
                        grupo['grado'].astype(str) == str(grado)
                    ]
                    catalogo.loc[idx_grado, 'pct'] = pct_equilibrado

    # Un precio 0, negativo o vacio no es un precio: se marca como faltante para
    # que se reporte en vez de valorar tallos a cero.
    catalogo.loc[~(catalogo['precio'] > 0), 'precio'] = np.nan

    # Misma fecha, variedad, longitud y mercado repetidos: gana la ultima fila,
    # tanto en precio como en %. Una linea repetida en una carga es una
    # correccion, no un tramo extra que se sume: sumar inflaria el ingreso, y
    # quedarse con la ultima a lo sumo lo deja corto y lo delata la validacion de
    # que los % sumen 100. Es la misma regla con que acumular_catalogo deduplica.
    catalogo = (
        catalogo.sort_values(['fecha', 'variedad', 'grado', 'mercado'])
        .groupby(['fecha', 'variedad', 'grado', 'mercado'], as_index=False)
        .agg(pct=('pct', _ultimo_valido), precio=('precio', _ultimo_valido))
    )
    # El % es propiedad de la variedad+longitud, no del mercado: si viene solo en
    # la fila de un mercado, aplica a todos.
    catalogo['pct'] = catalogo.groupby(
        ['fecha', 'variedad', 'grado'])['pct'].transform('max')
    catalogo = catalogo[catalogo['pct'] > 0]
    if catalogo.empty:
        raise ValueError(
            'No hay porcentajes validos por variedad y longitud en el catalogo.')

    por_longitud = catalogo.drop_duplicates(
        subset=['fecha', 'variedad', 'grado'])
    sumas = por_longitud.groupby(['fecha', 'variedad'], as_index=False)[
        'pct'].sum().rename(columns={'pct': 'suma_pct'})
    ultima_carga = por_longitud.groupby('variedad')['fecha'].max()
    fuera_de_rango = {
        str(fila['variedad']): float(fila['suma_pct'])
        for _, fila in sumas.iterrows()
        if ultima_carga[fila['variedad']] == fila['fecha']
        and abs(fila['suma_pct'] - 100.0) > TOLERANCIA_MIX_PCT
    }

    catalogo = catalogo.merge(sumas, on=['fecha', 'variedad'], how='left')
    divisor = catalogo['suma_pct'] if normalizar else 100.0
    catalogo['fraccion'] = catalogo['pct'] / divisor

    return catalogo[COLUMNAS_CATALOGO].reset_index(drop=True), fuera_de_rango


def _mix_vigente(catalogo: pd.DataFrame, corte: pd.Timestamp) -> pd.DataFrame:
    """Foto completa del % de cada variedad en su ultima carga hasta el corte."""
    historico = catalogo[catalogo['fecha'] <= corte]
    ultima = historico.groupby('variedad')['fecha'].transform('max')
    foto = historico[historico['fecha'] == ultima]
    return (
        foto.drop_duplicates(subset=['variedad', 'grado'])
        [['variedad', 'grado', 'pct', 'fraccion', 'fecha']]
        .rename(columns={'fecha': 'fecha_mix'})
        .reset_index(drop=True)
    )


def _precio_vigente(catalogo: pd.DataFrame, corte: pd.Timestamp) -> pd.DataFrame:
    """Ultimo precio conocido por variedad, longitud y mercado."""
    historico = catalogo[(catalogo['fecha'] <= corte)
                         & catalogo['precio'].notna()]
    if historico.empty:
        return pd.DataFrame(
            columns=['variedad', 'grado', 'mercado', 'precio', 'fecha_precio',
                     'antiguedad_dias'])
    vigentes = (
        historico.sort_values('fecha')
        .groupby(['variedad', 'grado', 'mercado'], as_index=False)
        .tail(1)
        [['variedad', 'grado', 'mercado', 'precio', 'fecha']]
        .rename(columns={'fecha': 'fecha_precio'})
        .copy()
    )
    vigentes['antiguedad_dias'] = (corte - vigentes['fecha_precio']).dt.days
    return vigentes.reset_index(drop=True)


def catalogo_vigente(catalogo: pd.DataFrame,
                     fecha_corte: pd.Timestamp | None = None) -> pd.DataFrame:
    """Cruza el mix vigente con el ultimo precio conocido de cada longitud.

    Las longitudes del mix que no tienen precio se conservan con precio nulo:
    son las que despues se reportan como sin valorar, en vez de desaparecer.
    """
    if catalogo is None or catalogo.empty:
        raise ValueError('No hay catalogo de precios para calcular ingresos.')

    corte = pd.Timestamp(fecha_corte).normalize() if fecha_corte is not None \
        else pd.Timestamp.today().normalize()
    if catalogo[catalogo['fecha'] <= corte].empty:
        raise ValueError(
            f'No hay catalogo con fecha menor o igual a {corte.date()}.')

    mix = _mix_vigente(catalogo, corte)
    precios = _precio_vigente(catalogo, corte)
    if precios.empty:
        vigente = mix.copy()
        for columna in ['mercado', 'precio', 'fecha_precio', 'antiguedad_dias']:
            vigente[columna] = np.nan
        return vigente
    return mix.merge(precios, on=['variedad', 'grado'], how='left')


def variedades_fuera_de_rango(vigente: pd.DataFrame) -> dict[str, float]:
    """Variedades cuyos % no suman 100 en el catalogo vigente.

    normalizar_catalogo ya lo reporta al cargar el archivo, pero el aviso tiene
    que seguir apareciendo en cada calculo posterior: la carga se hace una vez y
    el ingreso se recalcula muchas.
    """
    if vigente is None or vigente.empty:
        return {}
    por_longitud = vigente.drop_duplicates(subset=['variedad', 'grado'])
    sumas = por_longitud.groupby('variedad')['pct'].sum()
    return {
        str(variedad): float(suma)
        for variedad, suma in sumas.items()
        if abs(suma - 100.0) > TOLERANCIA_MIX_PCT
    }


def elegir_mercado(vigentes: pd.DataFrame,
                   estrategia: str = 'mejor_precio',
                   mercado_fijo: str | None = None) -> pd.DataFrame:
    """Resuelve un precio unico por variedad+longitud cuando hay varios mercados.

    - mejor_precio: toma el mercado que mas paga. Es el techo de ingreso y la
      referencia natural cuando el objetivo es maximizar.
    - mercado_fijo: valora todo contra un mercado concreto. Lo que ese mercado no
      compra queda sin precio y se reporta, no se valora con otro mercado.

    En ambos casos se conserva la brecha contra el segundo mejor mercado, que es
    lo que convierte la tabla en una decision comercial ("mover el 60cm de
    FREEDOM a EU vale +13%") y no solo en un total.
    """
    if vigentes is None or vigentes.empty:
        raise ValueError('No hay catalogo vigente para asignar mercado.')

    clave = ['variedad', 'grado']
    con_precio = vigentes[vigentes['precio'].notna()].copy()
    sin_precio = vigentes[vigentes['precio'].isna()].copy()
    if con_precio.empty:
        raise ValueError(
            'Ninguna longitud del catalogo vigente tiene precio utilizable.')

    if estrategia == 'mercado_fijo':
        if not mercado_fijo:
            raise ValueError(
                'Indica el mercado a usar con estrategia="mercado_fijo".')
        objetivo = str(mercado_fijo).strip().upper()
        elegidos = con_precio[con_precio['mercado'] == objetivo].copy()
        if elegidos.empty:
            disponibles = ', '.join(sorted(con_precio['mercado'].unique()))
            raise ValueError(
                f'El mercado "{objetivo}" no tiene precios. Disponibles: {disponibles}'
            )
        huerfanas = con_precio.merge(
            elegidos[clave].drop_duplicates(), on=clave, how='left',
            indicator=True)
        huerfanas = huerfanas[huerfanas['_merge'] == 'left_only'].drop(
            columns='_merge').drop_duplicates(subset=clave).copy()
        if not huerfanas.empty:
            huerfanas[['mercado', 'precio']] = np.nan
            sin_precio = pd.concat([sin_precio, huerfanas], ignore_index=True)
    elif estrategia == 'mejor_precio':
        elegidos = (
            con_precio.sort_values('precio')
            .groupby(clave, as_index=False)
            .tail(1)
            .copy()
        )
    else:
        raise ValueError(
            'estrategia debe ser "mejor_precio" o "mercado_fijo".')

    ordenado = con_precio.sort_values(
        clave + ['precio'], ascending=[True, True, False])
    comparacion = (
        ordenado.groupby(clave)
        .agg(
            precio_segundo_mercado=(
                'precio', lambda s: float(s.iloc[1]) if len(s) > 1 else np.nan),
            mercados_disponibles=('mercado', 'nunique'),
        )
        .reset_index()
    )

    elegidos = elegidos.merge(comparacion, on=clave, how='left')
    elegidos['brecha_pct'] = np.where(
        elegidos['precio_segundo_mercado'].notna()
        & (elegidos['precio_segundo_mercado'] > 0),
        100.0 * (elegidos['precio'] - elegidos['precio_segundo_mercado'])
        / elegidos['precio_segundo_mercado'],
        np.nan,
    )

    resultado = pd.concat([elegidos, sin_precio], ignore_index=True)
    return resultado.rename(
        columns={'mercado': 'mercado_elegido'}).reset_index(drop=True)


@dataclass
class Diagnostico:
    """Todo lo que quedo sin valorar. Sin esto el ingreso baja en silencio."""

    variedades_sin_catalogo: list[str] = field(default_factory=list)
    grados_sin_precio: list[str] = field(default_factory=list)
    mix_fuera_de_rango: dict[str, float] = field(default_factory=dict)
    precios_desactualizados: dict[str, int] = field(default_factory=dict)
    tallos_totales: float = 0.0
    tallos_valorados: float = 0.0

    @property
    def tallos_sin_valorar(self) -> float:
        return max(self.tallos_totales - self.tallos_valorados, 0.0)

    @property
    def cobertura_pct(self) -> float:
        if self.tallos_totales <= 0:
            return 0.0
        return 100.0 * self.tallos_valorados / self.tallos_totales

    @property
    def hay_alertas(self) -> bool:
        return bool(
            self.variedades_sin_catalogo
            or self.grados_sin_precio
            or self.mix_fuera_de_rango
            or self.precios_desactualizados
            or self.tallos_sin_valorar > 0
        )

    def mensajes(self) -> list[str]:
        avisos: list[str] = []
        if self.variedades_sin_catalogo:
            avisos.append(
                f'{len(self.variedades_sin_catalogo)} variedad(es) que no estan en el '
                'catalogo de precios: '
                + ', '.join(self.variedades_sin_catalogo[:8])
            )
        if self.grados_sin_precio:
            avisos.append(
                f'{len(self.grados_sin_precio)} combinacion(es) variedad-longitud '
                'sin precio: ' + ', '.join(self.grados_sin_precio[:8])
            )
        if self.mix_fuera_de_rango:
            detalle = ', '.join(
                f'{var} suma {suma:.1f}%'
                for var, suma in list(self.mix_fuera_de_rango.items())[:8]
            )
            avisos.append(f'Variedades cuyos % no suman 100: {detalle}')
        if self.precios_desactualizados:
            detalle = ', '.join(
                f'{clave} ({dias}d)'
                for clave, dias in list(self.precios_desactualizados.items())[:8]
            )
            avisos.append(
                f'Precios con mas de {DIAS_PRECIO_ANTIGUO} dias: {detalle}')
        if self.tallos_sin_valorar > 0:
            avisos.append(
                f'{self.tallos_sin_valorar:,.0f} tallos sin valorar '
                f'({self.cobertura_pct:.1f}% de cobertura)'
            )
        return avisos


@dataclass
class ResultadoIngresos:
    detalle: pd.DataFrame
    por_variedad: pd.DataFrame
    por_semana: pd.DataFrame
    totales: dict[str, float]
    precios_aplicados: pd.DataFrame
    diagnostico: Diagnostico


def _preparar_proyeccion(proyeccion: pd.DataFrame,
                         mapeo: dict[str, str] | None = None) -> pd.DataFrame:
    if proyeccion is None or proyeccion.empty:
        raise ValueError('No hay proyeccion de tallos para valorar.')

    col_variedad = _resolver(proyeccion, mapeo, 'variedad')
    # La columna de variedad no puede volver a elegirse como tallos ni semana.
    usadas = {col_variedad}
    col_tallos = _resolver(proyeccion, mapeo, 'tallos', excluir=usadas)
    usadas.add(col_tallos)
    col_semana = _resolver(
        proyeccion, mapeo, 'semana', requerido=False, excluir=usadas)

    base = pd.DataFrame({
        'variedad': _texto(proyeccion[col_variedad]),
        'tallos': pd.to_numeric(proyeccion[col_tallos], errors='coerce'),
    })
    base['semana'] = (
        proyeccion[col_semana].astype(str)
        if col_semana is not None else 'TOTAL'
    )
    base = base.dropna(subset=['tallos'])
    base = base[base['variedad'] != '']
    if base.empty:
        raise ValueError('La proyeccion no tiene tallos validos.')
    return base.groupby(['variedad', 'semana'], as_index=False)['tallos'].sum()


def calcular_ingresos(proyeccion: pd.DataFrame,
                      catalogo_precio: pd.DataFrame,
                      escenarios: dict[str, float] | None = None,
                      mix_fuera_de_rango: dict[str, float] | None = None,
                      mapeo_proyeccion: dict[str, str] | None = None) -> ResultadoIngresos:
    """Cruza proyeccion x catalogo (variedad+longitud) y devuelve ingresos.

    `catalogo_precio` es la salida de elegir_mercado: ya trae, por variedad y
    longitud, la fraccion de tallos y un precio unico.
    """
    escenarios = escenarios or ESCENARIOS_POR_DEFECTO
    proy = _preparar_proyeccion(proyeccion, mapeo_proyeccion)

    diagnostico = Diagnostico(
        tallos_totales=float(proy['tallos'].sum()),
        mix_fuera_de_rango=dict(mix_fuera_de_rango or {}),
    )
    diagnostico.variedades_sin_catalogo = sorted(
        set(proy['variedad']) - set(catalogo_precio['variedad']))

    cruce = proy.merge(catalogo_precio, on='variedad', how='inner')
    if cruce.empty:
        raise ValueError(
            'El cruce quedo vacio: ninguna variedad proyectada esta en el catalogo. '
            'Revisa que los nombres de variedad coincidan entre la proyeccion y el '
            'catalogo de precios.'
        )
    cruce['tallos_grado'] = cruce['tallos'] * cruce['fraccion']

    sin_precio = cruce['precio'].isna()
    diagnostico.grados_sin_precio = sorted({
        f'{fila.variedad} {fila.grado}'
        for fila in cruce[sin_precio].itertuples()
    })

    detalle = cruce[~sin_precio].copy()
    if detalle.empty:
        raise ValueError(
            'Ninguna longitud proyectada tiene precio en el catalogo vigente.')

    for nombre, variacion in escenarios.items():
        detalle[f'ingreso_{nombre}'] = (
            detalle['tallos_grado'] * detalle['precio'] * (1.0 + variacion)
        )

    diagnostico.tallos_valorados = float(detalle['tallos_grado'].sum())
    if 'antiguedad_dias' in detalle.columns:
        antiguos = detalle[detalle['antiguedad_dias'] > DIAS_PRECIO_ANTIGUO]
        diagnostico.precios_desactualizados = {
            f'{fila.variedad} {fila.grado}': int(fila.antiguedad_dias)
            for fila in antiguos.drop_duplicates(
                subset=['variedad', 'grado']).itertuples()
        }

    columnas_ingreso = [f'ingreso_{nombre}' for nombre in escenarios]
    agregaciones = {col: 'sum' for col in columnas_ingreso}
    agregaciones['tallos_grado'] = 'sum'

    por_variedad = (
        detalle.groupby('variedad', as_index=False)
        .agg(agregaciones)
        .sort_values('ingreso_base' if 'ingreso_base' in columnas_ingreso
                     else columnas_ingreso[0], ascending=False)
        .reset_index(drop=True)
    )
    por_variedad['rank'] = np.arange(1, len(por_variedad) + 1)

    por_semana = (
        detalle.groupby('semana', as_index=False)
        .agg(agregaciones)
        .sort_values('semana')
        .reset_index(drop=True)
    )

    totales = {col: float(detalle[col].sum()) for col in columnas_ingreso}
    totales['tallos_valorados'] = diagnostico.tallos_valorados

    return ResultadoIngresos(
        detalle=detalle,
        por_variedad=por_variedad,
        por_semana=por_semana,
        totales=totales,
        precios_aplicados=catalogo_precio,
        diagnostico=diagnostico,
    )


# ---------------------------------------------------------------------------
# Probabilidad del ingreso. Los escenarios pesimista/base/optimista son tres
# numeros elegidos a mano; no dicen que tan probable es cada uno. Aqui se simula
# el ingreso miles de veces moviendo las dos fuentes de incertidumbre reales
# (cuantos tallos y a que precio) para responder "que probabilidad hay de
# alcanzar X". Toda la simulacion es pandas/numpy: el LLM recibe las
# probabilidades ya calculadas.
# ---------------------------------------------------------------------------

def volatilidad_precios(catalogo: pd.DataFrame,
                        ventana_dias: int = 180,
                        minimo_cambios: int = 2,
                        fecha_corte: pd.Timestamp | None = None) -> pd.DataFrame:
    """Desviacion de la variacion relativa del precio, por variedad/longitud/mercado.

    Es la incertidumbre de precio medida del propio historico acumulado, no un
    supuesto: si el 60cm de una variedad se movio +8%, -5%, +3%, esa dispersion
    es su riesgo de precio.
    """
    vacio = pd.DataFrame(
        columns=['variedad', 'grado', 'mercado', 'sigma_precio', 'cambios'])
    if catalogo is None or catalogo.empty:
        return vacio

    corte = pd.Timestamp(fecha_corte).normalize() if fecha_corte is not None \
        else pd.Timestamp(catalogo['fecha'].max()).normalize()
    ventana = catalogo[
        (catalogo['fecha'] <= corte)
        & (catalogo['fecha'] >= corte - pd.Timedelta(days=ventana_dias))
        & catalogo['precio'].notna()
    ].sort_values('fecha')
    if ventana.empty:
        return vacio

    clave = ['variedad', 'grado', 'mercado']
    ventana = ventana.assign(
        variacion=ventana.groupby(clave)['precio'].pct_change())
    resumen = (
        ventana.dropna(subset=['variacion'])
        .groupby(clave, as_index=False)
        .agg(sigma_precio=('variacion', 'std'), cambios=('variacion', 'size'))
    )
    resumen = resumen[(resumen['cambios'] >= minimo_cambios)
                      & resumen['sigma_precio'].notna()]
    return resumen.reset_index(drop=True) if not resumen.empty else vacio


def volatilidad_proyeccion(proyeccion: pd.DataFrame,
                           mapeo: dict[str, str] | None = None,
                           minimo_observaciones: int = 3) -> pd.DataFrame:
    """Dispersion y sesgo del error relativo del modelo, por variedad.

    Sale del backtest que ya viene en la proyeccion (produccion real contra
    estimado). Si la proyeccion no trae la columna real, devuelve vacio y quien
    llame tiene que declarar un supuesto en vez de fingir que hubo medicion.
    """
    vacio = pd.DataFrame(
        columns=['variedad', 'sigma_tallos', 'sesgo_relativo', 'observaciones'])
    if proyeccion is None or proyeccion.empty:
        return vacio

    col_variedad = _resolver(proyeccion, mapeo, 'variedad', requerido=False)
    if col_variedad is None:
        return vacio
    col_estimado = _resolver(proyeccion, mapeo, 'tallos', requerido=False,
                             excluir={col_variedad})
    col_real = _resolver(proyeccion, mapeo, 'real', requerido=False,
                         excluir={col_variedad, col_estimado} - {None})
    if col_estimado is None or col_real is None:
        return vacio

    pares = pd.DataFrame({
        'variedad': _texto(proyeccion[col_variedad]),
        'estimado': pd.to_numeric(proyeccion[col_estimado], errors='coerce'),
        'real': pd.to_numeric(proyeccion[col_real], errors='coerce'),
    }).dropna()
    pares = pares[pares['estimado'] > 0]
    if pares.empty:
        return vacio

    pares['error_relativo'] = (
        pares['real'] - pares['estimado']) / pares['estimado']
    resumen = (
        pares.groupby('variedad', as_index=False)
        .agg(sigma_tallos=('error_relativo', 'std'),
             sesgo_relativo=('error_relativo', 'mean'),
             observaciones=('error_relativo', 'size'))
    )
    resumen = resumen[(resumen['observaciones'] >= minimo_observaciones)
                      & resumen['sigma_tallos'].notna()]
    return resumen.reset_index(drop=True) if not resumen.empty else vacio


def cargar_errores_modelo(ruta: str | Path | None = None) -> pd.Series:
    """Errores relativos del modelo, uno por caso evaluado.

    Convencion, la misma de ProyAst: error = (real - estimado) / real. Un error
    positivo es produccion que supero al estimado; uno negativo es un estimado
    que no se cumplio. Por eso el error esta acotado en +1 pero no por abajo: el
    modelo puede proyectar el triple de lo que despues se produjo.
    """
    destino = Path(ruta) if ruta is not None else (
        Path(__file__).with_name(CARPETA_EVALUACION) / ARCHIVO_ERRORES_MODELO)
    if not Path(destino).exists():
        return pd.Series(dtype=float, name='error_relativo')
    tabla = pd.read_csv(destino)
    if tabla.empty:
        return pd.Series(dtype=float, name='error_relativo')
    columna = 'error_relativo' if 'error_relativo' in tabla.columns \
        else tabla.columns[0]
    serie = pd.to_numeric(tabla[columna], errors='coerce').dropna()
    return serie.rename('error_relativo').reset_index(drop=True)


def factor_real_sobre_estimado(errores: pd.Series) -> pd.Series:
    """Convierte el error medido en el factor que multiplica al estimado.

    De error = (real - estimado) / real sale real = estimado / (1 - error). Sin
    esta inversion la simulacion aplicaria el error sobre la base equivocada y la
    asimetria quedaria al reves: en escala de error la cola larga es la negativa,
    en escala de factor es la positiva.
    """
    limpio = pd.to_numeric(pd.Series(errores), errors='coerce').dropna()
    return 1.0 / (1.0 - limpio.clip(upper=_ERROR_MAXIMO))


def probabilidad_cumplimiento(errores: pd.Series) -> float:
    """% de casos evaluados en que la produccion real alcanzo al estimado."""
    limpio = pd.to_numeric(pd.Series(errores), errors='coerce').dropna()
    if limpio.empty:
        return 0.0
    return float(100.0 * (limpio >= 0.0).mean())


def tabla_cumplimiento(errores: pd.Series,
                       tolerancias: tuple[float, ...] = TOLERANCIAS_CUMPLIMIENTO
                       ) -> pd.DataFrame:
    """Probabilidad de que el estimado caiga dentro de cada banda de error."""
    limpio = pd.to_numeric(pd.Series(errores), errors='coerce').dropna()
    if limpio.empty:
        return pd.DataFrame(columns=['tolerancia_pct', 'casos', 'probabilidad_pct'])
    absoluto = limpio.abs()
    filas = [
        {
            'tolerancia_pct': round(100.0 * tolerancia, 1),
            'casos': int((absoluto <= tolerancia).sum()),
            'probabilidad_pct': float(100.0 * (absoluto <= tolerancia).mean()),
        }
        for tolerancia in tolerancias
    ]
    return pd.DataFrame(filas)


@dataclass
class ResultadoProbabilidad:
    """Distribucion del ingreso, no un unico numero."""

    simulaciones: int
    referencia_base: float
    media: float
    desviacion: float
    percentiles: dict[str, float]
    por_variedad: pd.DataFrame
    por_semana: pd.DataFrame
    distribucion: np.ndarray
    supuestos: dict[str, str]
    cumplimiento: pd.DataFrame = field(default_factory=pd.DataFrame)
    casos_evaluados: int = 0

    def probabilidad_de_superar(self, meta: float) -> float:
        """% de simulaciones que alcanzan o superan la meta."""
        if self.distribucion.size == 0:
            return 0.0
        return float(100.0 * (self.distribucion >= float(meta)).mean())

    def intervalo(self, confianza: float = 0.90) -> tuple[float, float]:
        if not 0.0 < confianza < 1.0:
            raise ValueError('confianza debe estar entre 0 y 1.')
        cola = (1.0 - confianza) / 2.0
        return (float(np.quantile(self.distribucion, cola)),
                float(np.quantile(self.distribucion, 1.0 - cola)))

    def ingreso_comprometible(self, confianza: float = 0.80) -> float:
        """Ingreso que se alcanza con al menos esta probabilidad.

        Es el numero que se puede comprometer: con confianza 0.80, ocho de cada
        diez escenarios simulados quedan en este ingreso o por encima. Al subir la
        exigencia el monto baja, que es justo la decision de negocio.
        """
        if not 0.0 < confianza < 1.0:
            raise ValueError('confianza debe estar entre 0 y 1.')
        if self.distribucion.size == 0:
            return 0.0
        return float(np.quantile(self.distribucion, 1.0 - confianza))

    def curva_confianza(self, niveles: tuple[float, ...] = NIVELES_CONFIANZA
                        ) -> pd.DataFrame:
        """Ingreso comprometible en cada nivel de confianza."""
        return pd.DataFrame({
            'confianza_pct': [round(100.0 * nivel, 1) for nivel in niveles],
            'ingreso': [self.ingreso_comprometible(nivel) for nivel in niveles],
        })


def _sigma_precio_por_fila(detalle: pd.DataFrame, volatilidad: pd.DataFrame,
                           sigma_defecto: float) -> tuple[np.ndarray, str]:
    if volatilidad is None or volatilidad.empty:
        return (np.full(len(detalle), float(sigma_defecto)),
                f'supuesto ({sigma_defecto:.1%}): el catalogo no tiene historico '
                'suficiente para medir volatilidad de precio')

    llaves = detalle[['variedad', 'grado', 'mercado_elegido']].rename(
        columns={'mercado_elegido': 'mercado'})
    unido = llaves.merge(
        volatilidad[['variedad', 'grado', 'mercado', 'sigma_precio']],
        on=['variedad', 'grado', 'mercado'], how='left')
    mediana = float(volatilidad['sigma_precio'].median())
    propias = int(unido['sigma_precio'].notna().sum())
    sigma = unido['sigma_precio'].fillna(mediana).to_numpy(dtype=float)
    return sigma, (
        f'historico del catalogo: {propias}/{len(unido)} filas con volatilidad '
        f'propia, el resto con la mediana ({mediana:.1%})'
    )


def _sigma_tallos_por_variedad(variedades: np.ndarray, error_modelo: pd.DataFrame,
                               sigma_defecto: float) -> tuple[np.ndarray, str, float]:
    if error_modelo is None or error_modelo.empty:
        return (np.full(len(variedades), float(sigma_defecto)),
                f'supuesto ({sigma_defecto:.1%}): la proyeccion no trae produccion '
                'real para medir el error del modelo',
                0.0)

    por_variedad = error_modelo.set_index('variedad')['sigma_tallos']
    mediana = float(por_variedad.median())
    serie = pd.Series(variedades).map(por_variedad)
    propias = int(serie.notna().sum())
    sesgo = float(error_modelo['sesgo_relativo'].median())
    return (serie.fillna(mediana).to_numpy(dtype=float),
            f'backtest del modelo: {propias}/{len(variedades)} variedades con error '
            f'medido, el resto con la mediana ({mediana:.1%})',
            sesgo)


def _resumen_percentiles(acumulado: np.ndarray, etiquetas: np.ndarray,
                         nombre: str) -> pd.DataFrame:
    p05, p50, p95 = np.percentile(acumulado, [5, 50, 95], axis=1)
    return pd.DataFrame({
        nombre: etiquetas,
        'esperado': acumulado.mean(axis=1),
        'p05': p05,
        'p50': p50,
        'p95': p95,
    })


def simular_ingresos(resultado: ResultadoIngresos,
                     catalogo: pd.DataFrame | None = None,
                     proyeccion: pd.DataFrame | None = None,
                     errores_modelo: pd.Series | None = None,
                     simulaciones: int = 2000,
                     semilla: int = 20260810,
                     sigma_precio_defecto: float = 0.10,
                     sigma_tallos_defecto: float = 0.15,
                     escala_volatilidad_proyeccion: float = 1.0,
                     correlacion_precio: float = 0.6,
                     correlacion_tallos: float = 0.5,
                     mapeo_proyeccion: dict[str, str] | None = None,
                     lote: int = 500) -> ResultadoProbabilidad:
    """Simula el ingreso moviendo tallos y precios para obtener su distribucion.

    Las dos fuentes de incertidumbre se miden de los datos cuando existen, en
    este orden de preferencia para los tallos:

    1. `errores_modelo`: los errores medidos caso por caso (ver
       cargar_errores_modelo). Se remuestrean tal cual, sin asumir campana: la
       distribucion real es asimetrica y tiene cola larga, y una normal con la
       misma desviacion subestimaria justo los casos malos.
    2. `proyeccion`: el backtest que venga en la propia proyeccion, por variedad.
    3. `sigma_tallos_defecto`: supuesto declarado, cuando no hay nada medido.

    Para los precios, `catalogo` da la volatilidad por variedad y longitud. Lo
    que no se puede medir queda declarado en `supuestos`, para que nadie lea una
    probabilidad como si fuera medida.

    Las correlaciones no son un detalle fino: los precios de todas las longitudes
    se mueven en buena parte juntos (una caida de mercado los baja a todos) y la
    produccion tambien (un mes frio afecta a toda la finca). Simular cada fila
    independiente diversificaria el riesgo de forma ficticia y devolveria un
    intervalo mucho mas angosto que el real.

    La semilla es fija a proposito: el mismo catalogo y la misma proyeccion tienen
    que dar la misma probabilidad en cada recarga del panel.
    """
    detalle = resultado.detalle
    if detalle is None or detalle.empty:
        raise ValueError('No hay ingreso valorado para simular.')
    if simulaciones < 100:
        raise ValueError(
            'Usa al menos 100 simulaciones para que los percentiles signifiquen algo.')
    for nombre, valor in (('correlacion_precio', correlacion_precio),
                          ('correlacion_tallos', correlacion_tallos)):
        if not 0.0 <= valor <= 1.0:
            raise ValueError(f'{nombre} debe estar entre 0 y 1.')
    if escala_volatilidad_proyeccion <= 0.0:
        raise ValueError('escala_volatilidad_proyeccion debe ser mayor que 0.')

    codigos_var, variedades = pd.factorize(detalle['variedad'])
    codigos_sem, semanas = pd.factorize(detalle['semana'])

    volatilidad = volatilidad_precios(catalogo) if catalogo is not None \
        else pd.DataFrame()
    sigma_precio, fuente_precio = _sigma_precio_por_fila(
        detalle, volatilidad, sigma_precio_defecto)

    factores = np.asarray([], dtype=float)
    if errores_modelo is not None:
        factores = factor_real_sobre_estimado(
            errores_modelo).to_numpy(dtype=float)
    empirico = factores.size > 0

    sigma_tallos = np.zeros(len(variedades), dtype=float)
    sesgo = 0.0
    if empirico:
        if escala_volatilidad_proyeccion != 1.0:
            # Se estira la dispersion alrededor de la mediana: la forma medida y
            # el centro se conservan, solo cambia que tan lejos llegan las colas.
            centro = float(np.median(factores))
            factores = np.clip(
                centro + (factores - centro) * escala_volatilidad_proyeccion,
                0.0, None)
        fuente_tallos = (
            f'distribucion empirica de {factores.size} evaluaciones del modelo, '
            'remuestreadas sin suponer normalidad'
        )
    else:
        error_modelo = volatilidad_proyeccion(proyeccion, mapeo_proyeccion) \
            if proyeccion is not None else pd.DataFrame()
        sigma_tallos, fuente_tallos, sesgo = _sigma_tallos_por_variedad(
            np.asarray(variedades), error_modelo, sigma_tallos_defecto)
        sigma_tallos = sigma_tallos * escala_volatilidad_proyeccion

    base_fila = (detalle['tallos'] * detalle['fraccion']
                 * detalle['precio']).to_numpy(dtype=float)

    rng = np.random.default_rng(semilla)
    raiz_comun_p, raiz_propia_p = np.sqrt(
        correlacion_precio), np.sqrt(1.0 - correlacion_precio)
    raiz_comun_t, raiz_propia_t = np.sqrt(
        correlacion_tallos), np.sqrt(1.0 - correlacion_tallos)

    totales = np.empty(simulaciones, dtype=float)
    acum_var = np.empty((len(variedades), simulaciones), dtype=float)
    acum_sem = np.empty((len(semanas), simulaciones), dtype=float)

    inicio = 0
    while inicio < simulaciones:
        ancho = min(int(lote), simulaciones - inicio)
        # Un choque comun a todo el mercado mas uno propio de cada fila.
        choque_precio = sigma_precio[:, None] * (
            raiz_comun_p * rng.standard_normal(ancho)[None, :]
            + raiz_propia_p * rng.standard_normal((len(detalle), ancho))
        )
        if empirico:
            # Se remuestrean los factores medidos. Para correlacionar variedades
            # sin deformar la distribucion, cada variedad usa el sorteo comun con
            # probabilidad `correlacion_tallos` y uno propio con el resto: asi la
            # marginal sigue siendo exactamente la empirica.
            propio = factores[rng.integers(
                0, factores.size, size=(len(variedades), ancho))]
            comun = factores[rng.integers(
                0, factores.size, size=ancho)][None, :]
            factor_tallos = np.where(
                rng.random((len(variedades), ancho)) < correlacion_tallos,
                np.broadcast_to(comun, propio.shape), propio)
        else:
            choque_tallos = sigma_tallos[:, None] * (
                raiz_comun_t * rng.standard_normal(ancho)[None, :]
                + raiz_propia_t * rng.standard_normal((len(variedades), ancho))
            )
            factor_tallos = np.clip(1.0 + choque_tallos, 0.0, None)

        # Ni un precio ni una produccion pueden ser negativos.
        ingreso = (
            base_fila[:, None]
            * np.clip(1.0 + choque_precio, 0.0, None)
            * factor_tallos[codigos_var, :]
        )

        franja = slice(inicio, inicio + ancho)
        totales[franja] = ingreso.sum(axis=0)
        bloque_var = np.zeros((len(variedades), ancho), dtype=float)
        np.add.at(bloque_var, codigos_var, ingreso)
        acum_var[:, franja] = bloque_var
        bloque_sem = np.zeros((len(semanas), ancho), dtype=float)
        np.add.at(bloque_sem, codigos_sem, ingreso)
        acum_sem[:, franja] = bloque_sem
        inicio += ancho

    referencia = float(resultado.totales.get(
        'ingreso_base', float(base_fila.sum())))
    supuestos = {
        'precio': fuente_precio,
        'tallos': fuente_tallos,
        'correlacion': (
            f'choques correlacionados: precio {correlacion_precio:.0%}, '
            f'tallos {correlacion_tallos:.0%}'
        ),
    }
    if escala_volatilidad_proyeccion != 1.0:
        supuestos['escala'] = (
            f'volatilidad de la proyeccion escalada x{escala_volatilidad_proyeccion:.2f} '
            'sobre la medida: es un supuesto del usuario, no una medicion'
        )
    cumplimiento = pd.DataFrame()
    casos_evaluados = 0
    if empirico:
        cumplimiento = tabla_cumplimiento(errores_modelo)
        casos_evaluados = int(factores.size)
        supuestos['cumplimiento'] = (
            f'{probabilidad_cumplimiento(errores_modelo):.1f}% de los '
            f'{casos_evaluados} casos evaluados alcanzaron el estimado'
        )
        supuestos['centro'] = (
            f'factor real/estimado medido: mediana {float(np.median(factores)):.3f}, '
            f'media {float(factores.mean()):.3f}. Historicamente el modelo '
            'sub-proyecta, asi que la distribucion no queda centrada en el estimado'
        )
    elif abs(sesgo) > 0.05:
        supuestos['sesgo_modelo'] = (
            f'el modelo viene sesgado {sesgo:+.1%} contra la produccion real; la '
            'simulacion se centra en el estimado, asi que el P50 arrastra ese sesgo'
        )

    por_variedad = _resumen_percentiles(
        acum_var, np.asarray(variedades), 'variedad'
    ).sort_values('p50', ascending=False).reset_index(drop=True)
    por_semana = _resumen_percentiles(
        acum_sem, np.asarray(semanas), 'semana'
    ).sort_values('semana').reset_index(drop=True)

    return ResultadoProbabilidad(
        simulaciones=int(simulaciones),
        referencia_base=referencia,
        media=float(totales.mean()),
        desviacion=float(totales.std(ddof=1)),
        percentiles={
            'p05': float(np.percentile(totales, 5)),
            'p25': float(np.percentile(totales, 25)),
            'p50': float(np.percentile(totales, 50)),
            'p75': float(np.percentile(totales, 75)),
            'p95': float(np.percentile(totales, 95)),
        },
        por_variedad=por_variedad,
        por_semana=por_semana,
        distribucion=totales,
        supuestos=supuestos,
        cumplimiento=cumplimiento,
        casos_evaluados=casos_evaluados,
    )


def tendencia_precios(catalogo: pd.DataFrame, dias: int = 7,
                      fecha_corte: pd.Timestamp | None = None) -> pd.DataFrame:
    """Precio actual vs el de hace N dias, por variedad, longitud y mercado.

    Un promedio historico no sirve para decidir: lo que mueve la decision es el
    nivel de hoy y su cambio reciente.
    """
    if catalogo is None or catalogo.empty:
        return pd.DataFrame()

    corte = pd.Timestamp(fecha_corte).normalize() if fecha_corte is not None \
        else pd.Timestamp.today().normalize()
    actual = _precio_vigente(catalogo, corte)
    if actual.empty:
        return pd.DataFrame()
    actual = actual[['variedad', 'grado', 'mercado', 'precio', 'fecha_precio']].rename(
        columns={'precio': 'precio_actual', 'fecha_precio': 'fecha'})

    clave = ['variedad', 'grado', 'mercado']
    columna_previa = f'precio_hace_{dias}d'
    previo = _precio_vigente(catalogo, corte - pd.Timedelta(days=dias))
    if previo.empty:
        actual[columna_previa] = np.nan
        actual['delta_pct'] = np.nan
        return actual

    previo = previo[clave + ['precio']].rename(
        columns={'precio': columna_previa})
    comparado = actual.merge(previo, on=clave, how='left')
    base = comparado[columna_previa]
    comparado['delta_pct'] = np.where(
        base.notna() & (base > 0),
        100.0 * (comparado['precio_actual'] - base) / base,
        np.nan,
    )
    return comparado


# ---------------------------------------------------------------------------
# Persistencia: un solo archivo acumulado, una fila por fecha/variedad/longitud/mercado.
# ---------------------------------------------------------------------------

def _carpeta(carpeta: str | Path | None) -> Path:
    destino = Path(carpeta or CARPETA_DATOS_MERCADO)
    destino.mkdir(parents=True, exist_ok=True)
    return destino


def acumular_catalogo(nuevo: pd.DataFrame,
                      carpeta: str | Path | None = None) -> Path:
    """Agrega la carga al historico sin perder las cargas anteriores.

    Se acumula en vez de reemplazar para poder calcular tendencia de precios y
    para recalcular una semana pasada con el catalogo que estaba vigente.
    """
    destino = _carpeta(carpeta) / ARCHIVO_CATALOGO
    combinado = nuevo.copy()
    if destino.exists():
        previo = pd.read_csv(destino, parse_dates=['fecha'])
        combinado = pd.concat([previo, nuevo], ignore_index=True)
    combinado['fecha'] = pd.to_datetime(combinado['fecha'], errors='coerce')
    combinado = combinado.sort_values('fecha').drop_duplicates(
        subset=['fecha', 'variedad', 'grado', 'mercado'], keep='last')
    combinado.to_csv(destino, index=False)
    return destino


def cargar_catalogo(carpeta: str | Path | None = None) -> pd.DataFrame:
    destino = _carpeta(carpeta) / ARCHIVO_CATALOGO
    if not destino.exists():
        return pd.DataFrame(columns=COLUMNAS_CATALOGO)
    return pd.read_csv(destino, parse_dates=['fecha'])


def plantilla_catalogo() -> pd.DataFrame:
    """Ejemplo minimo del archivo unico, para arrancar un analisis nuevo.

    Se ofrece como descarga en la UI: es mas rapido que explicar el formato y
    fija los nombres de columna que el detector reconoce sin ayuda.
    """
    return pd.DataFrame({
        'Fecha': ['2026-08-10'] * 6,
        'Variedad': ['FREEDOM', 'FREEDOM', 'FREEDOM',
                     'MONDIAL', 'MONDIAL', 'MONDIAL'],
        'Longitud': ['50cm', '60cm', '70cm', '50cm', '60cm', '70cm'],
        'Mercado': ['USA', 'USA', 'EU', 'USA', 'USA', 'EU'],
        'Pct': [30, 50, 20, 25, 45, 30],
        'Precio': [0.42, 0.51, 0.63, 0.45, 0.55, 0.68],
    })


def _redondear_para_prompt(df: pd.DataFrame) -> pd.DataFrame:
    """Recorta decimales antes de mandar al modelo.

    Un 19.999999999999996 gasta tokens y ensucia la lectura sin aportar nada.
    """
    salida = df.copy()
    for columna in salida.columns:
        if not pd.api.types.is_float_dtype(salida[columna]):
            continue
        nombre = str(columna).lower()
        if 'tallos' in nombre:
            decimales = 0
        elif 'precio' in nombre:
            decimales = 4
        elif 'ingreso' in nombre:
            decimales = 2
        else:
            decimales = 1
        salida[columna] = salida[columna].round(decimales)
    return salida


def _bloque_probabilidad(probabilidad: ResultadoProbabilidad,
                         variedades_ranking: set[str]) -> list[str]:
    """Probabilidades ya calculadas. El modelo no debe estimar ninguna."""
    p05, p95 = probabilidad.intervalo(0.90)
    lineas = [
        '\nPROBABILIDAD DEL INGRESO '
        f'({probabilidad.simulaciones} simulaciones en pandas, no estimar de nuevo)',
        f'Ingreso esperado: {probabilidad.media:,.2f} '
        f'(desviacion {probabilidad.desviacion:,.2f})',
        f'Intervalo 90%: {p05:,.2f} a {p95:,.2f}',
        f'Mediana (P50): {probabilidad.percentiles["p50"]:,.2f}',
        'Probabilidad de alcanzar el ingreso base calculado '
        f'({probabilidad.referencia_base:,.2f}): '
        f'{probabilidad.probabilidad_de_superar(probabilidad.referencia_base):.1f}%',
    ]
    for etiqueta, factor in (('el 90% del base', 0.9), ('el 110% del base', 1.1)):
        meta = probabilidad.referencia_base * factor
        lineas.append(
            f'Probabilidad de superar {etiqueta} ({meta:,.2f}): '
            f'{probabilidad.probabilidad_de_superar(meta):.1f}%'
        )

    lineas.append(
        'Ingreso comprometible por nivel de confianza, es decir el ingreso que se '
        'alcanza con esa probabilidad (csv):')
    lineas.append(
        _redondear_para_prompt(probabilidad.curva_confianza()).to_csv(index=False))

    if not probabilidad.cumplimiento.empty:
        lineas.append(
            'Cumplimiento historico del modelo en '
            f'{probabilidad.casos_evaluados} evaluaciones: probabilidad de que el '
            'estimado caiga dentro de cada banda de error (csv):'
        )
        lineas.append(
            _redondear_para_prompt(probabilidad.cumplimiento).to_csv(index=False))

    rango = probabilidad.por_variedad
    if 'variedad' in rango.columns:
        rango = rango[rango['variedad'].isin(variedades_ranking)]
    if not rango.empty:
        lineas.append('Rango de ingreso por variedad (csv):')
        lineas.append(
            _redondear_para_prompt(
                rango[['variedad', 'p05', 'p50', 'p95']]).to_csv(index=False)
        )

    lineas.append('De donde sale la incertidumbre:')
    lineas.extend(f'- {concepto}: {fuente}'
                  for concepto, fuente in probabilidad.supuestos.items())
    return lineas


def resumen_para_llm(resultado: ResultadoIngresos,
                     tendencia: pd.DataFrame | None = None,
                     top: int = 15,
                     probabilidad: ResultadoProbabilidad | None = None) -> str:
    """Bloque compacto y ya calculado para incluir en el prompt.

    Reemplaza el envio de filas crudas: son ~40 lineas en vez de miles de filas,
    y el modelo no tiene que multiplicar nada. Con el precio por variedad la
    tabla de precios crece con el catalogo completo, asi que se recorta a las
    variedades del ranking.
    """
    lineas: list[str] = [
        'INGRESOS PROYECTADOS (calculados, no estimar de nuevo)']

    for nombre, valor in resultado.totales.items():
        if nombre.startswith('ingreso_'):
            lineas.append(
                f'Total {nombre.replace("ingreso_", "")}: {valor:,.2f}')
    lineas.append(
        f'Tallos valorados: {resultado.totales["tallos_valorados"]:,.0f}')
    lineas.append(
        f'Cobertura de la valoracion: {resultado.diagnostico.cobertura_pct:.1f}%')

    columna_orden = 'ingreso_base' if 'ingreso_base' in resultado.por_variedad.columns \
        else [c for c in resultado.por_variedad.columns if c.startswith('ingreso_')][0]
    ranking = resultado.por_variedad.head(top)
    lineas.append(f'\nTop {len(ranking)} variedades por ingreso (csv):')
    lineas.append(
        _redondear_para_prompt(
            ranking[['rank', 'variedad', 'tallos_grado', columna_orden]]
        ).to_csv(index=False)
    )

    variedades_ranking = set(ranking['variedad'])
    precios = resultado.precios_aplicados
    if 'variedad' in precios.columns:
        precios = precios[precios['variedad'].isin(variedades_ranking)]
    columnas_precio = [
        c for c in ['variedad', 'grado', 'mercado_elegido', 'precio',
                    'brecha_pct', 'antiguedad_dias']
        if c in precios.columns
    ]
    lineas.append('Precios aplicados por variedad y longitud (csv):')
    lineas.append(
        _redondear_para_prompt(precios[columnas_precio]).to_csv(index=False)
    )

    if tendencia is not None and not tendencia.empty:
        recorte = tendencia
        if 'variedad' in recorte.columns:
            recorte = recorte[recorte['variedad'].isin(variedades_ranking)]
        columnas_tendencia = [
            c for c in recorte.columns
            if c in ('variedad', 'grado', 'mercado', 'precio_actual', 'delta_pct')
            or c.startswith('precio_hace_')
        ]
        if not recorte.empty:
            lineas.append('Tendencia de precios (csv):')
            lineas.append(
                _redondear_para_prompt(
                    recorte[columnas_tendencia]
                ).to_csv(index=False)
            )

    if probabilidad is not None:
        lineas.extend(_bloque_probabilidad(probabilidad, variedades_ranking))

    avisos = resultado.diagnostico.mensajes()
    if avisos:
        lineas.append(
            'ADVERTENCIAS DE COBERTURA (mencionalas si afectan la respuesta):')
        lineas.extend(f'- {aviso}' for aviso in avisos)

    return '\n'.join(lineas)
