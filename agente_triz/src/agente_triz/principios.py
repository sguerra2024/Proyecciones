"""Catálogo de los 40 principios inventivos clásicos de TRIZ."""

PRINCIPIOS_INVENTIVOS = (
    ("01_segmentacion", "1 Segmentación"),
    ("02_extraccion", "2 Extracción"),
    ("03_calidad_local", "3 Calidad local"),
    ("04_asimetria", "4 Asimetría"),
    ("05_combinacion", "5 Consolidación"),
    ("06_universalidad", "6 Universalidad"),
    ("07_anidamiento", "7 Anidamiento"),
    ("08_contrapeso", "8 Contrapeso"),
    ("09_accion_previa_contraria", "9 Contramedida anticipada"),
    ("10_accion_previa", "10 Acción anticipada"),
    ("11_proteccion_previa", "11 Precaución previa"),
    ("12_equipotencialidad", "12 Equipotencialidad"),
    ("13_inversion", "13 Haga lo contrario"),
    ("14_esferoidicidad", "14 Esfericidad o curvatura"),
    ("15_dinamismo", "15 Dinamicidad"),
    ("16_accion_parcial_excesiva", "16 Acción parcial o excesiva"),
    ("17_otra_dimension", "17 Transición para una nueva dimensión"),
    ("18_vibracion_mecanica", "18 Vibración mecánica"),
    ("19_accion_periodica", "19 Acción periódica"),
    ("20_continuidad_accion_util", "20 Continuidad de la acción útil"),
    ("21_salto_rapido", "21 Haga rápido"),
    ("22_convertir_dano_beneficio", "22 Convierta el mal en beneficio"),
    ("23_retroalimentacion", "23 Retroalimentación"),
    ("24_mediador", "24 Mediador"),
    ("25_autoservicio", "25 Self service"),
    ("26_copia", "26 Copia"),
    ("27_objeto_barato_corta_duracion", "27 Descarte"),
    ("28_sustitucion_mecanica", "28 Sustitución del sistema mecánico"),
    ("29_neumatica_hidraulica", "29 Construcción neumática o hidráulica"),
    ("30_membranas_flexibles", "30 Membranas flexibles o films finos"),
    ("31_materiales_porosos", "31 Materiales porosos"),
    ("32_cambio_color", "32 Cambio de color"),
    ("33_homogeneidad", "33 Homogeneidad"),
    ("34_descartar_recuperar", "34 Descartar y recuperar"),
    ("35_cambio_parametros", "35 Transformación de propiedades"),
    ("36_transicion_fase", "36 Transición de fase"),
    ("37_expansion_termica", "37 Expansión térmica"),
    ("38_oxidantes_fuertes", "38 Oxidación acelerada"),
    ("39_atmosfera_inerte", "39 Ambiente inerte"),
    ("40_materiales_compuestos", "40 Materiales compuestos"),
)

DESCRIPCIONES_PRINCIPIOS = {
    "01_segmentacion": (
        "Divida el objeto en partes independientes. Haga un objeto "
        "desarmable para facilitar el ensamblaje o desmontaje. Aumente el "
        "grado de segmentación."
    ),
    "02_extraccion": (
        "Extraiga o remueva de un objeto la parte o propiedad que incomoda. "
        "Extraiga solamente la parte o propiedad necesaria."
    ),
    "03_calidad_local": (
        "Cambie la estructura de un objeto o del ambiente de homogéneo a "
        "heterogéneo. Haga que partes diferentes realicen funciones diferentes "
        "y coloque cada parte en las condiciones más favorables para su operación."
    ),
    "04_asimetria": (
        "Sustituya formas simétricas por formas asimétricas. Si un objeto es "
        "simétrico, aumente su grado de asimetría para adaptarlo mejor a su función."
    ),
    "05_combinacion": (
        "Integre en el espacio objetos similares u objetos destinados a "
        "operaciones continuas. Consolide en el tiempo operaciones hechas "
        "secuencialmente."
    ),
    "06_universalidad": (
        "Haga que un objeto realice múltiples funciones, eliminando la "
        "necesidad de otros objetos."
    ),
    "07_anidamiento": (
        "Encaje un objeto en otro, y estos a su vez en otro. Haga que un "
        "objeto pase por la cavidad de otro."
    ),
    "08_contrapeso": (
        "Para compensar el peso de un objeto, combínelo con otro que empuje "
        "contra la gravedad o hágalo interactuar con el ambiente, por ejemplo "
        "mediante fuerzas hidrodinámicas o aerodinámicas."
    ),
    "09_accion_previa_contraria": (
        "Si una acción tiene efectos útiles y perjudiciales, complétela con "
        "contramedidas anticipadas. Si el objeto trabaja con una carga "
        "excesiva o indeseada, compénsela aplicando una carga en sentido opuesto."
    ),
    "10_accion_previa": (
        "Realice por anticipado, parcial o completamente, los cambios "
        "requeridos en un objeto. Predisponga los objetos en el lugar más "
        "conveniente para que puedan entrar en uso sin pérdida de tiempo."
    ),
    "11_proteccion_previa": (
        "Implemente acciones de emergencia de antemano para compensar la baja "
        "confiabilidad de un objeto."
    ),
    "12_equipotencialidad": (
        "Cambie las condiciones de trabajo para que no sea necesario bajar "
        "o levantar el objeto."
    ),
    "13_inversion": (
        "En lugar de la acción requerida por un problema, implemente la "
        "acción opuesta. Torne fijas las partes móviles o móviles las partes "
        "fijas. Vire el objeto de punta a cabeza."
    ),
    "14_esferoidicidad": (
        "Sustituya piezas rectilíneas por curvilíneas, superficies planas por "
        "curvas y formas cúbicas por esféricas. Use discos, rollos, espirales "
        "y bolas. Sustituya movimientos rectilíneos por rotacionales."
    ),
    "15_dinamismo": (
        "Haga que las características de un objeto o del ambiente externo se "
        "ajusten automáticamente para ofrecer el desempeño óptimo en cada "
        "situación o etapa de operación."
    ),
    "16_accion_parcial_excesiva": (
        "Si es difícil obtener el 100 % de un efecto deseado, realice la "
        "acción parcialmente o en exceso."
    ),
    "17_otra_dimension": (
        "Cambie el movimiento en una línea por movimiento en una superficie, "
        "o el movimiento en una superficie por movimiento en el espacio. "
        "Disponga objetos en varias capas o niveles, incline el objeto o use "
        "el lado opuesto de una superficie."
    ),
    "18_vibracion_mecanica": (
        "Haga que un objeto vibre u oscile. Si ya vibra, aumente su frecuencia "
        "hasta la zona ultrasónica. Use la frecuencia de resonancia del objeto "
        "o vibradores piezoeléctricos."
    ),
    "19_accion_periodica": (
        "En vez de una acción continua, use acciones periódicas o pulsantes. "
        "Si una acción ya es periódica, cambie su frecuencia. Use las pausas "
        "entre pulsos para realizar otra acción."
    ),
    "20_continuidad_accion_util": (
        "Realice una acción continuamente, haciendo que todas las partes de "
        "un objeto operen a plena capacidad todo el tiempo. Elimine tiempos "
        "de espera o acciones intermitentes."
    ),
    "21_salto_rapido": (
        "Realice en alta velocidad las operaciones peligrosas o que causen "
        "daños."
    ),
    "22_convertir_dano_beneficio": (
        "Use factores perjudiciales, especialmente los del ambiente, para "
        "obtener un efecto positivo. Elimine un factor perjudicial "
        "combinándolo con otro o auméntelo hasta que deje de causar daño."
    ),
    "23_retroalimentacion": (
        "Introduzca retroalimentación en el proceso o sistema. Si ya se usa "
        "retroalimentación, cámbiela."
    ),
    "24_mediador": (
        "Use un objeto intermediario para transferir o realizar una acción. "
        "Una temporalmente el objeto original con otro que sea fácilmente removible."
    ),
    "25_autoservicio": (
        "Haga que el objeto se sirva a sí mismo y realice acciones "
        "complementarias o de reparación. Use residuos de materiales y energía."
    ),
    "26_copia": (
        "En lugar de trabajar directamente con un objeto frágil o incómodo "
        "de operar, use una copia simplificada y barata. Sustituya el objeto "
        "por una imagen óptica que pueda procesarse."
    ),
    "27_objeto_barato_corta_duracion": (
        "Sustituya un objeto por otro más barato, sacrificando otras propiedades."
    ),
    "28_sustitucion_mecanica": (
        "Sustituya el sistema mecánico por uno óptico, acústico, térmico u "
        "olfatorio. Use campos eléctricos, magnéticos o electromagnéticos "
        "para interactuar con el objeto."
    ),
    "29_neumatica_hidraulica": (
        "Sustituya las partes sólidas de un objeto por gas o líquido. Use "
        "aire o agua para inflar, amortiguar o proteger."
    ),
    "30_membranas_flexibles": (
        "Sustituya las construcciones convencionales por membranas o películas. "
        "Aísle el objeto del ambiente externo usando films finos o membranas "
        "flexibles."
    ),
    "31_materiales_porosos": (
        "Torne poroso un objeto o use elementos porosos complementarios. Si "
        "el objeto ya es poroso, llene anticipadamente los poros con alguna sustancia."
    ),
    "32_cambio_color": (
        "Cambie el color de un objeto o de su ambiente."
    ),
    "33_homogeneidad": (
        "Haga que los objetos que interactúan con un objeto principal sean "
        "del mismo material o de materiales con propiedades similares."
    ),
    "34_descartar_recuperar": (
        "Después de que un objeto haya realizado su función o se haga "
        "innecesario, modifíquelo o descártelo. Restaure las partes usadas "
        "durante su utilización."
    ),
    "35_cambio_parametros": (
        "Cambie el estado físico de un objeto, su concentración o densidad, "
        "su grado de flexibilidad o su temperatura."
    ),
    "36_transicion_fase": (
        "Use los fenómenos que ocurren durante los cambios de fase."
    ),
    "37_expansion_termica": (
        "Use la expansión o contracción térmica del material. También puede "
        "usar varios materiales con diferentes coeficientes de expansión térmica."
    ),
    "38_oxidantes_fuertes": (
        "Pase de un nivel de oxidación a otro más elevado."
    ),
    "39_atmosfera_inerte": (
        "Sustituya el ambiente normal por un ambiente inerte. Introduzca "
        "sustancias neutras o aditivos en un objeto, o realice el proceso en vacío."
    ),
    "40_materiales_compuestos": (
        "Cambie de material uniforme a material compuesto."
    ),
}

ETIQUETAS_PRINCIPIOS = dict(PRINCIPIOS_INVENTIVOS)
ID_POR_NOMBRE_PRINCIPIO = {
    nombre: identificador for identificador, nombre in PRINCIPIOS_INVENTIVOS
}
