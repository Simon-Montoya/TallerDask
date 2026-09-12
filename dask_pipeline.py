#!/usr/bin/env python3

import os
import re

import pandas as pd
import dask.dataframe as dd

from dask.distributed import Client


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

SCHEDULER_ADDRESS = "tcp://dask-scheduler:8786"

INPUT_PATH = (
    "/app/shared-data/raw/"
    "transactions_dirty_part_*.csv"
)

OUTPUT_PATH = (
    "/app/shared-data/processed/"
    "transactions_clean.parquet"
)


# Regex indicada por el taller.
# Busca un identificador numérico de exactamente 5 dígitos.
#
# Ejemplos:
#
# CLI-98234-A       -> 98234
# cli_98234_norm    -> 98234
# RAW#98234-[V2]    -> 98234
# CUST-98234-EXP    -> 98234
# 98234             -> 98234
#
# INVALID           -> sin resultado
# ANOMALOUS_STR     -> sin resultado

CUSTOMER_PATTERN = re.compile(
    r"(?:CLI-|cli_|RAW#|CUST-)?(\d{5})"
)


# ============================================================
# FUNCIÓN PARA CORREGIR MOJIBAKE
# ============================================================

def fix_mojibake(value):
    """
    Corrige textos que originalmente estaban en UTF-8
    pero fueron interpretados incorrectamente como Latin-1.

    Ejemplos:

    BogotÃ¡       -> Bogotá
    TransacciÃ³n  -> Transacción
    crÃ©dito      -> crédito
    """

    if pd.isna(value):
        return pd.NA

    text = str(value).strip()

    # Convertimos cadenas vacías en valores nulos.
    if text == "":
        return pd.NA

    # Indicadores frecuentes de mojibake.
    mojibake_markers = (
        "Ã",
        "Â",
        "â"
    )

    if any(marker in text for marker in mojibake_markers):

        try:
            return (
                text
                .encode("latin-1")
                .decode("utf-8")
            )

        except (
            UnicodeEncodeError,
            UnicodeDecodeError
        ):
            # Si no puede corregirse, conservamos
            # el texto original para no perder información.
            return text

    return text


# ============================================================
# FUNCIÓN DE LIMPIEZA POR PARTICIÓN
# ============================================================

def clean_partition(pdf):
    """
    Recibe una partición Pandas desde Dask.

    Cada worker ejecutará esta función sobre las
    particiones que le asigne el Scheduler.
    """

    pdf = pdf.copy()

    # --------------------------------------------------------
    # 1. Extraer código numérico del cliente
    # --------------------------------------------------------

    extracted_ids = (
        pdf["raw_customer_code"]
        .fillna("")
        .astype("string")
        .str.extract(
            CUSTOMER_PATTERN,
            expand=False
        )
    )

    # --------------------------------------------------------
    # 2. Crear código canónico CUST-XXXXX
    # --------------------------------------------------------

    valid_ids = extracted_ids.notna()

    pdf["customer_code"] = (
        "CUST-" + extracted_ids.fillna("00000")
    ).astype("string")

    # Los valores que no tenían identificador válido
    # reciben el token solicitado por el taller.

    pdf.loc[
        ~valid_ids,
        "customer_code"
    ] = "CUST-00000-ANOMALY"

    pdf["customer_code"] = (
        pdf["customer_code"]
        .astype("string")
    )

    # --------------------------------------------------------
    # 3. Crear indicador de anomalía
    # --------------------------------------------------------

    pdf["customer_code_is_anomaly"] = (
        pdf["customer_code"]
        == "CUST-00000-ANOMALY"
    ).astype("bool")

    # --------------------------------------------------------
    # 4. Corregir Mojibake
    # --------------------------------------------------------

    pdf["city_notes_clean"] = (
        pdf["city_notes_corrupted"]
        .map(fix_mojibake)
        .astype("string")
    )

    # Garantizamos explícitamente los tipos.
    # Esto evita problemas al escribir con PyArrow.

    pdf["transaction_id"] = (
        pdf["transaction_id"]
        .astype("string")
    )

    pdf["raw_customer_code"] = (
        pdf["raw_customer_code"]
        .astype("string")
    )

    pdf["city_notes_corrupted"] = (
        pdf["city_notes_corrupted"]
        .astype("string")
    )

    pdf["phone_raw"] = (
        pdf["phone_raw"]
        .astype("string")
    )

    pdf["business_category"] = (
        pdf["business_category"]
        .astype("string")
    )

    pdf["amount_usd"] = (
        pd.to_numeric(
            pdf["amount_usd"],
            errors="coerce"
        )
        .astype("float64")
    )

    return pdf


# ============================================================
# METADATA EXPLÍCITA
# ============================================================

def create_meta():
    """
    Define explícitamente la estructura resultante.

    No utilizamos ddf._meta para las columnas nuevas
    porque queremos evitar tipos genéricos 'object'
    que pueden producir errores con PyArrow.
    """

    return pd.DataFrame({

        "transaction_id":
            pd.Series(dtype="string"),

        "raw_customer_code":
            pd.Series(dtype="string"),

        "city_notes_corrupted":
            pd.Series(dtype="string"),

        "phone_raw":
            pd.Series(dtype="string"),

        "amount_usd":
            pd.Series(dtype="float64"),

        "business_category":
            pd.Series(dtype="string"),

        "customer_code":
            pd.Series(dtype="string"),

        "customer_code_is_anomaly":
            pd.Series(dtype="bool"),

        "city_notes_clean":
            pd.Series(dtype="string")
    })


# ============================================================
# PIPELINE PRINCIPAL
# ============================================================

def main():

    print()
    print("========================================")
    print(" PIPELINE DISTRIBUIDO DASK")
    print("========================================")
    print()

    # --------------------------------------------------------
    # 1. CONECTAR CON EL SCHEDULER
    # --------------------------------------------------------

    print("[1] Conectando al Scheduler Dask...")

    client = Client(
        SCHEDULER_ADDRESS
    )

    # Esperamos hasta que los tres workers estén disponibles.
    client.wait_for_workers(
        n_workers=3,
        timeout=60
    )

    scheduler_info = client.scheduler_info()

    workers = scheduler_info["workers"]

    print(
        f"[OK] Workers conectados: "
        f"{len(workers)}"
    )

    for index, worker_address in enumerate(
        workers.keys(),
        start=1
    ):
        print(
            f"     Worker {index}: "
            f"{worker_address}"
        )

    # --------------------------------------------------------
    # 2. COMPROBAR DIRECTORIOS
    # --------------------------------------------------------

    print()
    print("[2] Verificando directorios...")

    os.makedirs(
        "/app/shared-data/processed",
        exist_ok=True
    )

    raw_directory = (
        "/app/shared-data/raw"
    )

    if not os.path.exists(raw_directory):

        client.close()

        raise FileNotFoundError(
            "No existe el directorio "
            "/app/shared-data/raw"
        )

    files = [
        file
        for file in os.listdir(raw_directory)
        if file.endswith(".csv")
    ]

    print(
        f"[OK] Archivos CSV encontrados: "
        f"{len(files)}"
    )

    if len(files) != 6:

        print(
            "[ADVERTENCIA] Se esperaban 6 archivos CSV."
        )

    # --------------------------------------------------------
    # 3. LEER CSV CON DASK
    # --------------------------------------------------------

    print()
    print("[3] Leyendo archivos CSV con Dask...")

    ddf = dd.read_csv(

        INPUT_PATH,

        # Un archivo CSV se convierte en una partición.
        # Tenemos 6 CSV -> 6 particiones.
        blocksize=None,

        dtype={

            "transaction_id":
                "string",

            "raw_customer_code":
                "string",

            "city_notes_corrupted":
                "string",

            "phone_raw":
                "string",

            "amount_usd":
                "float64",

            "business_category":
                "string"
        }
    )

    print(
        f"[OK] Particiones Dask detectadas: "
        f"{ddf.npartitions}"
    )

    # --------------------------------------------------------
    # 4. CREAR DAG DE LIMPIEZA
    # --------------------------------------------------------

    print()
    print("[4] Construyendo DAG de limpieza...")

    meta = create_meta()

    cleaned_ddf = ddf.map_partitions(
        clean_partition,
        meta=meta
    )

    print("[OK] DAG construido correctamente.")

    print(
        "     Dask todavía NO ha ejecutado "
        "las transformaciones."
    )

    print(
        "     La ejecución es Lazy / Perezosa."
    )

    # --------------------------------------------------------
    # 5. ELIMINAR RESULTADO ANTERIOR SI EXISTE
    # --------------------------------------------------------

    if os.path.exists(OUTPUT_PATH):

        import shutil

        shutil.rmtree(
            OUTPUT_PATH,
            ignore_errors=True
        )

    # --------------------------------------------------------
    # 6. EJECUTAR PROCESAMIENTO DISTRIBUIDO
    # --------------------------------------------------------

    print()
    print("[5] Ejecutando procesamiento distribuido...")
    print()
    print(
        "    Abre el Dashboard mientras se ejecuta:"
    )
    print()
    print(
        "    http://localhost:8787/status"
    )
    print()

    # Esta operación dispara realmente el DAG.
    #
    # El Scheduler comienza a distribuir
    # las 6 particiones entre los 3 workers.

    cleaned_ddf.to_parquet(

        OUTPUT_PATH,

        engine="pyarrow",

        write_index=False,

        overwrite=True
    )

    print()
    print("[OK] Procesamiento distribuido finalizado.")

    print(
        "[OK] Resultado Parquet generado en:"
    )

    print(
        f"     {OUTPUT_PATH}"
    )

    # --------------------------------------------------------
    # 7. LEER RESULTADO PARQUET
    # --------------------------------------------------------

    print()
    print("[6] Leyendo resultado para validación...")

    result = dd.read_parquet(
        OUTPUT_PATH,
        engine="pyarrow"
    )

    # --------------------------------------------------------
    # 8. CONTAR REGISTROS
    # --------------------------------------------------------

    total_rows = (
        result
        .shape[0]
        .compute()
    )

    print(
        f"[OK] Total registros procesados: "
        f"{total_rows:,}"
    )

    # --------------------------------------------------------
    # 9. CONTAR ANOMALÍAS
    # --------------------------------------------------------

    anomaly_count = (
        result[
            "customer_code_is_anomaly"
        ]
        .sum()
        .compute()
    )

    print(
        f"[OK] Customer codes anómalos: "
        f"{anomaly_count:,}"
    )

    # --------------------------------------------------------
    # 10. VALIDAR MOJIBAKE
    # --------------------------------------------------------

    remaining_mojibake = (

        result[
            "city_notes_clean"
        ]

        .fillna("")

        .str.contains(
            r"Ã|Â|â",
            regex=True
        )

        .sum()

        .compute()
    )

    print(
        f"[OK] Registros con mojibake restante: "
        f"{remaining_mojibake:,}"
    )

    # --------------------------------------------------------
    # 11. VALIDAR CUSTOMER CODE
    # --------------------------------------------------------

    invalid_customer_codes = (

        ~result[
            "customer_code"
        ]
        .fillna("")
        .str.match(
            r"^CUST-\d{5}(?:-ANOMALY)?$"
        )

    ).sum().compute()

    print(
        f"[OK] Customer codes con formato inválido: "
        f"{invalid_customer_codes:,}"
    )

    # --------------------------------------------------------
    # 12. QUALITY GATES
    # --------------------------------------------------------

    print()
    print("[7] Ejecutando Quality Gates...")

    if total_rows != 300_000:

        client.close()

        raise ValueError(
            "QUALITY GATE FAILED: "
            "El resultado no contiene "
            "exactamente 300.000 registros."
        )

    print(
        "[PASS] El dataset contiene "
        "300.000 registros."
    )

    if remaining_mojibake != 0:

        client.close()

        raise ValueError(
            "QUALITY GATE FAILED: "
            "Todavía existen textos "
            "con Mojibake."
        )

    print(
        "[PASS] No quedan secuencias "
        "evidentes de Mojibake."
    )

    if invalid_customer_codes != 0:

        client.close()

        raise ValueError(
            "QUALITY GATE FAILED: "
            "Existen customer codes "
            "con formato incorrecto."
        )

    print(
        "[PASS] Todos los customer codes "
        "tienen formato válido."
    )

    # --------------------------------------------------------
    # 13. MOSTRAR EJEMPLOS
    # --------------------------------------------------------

    print()
    print("[8] Ejemplos del resultado:")
    print()

    sample = (

        result[
            [
                "transaction_id",
                "raw_customer_code",
                "customer_code",
                "city_notes_corrupted",
                "city_notes_clean"
            ]
        ]

        .head(
            10,
            npartitions=-1
        )
    )

    print(
        sample.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # FINAL
    # --------------------------------------------------------

    print()
    print("========================================")
    print(" PIPELINE COMPLETADO CORRECTAMENTE")
    print("========================================")
    print()

    print(
        f"Registros totales:       "
        f"{total_rows:,}"
    )

    print(
        f"Anomalías detectadas:    "
        f"{anomaly_count:,}"
    )

    print(
        f"Mojibake restante:       "
        f"{remaining_mojibake:,}"
    )

    print(
        f"Códigos inválidos:       "
        f"{invalid_customer_codes:,}"
    )

    print()
    print(
        "Salida:"
    )

    print(
        OUTPUT_PATH
    )

    print()

    client.close()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()