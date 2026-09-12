#!/usr/bin/env python3

import os
import subprocess

import dask.dataframe as dd

from prefect import flow, task


# ============================================================
# CONFIGURACIÓN
# ============================================================

RAW_DATA_PATH = "/app/shared-data/raw"

PROCESSED_DATA_PATH = (
    "/app/shared-data/processed/"
    "transactions_clean.parquet"
)

EXPECTED_ROWS = 300_000
EXPECTED_FILES = 6


# ============================================================
# TASK 1
# VERIFICAR DATOS CRUDOS
# ============================================================

@task(
    name="Validar archivos CSV",
    retries=2,
    retry_delay_seconds=3
)
def validate_raw_data():

    print()
    print("Validando archivos de entrada...")

    if not os.path.exists(RAW_DATA_PATH):
        raise FileNotFoundError(
            f"No existe {RAW_DATA_PATH}"
        )

    csv_files = [
        file
        for file in os.listdir(RAW_DATA_PATH)
        if file.endswith(".csv")
    ]

    print(
        f"Archivos CSV encontrados: "
        f"{len(csv_files)}"
    )

    if len(csv_files) != EXPECTED_FILES:

        raise ValueError(
            f"Se esperaban {EXPECTED_FILES} CSV, "
            f"pero se encontraron {len(csv_files)}."
        )

    print(
        "[PASS] Los 6 archivos CSV están disponibles."
    )

    return len(csv_files)


# ============================================================
# TASK 2
# EJECUTAR PIPELINE DASK
# ============================================================

@task(
    name="Ejecutar pipeline distribuido Dask",
    retries=2,
    retry_delay_seconds=5
)
def run_dask_pipeline():

    print()
    print(
        "Ejecutando pipeline distribuido Dask..."
    )

    process = subprocess.run(
        [
            "python",
            "/app/dask_pipeline.py"
        ],
        capture_output=True,
        text=True
    )

    # Mostrar salida de Dask dentro del log de Prefect.
    print(process.stdout)

    if process.returncode != 0:

        print(process.stderr)

        raise RuntimeError(
            "El pipeline distribuido Dask falló."
        )

    print(
        "[PASS] Pipeline Dask terminado correctamente."
    )

    return True


# ============================================================
# TASK 3
# VALIDAR PARQUET
# ============================================================

@task(
    name="Validar resultado Parquet",
    retries=2,
    retry_delay_seconds=3
)
def validate_parquet():

    print()
    print(
        "Validando dataset Parquet..."
    )

    if not os.path.exists(PROCESSED_DATA_PATH):

        raise FileNotFoundError(
            "No se encontró el dataset Parquet."
        )

    ddf = dd.read_parquet(
        PROCESSED_DATA_PATH,
        engine="pyarrow"
    )

    total_rows = (
        ddf
        .shape[0]
        .compute()
    )

    print(
        f"Registros encontrados: "
        f"{total_rows:,}"
    )

    if total_rows != EXPECTED_ROWS:

        raise ValueError(
            f"Se esperaban "
            f"{EXPECTED_ROWS:,} registros, "
            f"pero se encontraron "
            f"{total_rows:,}."
        )

    print(
        "[PASS] El Parquet contiene "
        "300.000 registros."
    )

    return total_rows


# ============================================================
# TASK 4
# QUALITY GATES
# ============================================================

@task(
    name="Quality Gates",
    retries=1
)
def quality_gates():

    print()
    print(
        "Ejecutando Quality Gates..."
    )

    ddf = dd.read_parquet(
        PROCESSED_DATA_PATH,
        engine="pyarrow"
    )

    # --------------------------------------------------------
    # Validar mojibake
    # --------------------------------------------------------

    mojibake_count = (

        ddf[
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
        f"Mojibake restante: "
        f"{mojibake_count}"
    )

    if mojibake_count != 0:

        raise ValueError(
            "Quality Gate FAILED: "
            "todavía existen textos con mojibake."
        )

    # --------------------------------------------------------
    # Validar customer_code
    # --------------------------------------------------------

    invalid_codes = (

        ~ddf[
            "customer_code"
        ]

        .fillna("")

        .str.match(
            r"^CUST-\d{5}(?:-ANOMALY)?$"
        )

    ).sum().compute()

    print(
        f"Códigos inválidos: "
        f"{invalid_codes}"
    )

    if invalid_codes != 0:

        raise ValueError(
            "Quality Gate FAILED: "
            "existen códigos con formato inválido."
        )

    anomalies = (

        ddf[
            "customer_code_is_anomaly"
        ]

        .sum()

        .compute()
    )

    print(
        f"Anomalías identificadas: "
        f"{anomalies:,}"
    )

    print()
    print(
        "[PASS] Todos los Quality Gates "
        "fueron superados."
    )

    return {
        "mojibake": mojibake_count,
        "invalid_codes": invalid_codes,
        "anomalies": anomalies
    }


# ============================================================
# FLOW PRINCIPAL
# ============================================================

@flow(
    name="Distributed Data Cleaning Pipeline",
    log_prints=True
)
def distributed_pipeline_flow():

    print()
    print("======================================")
    print(" PREFECT - PIPELINE DISTRIBUIDO")
    print("======================================")
    print()

    csv_count = validate_raw_data()

    pipeline_result = run_dask_pipeline(
        wait_for=[csv_count]
    )

    total_rows = validate_parquet(
        wait_for=[pipeline_result]
    )

    quality_result = quality_gates(
        wait_for=[total_rows]
    )

    print()
    print("======================================")
    print(" PREFECT FLOW COMPLETADO")
    print("======================================")
    print()

    return quality_result


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    distributed_pipeline_flow()