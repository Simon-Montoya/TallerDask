#!/usr/bin/env python3

import os
import random

import numpy as np
import pandas as pd


def generate_dirty_dataset(num_rows=300_000, output_dir="shared-data/raw"):

    os.makedirs(output_dir, exist_ok=True)

    print(f"[*] Iniciando generación de {num_rows:,} filas con ruido sintético...")

    # Semillas para obtener siempre resultados reproducibles
    np.random.seed(42)
    random.seed(42)

    # ---------------------------------------------------------
    # 1. raw_customer_code
    # ---------------------------------------------------------

    clean_ids = np.random.randint(
        10000,
        99999,
        size=num_rows
    )

    templates = [
        " CLI-{id}-A ",
        "cli_{id}_norm",
        "RAW#{id}-[V2]",
        "CUST-{id}-EXP",
        " {id} ",
        "ANOMALOUS_STR",
        "INVALID",
        " ",
        None
    ]

    weights = [
        0.45,
        0.20,
        0.15,
        0.10,
        0.04,
        0.02,
        0.02,
        0.01,
        0.01
    ]

    raw_customer_code = []

    for cid in clean_ids:

        template = random.choices(
            templates,
            weights=weights
        )[0]

        if template is None:
            raw_customer_code.append(np.nan)
        else:
            raw_customer_code.append(
                template.format(id=cid)
            )

    # ---------------------------------------------------------
    # 2. Textos con mojibake
    # ---------------------------------------------------------

    mojibake_phrases = [
        "TransacciÃ³n exitosa en BogotÃ¡ D.C.",
        "Cliente atendido en MedellÃ­n por garantÃ­a",
        "VerificaciÃ³n de crÃ©dito rechazada en PopayÃ¡n",
        "EnvÃ­o exprÃ©s hacia Cartagena de Indias",
        "ActualizaciÃ³n de direcciÃ³n en Cali (Valle)",
        "OperaciÃ³n pendiente de conciliaciÃ³n bancaria",
        "Sin observaciones registradas",
        "Ã±andÃº importado - paquete especial",
        " ",
        None
    ]

    phrase_weights = [
        0.25,
        0.20,
        0.15,
        0.15,
        0.10,
        0.08,
        0.03,
        0.02,
        0.01,
        0.01
    ]

    corrupted_notes = random.choices(
        mojibake_phrases,
        weights=phrase_weights,
        k=num_rows
    )

    # ---------------------------------------------------------
    # 3. Teléfonos caóticos
    # ---------------------------------------------------------

    phone_templates = [
        "+57 (310) {p1}-{p2}",
        "310.{p1}.{p2}",
        "TEL: 310{p1}{p2} Ext 402",
        "0057 310 {p1} {p2}",
        "310{p1}{p2}",
        "DESCONOCIDO",
        "N/A",
        "--",
        ""
    ]

    phone_weights = [
        0.35,
        0.25,
        0.15,
        0.10,
        0.08,
        0.03,
        0.02,
        0.01,
        0.01
    ]

    raw_phones = []

    for _ in range(num_rows):

        template = random.choices(
            phone_templates,
            weights=phone_weights
        )[0]

        phone = template.format(
            p1=random.randint(100, 999),
            p2=random.randint(1000, 9999)
        )

        raw_phones.append(phone)

    # ---------------------------------------------------------
    # 4. Métricas numéricas
    # ---------------------------------------------------------

    amounts = np.random.exponential(
        scale=150.0,
        size=num_rows
    )

    # 3% de valores NaN
    nan_mask = np.random.rand(num_rows) < 0.03
    amounts[nan_mask] = np.nan

    # ---------------------------------------------------------
    # 5. Categorías
    # ---------------------------------------------------------

    categories = random.choices(
        [
            "FINANCE",
            "LOGISTICS",
            "RETAIL",
            "HEALTH",
            "TECH"
        ],
        k=num_rows
    )

    # ---------------------------------------------------------
    # Construcción del DataFrame
    # ---------------------------------------------------------

    df = pd.DataFrame({
        "transaction_id": [
            f"TX-{i:07d}"
            for i in range(1, num_rows + 1)
        ],

        "raw_customer_code": raw_customer_code,

        "city_notes_corrupted": corrupted_notes,

        "phone_raw": raw_phones,

        "amount_usd": np.round(amounts, 2),

        "business_category": categories
    })

    # ---------------------------------------------------------
    # Dividir en 6 archivos CSV
    # ---------------------------------------------------------

    num_chunks = 6
    chunk_size = num_rows // num_chunks

    for idx in range(num_chunks):

        start = idx * chunk_size
        end = (idx + 1) * chunk_size

        chunk_df = df.iloc[start:end]

        chunk_path = os.path.join(
            output_dir,
            f"transactions_dirty_part_{idx + 1}.csv"
        )

        chunk_df.to_csv(
            chunk_path,
            index=False,
            encoding="utf-8"
        )

        print(
            f" -> Generada partición "
            f"{idx + 1}/{num_chunks}: "
            f"{chunk_path} "
            f"({len(chunk_df):,} filas)"
        )

    print(
        "\n[OK] Conjunto de datos generado "
        "correctamente en shared-data/raw/"
    )


if __name__ == "__main__":
    generate_dirty_dataset()