import pandas as pd
from rapidfuzz import process, fuzz


def clean_exercises(exercises_raw: dict) -> pd.DataFrame:
    df = pd.DataFrame(exercises_raw)

    def contar_instrucciones_es(x):
        if isinstance(x, dict):
            pasos_es = x.get('es')
            if isinstance(pasos_es, list):
                return len(pasos_es)
            pasos_en = x.get('en')
            if isinstance(pasos_en, list):
                return len(pasos_en)
        return 0

    def contar_musculos(muscle_group, secondary_muscles):
        total = 0
        if isinstance(muscle_group, str) and muscle_group.strip():
            total += 1
        if isinstance(secondary_muscles, list):
            total += len(secondary_muscles)
        return total

    df['num_instrucciones'] = df['instruction_steps'].apply(contar_instrucciones_es) if 'instruction_steps' in df.columns else 0
    df['num_musculos_involucrados'] = df.apply(
        lambda f: contar_musculos(f.get('muscle_group'), f.get('secondary_muscles')), axis=1
    )
    df['musculo_primario'] = df['muscle_group'] if 'muscle_group' in df.columns else ''
    return df


def enrich_with_external_metadata(exercises_limpio: pd.DataFrame, mega_gym_dataset: pd.DataFrame) -> pd.DataFrame:
    df_base = exercises_limpio.copy()
    df_mega = mega_gym_dataset.copy()

    df_base['name_clean'] = df_base['name'].str.lower().str.strip()
    df_mega['Title_clean'] = df_mega['Title'].str.lower().str.strip()
    opciones_mega = df_mega['Title_clean'].tolist()

    def buscar_mejor_coincidencia(query):
        match = process.extractOne(query, opciones_mega, scorer=fuzz.WRatio)
        return (match[0], match[1]) if match and match[1] >= 80 else (None, 0)

    resultados = df_base['name_clean'].apply(buscar_mejor_coincidencia)
    df_base['best_match'] = [res[0] for res in resultados]

    cobertura = (df_base['best_match'].notna().sum() / len(df_base)) * 100
    print(f"---> Porcentaje real de cobertura del match: {cobertura:.2f}% <---")

    df_enriched = pd.merge(
        df_base, df_mega[['Title_clean', 'Type', 'Level', 'Rating']],
        left_on='best_match', right_on='Title_clean', how='left'
    )
    df_enriched = df_enriched.drop(columns=['name_clean', 'Title_clean', 'best_match'])
    df_enriched[['Type', 'Level', 'Rating']] = df_enriched[['Type', 'Level', 'Rating']].fillna('No especificado').astype(str)
    return df_enriched