#!/usr/bin/env python3
import os
import sys
import pandas as pd

# arquivos esperados (ajuste se necessário)
BASE_DIR = 'organized'
TRAIN_PATH = os.path.join(BASE_DIR, 'train.csv')
VALIDATION_CANDIDATES = [os.path.join(BASE_DIR, 'validation.csv'),
                         os.path.join(BASE_DIR, 'val.csv')]
TEST_PATH = os.path.join(BASE_DIR, 'test.csv')

USECOLS = ['URL', 'label']
DTYPES = {'URL': 'string'}

def find_validation_path():
    for p in VALIDATION_CANDIDATES:
        if os.path.exists(p):
            return p
    return None

def load_two_cols(path):
    try:
        df = pd.read_csv(path, usecols=USECOLS, dtype=DTYPES)
    except ValueError as e:
        df_all = pd.read_csv(path, dtype=str)
        cols = {c.strip(): c for c in df_all.columns}
        if 'URL' in cols and 'label' in cols:
            df = df_all[[cols['URL'], cols['label']]].rename(columns={cols['URL']:'URL', cols['label']:'label'})
            df['URL'] = df['URL'].astype('string')
        else:
            raise RuntimeError(f"Arquivo {path} nao contem colunas 'URL' e 'label' reconheciveis.") from e
    # limpar NA nas colunas essenciais
    before = len(df)
    df = df.dropna(subset=['URL', 'label']).reset_index(drop=True)
    after = len(df)
    if after < before:
        print(f"[aviso] {path}: {before-after} linhas com URL/label faltando foram descartadas.")
    return df

def summarize(df, name):
    total = len(df)
    dup_urls = df['URL'].duplicated().sum()
    label_counts = df['label'].value_counts(dropna=False).to_dict()
    print(f"\n== {name} ==")
    print(f"Linhas totais: {total}")
    print(f"URLs duplicadas (mesma URL aparecendo mais de uma vez no mesmo arquivo): {dup_urls}")
    print("Distribuicao por label:")
    for k, v in label_counts.items():
        print(f"  {k} : {v}")
    # mostrar exemplos de duplicatas internas (até 5)
    if dup_urls > 0:
        dups = df[df['URL'].duplicated(keep=False)].sort_values('URL')
        print("\nExemplos de URLs duplicadas (ate 5):")
        for url in dups['URL'].unique()[:5]:
            rows = dups[dups['URL'] == url]
            print(f" URL: {url}  -> aparicoes: {len(rows)}  labels: {rows['label'].unique().tolist()}")
    return {
        'total': total,
        'dup_urls': dup_urls,
        'label_counts': label_counts,
        'urls_set': set(df['URL'].dropna().astype(str).tolist()),
        'url_to_label': dict(zip(df['URL'].astype(str).tolist(), df['label'].tolist()))
    }

def main():
    # verificar existência
    missing = []
    for p in [TRAIN_PATH, TEST_PATH]:
        if not os.path.exists(p):
            missing.append(p)
    val_path = find_validation_path()
    if val_path is None:
        missing.append(VALIDATION_CANDIDATES[0])  # prefer show the primary name
    else:
        # add val path to list to check existence (it exists)
        pass

    if missing:
        print("Erro: os seguintes arquivos nao foram encontrados:")
        for m in missing:
            print(" -", m)
        print("Coloque os arquivos na pasta 'organized/' ou ajuste os nomes no script.")
        sys.exit(1)

    # carregar
    print("Carregando arquivos (apenas colunas 'URL' e 'label')...")
    train_df = load_two_cols(TRAIN_PATH)
    val_df = load_two_cols(val_path) if val_path else None
    test_df = load_two_cols(TEST_PATH)

    # resumir
    train_info = summarize(train_df, "TRAIN")
    val_info = summarize(val_df, "VALIDATION")
    test_info = summarize(test_df, "TEST")

    # verificações cruzadas de URLs
    urls_train = train_info['urls_set']
    urls_val = val_info['urls_set']
    urls_test = test_info['urls_set']

    inter_tv = urls_train.intersection(urls_val)
    inter_tt = urls_train.intersection(urls_test)
    inter_vt = urls_val.intersection(urls_test)

    print("\n== Sobreposicao entre splits ==")
    print(f"TRAIN ∩ VALIDATION : {len(inter_tv)} URLs iguais")
    print(f"TRAIN ∩ TEST       : {len(inter_tt)} URLs iguais")
    print(f"VALIDATION ∩ TEST  : {len(inter_vt)} URLs iguais")

    if len(inter_tv) > 0:
        print("Exemplos (ate 5) de URLs em TRAIN e VALIDATION:")
        for url in list(inter_tv)[:5]:
            print(" -", url)

    if len(inter_tt) > 0:
        print("Exemplos (ate 5) de URLs em TRAIN e TEST:")
        for url in list(inter_tt)[:5]:
            print(" -", url)

    if len(inter_vt) > 0:
        print("Exemplos (ate 5) de URLs em VALIDATION e TEST:")
        for url in list(inter_vt)[:5]:
            print(" -", url)

    print("\n== Verificando conflitos de label entre arquivos (mesma URL com labels diferentes) ==")
    combined_map = {}
    conflicts = []

    def check_map(url_label_map, source_name):
        for url, lab in url_label_map.items():
            if url in combined_map and combined_map[url] != lab:
                conflicts.append((url, combined_map[url], lab, source_name))
            else:
                combined_map[url] = lab

    check_map(train_info['url_to_label'], 'TRAIN')
    check_map(val_info['url_to_label'], 'VALIDATION')
    check_map(test_info['url_to_label'], 'TEST')

    print(f"URLs com labels conflitantes entre arquivos: {len(conflicts)}")
    if conflicts:
        print("Exemplos (ate 5) de conflitos:")
        for url, lab1, lab2, src in conflicts[:5]:
            print(f" - URL: {url}  labels conflitantes: {lab1} vs {lab2} (ultimo visto em {src})")

    total_combined = train_info['total'] + val_info['total'] + test_info['total']
    print("\n== Proporcoes observadas (em relacao ao total combinado dos tres arquivos) ==")
    print(f"Total combinado: {total_combined}")
    if total_combined > 0:
        print(f"TRAIN  : {train_info['total']} ({train_info['total']/total_combined:.2%})")
        print(f"VALID  : {val_info['total']} ({val_info['total']/total_combined:.2%})")
        print(f"TEST   : {test_info['total']} ({test_info['total']/total_combined:.2%})")

    print("\nVerificacao concluida.")

if __name__ == '__main__':
    main()

