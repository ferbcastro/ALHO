import os
import pandas as pd
from sklearn.model_selection import train_test_split

# path info
INPUT_DIR = 'unorganized'
INPUT_FILENAME = 'PhiUSIIL_Phishing_URL_Dataset.csv'
INPUT_CSV = os.path.join(INPUT_DIR, INPUT_FILENAME)

# path info
OUTPUT_DIR = 'organized'
OUTPUT_TRAIN = os.path.join(OUTPUT_DIR, 'train.csv')
OUTPUT_VAL = os.path.join(OUTPUT_DIR, 'validation.csv')
OUTPUT_TEST = os.path.join(OUTPUT_DIR, 'test.csv')

USECOLS = ['URL', 'label']
DTYPES = {'URL': 'string', 'label': 'int8'}

# Ratio
TEST_FRAC = 0.10
VAL_FRAC = 0.20
TRAIN_FRAC = 0.70
RANDOM_STATE = 42

if not os.path.exists(INPUT_CSV):
    raise FileNotFoundError(f"Arquivo nao encontrado: {INPUT_CSV} (lembre de unzip)")

print("Lendo CSV...")
df = pd.read_csv(INPUT_CSV, usecols=USECOLS, dtype=DTYPES)

before = len(df)
df = df.drop_duplicates(subset=['URL']).reset_index(drop=True)
after = len(df)
print(f"Removidas {before - after} URLs duplicadas; dataset agora tem {after} linhas unicas.")

phishing = df[df['label'] == 0]
legit = df[df['label'] == 1]

n_min = min(len(phishing), len(legit))
print(f"Contagens originais -> phishing: {len(phishing)}, legitimas: {len(legit)}")
print(f"Balanceando usando n = {n_min}.")

phishing_sample = phishing.sample(n=n_min, random_state=RANDOM_STATE)
legit_sample = legit.sample(n=n_min, random_state=RANDOM_STATE)

df_balanced = pd.concat([phishing_sample, legit_sample], ignore_index=True)
df_balanced = df_balanced.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

total = len(df_balanced)
print(f"Dataset balanceado total: {total} (cada classe = {n_min})")

train_val, test = train_test_split(
    df_balanced,
    test_size=TEST_FRAC,
    stratify=df_balanced['label'],
    random_state=RANDOM_STATE
)

val_relative = VAL_FRAC / (1.0 - TEST_FRAC)
train, val = train_test_split(
    train_val,
    test_size=val_relative,
    stratify=train_val['label'],
    random_state=RANDOM_STATE
)


print(f"Treino: {len(train)} ({len(train)/total:.1%})")
print(f"Validacao: {len(val)} ({len(val)/total:.1%})")
print(f"Teste: {len(test)} ({len(test)/total:.1%})")

print("Distribuicao por label (treino):")
print(train['label'].value_counts())
print("Distribuicao por label (validation):")
print(val['label'].value_counts())
print("Distribuicao por label (test):")
print(test['label'].value_counts())

os.makedirs(OUTPUT_DIR, exist_ok=True)

train.to_csv(OUTPUT_TRAIN, index=False)
val.to_csv(OUTPUT_VAL, index=False)
test.to_csv(OUTPUT_TEST, index=False)

print(f"Arquivos salvos: {OUTPUT_TRAIN}, {OUTPUT_VAL}, {OUTPUT_TEST}")
