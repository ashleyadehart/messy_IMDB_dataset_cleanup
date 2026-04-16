import pandas as pd
import chardet

# -------------------------------
# STEP 1: DETECT ENCODING
# -------------------------------
def detect_encoding(file_path):
    print("Detecting file encoding...")

    with open(file_path, "rb") as f:
        raw_data = f.read()

    result = chardet.detect(raw_data)

    print("Encoding result:", result)

    # Warning for low confidence
    if result['confidence'] < 0.5:
        print("Warning: Low encoding confidence. File may contain mixed or corrupted encoding.")

    return result


# -------------------------------
# STEP 2: LOAD DATA
# -------------------------------
def load_data(file_path):
    print("\nLoading dataset...")

    try:
        df = pd.read_csv(file_path, encoding='utf-8', sep=None, engine='python')
        print("Loaded with UTF-8")
    except:
        df = pd.read_csv(file_path, encoding='latin1', sep=None, engine='python')
        print("Loaded with latin1 (fallback)")

    return df


# -------------------------------
# STEP 3: CLEAN COLUMN NAMES
# -------------------------------
def clean_column_names(df):
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    df = df.rename(columns={
        "imbd_title_id": "imdb_title_id",
        "original_titlê": "original_title",
        "genrë¨": "genre"
    })

    return df


# -------------------------------
# STEP 4: DROP UNUSED COLUMNS
# -------------------------------
def drop_unused_columns(df):
    return df.drop(columns=["unnamed:_8"], errors='ignore')


# -------------------------------
# STEP 5: HANDLE INITIAL MISSING VALUES
# -------------------------------
def handle_missing_values(df):
    df['content_rating'] = df['content_rating'].fillna("Not Rated")

    if not df['duration'].mode().empty:
        df['duration'] = df['duration'].fillna(df['duration'].mode()[0])
    else:
        df['duration'] = df['duration'].fillna("0")

    return df


# -------------------------------
# Step 6: NUMERIC CONVERSION
# -------------------------------
def safe_numeric(series, pattern=None):
    cleaned = series.astype(str)

    if pattern:
        cleaned = cleaned.str.replace(pattern, '', regex=True)

    return pd.to_numeric(cleaned, errors='coerce')


# -------------------------------
# STEP 7: CLEAN NUMERIC COLUMNS
# -------------------------------
def clean_numeric_columns(df):
    print("\nCleaning numeric columns...")

    df['votes'] = df['votes'].astype(str).str.replace('.', '', regex=False)
    df['votes'] = safe_numeric(df['votes'])

    df['score'] = (
        df['score']
        .astype(str)
        .str.replace(',', '.', regex=False)
        .str.replace('[^0-9.]', '', regex=True)
    )
    df['score'] = safe_numeric(df['score'])

    df['income'] = safe_numeric(df['income'], '[^0-9]')

    df['release_year'] = (
        df['release_year']
        .astype(str)
        .str.extract(r'(\d{4})')
    )
    df['release_year'] = safe_numeric(df['release_year'])

    df['duration'] = (
        df['duration']
        .astype(str)
        .str.extract(r'(\d+)')
    )
    df['duration'] = safe_numeric(df['duration'])

    return df


# -------------------------------
# STEP 8: FINAL CLEANUP
# -------------------------------
def final_cleanup(df):
    df = df.drop_duplicates().copy()

    df['genre'] = df['genre'].astype(str).str.lower().str.strip()
    df['country'] = df['country'].astype(str).str.strip().str.title()
    df['director'] = df['director'].astype(str).str.strip().str.title()
    df['original_title'] = df['original_title'].astype(str).str.strip()
    df['content_rating'] = df['content_rating'].astype(str).str.strip()

    df = df.reset_index(drop=True)

    return df


# -------------------------------
# STEP 9: FIX ENCODING (SAFE)
# -------------------------------
def fix_encoding(df):
    def safe_fix(text):
        if isinstance(text, str) and "Ã" in text:
            try:
                return text.encode('latin1').decode('utf-8')
            except:
                return text
        return text

    text_columns = ['original_title', 'director', 'genre', 'country', 'content_rating']

    for col in text_columns:
        df[col] = df[col].apply(safe_fix)

    return df


# -------------------------------
# STEP 10: HANDLE FINAL MISSING DATA
# -------------------------------
def handle_missing_final(df):
    print("\nHandling missing values...")

    df = df.dropna(how='all')

    df = df.dropna(subset=[
        'imdb_title_id',
        'original_title'
    ])

    df['release_year'] = df['release_year'].fillna(df['release_year'].median())
    df['duration'] = df['duration'].fillna(df['duration'].median())
    df['score'] = df['score'].fillna(df['score'].mean())

    return df


# -------------------------------
# STEP 11: ENFORCE FINAL DATA TYPES
# -------------------------------
def enforce_final_types(df):
    print("\nEnforcing final data types...")

    df['votes'] = df['votes'].astype('int64')
    df['release_year'] = df['release_year'].astype('int64')
    df['duration'] = df['duration'].astype('int64')

    if (df['income'].dropna() % 1 == 0).all():
        df['income'] = df['income'].astype('int64')

    return df


# -------------------------------
# STEP 12: VALIDATION
# -------------------------------
def validate_data(df):
    print("\n--- VALIDATION ---")

    print("\nDataset Info:")
    df.info()

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nDuplicate rows:", df.duplicated().sum())

    print("\nTotal rows:", len(df))

    print("\nColumn lengths:")
    for col in df.columns:
        print(col, len(df[col]))

    print("\nSample data:")
    print(df.head())


# -------------------------------
# STEP 13: SAVE DATA
# -------------------------------
def save_data(df, output_path):
    df.to_csv(output_path, index=False)
    print(f"\nClean dataset saved to: {output_path}")


# -------------------------------
# MAIN PIPELINE
# -------------------------------
def main():
    input_file = "messy_IMDB_dataset.csv"
    output_file = "clean_IMDB_dataset.csv"

    detect_encoding(input_file)

    df = load_data(input_file)
    df = clean_column_names(df)
    df = drop_unused_columns(df)
    df = handle_missing_values(df)
    df = clean_numeric_columns(df)
    df = final_cleanup(df)
    df = fix_encoding(df)
    df = handle_missing_final(df)
    df = enforce_final_types(df)

    validate_data(df)

    save_data(df, output_file)


# -------------------------------
# RUN SCRIPT
# -------------------------------
if __name__ == "__main__":
    main()