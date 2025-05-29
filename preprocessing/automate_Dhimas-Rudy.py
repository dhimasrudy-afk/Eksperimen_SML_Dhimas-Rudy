# automate_Dhimas-Rudy.py

import pandas as pd
import numpy as np
import os

def load_dataset(file_path):
    """
    Memuat dataset dari file CSV.
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Dataset berhasil dimuat dari {file_path}")
        return df
    except FileNotFoundError:
        print(f"Error: File {file_path} tidak ditemukan.")
        return None

def drop_unnecessary_columns(df, columns_to_drop):
    """
    Menghapus kolom-kolom yang tidak diperlukan dari DataFrame.
    """
    df_dropped = df.drop(columns=columns_to_drop, inplace=False) # Menggunakan inplace=False agar tidak mengubah df asli secara langsung
    print(f"Kolom yang dihapus: {columns_to_drop}")
    return df_dropped

def encode_categorical_columns(df, column_to_encode, prefix):
    """
    Melakukan one-hot encoding pada kolom kategorikal dan mengubah tipe datanya menjadi integer.
    """
    df_encoded = pd.get_dummies(df, columns=[column_to_encode], prefix=[prefix], dtype=int)
    # # Kode di notebook Anda secara eksplisit mengubah tipe kolom hasil get_dummies,
    # # namun dtype=int pada get_dummies seharusnya sudah cukup.
    # # Jika masih ada kolom F dan M sebagai boolean/object, baris berikut bisa ditambahkan:
    # if f'{prefix}_F' in df_encoded.columns:
    #     df_encoded[f'{prefix}_F'] = df_encoded[f'{prefix}_F'].astype(int)
    # if f'{prefix}_M' in df_encoded.columns:
    #     df_encoded[f'{prefix}_M'] = df_encoded[f'{prefix}_M'].astype(int)
    print(f"Kolom '{column_to_encode}' telah di-encode.")
    return df_encoded

def remove_outliers_iqr(df, column):
    """
    Menghapus outlier dari kolom tertentu menggunakan metode IQR.
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    df_filtered = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    print(f"Outlier dihapus dari kolom: {column}")
    return df_filtered

def transform_diagnosis_column(df, column_name='diagnosis'):
    """
    Mengubah nilai pada kolom 'diagnosis': 1 menjadi 0 (benign/control), dan nilai lainnya (2, 3) menjadi 1 (cancer).
    """
    df_transformed = df.copy()
    df_transformed[column_name] = df_transformed[column_name].apply(lambda x: 0 if x == 1 else 1)
    print(f"Kolom '{column_name}' telah ditransformasi (0 untuk benign/control, 1 untuk cancer).")
    return df_transformed

def preprocess_data(df_raw):
    """
    Fungsi utama untuk melakukan seluruh langkah preprocessing data.
    """
    # 1. Hapus kolom yang tidak diperlukan (sesuai notebook)
    columns_to_drop = ['sample_id', 'patient_cohort', 'sample_origin', 'stage', 'benign_sample_diagnosis', 'plasma_CA19_9', 'REG1A'] #
    df_processed = drop_unnecessary_columns(df_raw, columns_to_drop)

    # 2. Encoding data kategorikal (kolom 'sex')
    df_processed = encode_categorical_columns(df_processed, column_to_encode='sex', prefix='sex') #

    # 3. Penanganan Outlier (sesuai notebook untuk kolom numerik)
    numerical_cols_for_outlier = ['age', 'creatinine', 'LYVE1', 'REG1B', 'TFF1'] #
    df_temp_for_outlier_removal = df_processed.copy() #
    for col in numerical_cols_for_outlier:
        # Penting: Pastikan df_temp_for_outlier_removal diupdate di setiap iterasi
        # agar efek penghapusan outlier bersifat kumulatif seperti di notebook
        Q1 = df_temp_for_outlier_removal[col].quantile(0.25)
        Q3 = df_temp_for_outlier_removal[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_temp_for_outlier_removal = df_temp_for_outlier_removal[(df_temp_for_outlier_removal[col] >= lower_bound) & (df_temp_for_outlier_removal[col] <= upper_bound)] #
        print(f"Outlier dihapus dari kolom (dalam fungsi utama): {col}")
    
    df_processed = df_temp_for_outlier_removal.copy() #
    print(f"Jumlah baris setelah penghapusan semua outlier: {len(df_processed)}")

    # 4. Transformasi kolom 'diagnosis'
    df_processed = transform_diagnosis_column(df_processed, column_name='diagnosis') #
    
    print("Preprocessing data selesai.")
    return df_processed

# Bagian ini untuk menjalankan preprocessing jika file ini dijalankan secara langsung
if __name__ == "__main__":
    # Ganti dengan path ke dataset mentah Anda
    # Misalkan dataset mentah ada di folder '../namadataset_raw/debernardi_2020.csv'
    # relatif terhadap lokasi file automate_Dhimas-Rudy.py
    
    # Path dataset mentah (seperti di notebook Anda setelah diunduh dan dipindahkan)
    raw_dataset_path = "dataset_raw/debernardi_2020.csv"
    
    # Path untuk menyimpan dataset yang sudah diproses (sesuaikan dengan struktur folder yang diminta)
    # Misalnya, di dalam folder 'preprocessing' dengan nama 'debernardi_2020_preprocessed.csv'
    processed_dataset_path = "preprocessing/dataset_preprocessing/debernardi_2020_preprocessed.csv"
    os.makedirs(processed_dataset_path.rsplit('/', 1)[0], exist_ok=True)  # Membuat folder jika belum ada
    # Atau jika ingin disimpan di folder namadataset_preprocessing:
    # processed_dataset_path = "../namadataset_preprocessing/debernardi_2020_preprocessed.csv"


    # Memuat dataset mentah
    df_raw = load_dataset(raw_dataset_path)

    if df_raw is not None:
        # Melakukan preprocessing
        df_preprocessed = preprocess_data(df_raw.copy()) # Mengirim copy agar df_raw tetap utuh

        # Menampilkan info dari data yang sudah diproses
        print("\nInformasi DataFrame setelah preprocessing:")
        df_preprocessed.info()
        print("\nBeberapa baris pertama DataFrame setelah preprocessing:")
        print(df_preprocessed.head())
        print(f"\nJumlah baris dan kolom setelah preprocessing: {df_preprocessed.shape}")
        print("\nStatistik deskriptif setelah preprocessing:")
        print(df_preprocessed.describe(include='all'))
        print("\nJumlah nilai null setelah preprocessing:")
        print(df_preprocessed.isnull().sum())
        
        # Menyimpan dataset yang sudah diproses
        try:
            df_preprocessed.to_csv(processed_dataset_path, index=False)
            print(f"\nDataset yang sudah diproses disimpan di: {processed_dataset_path}")
        except Exception as e:
            print(f"Error saat menyimpan dataset yang sudah diproses: {e}")