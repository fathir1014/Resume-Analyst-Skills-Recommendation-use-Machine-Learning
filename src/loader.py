# loader.py
"""
Data loader utilities for Resume Analyst project.
- Loads Kaggle Resume.csv with robust validation
- Safe defaults to the project's canonical path
- Returns a clean DataFrame with only required columns
"""

from __future__ import annotations
from pathlib import Path
from typing import Iterable, Optional, Tuple, Union
import pandas as pd

# Default path assumes this file lives in src/ and data is at ../data/CV_kaggle/Resume/Resume.csv
DEFAULT_CSV_PATH = Path(__file__).resolve().parents[1] / "data" / "CV_kaggle" / "Resume" / "Resume.csv"
# __file__ -> lokasi fisik loader.py
# resolve() -> ubah ke path absolute, dalam kasus ini "C:\Users\HP\Desktop\Resume Analyst-Skills recommendation engine\Resume Analyst\src\loader.py"
# parents[1] -> naik satu root dari src ke Resume Analyst
REQUIRED_COLUMNS = ["Resume_str", "Category"]
# kolom wajib yang akan diambil, kalau kolom tersebut tidak ada di csv akan error
OPTIONAL_COLUMNS = ["ID", "Resume_html"]
# Kolom yang opsional jika mau diambil, jika tidak ada program akan tetap berjalan

# hasil akhir -> "...\Resume Analyst\data\CV_kaggle\Resume\Resume.csv"

def _validate_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    # df : pd.DataFrame  
    #   → parameter pertama, yaitu dataframe hasil load CSV.
    #   → nanti isinya tabel yang udah dibaca dari Resume.csv.
    #
    # required : Iterable[str]  
    #   → parameter kedua, berupa daftar string kolom yang wajib ada.
    #   → di kasus kita, isinya = ["Resume_str", "Category"].
    #
    # -> None  
    #   → artinya fungsi ini nggak mengembalikan nilai apa-apa (void function).
    #   → cuma ngecek & kalau ada yang salah langsung raise error.
    missing = [c for c in required if c not in df.columns]
    # Tujuan → cek apakah semua kolom required ada di df.columns.
    # df.columns = daftar nama kolom di dataframe.
    # Looping comprehension: ambil semua c (nama kolom wajib) yang tidak ditemukan di df.columns.
    # Hasil = list berisi nama kolom yang hilang.
    """ 
     Contoh:
     required = ["Resume_str", "Category"]
     df.columns = ["Resume", "Category"]
     missing = ["Resume_str"] """
    
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found columns: {list(df.columns)}")
    """
    if missing:
    Kondisi True kalau list missing tidak kosong (berarti ada kolom yang hilang).
    raise ValueError(...)
    - Lempar error (berhentikan program) dengan pesan yang jelas:
    - Kolom apa saja yang hilang (missing)
    - Kolom apa saja yang ditemukan (df.columns)
    - Ini disebut fail-fast strategy → lebih baik error di awal daripada model jalan dengan data salah format.
    """

def load_resume_csv(
    path: Union[str, Path, None] = None,
    usecols: Optional[Iterable[str]] = None,
    drop_duplicates: bool = True,
    min_text_len: int = 20,
    keep_optional: bool = False,
) -> pd.DataFrame:
    
    """
    - path : Union[str, Path, None] = None
        Argumen path bisa berupa string, Path, atau None.
        Jika None → otomatis pakai DEFAULT_CSV_PATH (data/Resume.csv bawaan project).

    - usecols : Optional[Iterable[str]] = None
        Pilih kolom yang mau diambil dari CSV.
        * None → ambil kolom wajib (['Resume_str', 'Category']).
        * Iterable[str] → misalnya ["Resume_str", "Category", "ID"] → hanya ambil itu.

    - drop_duplicates : bool = True
        Kalau True → hapus baris duplikat berdasarkan Resume_str.
        Kalau False → biarkan meskipun ada duplikat.

    - min_text_len : int = 20
        Buang baris yang teks resuménya terlalu pendek (<20 karakter).
        Bisa diset 0 kalau mau keep semua baris tanpa filter panjang.

    - keep_optional : bool = False
        Kalau True → ikut sertakan kolom opsional (ID, Resume_html) kalau ada di dataset.
        Kalau False → hanya ambil kolom wajib (atau sesuai usecols).
    
    Returns
    -------
    pd.DataFrame
        Dataframe bersih berisi kolom minimal ['Resume_str', 'Category'],
        sudah di-strip spasi, buang NA, filter teks pendek, dan drop duplikat.
    """

    csv_path = Path(path) if path is not None else DEFAULT_CSV_PATH
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found at: {csv_path}")
        """ 
        Kalau path tidak dikasih (None) → csv_path = DEFAULT_CSV_PATH

        Setelah csv_path ditentukan:
        - if not csv_path.exists(): cek apakah file benar-benar ada
         * Kalau ada → program lanjut ke langkah berikutnya
         * Kalau tidak ada → raise FileNotFoundError dan hentikan program

        """


    df = pd.read_csv(csv_path)
    _validate_columns(df, REQUIRED_COLUMNS)
    """
    - df = pd.read_csv(csv_path) → baca file CSV dan simpan ke DataFrame
    - _validate_columns(df, REQUIRED_COLUMNS) → cek apakah semua kolom wajib (Resume_str, Category) ada di df
    * Kalau lengkap → lanjut
    * Kalau ada yang hilang → raise ValueError
    """

    cols = list(REQUIRED_COLUMNS)
    # menambahkan kolom yang wajib ada ke dalam variabel cols, dijadikan sebagai list

    if keep_optional:
        for c in OPTIONAL_COLUMNS:
            if c in df.columns:
                cols.append(c)
    """
    - Kalau keep_optional=True → cek kolom opsional satu per satu dari OPTIONAL_COLUMNS
    - Kalau kolom tersebut memang ada di DataFrame (df.columns) → tambahkan ke daftar cols
    - Hasil akhirnya: cols berisi kolom wajib + kolom opsional (kalau ada dan diminta)
    """
    
    if usecols is not None:
        cols = list(usecols)
        """
    - Kalau user mengisi parameter usecols (bukan None),
      maka isi cols diganti sesuai yang diminta user.
    - Jadi aturan sebelumnya (kolom wajib + opsional) di-*override* penuh.
    - Contoh:
      * usecols=["Resume_str"] → hanya ambil kolom Resume_str
      * usecols=["Resume_str","Category","ID"] → ambil ketiganya
       """

    df = df[cols].copy()
    """
    Ambil hanya kolom yang ada di 'cols'
    .copy() bikin salinan baru, supaya:
    - aman dari SettingWithCopyWarning
    - perubahan berikutnya nggak memodifikasi view dari DataFrame asli
    """

    df["Resume_str"] = df["Resume_str"].astype(str).str.strip()
    df["Category"] = df["Category"].astype(str).str.strip()
    """
    Setiap nilai di kolom Resume_str dan Category:
    - dipaksa jadi string (.astype(str))
    - lalu di-strip (hapus spasi di awal & akhir teks, bukan hanya di akhir)
    """

    # Remove NAs
    df = df.dropna(subset=["Resume_str", "Category"])
    """
    - Hapus baris (row) yang punya nilai NaN di kolom Resume_str atau Category
    - subset=["Resume_str", "Category"] artinya hanya cek 2 kolom ini
    - Kalau salah satunya NaN → baris tersebut dibuang dari DataFrame
    """

    # Filter very short rows (likely parsing noise)
    if min_text_len is not None and min_text_len > 0:
        df = df[df["Resume_str"].str.len() >= int(min_text_len)]
    """
    - Filter baris berdasarkan panjang teks di kolom Resume_str
    - min_text_len = ambang batas minimal panjang teks (default 20 karakter)
    - Hanya baris dengan Resume_str >= min_text_len yang dipertahankan
    - Tujuannya: buang data resume yang terlalu pendek / noise
    """

    # Drop duplicates by text if requested
    if drop_duplicates:
        df = df.drop_duplicates(subset=["Resume_str"]).reset_index(drop=True)
    """
    - Hapus baris duplikat berdasarkan isi kolom Resume_str
    - subset=["Resume_str"] → hanya cek kolom Resume_str untuk menentukan duplikat
    - reset_index(drop=True) → setelah baris dibuang, index disusun ulang mulai dari 0
    - drop=True → supaya index lama tidak ikut disimpan sebagai kolom baru
    """

    # Normalize index
    df = df.reset_index(drop=True)
    return df


def stratified_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Stratified train/test split by Category.
    """
    from sklearn.model_selection import train_test_split
    X = df.drop(columns=["Category"])
    y = df["Category"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    train = X_train.copy()
    train["Category"] = y_train.values
    test = X_test.copy()
    test["Category"] = y_test.values
    return train, test
