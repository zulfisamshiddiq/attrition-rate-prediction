import pandas as pd
import joblib
import sys
import warnings
warnings.filterwarnings('ignore')

print("======================================================")
print("  Sistem Prediksi Attrition HR - Jaya Jaya Maju")
print("======================================================\n")

# 1. Load Model
try:
    model = joblib.load('model/rf_model.joblib') 
except FileNotFoundError:
    print("❌ Error: File rf_model.joblib tidak ditemukan.")
    sys.exit()

# 2. Load Database Karyawan
try:
    database = pd.read_csv('data/database_karyawan.csv')
except FileNotFoundError:
    print("❌ Error: File database_karyawan.csv tidak ditemukan.")
    sys.exit()

# 3. Interaksi dengan HR (Meminta Input ID)
print("Petunjuk: Anda bisa memasukkan lebih dari satu ID dengan memisahkannya menggunakan koma.")
print("Contoh: 1, 2, 45\n")
input_ids = input("👉 Masukkan EmployeeId yang ingin diprediksi: ")

# Membersihkan spasi dan memisahkan ID jika ada banyak
list_id_str = input_ids.split(',')

print("\nMemproses prediksi...\n")
print("------------------------------------------------------")

# 4. Looping untuk mencari dan memprediksi setiap ID
for id_str in list_id_str:
    try:
        emp_id = int(id_str.strip())
    except ValueError:
        print(f"⚠️ Input '{id_str}' bukan angka ID yang valid. Dilewati.")
        continue
    
    # Mencari data karyawan berdasarkan EmployeeId
    data_karyawan = database[database['EmployeeId'] == emp_id]
    
    if data_karyawan.empty:
        print(f"❌ Karyawan dengan EmployeeId {emp_id} tidak ditemukan di database.")
        continue
    
    # Menghapus kolom target dan ID sebelum diprediksi
    X_pred = data_karyawan.drop(columns=['EmployeeId', 'Attrition'], errors='ignore')
    
    # Mengubah data teks menjadi angka (One-Hot Encoding) agar sesuai dengan model
    X_pred = pd.get_dummies(X_pred)
    
    # Menyelaraskan jumlah dan nama kolom persis seperti saat model dilatih
    if hasattr(model, 'feature_names_in_'):
        X_pred = X_pred.reindex(columns=model.feature_names_in_, fill_value=0)
    
    # Melakukan Prediksi
    prediksi = model.predict(X_pred)
    
    # Menampilkan Hasil
    if prediksi[0] == 1:
        print(f"[{emp_id}] ⚠️ PERINGATAN : Karyawan ini BERESIKO RESIGN!")
    else:
        print(f"[{emp_id}] ✅ AMAN       : Karyawan ini diprediksi BERTAHAN.")

print("------------------------------------------------------")
print("\nSelesai...")