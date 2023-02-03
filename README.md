# Laporan Proyek Machine Learning - Gregorius Yuristama Nugraha

## Domain Proyek

Di zaman yang modern ini kebutuhan akan *gadget* terutama laptop merupakan hal yang sangat penting. Sehingga, hampir semua orang memerlukan laptop untuk memenuhi kebutuhan pekerjaannya. Dan jugakarena perkembangan teknologi di zaman modern yang sangat pesat ini, kebanyakan orang juga mungkin mengganti laptop mereka 1-2 tahun sekali karena ingin memiliki spesifikasi yang lebih mumpuni untuk menyelesaikan pekerjaan mereka. 

Di lain sisi, orang-orang yang tidak sanggup membeli laptop baru karena dirasa terlalu mahal akan lebih memilih membeli laptop bekas untuk pekerjaan mereka. Biasanya dalam membeli laptop bekas, yang paling dipertimbangkan adalah spesifikasi laptop tersebut apakah masih mumpuni untuk menyelesaikan pekerjaan mereka atau tidak.

Karena hal itulah banyak orang yang ingin menjual/membeli laptop lama untuk membantu mereka menyelesaikan pekerjaan/tugas mereka lebih cepat.
Dengan menggunakan algoritma *machine learning*, permasalahan ini dapat diselesaikan dengan memperkirakan harga laptop lama mereka berdasarkan spesifikasi laptop tersebut.



## Business Understanding

### Problem Statements

Bagaimana menentukan harga sebuah laptop bekas berdasarkan spesifikasi?

### Goals

Dengan menggunakan *predictive analysis* yang akurat berdasar spesifikasi dan harga laptop terjual di masa lalu.

### Solution Statements
Menggunakan 3 algoritma *predictive analysis*, yaitu : 

* ***K-Nearest Neighbor (KNN)*** : Algoritma *KNN* menggunakan ‘kesamaan fitur’ untuk memprediksi nilai dari setiap data yang baru. Cara kerja *KNN* adalah dengan membandingkan jarak satu sampel ke sampel pelatihan lain dengan memilih sejumlah k tetangga terdekat (dengan k adalah sebuah angka positif).
* ***Random Forest*** : Algoritma *random forest* adalah salah satu algoritma *supervised learning*.*Forest* yang dibangunnya adalah kumpulan *decision tree*.
* ***Adaboost*** : Algoritma yang menggunakan teknik *boosting* bekerja dengan membangun model dari data latih. Kemudian ia membuat model kedua yang bertugas memperbaiki kesalahan dari model pertama. Model ditambahkan sampai data latih terprediksi dengan baik atau telah mencapai jumlah maksimum model untuk ditambahkan. 

## Data Understanding

Untuk dapat membuat model *machine learning* yang maka diperlukan *dataset* berisi spesifikasi dan harga laptop terjual. Untuk *dataset* yang digunakan adalah : [*Laptop Price*](https://www.kaggle.com/datasets/muhammetvarl/laptop-price)


### Variabel-variabel pada Laptop Price dataset adalah sebagai berikut:
* Company : merupakan nama perusahaan pembuat laptop
* Product : berisi jenis dan seri laptop
* TypeName : merupakan tipe laptop (*Notebook*, *Ultrabook*, *Gaming*, dll.)
* Inches : ukuran layar laptop dalam inci
* ScreenResolution : resolusi layar laptop
* Cpu : *Central Processing Unit* (CPU) yang digunakan pada laptop
* Ram : ukuran *Random Access Memory* (RAM) yang digunakan pada laptop
* Memory : ukuran dan jenis *memory* pada laptop
* GPU : *Graphics Processing Unit* (GPU) yang digunakan pada laptop)
* OpSys : Sistem Operasi yang digunakan pada laptop
* Weight : berat laptop (kg)
* Price\_euros : harga laptop terjual (dalam euro)

Untuk dapat lebih memahami data dan bisa mendapatkan hasil yang lebih baik, dilakukan beberapa hal sebelum masuk ke tahapan *Data Preparation* :

1. **Mengatasi _missing value_** : mengecek apakah pada *dataset* terdapat data yang kosong atau tidak, jika ada maka dihapus.
2. **Mengecek tipe data** : jika tidak sesuai maka tipe data harus disesuaikan
3. **Memisahkan kolom** : memisahkan beberapa kolom yang memiliki 2 informasi atau lebih
4. **Mengatasi _outlier_** : mengecek apakah ada *outlier*, jika ada menghapus data yang berada di luar batas atas dan bawah
5. **Melakukan _Univariate Data Analysis_** : analisa data untuk variabel yang tidak saling berhubungan


![Screenshot 2023-01-17 at 5 40 18 PM](https://user-images.githubusercontent.com/102383943/212877867-0e431c08-bb00-484d-bf16-b4873f54f673.png)

Gambar 1. *Barplot* untuk kolom Company

Pada Gambar 1 dapat dilihat bahwa pada *dataset* laptop dengan merek HP merupakan merek yang paling banyak dijual


![Screenshot 2023-01-17 at 5 40 36 PM](https://user-images.githubusercontent.com/102383943/212877917-7a44d7e1-b9e9-4e52-9deb-9dcc25a9cc5d.png)

Gambar 2. *Barplot* untuk kolom OpSys

Pada Gambar 2 dapat dilihat bahwa sistem operasi cukup bermacam-macam dan yang paling banyak adalah laptop dengan sistem operasi *Windows* 10


![Screenshot 2023-01-17 at 5 40 50 PM](https://user-images.githubusercontent.com/102383943/212877964-21cae372-b797-4cc0-bf74-5c290487c5de.png)

Gambar 3. *Barplot* untuk kolom cpu_brand

Pada Gambar 3 dapat disimpulkan bahwa laptop dengan prosesor intel lebih banyak dijual dibanding AMD


![Screenshot 2023-01-17 at 5 41 37 PM](https://user-images.githubusercontent.com/102383943/212878167-01e53829-b397-4f22-83a1-e000fd6345ad.png)

Gambar 4. *Barplot* untuk kolom cpu_series

Dapat disimpulkan berdasarkan barplot diatas seri CPU yang paling banyak dijual adalah i5


![Screenshot 2023-01-17 at 5 41 54 PM](https://user-images.githubusercontent.com/102383943/212878240-50c6272f-60e1-43df-a797-4ae2944c26bb.png)

Gambar 5. *Barplot* untuk kolom memory\_1\_type

Dapat dilihat berdasarkan Gambar 5, kebanyakan laptop yang dijual adalah laptop yang menggunakan SSD

![Screenshot 2023-01-17 at 5 42 11 PM](https://user-images.githubusercontent.com/102383943/212878301-c8b1e3b3-3770-48f8-91ea-72324c301d04.png)

Gambar 6. Histogram untuk seluruh kolom numerik

Karena pada histogram memory\_count dan memory\_2\_size hanya memiliki 1 data saja maka kolom tersebut didrop

6. **Melakukan _Multivariate Data Analysis_** : Analisa data untuk variabel yang saling berkaitan


![Screenshot 2023-01-17 at 5 46 08 PM](https://user-images.githubusercontent.com/102383943/212879200-a296db56-05ac-4103-ba83-ea75844f4707.png)

Gambar 7. *Catplot* *price* terhadap Company

Ternyata berdasar Gambar 7, merek cukup berpengaruh terhadap harga. Dapat dilihat merek *chuwi* memiliki harga laptop paling murah sedangkan merk laptop *LG* memiliki harga tinggi

![Screenshot 2023-01-17 at 5 46 48 PM](https://user-images.githubusercontent.com/102383943/212879359-f7af52e3-c7e4-4e18-a2b0-f8c37f0facc4.png)

Gambar 8. *Catplot price* terhadap OpSys 

Dari Gambar 9 dapat disimpulkan bahwa sistem operasi laptop juga ternyata cukup berpengaruh terhadap harga, dapat dilihat laptop dengan sistem operasi *macOS* memiliki harga cukup tinggi dan yang paling tinggi adalah laptop dengan sistem operasi *Windows 7*

![Screenshot 2023-01-17 at 5 47 30 PM](https://user-images.githubusercontent.com/102383943/212879557-c0b9d47e-bc6d-4aa8-8bf8-4aaa4ef662d2.png)

Gambar 9. *Catplot price* terhadap TypeName

Berdasarkan Gambar 9, dapat dilihat bahwa tipe laptop juga memiliki pengaruh terhadap harga, tipe *notebook* memiliki harga paling murah sedangkan yang paling mahal adalah *workstation*

![Screenshot 2023-01-17 at 5 48 32 PM](https://user-images.githubusercontent.com/102383943/212879746-13b7073e-f93f-453a-9815-87b95d0c431b.png)

Gambar 10. *Catplot price* terhadap cpu_series

Dari Gambar 10 memperlihatkan bahwa seri CPU juga memiliki pengaruh terhadap harga, laptop dengan i7 cenderung memiliki harga lebih mahal dan seri M memiliki harga tertinggi

![Screenshot 2023-01-17 at 5 48 55 PM](https://user-images.githubusercontent.com/102383943/212879819-45b7b647-7a43-4c93-a969-13524fe7af71.png)

Gambar 11. *Catplot price* terhadap memory\_1\_type

Pada Gambar 11 dapat disimpulkan jenis *memory*/*storage* juga memiliki pengaruh terhadap harga, dimana laptop dengan penyimpanan SSD memiliki harga paling mahal disusul dengan jenis *Hybrid*


![Screenshot 2023-01-17 at 5 49 18 PM](https://user-images.githubusercontent.com/102383943/212879886-15fe83c5-9dc6-4cab-8b09-4bdce03998bb.png)

Gambar 12. *Catplot price* terhadap memory\_2\_type

Dari Gambar 12 memperlihatkan bahwa kolom memory\_2\_type hanya berisi 1 data saja, maka kolom tersebut di-*drop*.

![Screenshot 2023-01-17 at 5 44 47 PM](https://user-images.githubusercontent.com/102383943/212878894-c5c9048d-f728-4681-ade2-d30a316a8fd8.png)

Gambar 13. *Correlation Matrix*

Dari Gambar 13 dapat dilihat bahwa data yang memiliki korelasi paling besar terhadap harga laptop adalah besaran RAM. Lalu kolom *Inches*, weight\_float, dan memory\_1\_size memiliki korelasi yang kecil terhadap *price* maka kolom tersebut di-*drop*.

## Data Preparation
*Data preparation* yang dilakukan pada proyek ini adalah sebagai berikut :

1.  Melakukan *Encoding* Fitur Kategori : Karena model *machine learning* tidak dapat mengenali data yang bukan numerik maka perlu dilakukan *encoding* untuk fitur kategori dengan menggunakan *get\_dummies*.
2. *Train-Test-Split* : Membagi *dataset* menjadi data latih dan data validasi sebelum melatih model. 
3. Standarisasi data : Melakukan standarisasi untuk kolom yang bertipe numerik agar dapat menghasilkan performa yang lebih baik.

## Modeling
Proses *modelling* yang dilakukan pada proyek ini adalah dengan menggunakan 3 algoritma *machine learning*, yaitu *KNN*,* Random Forest*, dan *Adaboost*.

### *KNN*

Kelebihan : 

* Mudah diterapkan
* Mudah beradaptasi
* Memiliki sedikit *hyperparameter*

Kekurangan : 

* Tidak berfungsi dengan baik pada dataset berukuran besar
* Kurang cocok untuk dimensi tinggi
* Perlu standarisasi fitur
* Sensitif terhadap *noise data*, *missing values* dan *outliers*

Algoritma *KNN* pada proyek ini menggunakan parameter algorithm 'kd\_tree', dan dengan jumlah n\_neighbor 10. Dari parameter ini dihasilkan MSE pada *train* 56.727486 dan *test* 82.015829.

 Kemudian dilakukan juga *hyperparameter* *tuning* menggunakan *grid search* dengan parameter : 
 
* n\_neighbor (jumlah tetangga / k) : [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
* algorithm (algoritma yang digunakan untuk menghitung tetangga) : ["auto",  "ball\_tree", "kd\_tree", "brute"]

Setelah dilakukan *grid search*, didapatkan *algorithm*='ball\_tree' dan n\_neighbor=5 adalah parameter yang terbaik. Dan nilai MSE pada *test* menurun menjadi 78.19806140837794 terdapat peningkatan sebesar 4.88%

### *Random Forest*

Kelebihan : 

* Kuat terhadap data *outlier*.
* Bekerja dengan baik dengan data non-linear.
* Risiko *overfitting* lebih rendah.
* Berjalan secara efisien pada kumpulan data yang besar.
* Akurasi yang lebih baik daripada algoritma klasifikasi lainnya.

Kekurangan : 

* *Random Forest* cenderung bias saat berhadapan dengan variabel kategorikal.
* Waktu komputasi pada *dataset* berskala besar relatif lambat
* Tidak cocok untuk metode linier dengan banyak fitur *sparse*

Pada algoritma *Random Forest* proyek ini digunakan parameter n\_estimators=50, max\_depth=16, random\_state=55. Dari penggunaan parameter ini dihasilkan MSE pada *train* 11.861705 dan *test* 59.348965. 

Kemudian dilakukan *hyperparameter* *tuning* menggunakan *grid* *search* dengan parameter : 

* bootstrap (mengaktifkan sampel *bootstrap* pada *random forest* untuk membangun *tree*): [True, False]
* max\_depth (maksimal kedalaman dari *tree*) : [8, 16, 32, 64]
* max\_features (jumlah fitur yang dipertimbangkan untuk *split* terbaik) : ['auto', 'sqrt']
* min\_samples\_leaf (minimum sampel yang diperlukan untuk *leaf node*): [1, 2, 4]
* min\_samples\_split (sampel minimum yang diperlukan untuk memisah internal *node*): [2, 5, 10]
* n\_estimators (jumlah *tree* pada forest) : [60, 70, 80, 90, 100]

Setelah dilakukan *grid search* didapatkan parameter terbaik sebagai berikut : 

* bootstrap: False
* max\_depth: 32
* max\_features: 'sqrt'
* min\_samples_leaf: 1
* min\_samples_split: 5
* n\_estimators: 70

Dan setelah model terbaru dijalankan nilai dari MSE pada *test* menurun menjadi 56.9081351686941, terjadi peningkatan sebesar 5.92%

### Adaboost

Kelebihan : 

* Membutuhkan memori yang lebih sedikit
* Spesialisasi pada *weak* model

Kekurangan : 

*  Sensitif terhadap *noise* dan *outlier*

Pada algoritma *Adaboost* ini digunakan parameter learning\_rate=0.05. Dari parameter ini dihasilkan MSE pada *train* sebesar 79.021894 dan *test* 88.571281. 

Lalu dilakukan *Hyperparameter* *tuning* menggunakan *grid* *search* dengan parameter : 

* n\_estimators (jumlah maksimal *estimator* ketika *boosting* dihentikan): [500,1000,1500]
* learning\_rate (bobot yang diaplikasikan untuk setiap penggolong untuk setiap iterasi *boosting*) : [0.001,0.005, 0.01, 0.05, 0.1].

Setelah dilakukan *grid search* didapatkan parameter terbaik untuk learning\_rate adalah 0.01, n\_estimators sebesar 500. Nilai dari MSE menurun menjadi 87.49341186022788. Terjadi peningkatan sebesar 0.75%



## Evaluation

Metrik evaluasi yang digunakan pada proyek ini adalah *Mean Squared Error* (MSE) yang menghitung jumlah selisih kuadrat rata-rata nilai sebenarnya dengan nilai prediksi.

Persamaan MSE adalah sebagai berikut : 

$$ MSE = \frac{1}{N}\displaystyle\sum_{i=1}^{N}(y_{i} - y\textunderscore pred_{i})^2$$

Keterangan: 

*N = jumlah dataset*

*yi = nilai sebenarnya*

*y\_pred = nilai prediksi*

Semakin jauh/besar nilai prediksi dari nilai sebenarnya, maka nilai MSE juga akan semakin besar sedangkan jika nilai prediksi dekat/kecil nilai prediksi dari nilai sebenarnya maka nilai MSE semakin mengecil.

Sehingga algoritma yang baik adalah algoritma yang memiliki MSE kecil karena prediksi algoritma tersebut tidak melenceng jauh dari nilai sebenarnya. Dan dari hasil proyek ini dapat dilihat bahwa MSE terkecil adalah algoritma *random forest*, sehingga algoritma *random forest* adalah algoritma terbaik untuk proyek ini.

Tabel 1. Nilai MSE (*Mean Squared Error*) pada *train* dan *test* sebelum dilakukan *hyperparameter* *tuning*

|   |*train*|*test*|
|---|---|---|
|*KNN*|56.727486|82.015829|
|*Random Forest*|11.800786|60.274538|
|*Boosting*|78.511647|88.1475|

Tabel 2. Nilai MSE (*Mean Squared Error*) pada *train* dan *test* setelah dilakukan *hyperparameter* *tuning*

|   |*train*|*test*|
|---|---|---|
|***KNN***| 43.069575 | 78.198061 |
|***Random Forest***| 8.871217 | 56.908135 |
|***Boosting***| 77.785318 | 87.493412 |

Dari tabel 1 dan 2 dapat disimpulkan algoritma yang terbaik adalah *random forest* karena memiliki MSE pada *train* dan *test* terkecil. Dan juga ketiga algoritma tersebut setelah dilakukan *hyperparameter* *tuning* mengalami peningkatan dan peningkatan terbaik diraih oleh *random forest*.

## Conclusion

Dari hasil penelitian ini dapat dengan menggunakan *predictive analysis* dapat memperkirakan harga laptop berdasarkan spesifikasi dan harga di masa lalu. Lalu, algoritma *random forest* terbukti memiliki MSE yang cukup kecil sehingga cukup akurat untuk memprediksi harga laptop berdasarkan spesifikasi. Untuk penelitian selanjutnya dapat dilakukan improvisasi dengan menggunakan *dataset* yang lebih banyak dan juga yang sudah memiliki data penjualan laptop beserta spesifikasinya dengan mata uang rupiah.
