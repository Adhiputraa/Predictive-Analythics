# Predictive Analythics : Sistem Prediksi Penyakit Stroke Menggunakan Metode K-Nearest Neighbors Classifierr
*Laporan Proyek Machine Learning - Aditya Yoga Adhiputra*

## **Domain Proyek**
Domain proyek yang dijelaskan pada proyek machine learning kali ini yaitu tentang Kesehatan dengan judul Predictive Analythics : Sistem Prediksi Penyakit Stroke Menggunakan Metode K-Nearest Neighbors Classifier

### **Latar Belakang**

---

![stroke](https://user-images.githubusercontent.com/55022521/189844888-432d55ab-166d-4006-b933-ef292a4319ad.jpg)


Stroke merupakan salah satu penyakit yang menyebabkan kematian terbanyak di Indonesia. Pada tahun 2013 pravelensi stroke di Indonesia mencapai 12,1 per 1000 penduduk [[1]](https://www.jpnn.com/news/30-persen-penderita-stroke-usia-muda) .Hal ini membuat stroke menjadi penyakit keempat tertinggi yang diderita oleh masyarakat Indonesia. Sedangkan di seluruh dunia, stroke menduduki peringkat kedua sebagai penyakit yang menyebabkan kematian, dengan presentasi 11,13%, dari total kematian yang ada[[2]](https://jtiik.ub.ac.id/index.php/jtiik/article/view/190/0).

 Stroke kebanyakan diderita oleh orang yang berusia 40-an tahun. Namun saat ini, tidak menutup kemungkinan diusia muda juga terserang stroke. Data dari Rumah Sakit Saiful Anwar (RSSA) Kota Malang, sepanjang tahun 2016 penderita penyakit stroke 30% nya masih diusia muda, yaitu antara usia 18 – 40 tahun[[3]](https://pusdatin.kemkes.go.id/resources/download/general/Hasil%20Riskesdas%202013.pdf). Oleh karena itu penulis ingin membuat sebuah sistem prediksi yang dapat digunakan oleh masyarakat dalam mencegah terjadinya pentakit stroke.

## **Business Understanding**
### **Problem Statements**
Berdasarkan latar belakang yang menjadi tercipta nya penelitian ini, adapun rincian masalah yang dapat diselesaikan pada proyek ini adalah sebagai berikut :
*   Apa variabel yang paling berpengaruh dalam terhdapa seseorang yang mengalami penyakit stroke?
*   Apa model machine learning yang paling baik untuk memprediksi penyakit stroke?
*   Bagaimana kinerja sistem sederhana untuk memprediksi penyakit Stroke dengan menerapkan Random Forest berdasarkan data yang tersedia?

### **Goals**
Adapun tujuan dilakukan nya penelitian ini yaitu:
*   Pengolahan data dari beberapa variabel yang telah ditentukan dalam memprediksi penyakit stroke.
*   Sistem ini dapat digunakan oleh masyarakat untuk memprediksi penyakit stroke berdasarkan gejala-gejala yang ada.
*   Membandingkan beberapa algoritma yang digunakan dalam memprediksi penyakit stroke guna mendapatkan performa yang terbaik.

### **Solution Statements**
Dalam rangka mencapai tujuan penelitian yang ada, penulis akan membangun model prediksi dengan 4 buah model algoritma berbeda. Seluruh model akan dibandingkan dan akan dipilih satu yang terbaik dengan performa serta accuracy terbaik yang digunakan :

*   **Random Forest**
Random Forest adalah kombinasi prediktor pohon sedemikian rupa sehingga setiap pohon bergantung pada nilai vektor acak yang diambil sampelnya secara independen dan dengan distribusi yang sama untuk semua pohon di hutan. Kesalahan generalisasi untuk hutan konvergen sebagai batas karena jumlah pohon di hutan menjadi besar[[4]](https://doi.org/10.1023/A:1010933404324).

*   **Naive Bayes**
Naïve bayes adalah sebuah alat/metode untuk melakukan klasifikasi yang berakar pada teori probabilitas dan statistic yang ditemukan oleh ilmuan asal inggris yaitu thomas bayes. Ciri khas dari naïve bayes adalah metode klasifier ini memiliki asumsi yang kuat terhadap independensi dari masing-masing kondisi / kejadian. Pada metode naïve bayes, setiap kelas keputusan akan menghitung probabilitas dengan syarat bahwa kelas keputusan tersebut benar dan juga metode ini mengasumsikan bahwa atribut objek adalah pelaku independen[[5]](https://scikit-learn.org/stable/modules/naive_bayes.html).

*   **K-Nearest Neighbor**
K-Nearest Neighbor (KNN) adalah suatu metode yang menggunakan algoritma supervised dimana hasil dari query instance yang baru diklasifikan berdasarkan mayoritas dari kategori pada KNN. Tujuan dari algoritma KNN adalah untuk mengklasifikasi objek baru berdasarkan atribut dan training samples. Dimana hasil dari sampel uji yang baru diklasifikasikan berdasarkan mayoritas dari kategori pada KNN[[6]](http://labdas.si.fti.unand.ac.id/2022/03/20/penjelasan-cara-kerja-algoritma-k-nearest-neighbor-knn/).

*   **Logistic Regression**
Logistic Regression merupakan sebuah algoritma klasifikasi untuk mencari hubungan antara fitur (input) diskrit/kontinu dengan probabilitas hasil output diskrit tertentu[[7]](https://vincentmichael089.medium.com/machine-learning-2-logistic-regression-96b3d4e7b603).

## **Data Understanding**
Dataset yang penulis gunakan dalam proyek ini, yaitu Dataset dengan judul Stroke Prediction Dataset yang diambil pada laman Kaggle [[5]](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset). Dataset terebut berisikan 5110 data dengan 12 kolom. Berikut merupakan informasi lebih detail dari masing masing kolom dataset:
*   `id`: unique identifier
*   `gender`: "Male", "Female" or "Other"
*   `age`: age of the patient
*   `hypertension`: 0 if the patient doesn't have hypertension, 1 if the patient has hypertension
*   `heart_disease`: 0 if the patient doesn't have any heart diseases, 1 if the patient has a heart disease
*   `ever_married`: "No" or "Yes"
*   `work_type`: "children", "Govt_jov", "Never_worked", "Private" or "Self-employed"
*   `Residence_type`: "Rural" or "Urban"
*   `avg_glucose_level`: average glucose level in blood
*   `bmi`: body mass index
*   `smoking_status`: "formerly smoked", "never smoked", "smokes" or "Unknown"*
*   `stroke`: 1 if the patient had a stroke or 0 if not

![image](https://user-images.githubusercontent.com/55022521/189842969-a4ab8c9e-35a1-4b66-97cf-7ffc5827943d.png)
 
 **Correlation Matrix**
 Masing masing variabel memiliki korelasi terhadap target label yaitu 'stroke', variabel yang paling berpengaruh pada penelitian ini yaitu variabel 'age' sekitar 25%
 
![corelational matrix](https://user-images.githubusercontent.com/55022521/189872685-aa01ed7b-78b6-4f1c-b8b5-e77be697dca7.png)


## **Data Preparation**

Teknik yang penulis gunakan dalam tahap Data Preparation adalah sebagai berikut:  

*  **Encoding Categorical Feature**

Pada tahap ini, penulis melakukan proses encoding terhadap fitur kategori dengan menggunakan teknik One-Hot-Encoding yang merupakan teknik untuk merepresentasikan variabel atau fitur kategorikan ke dalam vektor biner. Guna mewujudkan hal tersebut, penulis menggunakan teknik LabelEncoder pada 
library Scikitlearn[[8]](https://medium.com/analytics-vidhya/one-hot-encoding-categorical-variables-what-is-it-why-is-it-how-is-it-6fd9ed3a161).

*   **Train-Test-Split**

Pada tahap ini, penulis melakukan pembagian dataset menjadi data latih dan data uji menggunakan train_test_split dari library Scikitlearn. Pembagian dataset ini bertujuan agar nantinya dapat digunakan untuk melatih dan mengevaluasi kinerja model. Pada proyek ini, 90% dataset digunakan untuk melatih model, dan 10% sisanya digunakan untuk mengevaluasi kinerja model.

*   **Standardization**

Pada tahap ini, penulis melakukan standardisasi menggunakan StandarScaler yang terdapat pada library sckitlearn. Standardisasi ini sangat berguna dalam menyeratakan skala pada data terutama data numerical

*   **Balancing the Dataset**

Pada tahap ini penulis melakukan over sampling. Dilakukan nya oversampling dikarenakan terjadi ketidakseimbangan pada data. Penulis menggunakan Metode SMOTE. Metode Synthetic Minority Over-samplingTechnique (SMOTE) merupakan metode yang populer diterapkan dalam rangka menangani ketidak seimbangan kelas. Teknik ini mensintesis sampel baru dari kelas minoritas untuk menyeimbangkan dataset dengan cara membuat instance baru dari minority class dengan pembentukan convex kombanasi dari instances yang saling berdekatan[[9]](https://mti.binus.ac.id/2018/06/08/synthetic-minority-over-sampling-technique-smote-algorithm-for-handling-imbalanced-data/).

## **Modeling**

Pada tahap ini, penulis membangun model prediksi dengan menggunakan 4 algoritma berbeda, yaitu Random Forest, Naive Bayes, K-Nearest Neighbor, Logistic Regression. Penelitian ini berupa klasifikasi maka penulis menggunakan metriks accuracy untuk melihat performa dari model dari ke-4 algoritma yang digunakan.

Berikut merupakan hasil dari modeling berupa metrik akurasi dari 4 algoritma.
matriks berupa matriks acc dari data training dan matriks dari data testing:

**AccuracyTraining** 

![image](https://user-images.githubusercontent.com/55022521/189843711-736ab3c1-a881-47bd-a51f-82080dee26f8.png)

**Accuracy Testing**

![image](https://user-images.githubusercontent.com/55022521/189843882-b1051474-5de2-4a3c-b0e6-9f74c3bfd87c.png)

## **Evaluation**

Berdasarkan hasil matriks accuracy score pada model prediksi dari ke-4 model diatas didapati bahwa model algoritma yang paling baik yaitu *KNeighborsClassifier*. Lalu kami melakukan testing secara manual terhadap model. Penulis mengambil data secara acar lalu melakukan test dan masih bisa ter-predict dengan benar.

**Confusion Matrix of KNeighborsClassifier**

![cmatrix](https://user-images.githubusercontent.com/55022521/189844280-b798258d-9bad-4de4-9989-37e363dcd3e2.png)

**Evauation Score**

![image](https://user-images.githubusercontent.com/55022521/189871832-e79c2227-fc1b-40cb-bfb0-51358f155bdc.png)

## **Kesimpulan**

Berdasarkan hasil penelitian diatas didapati bahwa pada saat training model penulis mendapatkan akurasi model yang sangat baik dari ke-4 model tersebut. Akan tetapi saat model dilakukan testing hasil akurasi nya kurang baik. 

Penulis menyimpulkan hal tersebut terjadi karena dataset yang digunakan dalam penilitian ini kurang begitu baik. Adanya Missing value pada kolom 'bmi' dan juga imbalanced pada data terutama pada kolom 'Stroke'. Penulis berusaha memperbaiki keseimbangan data menggunakan metode over sampling pada penelitian ini. Namun hal tersebut hanya memperbaiki training accuracy pada model saja, sehingga ketika dilakukan testing model dilakukan hasil nya tidak optimal karena model yang terlalu overfit.

## **References**


[[1]](https://www.jpnn.com/news/30-persen-penderita-stroke-usia-muda) ARISETIJONO, 2016. 30 Persen Penderita Stroke Usia Muda

https://www.jpnn.com/news/30-persenpenderita-stroke-usia-muda.


[[2]](https://jtiik.ub.ac.id/index.php/jtiik/article/view/190/0) SYAFIQ, MUHAMMAD, ACHMAD JAFAR A. K., RIZKA HUSNUN Z., DAESWARA 
JAUHARI, WANDA ATHIRA L., IMAM CHOLISSODIN, LAILIL MULFLIKHAH. 2016. Aplikasi Mobile (LIDE) untuk Diagnosis Tingkat Risiko Penyakit Stroke Menggunakan PTVPSO-SVM. Jurnal Teknologi Informasi dan Ilmu Komputer,Vol. 3, No.2, hlm. 147-155.

https://jtiik.ub.ac.id/index.php/jtiik/article/view/190/0


[[3]](https://pusdatin.kemkes.go.id/resources/download/general/Hasil%20Riskesdas%202013.pdf) BADAN PENELITIAN DAN PENGEMBANGAN KESEHATAN RI 2013. Laporan Riset Kesehatan Dasar (RISKESDAS) 2013. Jakarta : Kementerian Kesehatan Republik Indonesia,

https://pusdatin.kemkes.go.id/resources/download/general/Hasil%20Riskesdas%202013.pdf

[[4]](https://doi.org/10.1023/A:1010933404324) Breiman L (2001). “Random Forests”. Machine Learning. 45 (1): 5–32

https://doi.org/10.1023/A:1010933404324


[[5]](https://scikit-learn.org/stable/modules/naive_bayes.html). “Naive Bayes”. scikit-learn Machine Learning in Python 

https://scikit-learn.org/stable/modules/naive_bayes.html


[[6]](http://labdas.si.fti.unand.ac.id/2022/03/20/penjelasan-cara-kerja-algoritma-k-nearest-neighbor-knn/). Jamari. Untung, 2022. PENJELASAN CARA KERJA ALGORITMA K-NEAREST NEIGHBOR (KNN)
http://labdas.si.fti.unand.ac.id/2022/03/20/penjelasan-cara-kerja-algoritma-k-nearest-neighbor-knn/


[[7]](https://vincentmichael089.medium.com/machine-learning-2-logistic-regression-96b3d4e7b603). V. Michael, 2019. Machine Learning: Mengenal Logistic Regression

https://vincentmichael089.medium.com/machine-learning-2-logistic-regression-96b3d4e7b603

[[8]](https://medium.com/analytics-vidhya/one-hot-encoding-categorical-variables-what-is-it-why-is-it-how-is-it-6fd9ed3a161). Luna., “One-Hot Encoding Categorical Variables — What is it? Why is it? How is it?,” Medium, 2021, 

https://medium.com/analytics-vidhya/one-hot-encoding-categorical-variables-what-is-it-why-is-it-how-is-it-6fd9ed3a161.

[[9]](https://mti.binus.ac.id/2018/06/08/synthetic-minority-over-sampling-technique-smote-algorithm-for-handling-imbalanced-data/). ARWAN, VIRMAN ARDINA, LUDKI REZA ARIANA, FERICO SAMUEL, DUDI RAMDANI, ADITYA,  EVANS ANDITA SUKMANA, 2018. Synthetic Minority Over-sampling Technique (SMOTE) Algorithm For Handling Imbalanced Data 

https://mti.binus.ac.id/2018/06/08/synthetic-minority-over-sampling-technique-smote-algorithm-for-handling-imbalanced-data/

