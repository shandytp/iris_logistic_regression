# Notes

### TODO List
- [x] Apakah Lost Function dan Cost Function itu berbeda? [done]
- [x] Belajar tentang Baseline Model Classfication [done]
- [x] Belajar tentang Likelihood, Gradient Descent, Loss Function, Regularization (L1, L2) [done]
- [ ] Bikin code 
	- [ ] Baseline model dan compare dengan Logistic Regression
	- [ ] Loss Function
	- [ ] Gradient Descent


# Baseline Model

[Referensi 1](https://crunchingthedata.com/baseline-models-for-machine-learning/), [Referensi 2](https://stephenallwright.com/baseline-machine-learning-models/)

- Apa sih baseline model itu? merupakan simple model yang dibuat dalam waktu singkat
- Bentuknya bisa dalam bentuk stochastic models atau rule-based model
- Benefit dari membuat Baseline model:
	- **Understanding if the benefit worth the cost**
		Let's say, kita udah baseline model dan ternyata hasilnya cuma berbeda 5% dari model XGBoost. Apakah itu worth it? Tanpa bantuan benchmark dari baseline model, kita pasti akan menganggap model XGBoost ini udah bagus, tetapi karena ada context dari baseline model semua jadi berubah.
	- Assigning performance improvements

- Kenapa sih kok kita perlu menggunakan Baseline model di awal - awal?
  - Understand your data faster
	  - Kita bisa sneak peak pattern yg dibuat oleh baseline model ini. Sehingga kita bisa tahu bagian data mana yang sulit di klasifikasikan
	  - Identify low signal data
  - Compare your actual model to a benchmark.
  Hasil dari Baseline model yang kita buat bisa kita compare dengan model Machine Learning yg lain kita buat
	  - Utilize relative performance metrics
	  - Estimate the potential impact on business metrics
  - Iterate with speed
	  - Iterate on your model more quickly, karena setelah kita tau hasil dari baseline model, kita punya good benchmark untuk memodifikasi model kita pada bagian mananya.
	  - Progress to other project faster, karena kalo punya baseline model kita bisa langsung tau next step nya mau kemana daripada kita langsung bikin complex model

## Baseline Regression Models

- Mean atau Median. (dari target data)
Jika kasus kita adalah regresi, kita bisa langsung menggunakan Mean atau Median sebagai baseline model kita
- Conditional mean atau business logic
Cara nya masih sama seperti yang diatas, tetapi perbedaannya adalah kita memilih variable that **you believe to be most strongly associated with the outcome and build out some business logic**.
- Linear Regression (TIL kalo ternyata LinReg itu baseline model)

## Baseline Classification Models

- Mode/Modus, kalo proporsi kelas nya sama kita ambil random aja
- Conditional mode atau business logic
- Logistic Regression

## Apa yang harus dilakukan ketika baseline model kita much better daripada Machine Learning Model?

Extremely unlikely, tetapi jika terjadi besar kemungkinan ada kesalahan pada dataset seperti:
- Data Leakage
- Fitur yang digunakan tidak tepat

Rule of thumb untuk compare dengan baseline model:
- Beda dibawah 5% -- stick with the baseline model
- 5% s.d 10% -- ok, but depends on the use case
- Over 10% -- Good, stick with the machine learning model

- Performa bagus atau nggk itu bingung ngukur nya, apalagi cuma punya satu model. Ex: punya akurasi 75% itu bagus atau nggk, maka kita pake baseline model. 
- Baseline model itu model bodoh bodohan atau model malas. Enggk works di industri
- Goals kita mengalahkan baseline model
- Baseline model berbeda dengan simple model
- Human benchmark itu bentuk lain dari baseline model, ex: image classification. Kalo human lebih perform daripada model cukup kacau juga

# Cost Function
[Referensi 1](https://www.youtube.com/watch?v=ar8mUO3d05w&ab_channel=CodingLane), [Referensi 2](https://www.youtube.com/watch?v=t6MVuMavbBY&ab_channel=CodingLane)

- Error representation in Machine Learning
- Shows ow our model is predicting compared to original given dataset

cost function loglikehood --> gradient descent

# Regularization
[Referensi 1](https://towardsdatascience.com/the-basics-logistic-regression-and-regularization-828b0d2d206c)

Regularizations are also known as "shrinkage" methods, karena goals nya adalah reduce or shrink coefficients. Nanti ini akan reduce variance dari model. Apa goals dari reduce variance of a model? **Avoid overfit**

Over fit adalah ketika model kita gagal untuk generalize. Gampangannya, hasilnya bagus di training tapi kacau di data test atau unseen data

Lowering the variance of the model can improve the model's accuracy on unseen data

Terdapat dua jenis:
- Ridge (L2/Squared Penalty) 
	- Cara kerja nya itu mengambil nilai coefficient (bisa satu saja atau lebih tergantung case nya) dari Linear Regression terus di kuadratkan, dan didalamnya itu juga ada parameter **lambda** ($\lambda$). Lambda determines how severe the penalty
	- Minimize sum of squared residuals (kalo regression)
	- The higher the λvalue, the more coefficients in the regression will be pushed towards zero. But never exactyly zero
	- Ridge itu membantu reduce Variance by shrinking parameters and making our predictions less sensitive to them. Karena training data nya low bias
	- Ketika kita ingin menggunakan Ridge, setidaknya kita harus mempunyai data yang cukup. Semisal kita punya dua fitur yg digunakan, atleast kita harus punya dua data point, dst. Karena kalo semisal cuma satu data point aja, nanti bingung buat nentuin garis best fit nya itu seperti apa

$$\sum^{n}_{i}(y_i-\hat{y})^2+\lambda \sum^{p}_{j} \beta^{2}_{j}$$

- Lasso (L1/Absolute Penalty)
	- Mirip dengan Ridge, beda nya adalah nilai coefficient nya kita **absolutkan**
	- Tapi Lasso is much better untuk meminimalisir useless variables dari equation

$$\sum^{n}_{i}(y_i-\hat{y})^2+\lambda \sum^{p}_{j} ||\beta_{j}||$$

**Berarti kalo di Logistic Regression, sum of squared residuals nya diganti jadi Log Loss**
	
## Perbedaan antara L1 dan L2
- Kalo Ridge can only shrink the slope **close to 0** while Lasso regression can **shrink the slope all the way to 0** dengan memainkan parameter $\lambda$
- Lasso is much better untuk meminimalisir useless variables and makes the final equation simpler and easier to interpret
- Tetapi Ridge tends to do a little better when most variables are useful

Kalo tanpa regularisasi itu kita hanya meminimalisirkan nilai error saja, atau kalo pada rumus diatas itu hanya menghitung ruas kiri nya saja.
**Meminimilasirkan nilai error dan theta nya kalo pake regularisasi** 

# Log Loss

Alasan kenapa tidak pake MSE karena nanti akan terjebak di local minimum dan non convex

Fungsi dari Gradient Descent adalah membuat nilai error nya mendekati 0 atau converge

Definisi converge itu tergantung, bisa exact 0 atau mendekati 0

buat gradient descent dari scratch 

## Gradient Descent

![f6d26a4ad26ddefd931fc0c22dbb3d0d.png](../_resources/f6d26a4ad26ddefd931fc0c22dbb3d0d.png)

- Pada bagian kiri itu Slope nya Negatif dan bagian kanan slope nya Positif
- Kenapa begitu? Karena dari gradient atau slope. Jika pada bagian kiri hasilnya akan negatif dan kebalikannya
$$m=\frac{\Delta{y}}{\Delta{x}}$$

Rumus Gradient Descent

![Screenshot from 2023-01-17 16-43-55.png](../_resources/Screenshot%20from%202023-01-17%2016-43-55.png)

Goals dari Gradient Descent adalah mencari parameter $\theta$ yang paling optimum


### Referensi
- [Implementing logistic regression from scratch in Python](https://developer.ibm.com/articles/implementing-logistic-regression-from-scratch-in-python/)
- [Logistic Regression From Scratch in Python](https://towardsdatascience.com/logistic-regression-from-scratch-in-python-ec66603592e2)
- [Understanding the Logistic Regression and likelihood](https://stats.stackexchange.com/questions/304988/understanding-the-logistic-regression-and-likelihood)
- [Regularization in Machine Learning](https://towardsdatascience.com/regularization-in-machine-learning-76441ddcf99a)
- [What is Balance and Imbalance dataset](https://medium.com/analytics-vidhya/what-is-balance-and-imbalance-dataset-89e8d7f46bc5)
- [An Introduction to Balanced and Imbalanced Datasets in Machine Learning](https://encord.com/blog/an-introduction-to-balanced-and-imbalanced-datasets-in-machine-learning)
- [Fighting Overfitting With L1 or L2 Regularization: Which One Is Better?](https://neptune.ai/blog/fighting-overfitting-with-l1-or-l2-regularization)
- 
