import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="FaceShape")
class SpatialAttentionLayer(tf.keras.layers.Layer):
    """Custom layer untuk spatial attention (CBAM-style).
    
    Layer ini membantu model berfokus pada area spasial (koordinat piksel) yang 
    paling penting pada wajah (seperti rahang, tulang pipi, atau dahi) dan 
    mengabaikan latar belakang yang tidak relevan.
    """

    def __init__(self, kernel_size=7, **kwargs):
        # Menginisialisasi layer dan menyimpan ukuran kernel untuk operasi konvolusi
        super().__init__(**kwargs)
        self.kernel_size = kernel_size

    def build(self, input_shape):
        # Mendefinisikan sub-layer Conv2D untuk menghasilkan attention map tunggal (filters=1)
        # Menggunakan aktivasi Sigmoid agar nilai bobot perhatian berada di rentang 0 sampai 1
        self.conv = tf.keras.layers.Conv2D(
            filters=1,
            kernel_size=self.kernel_size,
            strides=1,
            padding="same",
            activation="sigmoid",
            kernel_initializer="glorot_uniform",
            name=self.name + "_conv"
        )

        # Membangun bobot internal Conv2D. Inputnya adalah hasil concat dari AvgPool dan MaxPool (2 channel)
        self.conv.build((input_shape[0], input_shape[1], input_shape[2], 2))
        super().build(input_shape)

    def call(self, inputs):
        # 1. Rata-rata nilai fitur di sepanjang dimensi channel (Channel-wise Average Pooling)
        avg_pool = tf.reduce_mean(
            inputs,
            axis=-1,
            keepdims=True
        )

        # 2. Ambil nilai fitur maksimum di sepanjang dimensi channel (Channel-wise Max Pooling)
        max_pool = tf.reduce_max(
            inputs,
            axis=-1,
            keepdims=True
        )

        # 3. Gabungkan kedua hasil pooling di sepanjang channel menjadi tensor berbentuk (H, W, 2)
        concat = tf.concat(
            [avg_pool, max_pool],
            axis=-1
        )

        # 4. Lewatkan hasil gabungan ke layer Conv2D untuk memetakan hubungan spasial menjadi 1 channel (Attention Map)
        attention_map = self.conv(concat)

        # 5. Kalikan matriks input asli dengan peta perhatian (element-wise multiplication)
        # Fitur yang penting akan diperkuat, sedangkan fitur tidak penting akan diperlemah
        return inputs * attention_map

    def get_config(self):
        # Menyimpan konfigurasi hyperparameter layer agar model bisa disimpan (.keras) dan dimuat kembali
        config = super().get_config()
        config.update({
            "kernel_size": self.kernel_size
        })
        return config


@tf.keras.utils.register_keras_serializable(package="FaceShape")
class TemperatureSoftmax(tf.keras.layers.Layer):
    """Layer softmax dengan parameter Temperature (T).
    
    Digunakan untuk mengontrol tingkat "kehalusan" distribusi probabilitas output.
    Nilai T > 1 membuat probabilitas antar-kelas menjadi lebih merata (soft), 
    sedangkan T < 1 membuatnya lebih tajam atau pasti (hard).
    """

    def __init__(self, temperature=1.0, **kwargs):
        # Menginisialisasi parameter temperature sebagai nilai float
        super().__init__(**kwargs)
        self.temperature = float(temperature)

    def call(self, logits):
        # Membagi nilai logits (output mentah model) dengan temperatur sebelum masuk ke rumus Softmax standar
        return tf.nn.softmax(logits / self.temperature, axis=-1)

    def get_config(self):
        # Menyimpan konfigurasi parameter untuk kebutuhan serialisasi objek Keras
        config = super().get_config()
        config.update({
            "temperature": self.temperature
        })
        return config


@tf.keras.utils.register_keras_serializable(package="FaceShape")
class FocalCrossEntropyLoss(tf.keras.losses.Loss):
    """Focal loss untuk penanganan ketidakseimbangan data (imbalance dataset) pada klasifikasi multi-class.
    
    Loss ini menurunkan bobot (down-weights) dari sampel yang mudah dipelajari (easy examples) 
    dan memaksa model untuk lebih fokus belajar dari sampel yang sulit diprediksi (hard examples).
    """

    def __init__(self, gamma=1.5, alpha=1.0, label_smoothing=0.0, **kwargs):
        # Inisialisasi dengan Reduction.NONE agar nilai loss dihitung per sampel terlebih dahulu
        super().__init__(
            reduction=tf.keras.losses.Reduction.NONE,
            **kwargs
        )

        self.gamma = gamma                # Parameter pemfokus (focussing parameter), makin tinggi makin fokus pada sampel sulit
        self.alpha = alpha                # Parameter penyeimbang bobot kelas (balancing factor)
        self.label_smoothing = label_smoothing  # Nilai penghalusan label untuk mencegah overfitting

    def call(self, y_true, y_pred):
        # Mengamankan nilai prediksi agar tidak menyentuh angka 0 atau 1 pas, menghindari error matematika log(0)
        y_pred = tf.clip_by_value(
            y_pred,
            tf.keras.backend.epsilon(),
            1.0 - tf.keras.backend.epsilon()
        )

        # Menerapkan Label Smoothing jika diaktifkan (> 0) agar target tidak terlalu bernilai 0 atau 1 ekstrem
        if self.label_smoothing > 0:
            num_classes = tf.cast(tf.shape(y_true)[-1], tf.float32)
            y_true = y_true * (1.0 - self.label_smoothing) + (
                self.label_smoothing / num_classes
            )

        # Menghitung nilai Categorical Cross Entropy standar untuk multi-class: minus dari jumlah (y_true * log(y_pred))
        cross_entropy = -tf.reduce_sum(
            y_true * tf.math.log(y_pred),
            axis=-1
        )

        # Mengambil probabilitas prediksi milik kelas yang bernilai benar (ground truth probability atau pt)
        true_class_prob = tf.reduce_sum(
            y_true * y_pred,
            axis=-1
        )

        # Menghitung Focal Weight dengan rumus: (1.0 - true_class_prob) pangkat gamma
        # Jika model percaya diri dan benar (pt mendekati 1), bobot ini mengecil (mendekati 0) -> Loss ditekan.
        # Jika model salah atau bingung (pt mendekati 0), bobot ini membesar (mendekati 1) -> Loss dipertahankan.
        focal_weight = tf.pow(
            1.0 - true_class_prob,
            self.gamma
        )

        # Mengembalikan total akhir bobot focal loss dikalikan nilai cross entropy asal
        return self.alpha * focal_weight * cross_entropy

    def get_config(self):
        # Menyimpan konfigurasi parameter loss untuk keperluan rekonstruksi saat model dimuat ulang
        config = super().get_config()
        config.update({
            "gamma": self.gamma,
            "alpha": self.alpha,
            "label_smoothing": self.label_smoothing,
        })
        return config