import tensorflow as tf
from huggingface_hub import hf_hub_download
import matplotlib.pyplot as plt

class InstanceNormalization(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon
    def build(self, input_shape):
        self.gamma = self.add_weight(shape=(input_shape[-1],), initializer="ones", trainable=True)
        self.beta  = self.add_weight(shape=(input_shape[-1],), initializer="zeros", trainable=True)
    def call(self, x):
        mean, var = tf.nn.moments(x, axes=[1,2], keepdims=True)
        return self.gamma * (x - mean)/tf.sqrt(var + self.epsilon) + self.beta

HF_MODEL_ID = "Coder-M/Bright"
MODEL_FILE = hf_hub_download(repo_id=HF_MODEL_ID, filename="bright_gan_generator_epoch45.h5")

generator = tf.keras.models.load_model(MODEL_FILE, compile=False,
                                       custom_objects={'InstanceNormalization': InstanceNormalization})
print("Generator loaded successfully!")

def brighten_from_path():
    img_path = input("Enter the path of the image: ")  # <-- input prompt
    return brighten_image(img_path)

def brighten_image(img_path):
    import tensorflow as tf
    import matplotlib.pyplot as plt

    img = tf.io.read_file(img_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [256, 256])
    img = (img * 2.0) - 1.0  # normalize to [-1,1]
    img_input = tf.expand_dims(img, 0)

    pred = generator(img_input, training=False)
    pred_img = (pred[0] + 1) / 2  # [-1,1] -> [0,1]

    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1)
    plt.title("Original")
    plt.imshow((img + 1) / 2)
    plt.axis("off")
    plt.subplot(1,2,2)
    plt.title("Brightened")
    plt.imshow(pred_img)
    plt.axis("off")
    plt.show()

    return pred_img

brightened_img = brighten_from_path()