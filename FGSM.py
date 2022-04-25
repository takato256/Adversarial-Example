import tensorflow as tf
import streamlit as st
from PIL import Image
from tensorflow.keras.applications.resnet_v2 import ResNet50V2, decode_predictions, preprocess_input

st.title('敵対的サンプル体験プログラム')
st.subheader('敵対的サンプルとは\n機械学習において誤分類を引き起こさせるために、ノイズを加えた画像のこと')

pretrained_model = ResNet50V2(include_top=True, weights='imagenet')
pretrained_model.trainable = True
decode_prediction = decode_predictions
loss_object = tf.keras.losses.CategoricalCrossentropy()

# 入力された画像の形式を整える
def preprocess(image):
  image = tf.cast(image, tf.float32)
  image = preprocess_input(image)
  image = image[None, ...]
  return image

def get_imagenet_label(probs):
  return decode_predictions(probs, top=1)[0][0]

def create_adversarial_pattern(input_image, input_label):
  with tf.GradientTape() as tape:
    tape.watch(input_image)
    prediction = pretrained_model(input_image)
    loss = loss_object(input_label, prediction)
    
  gradient = tape.gradient(loss, input_image)
  signed_grad = tf.sign(gradient)
  return signed_grad

def show_images(image, num):
  _, label, confidence = get_imagenet_label(pretrained_model.predict(image))
  image_ad = image[0]*0.5+0.5
  powerful_list = ['弱', 'やや弱', '中', '強']
  container_3 = st.container()
  container_3.image(image_ad.numpy(), caption = 'ノイズ:{}'.format(powerful_list[num]), use_column_width= 'auto')
  container_3.write("この画像は {} と認識されています。正解率は {:.2f}% です。".format(label, confidence*100))
    
uploaded_file = st.file_uploader("ここに画像ファイルをアップロードしてください(png, jpg, jpeg)", type=['png', 'jpg', 'jpeg'])
if uploaded_file != None:

    # オリジナル画像の下準備
    image = Image.open(uploaded_file)
    image_size = image.resize((224, 224))
    image_array = tf.keras.utils.img_to_array(image_size)
    image_tf = tf.convert_to_tensor(image_array)
    
    image = preprocess(image_tf)
    image_probs = pretrained_model.predict(image)

    _, image_class, class_confidence = get_imagenet_label(image_probs)
    
    # オリジナル画像を表示
    image_a = image[0]*0.5+0.5
    container_1 = st.container()
    container_1.image(image_a.numpy(), caption = 'オリジナル画像', use_column_width= 'auto',clamp=True)
    container_1.write("この画像は {} と認識されています。正解率は {:.2f}% です。".format(image_class, class_confidence*100))
    st.write('\n')

    labrador_retriever_index = 208
    label = tf.one_hot(labrador_retriever_index, image_probs.shape[-1])
    label = tf.reshape(label, (1, image_probs.shape[-1]))

    # ノイズ画像を表示
    perturbations = create_adversarial_pattern(image, label)
    image_b = perturbations[0]*0.5+0.5
    container_2 = st.container()
    container_2.image(image_b.numpy(), caption = 'ノイズ', use_column_width= 'auto')
    container_2.write("この画像はノイズです。これをオリジナル画像に加えます。")
    
    # 敵対的サンプルを表示
    epsilons = [0.01, 0.05, 0.1, 0.15]
    descriptions = [('Epsilon = {:0.3f}'.format(eps) if eps else 'Input') for eps in epsilons]
    for i, eps in enumerate(epsilons):
      adv_x = image + eps*perturbations
      adv_x = tf.clip_by_value(adv_x, -1, 1)
      show_images(adv_x, i)
      st.write('\n')