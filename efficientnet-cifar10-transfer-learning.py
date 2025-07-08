import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,GlobalAveragePooling2D,Dropout
from tensorflow.keras.datasets import cifar10
from tensorflow.keras import regularizers
from tensorflow.keras.utils import to_categorical

(x_train,y_train),(x_test,y_test)=cifar10.load_data()

x_train = tf.image.resize(x_train.astype('float32') / 255.0, [224, 224])
x_test = tf.image.resize(x_test.astype('float32') / 255.0, [224, 224])
y_train=to_categorical(y_train,10)
y_test=to_categorical(y_test,10)

base_model=EfficientNetB0(include_top=False, weights='imagenet', input_shape=(224,224,3))
base_model.trainable=False

model=Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128,activation='relu',kernel_regularizer=regularizers.l2(0.001)),
    Dropout(0.3),
    Dense(10,activation='softmax',kernel_regularizer=regularizers.l1(0.001))
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print("Training top layers only...")
model.fit(x_train, y_train, epochs=2, batch_size=30, validation_split=0.2)

# 7. Fine-tune: unfreeze last 20 layers
for layer in base_model.layers[-20:]:
    layer.trainable = True

# 8. Re-compile with lower LR for fine-tuning
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
print("🔧 Fine-tuning last 20 layers...")
model.fit(x_train, y_train, epochs=2, batch_size=30, validation_split=0.2)

# 9. Evaluate
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"\n✅ Final Test Accuracy: {test_acc:.4f}")