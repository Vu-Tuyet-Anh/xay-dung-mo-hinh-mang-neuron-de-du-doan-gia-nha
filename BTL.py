import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Đọc dữ liệu từ file CSV
df = pd.read_csv('Data_Set.csv', names=['serial', 'year_sold', 'age', 'distance', 'stores', 'latitude', 'longitude', 'price'])

# Kiểm tra giá trị bị thiếu và xử lý
df.fillna(df.mean(), inplace=True)
print("Số lượng giá trị bị thiếu sau khi xử lý:\n", df.isna().sum())

# Loại bỏ cột không cần thiết
df = df.iloc[:, 1:]

# Phân tích dữ liệu
sns.histplot(df['price'], kde=True)
plt.show()

plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.show()

# Chuẩn hóa dữ liệu
df_norm = (df - df.mean()) / df.std()

# Lưu giá trị trung bình và độ lệch chuẩn để chuyển đổi ngược
y_mean = df['price'].mean()
y_std = df['price'].std()

def convert_label_value(pred):
    return int(pred * y_std + y_mean)

# Chia tập dữ liệu
X = df_norm.iloc[:, :-1]
Y = df_norm.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X.values, Y.values, test_size=0.05, shuffle=True, random_state=0)

# Tạo mô hình với Dropout để tránh overfitting
def get_model():
    model = Sequential([
        Dense(64, input_shape=(X_train.shape[1],), activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(
        loss='mse',
        optimizer='adam'
    )
    return model

model = get_model()
model.summary()

# Thêm ModelCheckpoint để lưu mô hình tốt nhất
checkpoint = ModelCheckpoint("best_model.h5", save_best_only=True, monitor='val_loss', mode='min')
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Huấn luyện mô hình
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    batch_size=16,
    callbacks=[early_stopping, checkpoint],
    verbose=1
)

# Ghi nhận kết quả huấn luyện
history_df = pd.DataFrame(history.history)
history_df.to_csv("training_results.csv", index=False)

# Biểu đồ Loss
plt.plot(history.history['loss'], label='Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Quá trình huấn luyện mô hình")
plt.show()

# Dự đoán
y_pred = model.predict(X_test)
price_pred = [convert_label_value(y) for y in y_pred]
price_y_test = [convert_label_value(y) for y in y_test]

# Biểu đồ so sánh giá thực tế và dự đoán
plt.figure(figsize=(8, 6))
plt.scatter(price_y_test, price_pred, alpha=0.5, color='blue')
plt.plot([min(price_y_test), max(price_y_test)], [min(price_y_test), max(price_y_test)], color='red', linestyle='--')
plt.xlabel("Giá thực tế")
plt.ylabel("Giá dự đoán")
plt.title("So sánh giữa giá thực tế và giá dự đoán")
plt.show()
