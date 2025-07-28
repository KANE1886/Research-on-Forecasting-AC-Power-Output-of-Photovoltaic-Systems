import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# === 1. 加载数据 ===
df = pd.read_csv("Generation_data.csv")  # 替换成你的本地路径

features = ['Amb_Temp', 'WIND_Speed', 'IRR (W/m2)', 'DC Current in Amps',
            'AC Ir in Amps', 'AC Iy in Amps', 'AC Ib in Amps', 'MODULE_TEMP']
target = 'AC Power in Watts'

X = df[features].values
y = df[target].values

# === 2. 标准化与划分 ===
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

# 8:1:1 划分数据集
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# 添加时间步
X_train = X_train.reshape((-1, 1, X_train.shape[1]))
X_val = X_val.reshape((-1, 1, X_val.shape[1]))
X_test = X_test.reshape((-1, 1, X_test.shape[1]))

# === 3. 模型构建 ===
def build_model(input_shape):
    inputs = layers.Input(shape=input_shape)
    x = layers.MultiHeadAttention(num_heads=2, key_dim=4)(inputs, inputs)
    x = layers.LayerNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)  # 替代 RNN，压缩时间维度
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(1)(x)
    model = models.Model(inputs, outputs)
    optimizer = tf.keras.optimizers.Nadam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mae')
    return model

model = build_model((1, X_train.shape[2]))

# === 自定义回调，只保存验证集预测 ===
class ValPredictionLogger(callbacks.Callback):
    def __init__(self, X_val):
        super().__init__()
        self.X_val = X_val
        self.val_preds = []

    def on_epoch_end(self, epoch, logs=None):
        val_pred = self.model.predict(self.X_val, verbose=2).flatten()
        self.val_preds.append(val_pred)

val_logger = ValPredictionLogger(X_val)

# === 4. 训练 ===
model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=150,
    batch_size=64,
    verbose=2,
    callbacks=[val_logger]
)

# === 5. 计算验证集各epoch指标并保存 ===
def calc_metrics(y_true, y_pred):
    return {
        'MAE': mean_absolute_error(y_true, y_pred),
        'MAPE': mean_absolute_percentage_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'R2': r2_score(y_true, y_pred)
    }

def inverse_transform(scaler, data):
    return scaler.inverse_transform(data.reshape(-1, 1)).flatten()

history_metrics = {
    'epoch': [],
    'val_MAE': [],
    'val_MAPE': [],
    'val_RMSE': [],
    'val_R2': [],
}

y_val_true = inverse_transform(scaler_y, y_val)

for i, val_pred_scaled in enumerate(val_logger.val_preds, 1):
    y_val_pred = inverse_transform(scaler_y, val_pred_scaled)
    val_m = calc_metrics(y_val_true, y_val_pred)

    history_metrics['epoch'].append(i)
    history_metrics['val_MAE'].append(val_m['MAE'])
    history_metrics['val_MAPE'].append(val_m['MAPE'])
    history_metrics['val_RMSE'].append(val_m['RMSE'])
    history_metrics['val_R2'].append(val_m['R2'])

# === 6. 保存指标到 Excel，合并为 Transformer 列 ===
def save_metric_with_merge(file_name, metric_key):
    new_col_name = f"Transformer_{metric_key.split('_')[1]}"
    new_df = pd.DataFrame({
        'epoch': history_metrics['epoch'],
        new_col_name: history_metrics[metric_key]
    })

    if os.path.exists(file_name):
        old_df = pd.read_excel(file_name)
        merged_df = pd.merge(old_df, new_df, on='epoch', how='outer')
    else:
        merged_df = new_df

    merged_df.to_excel(file_name, index=False)

# 保存四个指标
save_metric_with_merge("val_MAE.xlsx", "val_MAE")
save_metric_with_merge("val_MAPE.xlsx", "val_MAPE")
save_metric_with_merge("val_RMSE.xlsx", "val_RMSE")
save_metric_with_merge("val_R2.xlsx", "val_R2")