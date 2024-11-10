import os
import pandas as pd
from pymfe.mfe import MFE
from sklearn.preprocessing import LabelEncoder

# 数据集文件夹路径
data_folder = "../dataset/single/"

output_folder = "../meta_features/single"
os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(data_folder):
    if filename.endswith(".data"):
        file_path = os.path.join(data_folder, filename)

        df = pd.read_csv(file_path, delim_whitespace=True, header=None)

        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

        for col in X.columns:
            if X[col].dtype == 'object':
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col])

        X = X.values  # 特征矩阵
        y = y.values  # 标签向量

        mfe = MFE(groups=["general", "statistical", "info-theory", "model-based", "landmarking"])
        mfe.fit(X, y)
        ft, ft_vals = mfe.extract()

        meta_features_df = pd.DataFrame([ft_vals], columns=ft)

        output_filename = f"{filename.split('.')[0]}_meta_features.csv"
        output_path = os.path.join(output_folder, output_filename)

        meta_features_df.to_csv(output_path, index=False)

        print(f"Meta-features for {filename} extracted and saved to {output_filename}.")
