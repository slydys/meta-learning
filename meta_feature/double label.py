import os
import pandas as pd
import numpy as np
from pymfe.mfe import MFE
from sklearn.preprocessing import LabelEncoder

data_folder = "../dataset/double/"
output_folder = "../meta_features/double"
os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(data_folder):
    if filename.endswith(".data") or filename.endswith(".csv") or filename.endswith(".txt"):
        file_path = os.path.join(data_folder, filename)

        data = pd.read_csv(file_path, sep="\t", header=None)

        for col in data.columns[1:]:
            if data[col].dtype == 'object':
                data[col] = LabelEncoder().fit_transform(data[col])

        X = data.iloc[:, :-2].values
        Y = data.iloc[:, -2:].values

        Y_combined = np.sum(Y, axis=1)  # 将最后两列的值相加，得到一个新的标签列

        mfe = MFE(groups=["statistical", "info-theory", "model-based"])

        mfe.fit(X, Y_combined)
        ft, ft_vals = mfe.extract()

        meta_features_df = pd.DataFrame([ft_vals], columns=ft)

        output_filename = f"{filename.split('.')[0]}_meta_features.csv"
        output_path = os.path.join(output_folder, output_filename)

        meta_features_df.to_csv(output_path, index=False)

        print(f"Meta-features for {filename} extracted and saved to {output_filename}.")
