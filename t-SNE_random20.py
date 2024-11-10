import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import arff
import pandas as pd
from sklearn.manifold import TSNE


def load_arff_data(file_path):
    data, meta = arff.loadarff(file_path)
    df = pd.DataFrame(data)
    features = df.iloc[:, :-1].values  # 特征
    labels = df.iloc[:, -1].astype(int).values  # 标签 最后一列
    return features, labels


def create_image(X, labels, output_path, img_width=256, img_height=256, random_state=None):
    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=random_state)
    X_embedded = tsne.fit_transform(X)

    unique_labels = np.unique(labels)
    label_colors = {label: (np.random.randint(0, 256),
                            np.random.randint(0, 256),
                            np.random.randint(0, 256)) for label in unique_labels}

    image = np.ones((img_height, img_width, 3), dtype=np.uint8) * 255  # 背景设为白色

    x_min, x_max = X_embedded[:, 0].min(), X_embedded[:, 0].max()
    y_min, y_max = X_embedded[:, 1].min(), X_embedded[:, 1].max()

    def normalize(value, min_val, max_val, scale):
        return int((value - min_val) / (max_val - min_val) * (scale - 1))

    drawn_positions = set()

    for i, (x, y) in enumerate(X_embedded):
        x_norm = normalize(x, x_min, x_max, img_width)
        y_norm = normalize(y, y_min, y_max, img_height)
        position = (x_norm, y_norm)

        if position in drawn_positions:
            continue

        drawn_positions.add(position)
        color = label_colors[labels[i]]
        image[y_norm, x_norm] = color

    plt.imsave(output_path, image)
    #print(image.shape)


def arff_files(input_folder, output_folder, num_images=20):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith('.arff'):
            file_path = os.path.join(input_folder, filename)
            dataset_name = os.path.splitext(filename)[0]

            dataset_folder = os.path.join(output_folder, dataset_name)
            if not os.path.exists(dataset_folder):
                os.makedirs(dataset_folder)

            X, labels = load_arff_data(file_path)

            for i in range(num_images):
                output_path = os.path.join(dataset_folder, f"{dataset_name}_image_{i + 1}.png")
                create_image(X, labels, output_path, random_state=None)
                print(f"Generated image {i + 1} for dataset: {dataset_name}")


input_folder = 'dataset/data_arff'
output_folder = 't-SNE_pictures'

arff_files(input_folder, output_folder)
