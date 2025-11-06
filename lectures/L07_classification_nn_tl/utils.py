import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.manifold import TSNE


def load_data():
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    
    classes = np.asarray([3, 5])
    x_train, y_train = _get_dogs_and_cats(x_train, y_train, classes)
    x_test, y_test = _get_dogs_and_cats(x_test, y_test, classes)

    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")
    return (x_train, y_train), (x_test, y_test)


def _get_dogs_and_cats(data, target, classes):
    mask = np.in1d(target[:, 0], classes)
    y = np.where(target[mask] == 3, 0, 1)
    X = data[mask]
    return X, y


def plot_images(images, labels):
    assert np.allclose(labels, np.round(labels, 0)), f"Only zeros or ones are expected, but got: {list(labels.ravel())}"
    stringify = lambda x: "dog" if x == 1 else "cat"

    plt.figure(figsize=(10, 10))
    for idx, (image, label) in enumerate(zip(images, labels)):
        ax = plt.subplot(3, 3, idx + 1)
        plt.imshow(image.astype("uint8"))
        plt.title(stringify(label))
        plt.axis("off")
        

def plot_accuracy(history):
    # summarize history for accuracy
    plt.grid(linestyle="--")
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
    
    
def plot_loss(history):
    # summarize history for loss
    plt.grid(linestyle="--")
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
    
    
def plot_clusters(images, labels, features):
    # inspired by: https://e3oroush.github.io/tsne-visualization/
    num_images_to_plot = 1000
    sort_order = sorted(np.random.randint(0, len(labels), size=num_images_to_plot))
    images = [images[i] for i in sort_order]
    labels = [labels[i] for i in sort_order]
    features = [features[i] for i in sort_order]
    
    X = np.array(features)
    tsne = TSNE(n_components=2, learning_rate=200, perplexity=30, n_iter=2000, verbose=0).fit_transform(X)
    
    tx, ty = tsne[:,0], tsne[:,1]
    tx = (tx-np.min(tx)) / (np.max(tx) - np.min(tx))
    ty = (ty-np.min(ty)) / (np.max(ty) - np.min(ty))
    
    width = 4000
    height = 3000
    max_dim = 100

    full_image = Image.new('RGBA', (width, height))
    for img, x, y in zip(images, tx, ty):
        tile = Image.fromarray(img.astype("uint8"), 'RGB')
        rs = max(0.4, tile.width/max_dim, tile.height/max_dim)
        tile = tile.resize((int(tile.width/rs), int(tile.height/rs)), Image.ANTIALIAS)
        full_image.paste(tile, (int((width-max_dim)*x), int((height-max_dim)*y)), mask=tile.convert('RGBA'))

    plt.figure(figsize=(16, 12))
    plt.title("Images")
    plt.imshow(full_image)
    
    width = 4000
    height = 3000
    max_dim = 100

    full_image = Image.new('RGBA', (width, height))
    for lbl, x, y in zip(labels, tx, ty):
        img = np.zeros(shape=images[0].shape)
        if lbl == 0:
            img[:, :, 0] = 255
        else:
            img[:, :, -1] = 255 

        tile = Image.fromarray(img.astype("uint8"), 'RGB')    
        rs = max(0.4, tile.width/max_dim, tile.height/max_dim)
        tile = tile.resize((int(tile.width/rs), int(tile.height/rs)), Image.ANTIALIAS)
        full_image.paste(tile, (int((width-max_dim)*x), int((height-max_dim)*y)), mask=tile.convert('RGBA'))

    plt.figure(figsize = (16,12))
    plt.title("Labels")
    plt.imshow(full_image)