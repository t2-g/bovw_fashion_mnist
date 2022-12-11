import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from scipy.cluster.vq import kmeans, vq
from scipy.spatial.distance import cdist
import cv2
import joblib
import pylab as pl
plt.style.use('dark_background')

image_arr = []
image_classes = []

visual_words = []
voc = [[]]


def KMeansClustering(X, k):
  # init centroids
    n = X.shape[0]
    centers = X[np.random.choice(n, k, replace=False)]
    closest = np.zeros(n).astype(int)

    while True:
        old_closest = closest.copy()

        #  update cluster membership
        distances = cdist(X, centers)
        closest = np.argmin(distances, axis=1)

        #  update centroids
        for i in range(k):
            centers[i, :] = X[closest == i].mean(axis=0)

        # break if converged
        if all(closest == old_closest):
            break

    return closest, centers


def CreateDictionary():
    data_set = pd.read_csv('data/fashion-mnist_train.csv')
    global image_arr, image_classes
    image_label = data_set['label']
    for x in image_label:
        image_classes.append(x)
    d = data_set.drop("label", axis=1)
    # print(d.shape)
    # print(type(image_classes))

    for i in range(0, d.shape[0]):
        grid_data = np.asarray(d.iloc[i]).reshape(28, 28)
        image_arr.append(np.uint8(grid_data))
    # print(len(image_arr))

    key_points = []
    des = []
    sift = cv2.xfeatures2d.SIFT_create()
    image_classes2 = []
    # print(len(image_arr))
    for i in range(len(image_arr)):
        keypoints, descriptor = sift.detectAndCompute(image_arr[i], None)
        if descriptor is None:
            continue
        image_classes2.append(image_classes[i])
        key_points.append(keypoints)
        des.append(descriptor)

    # Stacking descriptors 
    descriptors = np.vstack(des)
    descriptors_float = descriptors.astype(float)

    # k-means clustering result from elbow method
    k = 20
    voc, _ = kmeans(descriptors_float, k, 1)

    # Calculate the histogram of features and represent them as vector
 
    im_features = np.zeros((len(image_classes2), k), "float32")
    for i in range(len(image_classes2)):
        words, _ = vq(des[i], voc)
        visual_words.append(words)
        for w in words:
            im_features[i][w] += 1

    from sklearn.preprocessing import StandardScaler
    stdSlr = StandardScaler().fit(im_features)
    im_features = stdSlr.transform(im_features)

    from sklearn.svm import LinearSVC
    svc_classifier = LinearSVC(max_iter=1000)  # Default of 100 is not converging
    svc_classifier.fit(im_features, np.array(image_classes2))

    training_names = ["T-shirt/top", "Trouser", "Pullover", "Dress",
                      "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

    # Saved the histograms and Joblib dumps Python object into one file
    joblib.dump((svc_classifier, training_names, stdSlr, k, voc), "bovw.pkl", compress=3)


def ComputeHistogram(visual_words, voc):
    frequency_vectors = []
    k = 20
    for img_visual_words in visual_words:
        # create a frequency vector for each image
        img_frequency_vector = np.zeros(k)
        for word in img_visual_words:
            img_frequency_vector[word] += 1
        frequency_vectors.append(img_frequency_vector)
    # stack together in numpy array
    N = 55716

    # performing tfidf method 
    frequency_vectors = np.stack(frequency_vectors)
    df = np.sum(frequency_vectors > 0, axis=0)
    idf = np.log(N / df)
    tfidf = frequency_vectors * idf
    # plotting all the histograms 
    plt.bar(list(range(k)), tfidf[0])
    plt.show()


def MatchHistogram():
    clf, classes_names, stdSlr, k, voc = joblib.load("bovw.pkl")

    data_set = pd.read_csv('data/fashion-mnist_test.csv')

    # similar to the above create dictionary function
    image_classes = []
    image_label = data_set['label']
    for x in image_label:
        image_classes.append(x)
    d = data_set.drop("label", axis=1)
    image_arr = []
    for i in range(0, d.shape[0]):
        grid_data = d.iloc[i].to_numpy().reshape(28, 28)
        image_arr.append(grid_data.astype(np.uint8))
    key = []
    des = []
    sift = cv2.xfeatures2d.SIFT_create(100)
    image_classes2 = []
    for i in range(len(image_arr)):
        keypoints, descriptor = sift.detectAndCompute(image_arr[i], None)
        if descriptor is None:
            continue
        image_classes2.append(image_classes[i])
        key.append(keypoints)
        des.append(descriptor)
    im_features = np.zeros((len(image_classes2), k), "float32")
    for i in range(len(image_classes2)):
        words, _ = vq(des[i], voc)
        for w in words:
            im_features[i][w] += 1

    im_features = stdSlr.transform(im_features)

    true_class = [classes_names[i] for i in image_classes2]
    # prediction of class names
    predictions = [classes_names[i] for i in clf.predict(im_features)]

    count = 0
    for i in range(len(true_class)):
        if (true_class[i] == predictions[i]):
            count += 1

    # for accuracy ,print the confusion matrix
    
    def showconfusionmatrix(cm):
        pl.matshow(cm)
        pl.title('Confusion matrix')
        pl.colorbar()
        pl.show()
    print("Total Hits Are:", count)
    accuracy = accuracy_score(true_class, predictions)
    print("accuracy = ", accuracy)
    cm = confusion_matrix(true_class, predictions)
    print(cm)
    df = pd.DataFrame([true_class,predictions], index=['True Classes', 'Predicted Classes']).T
    df.to_csv('answer.csv')
    showconfusionmatrix(cm)


CreateDictionary()
ComputeHistogram(visual_words, voc)
MatchHistogram()
