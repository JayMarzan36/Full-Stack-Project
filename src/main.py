import matplotlib.pyplot as plt

import numpy as np

from scipy.linalg import svd

from parseFile import loadFile, countWords


def showTopWords(U, S, words, k, int=3, topN=5) -> None:
    for i in range(k):
        print(f"Top words for Concept {i+1}:")

        concept_vector = U[:, i]

        top_indices = np.argsort(np.abs(concept_vector))[::-1][:10]

        for idx in top_indices:
            print(f"{words[idx]} -> {concept_vector[idx]:.4f}")
        print()


def getCosineSimilarity(v1: list, v2: list) -> float:
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0

    return dot_product / (norm_v1 * norm_v2)


def vis2d(matrix: list) -> None:
    graph = plt.matshow(matrix)

    plt.gcf().set_size_inches(10, 8)

    plt.subplots_adjust(bottom=0.5, right=0.8, wspace=0.3, hspace=0.3)

    plt.yticks(range(len(words)), words)
    plt.xticks(range(len(testFiles)), testFiles)
    plt.colorbar()
    plt.show()


def parseAllFiles(filePaths: list) -> dict:
    allWordCounts = {}
    for filePath in filePaths:
        content = loadFile(filePath)
        wordCount = countWords(content)
        allWordCounts[filePath] = wordCount
    return allWordCounts


def makeMatrix(allWordCounts: dict) -> list:
    allWords = sorted(set().union(*allWordCounts.values()))

    matrix = []
    for filePath, wordCount in allWordCounts.items():
        row = [wordCount.get(word, 0) for word in allWords]
        matrix.append(row)

    return matrix, list(allWords)


if __name__ == "__main__":
    testFiles = [
        "Full-Stack-Project/src/testDocuments/test1.txt",
        "Full-Stack-Project/src/testDocuments/test2.txt",
        "Full-Stack-Project/src/testDocuments/test3.txt",
        "Full-Stack-Project/src/testDocuments/test4.txt",
    ]

    allWordCounts = parseAllFiles(testFiles)

    matrix, words = makeMatrix(allWordCounts)

    matrix = np.array(matrix).T

    # vis2d(matrix)

    U, S, Vt = svd(matrix, full_matrices=False)

    k = 3

    Sk = np.diag(S[:k])
    Vk = Vt[:k, :]
    Dk = np.dot(Sk, Vk).T

    print("Dk shape:", Dk.shape)
    print("testFiles length:", len(testFiles))

    showTopWords(U, S, words, k=3)

    # print(getCosineSimilarity(Dk[0], Dk[1]))  # Cosine similarity between first two documents

    # print(Sk, end="\n")  # Term vector space
    # print(Vk, end="\n")  # Document vector space
    # print(Dk, end="\n")  # Singular values

    origin = np.zeros((3, 3))

    fig = plt.figure(figsize=(8, 6))

    ax = fig.add_subplot(111, projection="3d")

    DkNorm = Dk / np.linalg.norm(Dk, axis=1)[:, np.newaxis]

    scaleFactor = 0.1
    DkScaled = DkNorm * scaleFactor

    for idx, docVector in enumerate(DkScaled):
        ax.text(
            docVector[0],
            docVector[1],
            docVector[2],
            testFiles[idx],
            size=10,
            zorder=1,
            color="k",
        )

    origin = np.zeros((Dk.shape[0], 3))

    ax.quiver(
        origin[:, 0],
        origin[:, 1],
        origin[:, 2],
        DkScaled[:, 0],
        DkScaled[:, 1],
        DkScaled[:, 2],
        color="r",
    )

    ax.set_xlabel("Concept 1")

    ax.set_ylabel("Concept 2")

    ax.set_zlabel("Concept 3")

    ax.set_title("Documents in Semantic Space")

    plt.show()
