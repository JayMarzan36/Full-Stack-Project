import os


def loadFile(filePath: str) -> str:
    fullPath = os.path.join(os.getcwd(), filePath)

    with open(fullPath, "r") as file:

        content = file.read()

    return content


def countWords(content: str) -> dict:
    words = content.split()

    wordCount = {}

    for word in words:

        word = word.lower()
        
        word = word.strip("-=_+<>.,!?()[]{};:\"'")

        if word in wordCount:

            wordCount[word] += 1

        else:

            wordCount[word] = 1

    return wordCount


if __name__ == "__main__":
    filePath = "testDocuments/test1.txt"

    if os.path.exists(filePath):

        content = loadFile(filePath)

        wordCount = countWords(content)

        print(wordCount)

    else:

        print(f"File {filePath} does not exist.")
