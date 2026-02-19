import gzip
import math
import sys

inFilePath = sys.argv[1]
outFilePath = sys.argv[2]

def averageNumberLists(x):
    average = []

    for i in range(len(x[0])):
        values = [float(y[i]) for y in x]
        average.append(sum(values) / float(len(values)))

    return [average]

with gzip.open(inFilePath) as inFile:
    with gzip.open(outFilePath, "w") as outFile:
        # Save header row with "Sample" as the first column name.
        header_row = inFile.readline().decode().rstrip("\n").split("\t")
        header_row[0] = "Sample"
        outFile.write(("\t".join(header_row) + "\n").encode())

        expressionDict = {}

        # Average replicate samples
        for line in inFile:
            row = line.decode().rstrip("\n").split("\t")
            sampleID = row[0][:12]

            # Initialize the dictionary if not data is present for that sample
            if sampleID not in expressionDict:
                expressionDict[sampleID] = []

            # Store the expression values in the dictionary
            expressionDict[sampleID].append([float(x) for x in row[1:]])

        for sampleID in expressionDict.keys():
            if len(expressionDict[sampleID]) > 1:
                expressionDict[sampleID] = averageNumberLists(expressionDict[sampleID])

            outFile.write(("\t".join([sampleID] + [str(math.log2(x + 1)) for x in expressionDict[sampleID][0]]) + "\n").encode())
