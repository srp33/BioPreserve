import gzip
import sys

inFilePath = sys.argv[1]
outFilePath = sys.argv[2]

matrix = []

with gzip.open(inFilePath) as inFile:
    for line in inFile:
        matrix.append(line.decode().rstrip().split("\t"))

matrix = zip(*matrix)

with gzip.open(outFilePath, "w") as outFile:
    for row in matrix:
        outFile.write(("\t".join(row) + "\n").encode())
