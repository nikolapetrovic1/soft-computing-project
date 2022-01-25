import glob
import os

# Current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

print(current_dir)

current_dir = 'images'

# Percentage of images to be used for the test set
percentage_test = 10

# Create and/or truncate train.txt and test.txt
# file_train = open('data/train.txt', 'w')
file_test = open('data/test1.txt', 'w')

# index_test = round(100 / percentage_test)
for pathAndFilename in glob.iglob(os.path.join(current_dir, "*.jpg")):
    title, ext = os.path.splitext(os.path.basename(pathAndFilename))
    file_test.write("data/test" + "/" + title + '.jpg' + "\n")
    # if counter == index_test:
    #     counter = 1
    #     file_test.write("data/obj" + "/" + title + '.jpg' + "\n")
    # else:
