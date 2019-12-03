import sys 
from choice import predict
from ensemble_models import filenames
from numpy import argmax,zeros

print(len(filenames))
training_label = sys.argv[1]
output_file = sys.argv[2]
total_score = zeros([500,5])

if training_label == "ensemble": 
    for label in filenames:
        print(label)
        _,score = predict(label)
        total_score += score
else :
    _, total_score = predict(training_label)
print(total_score)
output = argmax(total_score,1)
with open(output_file,"w") as file:
    file.write("id,Ans\n")
    for i,j in enumerate(output):
        file.write("%d,%d\n"%(i+1,j))
file.close()
