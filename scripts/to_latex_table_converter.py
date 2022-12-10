import re

table = """
Classifier 	Accuracy 	Log Loss
DecisionTreeClassifier 	33.867089 	22.852851
RandomForestClassifier 	86.135023 	0.685187
GradientBoostingClassifier 	89.341864 	0.376236
GaussianNB 	8.761579 	5.609309
KNeighborsClassifier 	89.955570 	3.486562
AdaBoostClassifier 	90.237217 	0.683156
"""

lines = table.split('\n')
output = ''
for line in lines:
  line = re.sub('_', ' ', line)
  output += re.sub(' *.\t', " & ", line)
  output += '\n  \\\\\hline \n'
print(output)