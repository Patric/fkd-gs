import re

table = """
eigenvector_score 	0.244 	3560.221 	0.000 	19.645 	0.000 	-0.066 	0.163
harmonic_closeness_centrality 	0.263 	2828.637 	0.000 	3.670 	0.055 	-0.059 	0.168
hits_hub 	0.103 	3.883 	0.049 	0.077 	0.782 	-0.002 	0.069
hits_auth 	0.261 	1850.662 	0.000 	9.523 	0.002 	-0.048 	0.159
closeness_score 	0.260 	1961.225 	0.000 	2.354 	0.125 	-0.049 	0.174
page_rank_score 	0.266 	4.536 	0.033 	306.354 	0.000 	-0.002 	0.136
article_rank_score 	0.271 	61.463 	0.000 	164.771 	0.000 	-0.009 	0.131
"""

lines = table.split('\n')
output = ''
for line in lines:
  line = re.sub('_', ' ', line)
  output += re.sub(' *.\t', " & ", line)
  output += '\n  \\\\\hline \n'
print(output)