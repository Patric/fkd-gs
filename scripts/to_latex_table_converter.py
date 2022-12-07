import re

table = """
eigenvector_score 	0.245 	1950.344 	0.000 	9.942 	0.002 	-0.037 	0.001
harmonic_closeness_centrality 	0.271 	196599.696 	0.000 	8538.933 	0.000 	-0.352 	0.353
hits_hub 	0.190 	26842.827 	0.000 	981.298 	0.000 	-0.138 	0.286
hits_auth 	0.248 	693.709 	0.000 	3.032 	0.082 	0.022 	0.036
betweenness_score 	0.002 	2.082 	0.149 	4205253238.833 	0.000 	0.001 	0.000
closeness_score 	0.271 	190243.421 	0.000 	7738.225 	0.000 	-0.347 	0.289
page_rank_score 	0.264 	3580.209 	0.000 	22986.895 	0.000 	-0.051 	0.000
article_rank_score 	0.260 	1308.763 	0.000 	708.467 	0.000 	-0.031 	0.000
outDegree 	0.114 	51.639 	0.000 	3397591.733 	0.000 	-0.006 	0.018
inDegree 	0.178 	11221.404 	0.000 	3397591.733 	0.000 	-0.089 	0.007
degree 	0.053 	204.437 	0.000 	6795183.465 	0.000 	-0.012 	0.010
"""

lines = table.split('\n')
output = ''
for line in lines:
  line = re.sub('_', ' ', line)
  output += re.sub(' *.\t', " & ", line)
  output += '\n  \\\\\hline \n'
print(output)