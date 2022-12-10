
// create graph projection
CALL gds.graph.project('retweet_network','User','retweeted_post_of');

// verify datasets, mean should be like in GNN-fake-news-detection on github

CALL gds.wcc.stats('retweet_network')
YIELD componentCount, componentDistribution
RETURN componentCount, 
       componentDistribution.min as min,
       componentDistribution.max as max,
       componentDistribution.mean as mean


// export betweenness

CALL apoc.export.csv.query("
CALL gds.betweenness.stream('retweet_network')
YIELD nodeId, score
MATCH (user: User) WHERE id(user) = nodeId
RETURN user.label, score as betweenness_score
ORDER BY nodeId ASC", 'betweenness_to_label.csv', {});

// export closenessWasserman and Faust

CALL apoc.export.csv.query("
CALL gds.beta.closeness.stream('retweet_network', {useWassermanFaust: true})
YIELD nodeId, score
MATCH (user: User) WHERE id(user) = nodeId
RETURN user.label, score as closeness_score
ORDER BY nodeId ASC", 'closeness_to_label.csv', {});


// export harmonic closeness

CALL apoc.export.csv.query("
CALL gds.alpha.closeness.harmonic.stream('retweet_network')
YIELD nodeId, centrality
MATCH (user: User) WHERE id(user) = nodeId
RETURN user.label, centrality as harmonic_closeness_centrality
ORDER BY nodeId ASC", 'harmonic_closeness_to_label.csv', {});

// export degree

CALL apoc.export.csv.query("MATCH (user:User)
WITH user, id(user) as nodeId, size([(user)-[:retweeted_post_of]->() | user]) as outDegree, size([(user)<-[:retweeted_post_of]-() | user]) as inDegree
WHERE id(user) = nodeId
RETURN user.label, outDegree, inDegree, inDegree + outDegree as degree
ORDER BY nodeId ASC", 'degree_to_label.csv', {});

// PageRank

CALL apoc.export.csv.query("CALL gds.pageRank.stream('retweet_network', {maxIterations: 20, dampingFactor: 0.85})
YIELD nodeId, score
MATCH (user: User) WHERE id(user) = nodeId
RETURN user.label, score as page_rank_score
ORDER BY nodeId ASC", 'page_rank_to_label.csv', {});


// ArticleRank
CALL apoc.export.csv.query("CALL gds.articleRank.stream('retweet_network', {maxIterations: 20, dampingFactor: 0.85})
YIELD nodeId, score
MATCH (user: User) WHERE id(user) = nodeId
RETURN user.label, score as article_rank_score
ORDER BY nodeId ASC", 'article_rank_to_label.csv', {});

// Eigenvector
CALL apoc.export.csv.query("CALL gds.eigenvector.stream('retweet_network', {maxIterations: 40})
YIELD nodeId, score
MATCH (user: User) WHERE id(user) = nodeId
RETURN user.label, score as eigenvector_score
ORDER BY nodeId ASC", 'eigenvector_to_label.csv', {});

// HITS
CALL apoc.export.csv.query("CALL gds.alpha.hits.stream('retweet_network', {hitsIterations: 50})
YIELD nodeId, values
MATCH (user: User) WHERE id(user) = nodeId
RETURN user.label, values.hub as hits_hub, values.auth as hits_auth
ORDER BY nodeId ASC", 'hits_to_label.csv', {});


