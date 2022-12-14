CALL gds.graph.project.cypher(
  'tweets_interactions_network',
  'MATCH (t1: Tweet) WHERE t1:Tweet RETURN id(t1) AS id',
  'MATCH (t1)-[r:retweetedStatus|inReplyToStatus]->(t2) RETURN id(t1) AS source, id(t2) AS target')

// 	nodeCount	relationshipCount	projectMillis
//	12 296 167	4 391 763	 27499


CALL apoc.export.csv.query("
CALL gds.betweenness.stream('tweets_interactions_network')
YIELD nodeId, score
MATCH (user: User) WHERE id(user) = nodeId
RETURN user.label, score as betweenness_score
ORDER BY nodeId ASC", 'betweenness_to_label.csv', {});

// export closeness correlation data Wasserman and Faust

CALL apoc.export.csv.query("
CALL gds.beta.closeness.stream('tweets_interactions_network', {useWassermanFaust: true})
YIELD nodeId, score
MATCH (user: User) WHERE id(user) = nodeId
RETURN user.label, score as closeness_score
ORDER BY nodeId ASC", 'closeness_to_label.csv', {});

// export harmonic closeness

CALL apoc.export.csv.query("
CALL gds.alpha.closeness.harmonic.stream('tweets_interactions_network')
YIELD nodeId, centrality
MATCH (user: User) WHERE id(user) = nodeId
RETURN user.label, centrality as harmonic_closeness_centrality
ORDER BY nodeId ASC", 'harmonic_closeness_to_label.csv', {});

// export degree correlation data

CALL apoc.export.csv.query("MATCH (user:User)
WITH user, id(user) as nodeId, size([(user)-[:retweeted_post_of]->() | user]) as outDegree, size([(user)<-[:retweeted_post_of]-() | user]) as inDegree
WHERE id(user) = nodeId
RETURN user.label, outDegree, inDegree, inDegree + outDegree as degree
ORDER BY nodeId ASC", 'degree_to_label.csv', {});

// PageRank

CALL apoc.export.csv.query("CALL gds.pageRank.stream('tweets_interactions_network', {maxIterations: 20, dampingFactor: 0.85})
YIELD nodeId, score
MATCH (user: User) WHERE id(user) = nodeId
RETURN user.label, score as page_rank_score
ORDER BY nodeId ASC", 'page_rank_to_label.csv', {});

// ArticleRank

CALL apoc.export.csv.query("CALL gds.articleRank.stream('tweets_interactions_network', {maxIterations: 20, dampingFactor: 0.85})
YIELD nodeId, score
MATCH (user: User) WHERE id(user) = nodeId
RETURN user.label, score as article_rank_score
ORDER BY nodeId ASC", 'article_rank_to_label.csv', {});


// Eigenvector

CALL apoc.export.csv.query("CALL gds.eigenvector.stream('tweets_interactions_network', {maxIterations: 40})
YIELD nodeId, score
MATCH (user: User) WHERE id(user) = nodeId
RETURN user.label, score as eigenvector_score
ORDER BY nodeId ASC", 'eigenvector_to_label.csv', {});

// HITS

CALL apoc.export.csv.query("CALL gds.alpha.hits.stream('tweets_interactions_network', {hitsIterations: 50})
YIELD nodeId, values
MATCH (user: User) WHERE id(user) = nodeId
RETURN user.label, values.hub as hits_hub, values.auth as hits_auth
ORDER BY nodeId ASC", 'hits_to_label.csv', {});
