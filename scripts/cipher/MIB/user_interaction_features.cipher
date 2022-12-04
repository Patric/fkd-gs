CALL gds.graph.project('interaction_network','User','interacted');

CALL apoc.export.csv.query("
CALL gds.betweenness.stream('interaction_network')
YIELD nodeId, score
MATCH (user: User) WHERE id(user) = nodeId
RETURN user.label, score as betweenness_score
ORDER BY nodeId ASC", 'betweenness_to_label.csv', {});

// export closeness correlation data Wasserman and Faust

CALL apoc.export.csv.query("
CALL gds.beta.closeness.stream('interaction_network', {useWassermanFaust: true})
YIELD nodeId, score
MATCH (user: User) WHERE id(user) = nodeId
RETURN user.label, score as closeness_score
ORDER BY nodeId ASC", 'closeness_to_label.csv', {});

// export harmonic closeness

CALL apoc.export.csv.query("
CALL gds.alpha.closeness.harmonic.stream('interaction_network')
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

CALL apoc.export.csv.query("CALL gds.pageRank.stream('interaction_network', {maxIterations: 20, dampingFactor: 0.85})
YIELD nodeId, score
MATCH (user: User) WHERE id(user) = nodeId
RETURN user.label, score as page_rank_score
ORDER BY nodeId ASC", 'page_rank_to_label.csv', {});

// ArticleRank

CALL apoc.export.csv.query("CALL gds.articleRank.stream('interaction_network', {maxIterations: 20, dampingFactor: 0.85})
YIELD nodeId, score
MATCH (user: User) WHERE id(user) = nodeId
RETURN user.label, score as article_rank_score
ORDER BY nodeId ASC", 'article_rank_to_label.csv', {});

// Eigenvector

CALL apoc.export.csv.query("CALL gds.eigenvector.stream('interaction_network', {maxIterations: 40})
YIELD nodeId, score
MATCH (user: User) WHERE id(user) = nodeId
RETURN user.label, score as eigenvector_score
ORDER BY nodeId ASC", 'eigenvector_to_label.csv', {});

// HITS

CALL apoc.export.csv.query("CALL gds.alpha.hits.stream('interaction_network', {hitsIterations: 50})
YIELD nodeId, values
MATCH (user: User) WHERE id(user) = nodeId
RETURN user.label, values.hub as hits_hub, values.auth as hits_auth
ORDER BY nodeId ASC", 'hits_to_label.csv', {});
