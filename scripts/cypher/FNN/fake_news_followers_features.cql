//CALL gds.alpha.graph.sample.rwr(
//  'follower_network_sampled_to_0_3',
//  'follower_network',
//  { samplingRatio: 0.3 }
//);

// CALL gds.beta.listProgress()

CALL gds.graph.project('follower_network','User','follows');

CALL gds.alpha.graph.sample.rwr(
  'follower_network_sampled_with_stratification',
  'follower_network',
  { samplingRatio: 0.5 }
);

// export closeness Wasserman and Faust

CALL apoc.export.csv.query("
CALL gds.beta.closeness.stream('follower_network_sampled_with_stratification', {useWassermanFaust: true})
YIELD nodeId, score
MATCH (user: User) WHERE id(user) = nodeId
RETURN user.label, score as closeness_score
ORDER BY nodeId ASC", 'closeness_to_label.csv', {});

// export harmonic closeness

CALL apoc.export.csv.query("
CALL gds.alpha.closeness.harmonic.stream('follower_network_sampled_with_stratification')
YIELD nodeId, centrality
MATCH (user: User) WHERE id(user) = nodeId
RETURN user.label, centrality as harmonic_closeness_centrality
ORDER BY nodeId ASC", 'harmonic_closeness_to_label.csv', {});

// export betweenness

CALL apoc.export.csv.query("
CALL gds.betweenness.stream('follower_network_sampled_with_stratification')
YIELD nodeId, score
MATCH (user: User) WHERE id(user) = nodeId
RETURN user.label, score as betweenness_score
ORDER BY nodeId ASC", 'betweenness_to_label.csv', {});


// export degree

CALL apoc.export.csv.query("MATCH (user:User)
WITH user, id(user) as nodeId, size([(user)-[:follows]->() | user]) as outDegree, size([(user)<-[:follows]-() | user]) as inDegree
WHERE id(user) = nodeId
RETURN user.label, outDegree, inDegree, inDegree + outDegree as degree
ORDER BY nodeId ASC", 'degree_to_label.csv', {});

// PageRank

CALL apoc.export.csv.query("CALL gds.pageRank.stream('follower_network_sampled_with_stratification', {maxIterations: 20, dampingFactor: 0.85})
YIELD nodeId, score
MATCH (user: User) WHERE id(user) = nodeId
RETURN user.label, score as page_rank_score
ORDER BY nodeId ASC", 'page_rank_to_label.csv', {});

// ArticleRank
CALL apoc.export.csv.query("CALL gds.articleRank.stream('follower_network_sampled_with_stratification', {maxIterations: 20, dampingFactor: 0.85})
YIELD nodeId, score
MATCH (user: User) WHERE id(user) = nodeId
RETURN user.label, score as article_rank_score
ORDER BY nodeId ASC", 'article_rank_to_label.csv', {});


// Eigenvector
CALL apoc.export.csv.query("CALL gds.eigenvector.stream('follower_network_sampled_with_stratification', {maxIterations: 40})
YIELD nodeId, score
MATCH (user: User) WHERE id(user) = nodeId
RETURN user.label, score as eigenvector_score
ORDER BY nodeId ASC", 'eigenvector_to_label.csv', {});

// HITS
CALL apoc.export.csv.query("CALL gds.alpha.hits.stream('follower_network_sampled_with_stratification', {hitsIterations: 50})
YIELD nodeId, values
MATCH (user: User) WHERE id(user) = nodeId
RETURN user.label, values.hub as hits_hub, values.auth as hits_auth
ORDER BY nodeId ASC", 'hits_to_label.csv', {});

