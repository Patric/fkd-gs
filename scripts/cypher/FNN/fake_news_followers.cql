CREATE CONSTRAINT FOR (u:User) REQUIRE u.twitterId IS UNIQUE;


:auto LOAD CSV WITH HEADERS FROM "file:///users_following.csv" AS csvLine1
CALL {
WITH csvLine1
MERGE (user:User
 { 
    twitterId: csvLine1.user_id
 })
 ON CREATE SET user.label = csvLine1.label
MERGE (following:User
 { 
    twitterId: csvLine1.following
 })
ON CREATE SET following.label = csvLine1.label
CREATE (user)-[:follows]->(following)
} IN TRANSACTIONS OF 10000 ROWS;

:auto LOAD CSV WITH HEADERS FROM "file:///users_followers.csv" AS csvLine1
CALL {
WITH csvLine1
MERGE (user:User
 { 
    twitterId: csvLine1.user_id
 })
 ON CREATE SET user.label = csvLine1.label
MERGE (follower:User
 { 
    twitterId: csvLine1.followers
 })
  ON CREATE SET follower.label = csvLine1.label
CREATE (follower)-[:follows]->(user)
} IN TRANSACTIONS OF 10000 ROWS;

:auto MATCH (user:User)
WHERE size([(user)-[:follows]->() | user]) + size([(user)<-[:follows]-() | user]) <= 1
with user LIMIT 1000000
CALL { with user
DETACH DELETE user } IN TRANSACTIONS OF 10000 ROWS

:auto MATCH (user:User)
WHERE size([(user)-[:follows]->() | user]) + size([(user)<-[:follows]-() | user]) <= 1
CALL { with user
DETACH DELETE user } IN TRANSACTIONS OF 10000 ROWS

MATCH (user:User)
WHERE size([(user)-[:follows]->() | user]) + size([(user)<-[:follows]-() | user]) < 1
DETACH DELETE user

// After filtering out noise (node <= 1 degree)
// 2 713 356

// 6 240 964


// componentCount	min	maximage.png	mean
// 38	1	2712927	71403.92105263157

// fromGraphName	startNodeCount	graphName	nodeCount	relationshipCount	projectMillis
// "follower_network"	2003	"follower_network_sampled_with_stratification"	1356663	9094012	366192



// nodeProjection	relationshipProjection	graphName	nodeCount	relationshipCount	projectMillis
// {
//   "User": {
//     "label": "User",
//     "properties": {

//     }
//   }
// }
// {
//   "follows": {
//     "orientation": "NATURAL",
//     "aggregation": "DEFAULT",
//     "type": "follows",
//     "properties": {

//     }
//   }
// }
// "follower_network"	2713356	12660581	12532

