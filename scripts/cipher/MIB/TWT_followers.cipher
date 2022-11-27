CREATE CONSTRAINT FOR (u:User) REQUIRE u.twitterId IS UNIQUE;

:auto LOAD CSV WITH HEADERS FROM "file:///TWT.csv/followers.csv" AS csvLine1
CALL {
WITH csvLine1
MERGE (user:User
 { 
    twitterId: 'TWT'+csvLine1.source_id,
    label: 0
 })
MERGE (follower:User
 { 
    twitterId: 'TWT'+csvLine1.target_id,
    label: 0
 })
CREATE (follower)-[:follows]->(user)
} IN TRANSACTIONS OF 10000 ROWS;


:auto LOAD CSV WITH HEADERS FROM "file:///TWT.csv/friends.csv" AS csvLine1
CALL {
WITH csvLine1
MERGE (user:User
 { 
    twitterId: 'TWT'+csvLine1.source_id,
    label: 0
 })
MERGE (follower:User
 { 
    twitterId: 'TWT'+csvLine1.target_id,
    label: 0
 })
MERGE (follower)-[:follows]-(user)
} IN TRANSACTIONS OF 10000 ROWS;