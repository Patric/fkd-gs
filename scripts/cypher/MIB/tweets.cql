CREATE CONSTRAINT FOR (u:User) REQUIRE u.twitterId IS UNIQUE;
CREATE CONSTRAINT FOR (t:Tweet) REQUIRE t.twitterId IS UNIQUE;


:auto LOAD CSV WITH HEADERS FROM "file:///tweets_processed.csv" AS csvLine
CALL {
    
WITH csvLine
MERGE (u:User { twitterId: csvLine.user_id })
CREATE (t: Tweet
 {  twitterId: csvLine.id,
    text: coalesce(csvLine.text, "unknown"),
    source: coalesce(csvLine.source, "unknown"),
    createdAt: coalesce(date(datetime({ epochmillis: 
apoc.date.parse(csvLine.timestamp, "ms", "yyyy-MM-dd HH:mm:ss") })), "unknown"),
    possibly_sensitive: coalesce(csvLine.possibly_sensitive, "unknown"),
    place: coalesce(csvLine.place, "unknown")
 }
)
CREATE (t)-[:postedBy]->(u)
} IN TRANSACTIONS OF 50000 ROWS



LOAD CSV WITH HEADERS FROM "file:///tweets_processed.csv" AS csvLine
WITH csvLine.in_reply_to_status_id as in_reply_to_status_id 
WHERE in_reply_to_status_id <> '0'
MERGE (t2: Tweet { twitterId: in_reply_to_status_id});

LOAD CSV WITH HEADERS FROM "file:///tweets_processed.csv" AS csvLine
WITH csvLine.retweeted_status_id as retweeted_status_id
WHERE retweeted_status_id <> '0'
MERGE (t3: Tweet { twitterId: retweeted_status_id});



:auto LOAD CSV WITH HEADERS FROM "file:///tweets_processed.csv" AS csvLine
CALL {
WITH csvLine
MATCH (t1: Tweet { twitterId: csvLine.id })
MATCH (t2: Tweet { twitterId: csvLine.in_reply_to_status_id})
MERGE (t1)-[:inReplyToStatus { screenName: coalesce(csvLine.in_reply_to_screen_name, 'unknown') }]->(t2)
} IN TRANSACTIONS OF 50000 ROWS

:auto LOAD CSV WITH HEADERS FROM "file:///tweets_processed.csv" AS csvLine
CALL {
WITH csvLine
MATCH (t1: Tweet { twitterId: csvLine.id })
MATCH (t2: Tweet { twitterId: csvLine.retweeted_status_id})
MERGE (t1)-[:retweetedStatus]->(t2)
} IN TRANSACTIONS OF 50000 ROWS

// Filter out noise (nodes with 1 or less degree)

:auto MATCH (t:Tweet)
WHERE size([(t)-[:retweetedStatus|inReplyToStatus]->() | t]) + size([(t)<-[:retweetedStatus|inReplyToStatus]-() | t]) <= 1
with t LIMIT 1000000
CALL { with t
DETACH DELETE t } IN TRANSACTIONS OF 10000 ROWS

// Create interaction relations

MATCH (u1: User)<-[:postedBy]-(t1: Tweet)-[:inReplyToStatus]->(t2: Tweet)-[:postedBy]->(u2: User)
MERGE (u1)-[i:interacted {inReplyToStatus: true}]->(u2)
RETURN u1, u2, i

MATCH (u1: User)<-[:postedBy]-(t1: Tweet)-[:retweetedStatus]->(t2: Tweet)-[:postedBy]->(u2: User)
MERGE (u1)-[i:interacted]->(u2)
ON MATCH
  SET
    i.retweetedStatus = true

:auto MATCH (t: Tweet)
WHERE size([(t)-[:retweetedStatus|inReplyToStatus]->() | t]) + size([(t)<-[:retweetedStatus|inReplyToStatus]-() | t]) <= 1
with t LIMIT 3000000
CALL { with t DETACH DELETE t } in TRANSACTIONS of 10000 ROWS
