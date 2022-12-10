:auto LOAD CSV WITH HEADERS FROM "file:///nodes_to_graph_id.csv" AS csvLine1
CALL {
WITH csvLine1
MERGE (u:User
 { 
    node_id: coalesce(csvLine1.user_node_id, "No user_node_id")
 })
ON CREATE SET u.graph_id = csvLine1.graph_id, u.label = csvLine1.label
} IN TRANSACTIONS OF 10000 ROWS;

:auto LOAD CSV WITH HEADERS FROM "file:///edges.csv" AS csvLine2
CALL {
WITH csvLine2
MATCH (u1:User { node_id: csvLine2.from })
MATCH (u2:User { node_id: csvLine2.to })
MERGE (u1)-[:retweeted_post_of]->(u2)
} IN TRANSACTIONS OF 10000 ROWS;
