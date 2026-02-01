import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv
import os
import json
import sys

# Load environment variables from .env
load_dotenv()

# Fetch connection variables
USER = os.getenv("user")
PASSWORD = os.getenv("password")
HOST = os.getenv("host")
PORT = os.getenv("port")
DBNAME = os.getenv("dbname")

def get_connection():
    """Create and return a database connection"""
    try:
        connection = psycopg2.connect(
            user=USER,
            password=PASSWORD,
            host=HOST,
            port=PORT,
            dbname=DBNAME
        )
        return connection
    except Exception as e:
        raise Exception(f"Failed to connect to database: {e}")

def upload_nodes(nodes, connection):
    """Bulk insert nodes into the database"""
    cursor = connection.cursor()
    
    # Prepare data for insertion
    node_data = [
        (
            node.get("node_id"),
            node.get("utility_qubits", 0),
            node.get("bonus_bell_pairs", 0),
            node.get("capacity", 0),
            node.get("owned", 0)
        )
        for node in nodes
    ]
    
    # Insert nodes using execute_values for efficiency
    insert_query = """
        INSERT INTO nodes (node_id, utility_qubits, bonus_bell_pairs, capacity, owned)
        VALUES %s
        ON CONFLICT (node_id) DO UPDATE SET
            utility_qubits = EXCLUDED.utility_qubits,
            bonus_bell_pairs = EXCLUDED.bonus_bell_pairs,
            capacity = EXCLUDED.capacity,
            owned = EXCLUDED.owned
        RETURNING id, node_id;
    """
    
    execute_values(cursor, insert_query, node_data)
    results = cursor.fetchall()
    connection.commit()
    
    print(f"✓ Uploaded {len(results)} nodes")
    cursor.close()
    return results

def get_node_id_map(connection):
    """Fetch all nodes and create a mapping from node_id to database id"""
    cursor = connection.cursor()
    cursor.execute('SELECT id, node_id FROM nodes')
    nodes = cursor.fetchall()
    cursor.close()
    
    # Create mapping: node_id -> id
    return {node_id: id for id, node_id in nodes}

def upload_edges(edges, node_id_map, connection):
    """Bulk insert edges into the database"""
    cursor = connection.cursor()
    edge_data = []
    
    skipped = 0
    
    for edge in edges:
        # Support multiple edge formats
        from_n = None
        to_n = None
        meta = {}
        
        if isinstance(edge, list) and len(edge) >= 2:
            # Format: [from_node_id, to_node_id] or [from, to, metadata]
            from_n = edge[0]
            to_n = edge[1]
            meta = edge[2] if len(edge) > 2 and isinstance(edge[2], dict) else {}
        elif isinstance(edge, dict):
            # Check if edge_id exists as an array [from, to]
            if "edge_id" in edge and isinstance(edge["edge_id"], list) and len(edge["edge_id"]) >= 2:
                from_n = edge["edge_id"][0]
                to_n = edge["edge_id"][1]
                meta = edge
            else:
                # Try other common field names
                from_n = (edge.get("from") or edge.get("from_node") or edge.get("from_node_id") or 
                         edge.get("source") or edge.get("start"))
                to_n = (edge.get("to") or edge.get("to_node") or edge.get("to_node_id") or 
                       edge.get("target") or edge.get("end"))
                meta = edge
        
        if from_n is None or to_n is None:
            skipped += 1
            if skipped <= 3:
                print(f"⚠ Skipping edge - could not find from/to nodes: {edge}")
            continue

        if from_n not in node_id_map or to_n not in node_id_map:
            skipped += 1
            if skipped <= 3:
                print(f"⚠ Skipping edge with unknown node(s): '{from_n}' -> '{to_n}'")
                print(f"   Available node_ids sample: {list(node_id_map.keys())[:5]}")
            continue
        
        from_id = node_id_map[from_n]
        to_id = node_id_map[to_n]
        
        if from_id == to_id:
            skipped += 1
            if skipped <= 3:
                print(f"⚠ Skipping self-loop edge: {from_n} -> {to_n}")
            continue

        edge_data.append((
            from_id,
            to_id,
            meta.get("base_threshold", 0.0),
            meta.get("difficulty_rating", 0),
            meta.get("successful_attempts", 0),
            meta.get("x_weight", 0.0),
            meta.get("y_weight", 0.0)
        ))
    
    if skipped > 3:
        print(f"⚠ ... and {skipped - 3} more edges skipped (total {skipped} skipped)")
    
    if not edge_data:
        print("⚠ No valid edges to upload")
        return
    
    # Insert edges in batches
    BATCH_SIZE = 500
    total_uploaded = 0
    
    insert_query = """
        INSERT INTO edges (
            from_node_id, to_node_id, base_threshold, 
            difficulty_rating, successful_attempts, x_weight, y_weight
        )
        VALUES %s
        ON CONFLICT DO NOTHING;
    """
    
    for i in range(0, len(edge_data), BATCH_SIZE):
        batch = edge_data[i:i+BATCH_SIZE]
        execute_values(cursor, insert_query, batch)
        connection.commit()
        total_uploaded += len(batch)
        print(f"✓ Uploaded {len(batch)} edges (batch {i//BATCH_SIZE + 1})")
    
    print(f"✓ Total edges uploaded: {total_uploaded}")
    cursor.close()

def main():
    if len(sys.argv) < 2:
        print("Usage: python upload_to_supabase.py path/to/graph.json")
        sys.exit(1)
    
    graph_file = sys.argv[1]
    
    if not os.path.exists(graph_file):
        print(f"Error: File '{graph_file}' not found")
        sys.exit(1)
    
    print(f"Loading graph data from {graph_file}...")
    with open(graph_file, "r") as f:
        doc = json.load(f)
    
    # Extract graph data (adjust based on your JSON structure)
    graph = None
    if isinstance(doc, dict) and "graph" in doc and isinstance(doc["graph"], list):
        graph = doc["graph"][0].get("data") if len(doc["graph"]) > 0 else None
    if graph is None:
        # Fallback: if nodes/edges at top-level
        graph = doc if ("nodes" in doc and "edges" in doc) else None
    if graph is None:
        raise Exception("Could not find graph data. Expected structure with 'nodes' and 'edges'")

    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    
    print(f"Found {len(nodes)} nodes and {len(edges)} edges")

    if not nodes:
        print("⚠ No nodes to upload.")
        return
    
    # Connect to database
    print("\nConnecting to database...")
    connection = get_connection()
    print("✓ Connection successful!")
    
    try:
        # Upload nodes
        print("\nUploading nodes...")
        upload_nodes(nodes, connection)
        
        # Get node ID mapping
        print("\nFetching node ID mapping...")
        node_id_map = get_node_id_map(connection)
        print(f"✓ Mapped {len(node_id_map)} nodes")
        
        # Upload edges
        print("\nUploading edges...")
        upload_edges(edges, node_id_map, connection)
        
        print("\n✓ Upload complete!")
        
    except Exception as e:
        connection.rollback()
        print(f"\n✗ Error during upload: {e}")
        raise
    finally:
        connection.close()
        print("Connection closed.")

if __name__ == "__main__":
    main()