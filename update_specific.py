import psycopg2
from dotenv import load_dotenv
import os

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

# ========== UPDATE SPECIFIC NODE ==========

def update_node_by_node_id(connection, node_id, **fields):
    """
    Update a specific node by its node_id
    
    Example usage:
        update_node_by_node_id(conn, "College Park, MD", owned=1, capacity=100)
    """
    cursor = connection.cursor()
    
    # Build the UPDATE query dynamically based on fields provided
    set_parts = []
    values = []
    for field, value in fields.items():
        set_parts.append(f"{field} = %s")
        values.append(value)
    
    values.append(node_id)  # For the WHERE clause
    
    query = f"""
        UPDATE nodes 
        SET {', '.join(set_parts)}
        WHERE node_id = %s
    """
    
    cursor.execute(query, values)
    connection.commit()
    rows_updated = cursor.rowcount
    cursor.close()
    
    print(f"✓ Updated {rows_updated} node(s) with node_id='{node_id}'")
    return rows_updated

def update_node_by_id(connection, id, **fields):
    """
    Update a specific node by its database id
    
    Example usage:
        update_node_by_id(conn, 5, owned=1, utility_qubits=10)
    """
    cursor = connection.cursor()
    
    set_parts = []
    values = []
    for field, value in fields.items():
        set_parts.append(f"{field} = %s")
        values.append(value)
    
    values.append(id)
    
    query = f"""
        UPDATE nodes 
        SET {', '.join(set_parts)}
        WHERE id = %s
    """
    
    cursor.execute(query, values)
    connection.commit()
    rows_updated = cursor.rowcount
    cursor.close()
    
    print(f"✓ Updated {rows_updated} node(s) with id={id}")
    return rows_updated

# ========== UPDATE SPECIFIC EDGE ==========

def update_edge_by_node_ids(connection, from_node_id, to_node_id, **fields):
    """
    Update a specific edge by the node_ids of the nodes it connects
    
    Example usage:
        update_edge_by_node_ids(conn, "College Park, MD", "Washington, DC", 
                                base_threshold=0.95, difficulty_rating=2)
    """
    cursor = connection.cursor()
    
    # First get the database IDs for the nodes
    cursor.execute("""
        SELECT id FROM nodes WHERE node_id = %s
    """, (from_node_id,))
    from_result = cursor.fetchone()
    
    cursor.execute("""
        SELECT id FROM nodes WHERE node_id = %s
    """, (to_node_id,))
    to_result = cursor.fetchone()
    
    if not from_result or not to_result:
        print(f"✗ Could not find nodes: '{from_node_id}' or '{to_node_id}'")
        cursor.close()
        return 0
    
    from_id = from_result[0]
    to_id = to_result[0]
    
    # Build the UPDATE query
    set_parts = []
    values = []
    for field, value in fields.items():
        set_parts.append(f"{field} = %s")
        values.append(value)
    
    values.extend([from_id, to_id])
    
    query = f"""
        UPDATE edges 
        SET {', '.join(set_parts)}
        WHERE from_node_id = %s AND to_node_id = %s
    """
    
    cursor.execute(query, values)
    connection.commit()
    rows_updated = cursor.rowcount
    cursor.close()
    
    print(f"✓ Updated {rows_updated} edge(s) from '{from_node_id}' to '{to_node_id}'")
    return rows_updated

def update_edge_by_id(connection, edge_id, **fields):
    """
    Update a specific edge by its database id
    
    Example usage:
        update_edge_by_id(conn, 42, x_weight=0.5, y_weight=0.3)
    """
    cursor = connection.cursor()
    
    set_parts = []
    values = []
    for field, value in fields.items():
        set_parts.append(f"{field} = %s")
        values.append(value)
    
    values.append(edge_id)
    
    query = f"""
        UPDATE edges 
        SET {', '.join(set_parts)}
        WHERE id = %s
    """
    
    cursor.execute(query, values)
    connection.commit()
    rows_updated = cursor.rowcount
    cursor.close()
    
    print(f"✓ Updated {rows_updated} edge(s) with id={edge_id}")
    return rows_updated

# ========== QUERY/READ FUNCTIONS ==========

def get_node(connection, node_id):
    """Get a node by its node_id"""
    cursor = connection.cursor()
    cursor.execute("""
        SELECT id, node_id, utility_qubits, bonus_bell_pairs, capacity, owned
        FROM nodes
        WHERE node_id = %s
    """, (node_id,))
    result = cursor.fetchone()
    cursor.close()
    
    if result:
        return {
            "id": result[0],
            "node_id": result[1],
            "utility_qubits": result[2],
            "bonus_bell_pairs": result[3],
            "capacity": result[4],
            "owned": result[5]
        }
    return None

def get_edge(connection, from_node_id, to_node_id):
    """Get an edge by the node_ids it connects"""
    cursor = connection.cursor()
    cursor.execute("""
        SELECT e.id, e.base_threshold, e.difficulty_rating, 
               e.successful_attempts, e.x_weight, e.y_weight
        FROM edges e
        JOIN nodes n1 ON e.from_node_id = n1.id
        JOIN nodes n2 ON e.to_node_id = n2.id
        WHERE n1.node_id = %s AND n2.node_id = %s
    """, (from_node_id, to_node_id))
    result = cursor.fetchone()
    cursor.close()
    
    if result:
        return {
            "id": result[0],
            "base_threshold": result[1],
            "difficulty_rating": result[2],
            "successful_attempts": result[3],
            "x_weight": result[4],
            "y_weight": result[5]
        }
    return None

# ========== EXAMPLE USAGE ==========

def main():
    print("Connecting to database...")
    connection = get_connection()
    print("✓ Connection successful!\n")
    
    try:
        # EXAMPLE 1: Update a specific node by node_id
        print("Example 1: Update node 'College Park, MD'")
        update_node_by_node_id(
            connection, 
            "College Park, MD",
            owned=1,
            capacity=150
        )
        
        # EXAMPLE 2: Update a specific edge between two cities
        print("\nExample 2: Update edge from 'College Park, MD' to 'Washington, DC'")
        update_edge_by_node_ids(
            connection,
            "College Park, MD",
            "Washington, DC",
            base_threshold=0.95,
            x_weight=0.5,
            y_weight=0.3
        )
        
        # EXAMPLE 3: Read a node to verify changes
        print("\nExample 3: Read node 'College Park, MD'")
        node = get_node(connection, "College Park, MD")
        if node:
            print(f"Node data: {node}")
        
        # EXAMPLE 4: Read an edge to verify changes
        print("\nExample 4: Read edge")
        edge = get_edge(connection, "College Park, MD", "Washington, DC")
        if edge:
            print(f"Edge data: {edge}")
        
        print("\n✓ All examples complete!")
        
    except Exception as e:
        connection.rollback()
        print(f"\n✗ Error: {e}")
        raise
    finally:
        connection.close()
        print("\nConnection closed.")

if __name__ == "__main__":
    print("=" * 60)
    print("UPDATE SPECIFIC NODE OR EDGE - Example Script")
    print("=" * 60)
    print()
    print("This script demonstrates how to update specific nodes and edges.")
    print("Modify the main() function to customize your updates.")
    print()
    
    confirm = input("Run the examples? (yes/no): ")
    
    if confirm.lower() in ['yes', 'y']:
        main()
    else:
        print("\nScript cancelled.")
        print("\nYou can import these functions in your own scripts:")
        print("  from update_specific import update_node_by_node_id, update_edge_by_node_ids")