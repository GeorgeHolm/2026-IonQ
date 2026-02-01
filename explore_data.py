from client import GameClient
from visualization import GraphTool
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.classical import expr
import json
from pathlib import Path
import psycopg2
from dotenv import load_dotenv
import os
import time


# Load environment variables for Supabase connection
load_dotenv()


SESSION_FILE = Path("npqc_session.json")


def get_db_connection():
   """Create and return a database connection"""
   try:
       connection = psycopg2.connect(
           user=os.getenv("user"),
           password=os.getenv("password"),
           host=os.getenv("host"),
           port=os.getenv("port"),
           dbname=os.getenv("dbname")
       )
       return connection
   except Exception as e:
       print(f"⚠ Database connection failed: {e}")
       return None


def _ensure_node_locks_table(conn) -> None:
   """Ensure a lightweight coordination table exists (doesn't require Prisma migrations)."""
   cur = conn.cursor()
   cur.execute(
       """
       CREATE TABLE IF NOT EXISTS node_locks (
           node_id TEXT PRIMARY KEY,
           owner TEXT NOT NULL,
           status TEXT NOT NULL,
           updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
       )
       """
   )
   conn.commit()
   cur.close()




def _debug_enabled() -> bool:
   return os.getenv("IONQ_DEBUG", "").strip().lower() in {"1", "true", "yes", "y"}




def get_node_lock(node_id: str):
   """Return lock row dict or None."""
   conn = get_db_connection()
   if not conn:
       return None
   try:
       _ensure_node_locks_table(conn)
       cur = conn.cursor()
       cur.execute(
           "SELECT node_id, owner, status, updated_at FROM node_locks WHERE node_id = %s",
           (node_id,),
       )
       row = cur.fetchone()
       cur.close()
       conn.close()
       if not row:
           return None
       return {"node_id": row[0], "owner": row[1], "status": row[2], "updated_at": row[3]}
   except Exception as e:
       if _debug_enabled():
           print(f"  ✗ Database error reading lock: {e}")
       try:
           conn.close()
       except Exception:
           pass
       return None




def try_lock_node(node_id: str, *, owner: str, status: str, ttl_seconds: int) -> bool:
   """
   Best-effort team coordination lock.
   Succeeds if:
     - no lock exists, OR
     - existing lock is expired (updated_at older than ttl_seconds), OR
     - lock is already owned by `owner` (refresh).
   """
   conn = get_db_connection()
   if not conn:
       return True  # don't brick gameplay if DB is down
   try:
       _ensure_node_locks_table(conn)
       cur = conn.cursor()
       cur.execute(
           """
           INSERT INTO node_locks (node_id, owner, status, updated_at)
           VALUES (%s, %s, %s, NOW())
           ON CONFLICT (node_id) DO UPDATE
           SET owner = EXCLUDED.owner,
               status = EXCLUDED.status,
               updated_at = NOW()
           WHERE
               node_locks.owner = EXCLUDED.owner
               OR node_locks.updated_at < (NOW() - (%s * INTERVAL '1 second'))
           """,
           (node_id, owner, status, int(ttl_seconds)),
       )
       ok = (cur.rowcount or 0) > 0
       conn.commit()
       cur.close()
       conn.close()
       return ok
   except Exception as e:
       if _debug_enabled():
           print(f"  ✗ Database error locking node: {e}")
       try:
           conn.close()
       except Exception:
           pass
       return True  # fail-open




def clear_node_lock(node_id: str, *, owner: str) -> None:
   conn = get_db_connection()
   if not conn:
       return
   try:
       _ensure_node_locks_table(conn)
       cur = conn.cursor()
       cur.execute("DELETE FROM node_locks WHERE node_id = %s AND owner = %s", (node_id, owner))
       conn.commit()
       cur.close()
       conn.close()
   except Exception:
       try:
           conn.close()
       except Exception:
           pass




def get_node_owned_status(node_id):
   """Get a node's owned status from Supabase
   Returns: None (not found), 0 (not owned), 1 (owned), 2 (skipped due to failures)
   """
   conn = get_db_connection()
   if not conn:
       return None
  
   try:
       cursor = conn.cursor()
       cursor.execute("""
           SELECT owned
           FROM nodes
           WHERE node_id = %s
       """, (node_id,))
       result = cursor.fetchone()
       cursor.close()
       conn.close()
      
       if result:
           return result[0]
       else:
           return None
   except Exception as e:
       print(f"  ✗ Database error getting node status: {e}")
       conn.close()
       return None
   
def get_edges():
   """Get edges from Supabase
   Returns: None (not found), 0 (not owned), 1 (owned), 2 (skipped due to failures)
   """
   conn = get_db_connection()
   if not conn:
       return None
  
   try:
       cursor = conn.cursor()
       cursor.execute("""
           SELECT id, from_node_id, to_node_id, x_weight, y_weight
           FROM edges
       """)
       result = cursor.fetchall()
       cursor.close()
       conn.close()
      
       edges = []
       edges.append("id,from_node_id,to_node_id,x_weight,y_weight")
       for row in result:
              edges.append(f"{row[0]},{row[1]},{row[2]},{row[3]},{row[4]}")

       if edges:
            return "\n".join(edges)
       else:
           return None
   except Exception as e:
       print(f"  ✗ Database error getting node status: {e}")
       conn.close()
       return None

def get_nodes():
   """Get nodes from Supabase
   Returns: None (not found), 0 (not owned), 1 (owned), 2 (skipped due to failures)
   """
   conn = get_db_connection()
   if not conn:
       return None
  
   try:
       cursor = conn.cursor()
       cursor.execute("""
           SELECT id, node_id
           FROM nodes
       """)
       result = cursor.fetchall()
       cursor.close()
       conn.close()
      
       edges = []
       edges.append("id,node_id")
       for row in result:
              edges.append(f"{row[0]},{row[1]}")

       if edges:
            return "\n".join(edges)
       else:
           return None
   except Exception as e:
       print(f"  ✗ Database error getting node status: {e}")
       conn.close()
       return None

def update_node_owned(node_id, owned=1):
   """Update a node's owned status in Supabase
   owned=0: not owned
   owned=1: successfully owned
   owned=2: skipped (too many failures)
   """
   conn = get_db_connection()
   if not conn:
       return False
  
   try:
       cursor = conn.cursor()
       cursor.execute("""
           UPDATE nodes
           SET owned = %s
           WHERE node_id = %s
       """, (owned, node_id))
       conn.commit()
       rows_updated = cursor.rowcount
       cursor.close()
       conn.close()
      
       if rows_updated > 0:
           status_msg = "owned (success)" if owned == 1 else "skipped (too many fails)" if owned == 2 else f"status={owned}"
           print(f"  📊 Database: Set node '{node_id}' {status_msg}")
           return True
       else:
           print(f"  ⚠ Database: Node '{node_id}' not found")
           return False
   except Exception as e:
       print(f"  ✗ Database error updating node: {e}")
       conn.close()
       return False


def update_edge_weights(from_node_id, to_node_id, x_weight=None, y_weight=None):
   """Update edge weights in Supabase after probing"""
   conn = get_db_connection()
   if not conn:
       return False
  
   try:
       cursor = conn.cursor()
      
       # Get node IDs
       cursor.execute("SELECT id FROM nodes WHERE node_id = %s", (from_node_id,))
       from_result = cursor.fetchone()
      
       cursor.execute("SELECT id FROM nodes WHERE node_id = %s", (to_node_id,))
       to_result = cursor.fetchone()
      
       if not from_result or not to_result:
           print(f"  ⚠ Database: Could not find nodes for edge")
           cursor.close()
           conn.close()
           return False
      
       from_id = from_result[0]
       to_id = to_result[0]
      
       # Update edge weights
       updates = []
       values = []
      
       if x_weight is not None:
           updates.append("x_weight = %s")
           values.append(x_weight)
      
       if y_weight is not None:
           updates.append("y_weight = %s")
           values.append(y_weight)
      
       if not updates:
           cursor.close()
           conn.close()
           return False
      
       values.extend([from_id, to_id])
      
       query = f"""
           UPDATE edges
           SET {', '.join(updates)}
           WHERE from_node_id = %s AND to_node_id = %s
       """
      
       cursor.execute(query, values)
       conn.commit()
       rows_updated = cursor.rowcount
       cursor.close()
       conn.close()
      
       if rows_updated > 0:
           weight_str = f"x={x_weight:.3f}" if x_weight is not None else ""
           if y_weight is not None:
               weight_str += f" y={y_weight:.3f}" if weight_str else f"y={y_weight:.3f}"
           print(f"  📊 Database: Updated edge ({from_node_id}, {to_node_id}) {weight_str}")
           return True
       else:
           print(f"  ⚠ Database: Edge not found")
           return False
   except Exception as e:
       print(f"  ✗ Database error updating edge: {e}")
       conn.close()
       return False
   

print(get_nodes())