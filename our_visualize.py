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


def save_session(client):
   if client.api_token:
       with open(SESSION_FILE, "w") as f:
           json.dump({"api_token": client.api_token, "player_id": client.player_id, "name": client.name}, f)
       print(f"Session saved.")


def load_session():
   if not SESSION_FILE.exists():
       return None
  
   try:
       with open(SESSION_FILE) as f:
           data = json.load(f)
   except (json.JSONDecodeError, ValueError):
       # Session file is corrupted or empty
       print("Warning: session.json is corrupted. Starting fresh.")
       SESSION_FILE.unlink()  # Delete the corrupted file
       return None
  
   client = GameClient(api_token=data.get("api_token"))
   client.player_id = data.get("player_id")
   client.name = data.get("name")
  
   try:
       status = client.get_status()
       if status:
           print(f"Resumed: {client.player_id} | Score: {status.get('score', 0)} | Budget: {status.get('budget', 0)}")
           return client
   except:
       print("Warning: Could not resume session. Starting fresh.")
       return None
  
   return None


# [Rest of your existing code continues here...]
from typing import Literal, Tuple, Union




def _pair_qubits(num_pairs: int, pair_idx: int) -> Tuple[int, int]:
  """Return (alice_qubit, bob_qubit) for the given Bell-pair index."""
  if num_pairs < 1:
      raise ValueError('num_pairs must be >= 1')
  if not (0 <= pair_idx < num_pairs):
      raise ValueError(f'pair_idx must be in [0, {num_pairs-1}]')
  alice = pair_idx
  bob = 2 * num_pairs - 1 - pair_idx
  return alice, bob




def _apply_bilateral_cnot(qc: QuantumCircuit, *, keep: Tuple[int, int], discard: Tuple[int, int]) -> None:
  ka, kb = keep
  da, db = discard
  qc.cx(ka, da)  # Alice: keep -> discard
  qc.cx(kb, db)  # Bob:   keep -> discard




def _append_single_round_flag(
  qc: QuantumCircuit,
  *,
  discard: Tuple[int, int],
  cr: ClassicalRegister,
  meas_a_bit: int,
  meas_b_bit: int,
  flag_bit: int,
) -> None:
  """Single-round postselection: flag = meas_a XOR meas_b."""
  da, db = discard


  qc.measure(da, cr[meas_a_bit])
  qc.measure(db, cr[meas_b_bit])
  # Use a real-time classical XOR to compute the postselection flag.
  # This avoids quantum feedforward and qubit resets, which may be unsupported server-side.
  qc.store(cr[flag_bit], expr.bit_xor(cr[meas_a_bit], cr[meas_b_bit]))




def _append_round_and_or_flag(
  qc: QuantumCircuit,
  *,
  discard: Tuple[int, int],
  cr: ClassicalRegister,
  meas_a_bit: int,
  meas_b_bit: int,
  mismatch_bit: int,
  flag_bit: int,
) -> None:
  """Multi-round postselection: flag := flag OR (meas_a XOR meas_b)."""
  da, db = discard


  qc.measure(da, cr[meas_a_bit])
  qc.measure(db, cr[meas_b_bit])
  qc.store(cr[mismatch_bit], expr.bit_xor(cr[meas_a_bit], cr[meas_b_bit]))
  qc.store(cr[flag_bit], expr.bit_or(cr[flag_bit], cr[mismatch_bit]))




PurificationVariant = Literal['bit', 'phase']
Rounds = Union[int, Literal['max']]




def create_purification_circuit(
  num_bell_pairs: int,
  *,
  variant: PurificationVariant = 'bit',
  rounds: Rounds = 1,
) -> Tuple[QuantumCircuit, int]:
  """Create a purification circuit for the game.


  Returns: (qc, flag_bit_index)


  - Output pair is always on qubits (N-1, N) as required by the rules.
  - `variant='bit'` does a Z-basis parity check (suppresses X/bit-flips).
  - `variant='phase'` does an X-basis parity check (suppresses Z/phase-flips).
  - For `rounds > 1` we chain multiple parity checks using classical logic in the same circuit.
  """


  if num_bell_pairs < 1:
      raise ValueError('num_bell_pairs must be >= 1')


  n_qubits = 2 * num_bell_pairs


  # Trivial case: N=1 (no distillation possible). Keep all shots by leaving flag=0.
  if num_bell_pairs == 1:
      cr = ClassicalRegister(1, 'c')
      qr = QuantumRegister(n_qubits, 'q')
      qc = QuantumCircuit(qr, cr)
      return qc, 0


  keep = _pair_qubits(num_bell_pairs, num_bell_pairs - 1)
  ka, kb = keep


  # Resolve how many rounds we want.
  if rounds == 'max':
      desired_rounds = 10**9
  else:
      desired_rounds = int(rounds)
      if desired_rounds < 1:
          raise ValueError('rounds must be >= 1')


  if desired_rounds == 1:
      # Use the outermost pair as the discard (matches the tutorial layout for N=2).
      discard_idxs = [0]
  else:
      max_rounds = num_bell_pairs - 1
      discard_idxs = list(range(num_bell_pairs - 2, -1, -1))  # N-2, ..., 0
      discard_idxs = discard_idxs[: min(desired_rounds, max_rounds)]


      # Multi-round circuits must NOT repeatedly overwrite/accumulate a single classical flag bit
      # because the backend may not respect sequential classical updates within one circuit.
      # Instead, record each round's mismatch into a distinct bit and compute the final flag once.
      r = len(discard_idxs)
      # Layout per round i:
      # c[3*i+0] Alice measurement
      # c[3*i+1] Bob measurement
      # c[3*i+2] mismatch = a XOR b
      # Final: c[3*r] flag = OR over all mismatches
      cr = ClassicalRegister(3 * r + 1, 'c')
      qr = QuantumRegister(n_qubits, 'q')
      qc = QuantumCircuit(qr, cr)


      mismatch_bits: list[int] = []


      for round_i, idx in enumerate(discard_idxs):
          discard = _pair_qubits(num_bell_pairs, idx)
          da, db = discard


          meas_a_bit = 3 * round_i + 0
          meas_b_bit = 3 * round_i + 1
          mismatch_bit = 3 * round_i + 2
          mismatch_bits.append(mismatch_bit)


          if variant == 'phase':
              qc.h(qr[ka]); qc.h(qr[kb])
              qc.h(qr[da]); qc.h(qr[db])


          _apply_bilateral_cnot(qc, keep=keep, discard=discard)


          qc.measure(da, cr[meas_a_bit])
          qc.measure(db, cr[meas_b_bit])
          qc.store(cr[mismatch_bit], expr.bit_xor(cr[meas_a_bit], cr[meas_b_bit]))


          if variant == 'phase':
              qc.h(qr[ka]); qc.h(qr[kb])


      flag_bit = 3 * r
      flag_expr = cr[mismatch_bits[0]]
      for mb in mismatch_bits[1:]:
          flag_expr = expr.bit_or(flag_expr, cr[mb])
      qc.store(cr[flag_bit], flag_expr)


      return qc, flag_bit


  # Single-round path
  # Classical layout:
  # c[0] Alice discard measurement
  # c[1] Bob discard measurement
  # c[2] scratch
  # c[3] flag (0=keep, 1=discard)
  cr = ClassicalRegister(4, 'c')
  qr = QuantumRegister(n_qubits, 'q')
  qc = QuantumCircuit(qr, cr)


  discard = _pair_qubits(num_bell_pairs, discard_idxs[0])
  da, db = discard


  if variant == 'phase':
      qc.h(qr[ka]); qc.h(qr[kb])
      qc.h(qr[da]); qc.h(qr[db])


  _apply_bilateral_cnot(qc, keep=keep, discard=discard)


  _append_single_round_flag(
      qc,
      discard=discard,
      cr=cr,
      meas_a_bit=0,
      meas_b_bit=1,
      flag_bit=3,
  )


  if variant == 'phase':
      qc.h(qr[ka]); qc.h(qr[kb])


  return qc, 3




# --- N=3 parity-check (syndrome-style) variant ---
# Uses the two extra pairs as parity-check ancillas against the output pair.


TriplePostselect = Literal["or", "and2"]  # discard if any mismatch vs only if both mismatches




def _append_mismatch_bit(
  qc: QuantumCircuit,
  *,
  discard: Tuple[int, int],
  cr: ClassicalRegister,
  meas_a_bit: int,
  meas_b_bit: int,
  out_bit: int,
) -> None:
  """Measure discard pair and write mismatch = meas_a XOR meas_b into cr[out_bit]."""
  da, db = discard


  qc.measure(da, cr[meas_a_bit])
  qc.measure(db, cr[meas_b_bit])
  qc.store(cr[out_bit], expr.bit_xor(cr[meas_a_bit], cr[meas_b_bit]))




def create_three_pair_syndrome_circuit(
  *,
  orientation: Literal["bit", "phase"] = "bit",
  postselect: TriplePostselect = "and2",
) -> Tuple[QuantumCircuit, int]:
  """
  N=3 "syndrome-style" parity checks against the required output pair (pair 3).
  - orientation='bit': Z-basis parity checks (targets X/bit-flips)
  - orientation='phase': X-basis parity checks (targets Z/phase-flips via H wrappers)
  - postselect='and2': discard only if BOTH parity checks mismatch (higher P)
    postselect='or': discard if ANY mismatch (lower P, usually higher F)


  Output pair is always on qubits (N-1, N) = (2, 3) for N=3.
  """
  num_bell_pairs = 3
  n_qubits = 2 * num_bell_pairs


  # c[0],c[1] scratch meas; c[2]=mismatch_vs_pair1; c[3]=mismatch_vs_pair0; c[4]=flag (0=keep)
  cr = ClassicalRegister(5, "c")
  qr = QuantumRegister(n_qubits, "q")
  qc = QuantumCircuit(qr, cr)


  keep = _pair_qubits(num_bell_pairs, num_bell_pairs - 1)  # (2,3)
  ka, kb = keep


  for disc_idx, mismatch_bit in [(1, 2), (0, 3)]:
      discard = _pair_qubits(num_bell_pairs, disc_idx)
      da, db = discard


      if orientation == "phase":
          qc.h(qr[ka]); qc.h(qr[kb])
          qc.h(qr[da]); qc.h(qr[db])


      _apply_bilateral_cnot(qc, keep=keep, discard=discard)
      _append_mismatch_bit(qc, discard=discard, cr=cr, meas_a_bit=0, meas_b_bit=1, out_bit=mismatch_bit)


      if orientation == "phase":
          qc.h(qr[ka]); qc.h(qr[kb])


  flag_bit = 4


  if postselect == "or":
      qc.store(cr[flag_bit], expr.bit_or(cr[2], cr[3]))
  else:
      qc.store(cr[flag_bit], expr.bit_and(cr[2], cr[3]))


  return qc, flag_bit




# Backwards-compatible helpers used by the rest of the notebook.


def create_distillation_circuit() -> QuantumCircuit:
  qc, _flag_bit = create_purification_circuit(2, variant='bit', rounds=1)
  return qc




def create_distillation_circuit_1() -> QuantumCircuit:
  qc, _flag_bit = create_purification_circuit(1)
  return qc




# Examples (optional):
SHOW_EXAMPLES = True




def pick_variant_from_noise(f_x: float, f_z_y: float, *, margin: float = 0.05) -> PurificationVariant:


  if f_x > f_z_y + margin:
      return 'bit'
  if f_z_y > f_x + margin:
      return 'phase'
  return 'bit'




if SHOW_EXAMPLES:
  qc2_bit, fb2 = create_purification_circuit(2, variant='bit', rounds=1)
  qc2_phase, fb2p = create_purification_circuit(2, variant='phase', rounds=1)
  qc4_bit, fb4 = create_purification_circuit(4, variant='bit', rounds='max')
  qc4_phase, fb4p = create_purification_circuit(4, variant='phase', rounds='max')


  print('N=2 bit flag_bit =', fb2)
  print(qc2_bit.draw(output='mpl', filename='bit_2.png'))
  print('N=4 bit flag_bit =', fb4)
  print(qc4_bit.draw(output='mpl', filename='bit_4.png'))
  print('N=2 phase flag_bit =', fb2p)
  print(qc2_phase.draw(output='mpl', filename='phase_2.png'))
  print('N=4 phase (multi-round) flag_bit =', fb4)
  print(qc4_phase.draw(output='mpl', filename='phase_4.png'))

