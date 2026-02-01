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


# Try to resume existing session
client = load_session()


if not client:
   print("No saved session. Register below.")


if client and client.api_token:
   print(f"Already registered as {client.player_id}")
   status = client.get_status()
else:
   client = GameClient()
  
   # CHANGE THESE to your unique values
   PLAYER_ID = "npqc_1"
   PLAYER_NAME = "NPQC Player 1"
  
   result = client.register(PLAYER_ID, PLAYER_NAME, location="remote")
  
   if result.get("ok"):
       print(f"Registered! Token: {client.api_token[:20]}...")
       save_session(client)
       # Registration response contains starting_candidates
       status = result.get("data", {})
   else:
       print(f"Failed: {result.get('error', {}).get('message')}")
       status = {}


# Only get fresh status if we don't have starting_candidates (i.e., resumed session)
if not status.get('starting_candidates'):
   status = client.get_status()
  
print(f"\nCurrent status keys: {list(status.keys())}")


# SELECT STARTING NODE
if status.get('starting_node'):
   print(f"\n✓ Starting node already selected: {status['starting_node']}")
   print(f"Budget: {status['budget']} | Score: {status['score']}")
else:
   print("\n" + "="*60)
   print("SELECT YOUR STARTING NODE")
   print("="*60)
  
   candidates = status.get("starting_candidates", [])
  
   if not candidates:
       print("No starting candidates available!")
   else:
       # Rank candidates by a simple score: utility_qubits + 2*bonus_bell_pairs
       ranked_candidates = []
       for c in candidates:
           utility = c.get('utility_qubits', 0)
           bonus = c.get('bonus_bell_pairs', 0)
           score = utility + (2 * bonus)  # Weight bonus more heavily
           ranked_candidates.append({
               'node_id': c['node_id'],
               'utility_qubits': utility,
               'bonus_bell_pairs': bonus,
               'score': score,
               'raw': c
           })
      
       # Sort by score descending
       ranked_candidates.sort(key=lambda x: x['score'], reverse=True)
      
       print("\nAvailable starting nodes (ranked by value):")
       print("-" * 60)
       for i, c in enumerate(ranked_candidates):
           print(f"{i+1}. {c['node_id']}")
           print(f"   Utility Qubits: {c['utility_qubits']}")
           print(f"   Bonus Bell Pairs: +{c['bonus_bell_pairs']}")
           print(f"   Score: {c['score']}")
           print()
      
       # Automatic selection: choose the highest-ranked node
       AUTO_SELECT = False  # Set to False for manual selection
      
       if AUTO_SELECT:
           best = ranked_candidates[0]
           print(f"Auto-selecting best node: {best['node_id']}")
           result = client.select_starting_node(best['node_id'])
          
           if result.get('ok'):
               print(f"✓ Successfully selected starting node: {best['node_id']}")
               save_session(client)
           else:
               print(f"✗ Failed to select starting node: {result.get('error', {}).get('message')}")
       else:
           # Manual selection
           while True:
               try:
                   choice = input(f"Select starting node (1-{len(ranked_candidates)}) or 'q' to quit: ").strip()
                  
                   if choice.lower() == 'q':
                       print("Exiting without selecting a starting node.")
                       break
                  
                   idx = int(choice) - 1
                   if 0 <= idx < len(ranked_candidates):
                       selected = ranked_candidates[idx]
                       print(f"\nYou selected: {selected['node_id']}")
                       confirm = input("Confirm? (yes/no): ").strip().lower()
                      
                       if confirm in ['yes', 'y']:
                           result = client.select_starting_node(selected['node_id'])
                          
                           if result.get('ok'):
                               print(f"✓ Successfully selected starting node: {selected['node_id']}")
                               save_session(client)
                               break
                           else:
                               print(f"✗ Failed to select: {result.get('error', {}).get('message')}")
                       else:
                           print("Selection cancelled. Try again.")
                   else:
                       print(f"Invalid choice. Please enter a number between 1 and {len(ranked_candidates)}.")
               except ValueError:
                   print("Invalid input. Please enter a number.")
               except KeyboardInterrupt:
                   print("\nExiting.")
                   break


# Update status after selection
status = client.get_status()
if status.get('starting_node'):
   print(f"\n✓ Game ready! Starting from: {status['starting_node']}")
   print(f"Budget: {status['budget']} | Score: {status['score']}")




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
SHOW_EXAMPLES = False




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


  print('N=2 bit flag_bit =', fb2)
  print(qc2_bit.draw(output='text'))
  print('N=2 phase flag_bit =', fb2p)
  print(qc2_phase.draw(output='text'))
  print('N=4 bit (multi-round) flag_bit =', fb4)
  print(qc4_bit.draw(output='text'))




# --- Recon helpers (implemented via /claim_edge) ---
# Note: these are real claim attempts using num_bell_pairs=1.
# If a probe beats the threshold it will claim the edge and spend 1 bell pair.


def probe_edge_attempt(client: GameClient, edge_id, *, apply_x: bool = False):
  """Attempt a 1-bell-pair probe on an edge.


  Returns the server response dict (or None).
  """


  edge = tuple(edge_id) if isinstance(edge_id, list) else edge_id


  qr = QuantumRegister(2, 'q')
  cr = ClassicalRegister(1, 'c')
  qc = QuantumCircuit(qr, cr)


  if apply_x:
      qc.x(qr[0])


  result = client.claim_edge(edge, qc, 0, 1)
  if not result.get('ok'):
      print(f"{edge}: PROBE ERROR - {result.get('error', {}).get('message')}")
      return None


  d = result['data']
  status = 'CLAIMED' if d.get('success') else 'probed'
  fidelity = d.get('fidelity', 0)
  print(f"{edge}: {status} F={fidelity:.3f} P={d.get('success_probability', 0):.3f} thr={d.get('threshold', 0):.3f}")
 
  # Save to database based on probe type
  from_node, to_node = edge
  if apply_x:
      # This is the X-probe, save as x_weight
      update_edge_weights(from_node, to_node, x_weight=float(fidelity))
  else:
      # This is the raw probe, save as y_weight 
      update_edge_weights(from_node, to_node, y_weight=float(fidelity))
 
  return result




def recon_noise_type_for_edge(client: GameClient, edge_id, *, margin: float = 0.05):
  """Run the two probes (blank, then X) unless we claim early.


  Returns:
    - {'claimed': True, 'result': <claim result>} if either probe claims the edge
    - {'claimed': False, 'variant': 'bit'|'phase', 'raw': f_raw, 'x': f_x, 'z_y': f_z_y} otherwise
  """


  edge = tuple(edge_id) if isinstance(edge_id, list) else edge_id


  print(f"Recon {edge}")
  r0 = probe_edge_attempt(client, edge, apply_x=False)
  if r0 and r0.get('data', {}).get('success'):
      return {'claimed': True, 'result': r0}


  r1 = probe_edge_attempt(client, edge, apply_x=True)
  if r1 and r1.get('data', {}).get('success'):
      return {'claimed': True, 'result': r1}


  if not r0 or not r1:
      return {'claimed': False, 'variant': 'bit', 'raw': None, 'x': None, 'z_y': None}


  f_raw = float(r0['data'].get('fidelity', 0) or 0)
  f_x = float(r1['data'].get('fidelity', 0) or 0)
  f_z_y = max(0.0, 1.0 - f_raw - f_x)
  if f_x > f_z_y + margin:
      bias = 'x'
  elif f_z_y > f_x + margin:
      bias = 'z'
  else:
      bias = 'mixed'


  variant = pick_variant_from_noise(f_x, f_z_y, margin=margin)


  print(f"Bias={bias} pick={variant} raw={f_raw:.3f} x={f_x:.3f} z+y={f_z_y:.3f}")


  return {'claimed': False, 'bias': bias, 'variant': variant, 'raw': f_raw, 'x': f_x, 'z_y': f_z_y}




def recon_noise_type_for_edge_after_raw(client: GameClient, edge_id, *, f_raw: float, margin: float = 0.05):
  """Noise recon when you've already done the blank (no-gate) N=1 attempt.


  This runs only the X-probe (apply_x=True) unless it claims early.


  Returns the same shape as recon_noise_type_for_edge.
  """


  edge = tuple(edge_id) if isinstance(edge_id, list) else edge_id


  r1 = probe_edge_attempt(client, edge, apply_x=True)
  if r1 and r1.get('data', {}).get('success'):
      return {'claimed': True, 'result': r1}


  if not r1:
      return {'claimed': False, 'variant': 'bit', 'raw': f_raw, 'x': None, 'z_y': None}


  f_x = float(r1['data'].get('fidelity', 0) or 0)
  f_z_y = max(0.0, 1.0 - float(f_raw) - f_x)


  variant = pick_variant_from_noise(f_x, f_z_y, margin=margin)


  print(f"pick={variant} raw={float(f_raw):.3f} x={f_x:.3f} z+y={f_z_y:.3f}")


  return {'claimed': False, 'variant': variant, 'raw': float(f_raw), 'x': f_x, 'z_y': f_z_y}




# --- DEJMPS-style variant (2->1 recurrence) ---
# Adds local phase rotations around the standard parity-check step.
# Often helps on mixed/asymmetric noise, but not guaranteed to dominate on fidelity×p_success.


DEJMPSOrientation = Literal['bit', 'phase']




def create_dejmps_circuit(
  num_bell_pairs: int,
  *,
  orientation: DEJMPSOrientation = 'bit',
  rounds: Rounds = 1,
) -> Tuple[QuantumCircuit, int]:
  """DEJMPS-style recurrence.


  Implementation notes:
  - Same keep/discard structure as create_purification_circuit.
  - Adds S on Alice qubits and Sdg on Bob qubits on BOTH the keep and discard pairs per round.
  - orientation='phase' wraps the round in Hadamards (X-basis parity check).


  Returns: (qc, flag_bit)
  """


  if num_bell_pairs < 2:
      # No real distillation possible; fall back to the base builder.
      return create_purification_circuit(num_bell_pairs, variant='bit', rounds=1)


  # Build the base circuit structure but inline the per-round loop so we can add rotations.
  n_qubits = 2 * num_bell_pairs


  keep = _pair_qubits(num_bell_pairs, num_bell_pairs - 1)
  ka, kb = keep


  # Resolve rounds similarly to create_purification_circuit
  if rounds == 'max':
      desired_rounds = 10**9
  else:
      desired_rounds = int(rounds)
      if desired_rounds < 1:
          raise ValueError('rounds must be >= 1')


  if desired_rounds == 1:
      discard_idxs = [0]
  else:
      max_rounds = num_bell_pairs - 1
      discard_idxs = list(range(num_bell_pairs - 2, -1, -1))
      discard_idxs = discard_idxs[: min(desired_rounds, max_rounds)]


  multi_round = len(discard_idxs) > 1
  if multi_round:
      r = len(discard_idxs)
      cr = ClassicalRegister(3 * r + 1, 'c')
      flag_bit = 3 * r
  else:
      cr = ClassicalRegister(4, 'c')
      flag_bit = 3
  qr = QuantumRegister(n_qubits, 'q')
  qc = QuantumCircuit(qr, cr)


  mismatch_bits: list[int] = []


  for round_i, idx in enumerate(discard_idxs):
      discard = _pair_qubits(num_bell_pairs, idx)
      da, db = discard


      # Optional phase-orientation wrapper
      if orientation == 'phase':
          qc.h(qr[ka]); qc.h(qr[kb])
          qc.h(qr[da]); qc.h(qr[db])


      # DEJMPS-style local phase rotations
      qc.s(qr[ka]); qc.s(qr[da])
      qc.sdg(qr[kb]); qc.sdg(qr[db])


      _apply_bilateral_cnot(qc, keep=keep, discard=discard)


      if multi_round:
          meas_a_bit = 3 * round_i + 0
          meas_b_bit = 3 * round_i + 1
          mismatch_bit = 3 * round_i + 2
          mismatch_bits.append(mismatch_bit)


          qc.measure(da, cr[meas_a_bit])
          qc.measure(db, cr[meas_b_bit])
          qc.store(cr[mismatch_bit], expr.bit_xor(cr[meas_a_bit], cr[meas_b_bit]))
      else:
          _append_single_round_flag(qc, discard=discard, cr=cr, meas_a_bit=0, meas_b_bit=1, flag_bit=flag_bit)


      if orientation == 'phase':
          qc.h(qr[ka]); qc.h(qr[kb])


  if multi_round:
      flag_expr = cr[mismatch_bits[0]]
      for mb in mismatch_bits[1:]:
          flag_expr = expr.bit_or(flag_expr, cr[mb])
      qc.store(cr[flag_bit], flag_expr)


  return qc, flag_bit




# Step 5: Rank claimable edges (notebook-local copy)
#
# This is copied from `strategy.py` so the notebook doesn't depend on that file.


from collections import deque, defaultdict
from dataclasses import dataclass
from math import inf
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Set, Tuple


NodeId = str


@dataclass(frozen=True)
class GraphIndex:
  nodes: Dict[NodeId, Dict[str, Any]]
  adj: Dict[NodeId, List[NodeId]]
  bonus_nodes: Set[NodeId]




def build_graph_index(graph_data: Dict[str, Any]) -> GraphIndex:
  nodes = {n["node_id"]: n for n in graph_data.get("nodes", [])}
  adj: DefaultDict[NodeId, List[NodeId]] = defaultdict(list)
  for edge in graph_data.get("edges", []):
      n1, n2 = edge["edge_id"]
      adj[n1].append(n2)
      adj[n2].append(n1)


  bonus_nodes = {
      node_id
      for node_id, n in nodes.items()
      if int(n.get("bonus_bell_pairs", 0) or 0) > 0
  }
  return GraphIndex(nodes=nodes, adj=dict(adj), bonus_nodes=bonus_nodes)




def multi_source_bfs_distances(adj: Dict[NodeId, List[NodeId]], sources: Iterable[NodeId]) -> Dict[NodeId, int]:
  sources = list(sources)
  dist: Dict[NodeId, int] = {}
  q: deque[NodeId] = deque()
  for s in sources:
      dist[s] = 0
      q.append(s)


  while q:
      u = q.popleft()
      du = dist[u]
      for v in adj.get(u, []):
          if v not in dist:
              dist[v] = du + 1
              q.append(v)
  return dist




def get_owned_bonus_total(nodes: Dict[NodeId, Dict[str, Any]], owned_nodes: Iterable[NodeId]) -> int:
  total = 0
  for node_id in owned_nodes:
      total += int(nodes.get(node_id, {}).get("bonus_bell_pairs", 0) or 0)
  return total




def budget_safety_margin(*, budget: int, owned_bonus_total: int) -> int:
  return int(budget) - int(owned_bonus_total)




def can_afford_success(*, budget: int, success_cost: int) -> bool:
  return (int(budget) - int(success_cost)) > 0




def is_risky_spend(*, budget: int, owned_bonus_total: int, success_cost: int, min_safety_margin: int = 8) -> bool:
  return budget_safety_margin(budget=budget, owned_bonus_total=owned_bonus_total) - int(success_cost) < int(min_safety_margin)




def _edge_endpoints(edge: Dict[str, Any], owned: Set[NodeId]) -> Tuple[Optional[NodeId], Optional[NodeId]]:
  n1, n2 = edge["edge_id"]
  if (n1 in owned) and (n2 not in owned):
      return n1, n2
  if (n2 in owned) and (n1 not in owned):
      return n2, n1
  return None, None




def rank_claimable_edges(client: Any, *, assumed_success_cost: int = 2) -> List[Dict[str, Any]]:
  status = client.get_status()
  budget = int(status.get("budget", 0) or 0)
  owned: Set[NodeId] = set(status.get("owned_nodes", []) or [])
  owner_id = str(getattr(client, "player_id", "") or "")
  ignore_own_skips = os.getenv("IONQ_IGNORE_OWN_SKIPS", "").strip().lower() in {"1", "true", "yes", "y"}


  graph = client.get_cached_graph()
  gi = build_graph_index(graph)


  owned_bonus_total = get_owned_bonus_total(gi.nodes, owned)
  safety = budget_safety_margin(budget=budget, owned_bonus_total=owned_bonus_total)


  claimable = client.get_claimable_edges()
  if not claimable:
      return []


  dist_to_bonus = multi_source_bfs_distances(gi.adj, gi.bonus_nodes) if gi.bonus_nodes else {}


  bonus_weight = 5.0 if safety < 20 else 2.0
  dist_weight = 3.0 if safety < 20 else 1.0
  defense_weight = 4.0 if safety < 20 else 1.0


  ranked: List[Dict[str, Any]] = []
  for edge in claimable:
      owned_node, new_node = _edge_endpoints(edge, owned)
      if not owned_node or not new_node:
          continue


      # Team coordination: skip nodes locked by another runner recently.
      lock = get_node_lock(new_node)
      if lock and lock.get("owner") and lock.get("owner") != owner_id:
          # If someone else recently touched this node, avoid it; let the TTL handle staleness.
          print(f"  ⊗ Skipping edge to {new_node} (locked by {lock.get('owner')} status={lock.get('status')})")
          continue
      if lock and lock.get("status") in {"skipped", "cooldown"} and not ignore_own_skips:
          # Cooldown markers are advisory for our own runner; for other runners they are hard blocks.
          if lock.get("owner") and lock.get("owner") != owner_id:
              print(f"  ⊗ Skipping edge to {new_node} (cooldown status={lock.get('status')} owner={lock.get('owner')})")
              continue


      new_info = gi.nodes.get(new_node, {})
      owned_info = gi.nodes.get(owned_node, {})


      new_utility = int(new_info.get("utility_qubits", 0) or 0)
      new_bonus = int(new_info.get("bonus_bell_pairs", 0) or 0)
      owned_bonus = int(owned_info.get("bonus_bell_pairs", 0) or 0)


      base_threshold = float(edge.get("base_threshold", 1.0) or 1.0)
      difficulty = float(edge.get("difficulty_rating", 0) or 0)


      d_bonus = dist_to_bonus.get(new_node, inf)


      ease = -6.0 * base_threshold - 0.75 * difficulty
      immediate_value = (1.5 * new_utility) + (bonus_weight * new_bonus)
      path_value = -dist_weight * (d_bonus if d_bonus != inf else 50)
      defense_value = defense_weight * owned_bonus
      score = immediate_value + path_value + defense_value + ease


      safe_to_spend = can_afford_success(budget=budget, success_cost=int(assumed_success_cost))
      risky = is_risky_spend(
          budget=budget,
          owned_bonus_total=owned_bonus_total,
          success_cost=int(assumed_success_cost),
          min_safety_margin=8,
      )


      ranked.append(
          {
              "edge_id": tuple(edge["edge_id"]),
              "owned_node": owned_node,
              "new_node": new_node,
              "new_utility": new_utility,
              "new_bonus": new_bonus,
              "owned_bonus": owned_bonus,
              "base_threshold": base_threshold,
              "difficulty_rating": difficulty,
              "score": float(score),
              "budget": budget,
              "owned_bonus_total": owned_bonus_total,
              "safety_margin": safety,
              "recommended_success_cost": int(assumed_success_cost),
              "safe_to_spend_now": bool(safe_to_spend),
              "risky_spend": bool(risky),
              "raw_edge": edge,
          }
      )


  ranked.sort(key=lambda r: r["score"], reverse=True)
  return ranked




status = client.get_status()
budget = int(status.get('budget', 0) or 0)
assumed_cost = 1 if budget <= 2 else 2


# MAIN GAME LOOP - Run until we're out of bell pairs
print("\n" + "="*60)
print("STARTING MAIN GAME LOOP")
print("="*60)


_edge_fail_counts = {}
MAX_FAILS_BEFORE_SKIP = 2  # Configurable threshold
iteration = 0
LOCK_TTL_SECONDS = int(os.getenv("IONQ_LOCK_TTL_SECONDS", "900") or 900)  # 15 min
SKIP_TTL_SECONDS = int(os.getenv("IONQ_SKIP_TTL_SECONDS", "600") or 600)  # 10 min
NODE_RETRY_COOLDOWN_SECONDS = int(os.getenv("IONQ_NODE_RETRY_COOLDOWN_SECONDS", "120") or 120)  # 2 min
_node_cooldown_until: Dict[str, float] = {}
_node_recent_until: Dict[str, float] = {}


while True:
   iteration += 1
   status = client.get_status()
   budget = int(status.get('budget', 0) or 0)


   # Debug: verify whether "no claimable edges" is server-side or local filtering.
   try:
       owned_nodes = status.get("owned_nodes", []) or []
       print(f"[debug] owned_nodes={len(owned_nodes)} is_active={status.get('is_active')} budget={budget}")
       print(f"[debug] server_claimable_edges={len(client.get_claimable_edges())}")
   except Exception as e:
       print(f"[debug] claimable/owned debug failed: {e}")
  
   print(f"\n{'='*60}")
   print(f"ITERATION {iteration} - Budget: {budget}")
   print(f"{'='*60}")
  
   # Check if we're out of budget
   if budget <= 0:
       print("\n✗ OUT OF BUDGET! Game over.")
       break
  
   # Get ranked edges (this now filters out owned nodes automatically)
   assumed_cost = 1 if budget <= 2 else 2
   ranked = rank_claimable_edges(client, assumed_success_cost=assumed_cost)
  
   if not ranked:
       print("No claimable edges (all reachable nodes are owned or skipped). Game over.")
       break
  
   # Show top edges
   meta = ranked[0]
   print(f"Budget={meta['budget']} safety={meta['safety_margin']} bonus_total={meta['owned_bonus_total']}")
   print(f"\nTop {min(5, len(ranked))} edges:")
   for i, r in enumerate(ranked[:5]):
       print(
           f"{i:2d}: {r['edge_id']} thr={r['base_threshold']:.3f} diff={r['difficulty_rating']} "
           f"U={r['new_utility']} +B={r['new_bonus']} safe={r['safe_to_spend_now']}"
       )
  
   # Filter to spendable edges
   now = time.time()
   spendable = [r for r in ranked if r.get('safe_to_spend_now')]
   # Prefer nodes not currently in our local cooldown window.
   non_cooldown = [
       r for r in spendable if float(_node_cooldown_until.get(r["new_node"], 0) or 0) <= now
   ]
   if non_cooldown:
       spendable = non_cooldown
   # Local anti-stall: prefer edges we haven't just targeted.
   non_recent = [r for r in spendable if float(_node_recent_until.get(r["new_node"], 0) or 0) <= now]
   if non_recent:
       spendable = non_recent
   if not spendable:
       print("\n✗ No spendable edges (budget too low). Game over.")
       break
  
   # Target the best spendable edge
   target = spendable[0]
   edge_id = tuple(target['edge_id'])
   new_node = target['new_node']
   owner_id = str(getattr(client, "player_id", "") or "")
  
   print(f"\n→ Targeting edge {edge_id} thr={target['base_threshold']:.3f} -> {new_node}")


   # In-run cooldown: don't hammer the same node forever even if we're the locker.
   cooldown_until = float(_node_cooldown_until.get(new_node, 0) or 0)
   if cooldown_until > now:
       print(f"⚠ Node {new_node} in local cooldown for {int(cooldown_until - now)}s; skipping.")
       continue


   # Coordinate with teammates: lock the node before spending time probing/attempting.
   if owner_id:
       lock = get_node_lock(new_node)
       if lock and lock.get("status") in {"skipped", "cooldown"}:
           if lock.get("owner") and lock.get("owner") != owner_id:
               print(f"⚠ Node {new_node} is in cooldown (status={lock.get('status')}); skipping.")
               continue
       if not try_lock_node(new_node, owner=owner_id, status="attempting", ttl_seconds=LOCK_TTL_SECONDS):
           print(f"⚠ Node {new_node} is locked by another runner; skipping.")
           continue
  
   # Check current fail count for this edge
   current_fails = int(_edge_fail_counts.get(edge_id, 0) or 0)
   if current_fails >= MAX_FAILS_BEFORE_SKIP:
       print(f"⚠ Edge {edge_id} has failed {current_fails} times - marking node {new_node} as skipped (team TTL)")
       if owner_id:
           try_lock_node(new_node, owner=owner_id, status="skipped", ttl_seconds=SKIP_TTL_SECONDS)
       _node_cooldown_until[new_node] = time.time() + float(SKIP_TTL_SECONDS)
       _node_recent_until[new_node] = time.time() + float(SKIP_TTL_SECONDS)
       _edge_fail_counts[edge_id] = 0  # Reset counter for this edge
       continue  # Move to next iteration
  
   # Knobs
   ENABLE_RECON = True
   RECON_MARGIN = 0.05
   FAILS_BEFORE_ESCALATE = 10
  
   bias = 'mixed'
   if ENABLE_RECON:
       recon = recon_noise_type_for_edge(client, edge_id, margin=RECON_MARGIN)
       if recon.get('claimed'):
           print('✓ Claimed via probe!')
           # Update database: mark the new node as owned
           update_node_owned(new_node, owned=1)
           if owner_id:
               try_lock_node(new_node, owner=owner_id, status="owned", ttl_seconds=10**9)
           _edge_fail_counts[edge_id] = 0  # Reset fail counter on success
           _node_recent_until[new_node] = time.time() + float(NODE_RETRY_COOLDOWN_SECONDS)
           continue  # Skip to next iteration
       else:
           bias = recon.get('bias', 'mixed')
   else:
       recon = {'claimed': False}
  
   # Refresh budget after recon (in case probes spent budget)
   status = client.get_status()
   budget = int(status.get('budget', 0) or 0)
  
   allow_n2 = budget > 2
   allow_n3 = budget > 3
   allow_n4 = budget > 4
  
   if not allow_n2:
       print('✗ Budget too low for N=2 after recon. Stopping.')
       break
  
   fails = int(_edge_fail_counts.get(edge_id, 0) or 0)
   use_n4 = (fails >= FAILS_BEFORE_ESCALATE) and allow_n4
   if (fails >= FAILS_BEFORE_ESCALATE) and not allow_n4:
       print(f"⚠ Escalate blocked (fails={fails}, budget={budget}).")
  
   num_bell_pairs = 4 if use_n4 else 2
  
   if bias == 'x':
       attempt_labels = ['bit', 'bit3r', 'dejmps-bit3r', 'syn3-bit-and2', 'dejmps-bit', 'phase', 'syn3-phase-and2']
   elif bias == 'z':
       attempt_labels = ['phase', 'phase3r', 'dejmps-phase3r', 'syn3-phase-and2', 'dejmps-phase', 'bit', 'syn3-bit-and2']
   else:
       attempt_labels = ['dejmps-bit3r', 'syn3-bit-and2', 'dejmps-phase3r', 'syn3-phase-and2', 'bit3r', 'phase3r', 'dejmps-bit', 'dejmps-phase', 'bit', 'phase']
  
   any_success = False
   full_miss = True
  
   for label in attempt_labels:
       if label == 'bit':
           attempt_n = num_bell_pairs
           circuit, flag_bit = create_purification_circuit(attempt_n, variant='bit', rounds=1)
       elif label == 'bit3r':
           if not allow_n3:
               continue
           attempt_n = 3
           circuit, flag_bit = create_purification_circuit(attempt_n, variant='bit', rounds='max')
       elif label == 'phase':
           attempt_n = num_bell_pairs
           circuit, flag_bit = create_purification_circuit(attempt_n, variant='phase', rounds=1)
       elif label == 'phase3r':
           if not allow_n3:
               continue
           attempt_n = 3
           circuit, flag_bit = create_purification_circuit(attempt_n, variant='phase', rounds='max')
       elif label == 'dejmps-bit':
           attempt_n = num_bell_pairs
           circuit, flag_bit = create_dejmps_circuit(attempt_n, orientation='bit', rounds=1)
       elif label == 'dejmps-bit3r':
           if not allow_n3:
               continue
           attempt_n = 3
           circuit, flag_bit = create_dejmps_circuit(attempt_n, orientation='bit', rounds='max')
       elif label == 'dejmps-phase':
           attempt_n = num_bell_pairs
           circuit, flag_bit = create_dejmps_circuit(attempt_n, orientation='phase', rounds=1)
       elif label == 'dejmps-phase3r':
           if not allow_n3:
               continue
           attempt_n = 3
           circuit, flag_bit = create_dejmps_circuit(attempt_n, orientation='phase', rounds='max')
       elif label == 'syn3-bit-and2':
           attempt_n = 3
           circuit, flag_bit = create_three_pair_syndrome_circuit(orientation='bit', postselect='and2')
       elif label == 'syn3-phase-and2':
           attempt_n = 3
           circuit, flag_bit = create_three_pair_syndrome_circuit(orientation='phase', postselect='and2')
       else:
           raise ValueError(f'Unknown attempt label: {label}')


       # Avoid a successful claim that would end the game (budget <= 0).
       if (budget - int(attempt_n)) <= 0:
           print(f"  Skip {label} N={attempt_n} (budget={budget} would go <= 0 on success)")
           continue
      
       print(f"  Try {label} N={attempt_n}")
       result = client.claim_edge(edge_id, circuit, flag_bit, attempt_n)
      
       if not result.get('ok'):
           print(f"  ✗ Error: {result.get('error', {}).get('message')}")
           full_miss = False
           continue
      
       data = result['data']
       print(
           f"    success={data.get('success')} F={data.get('fidelity', 0):.4f} "
           f"P={data.get('success_probability', 0):.4f} thr={data.get('threshold', 0):.4f}"
       )
      
       if data.get('success'):
           any_success = True
           full_miss = False
           print(f"  ✓ SUCCESS with {label}!")
          
           # Update database: mark the new node as owned
           update_node_owned(new_node, owned=1)
           if owner_id:
               try_lock_node(new_node, owner=owner_id, status="owned", ttl_seconds=10**9)
           _edge_fail_counts[edge_id] = 0  # Reset fail counter on success
           _node_recent_until[new_node] = time.time() + float(NODE_RETRY_COOLDOWN_SECONDS)
          
           break
  
   if any_success:
       # Already reset in the success block above
       pass
   elif full_miss:
       _edge_fail_counts[edge_id] = fails + 1
       print(f"  ⚠ All attempts failed. fails[{edge_id}]={_edge_fail_counts[edge_id]}")
       _node_recent_until[new_node] = time.time() + float(NODE_RETRY_COOLDOWN_SECONDS)
       if owner_id:
           # Release the "attempting" lock into a short cooldown so teammates aren't blocked for 15 minutes.
           try_lock_node(new_node, owner=owner_id, status="cooldown", ttl_seconds=NODE_RETRY_COOLDOWN_SECONDS)
      
       # Check if we've hit the max failure threshold
       if _edge_fail_counts[edge_id] >= MAX_FAILS_BEFORE_SKIP:
           print(f"  ⚠ Reached {MAX_FAILS_BEFORE_SKIP} failures - will skip this node next iteration")
           if owner_id:
               try_lock_node(new_node, owner=owner_id, status="skipped", ttl_seconds=SKIP_TTL_SECONDS)
           _node_cooldown_until[new_node] = time.time() + float(SKIP_TTL_SECONDS)


print("\n" + "="*60)
print("GAME LOOP COMPLETE")
print("="*60)




client.print_status()





