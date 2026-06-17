"""
qrc_core.py
===========
Three simulation backends, selectable at QuantumReservoir construction time:

  backend="statevector"   — exact Qiskit Statevector (original behaviour, no shots)
  backend="aer_ideal"     — AerSimulator, shot-based but zero noise
                            (lets you verify Aer matches statevector before adding noise)
  backend="aer_noise"     — AerSimulator + NoiseModel
                            (depolarizing gate error + readout error, both controllable)

Public API is unchanged from the original:
  build_observables(n_qubits)                          → list[SparsePauliOp]
  QuantumReservoir(n_qubits, n_layers, seeds, ...)
      .get_combined_latent(input_sequence)             → np.ndarray
  apply_hankel(latent_history, window)                 → np.ndarray
  prepare_latent_history(inputs, reservoir, lookback)  → np.ndarray
  prepare_hankel_features(inputs, reservoir, lookback, window) → np.ndarray

Noise parameters (only used when backend="aer_noise")
------------------------------------------------------
  depolar_error : float   — single-qubit depolarizing probability  (e.g. 0.01 = 1 %)
  cx_error      : float   — two-qubit (CX) depolarizing probability (e.g. 0.02 = 2 %)
  readout_error : float   — symmetric bit-flip probability on measurement (e.g. 0.01)
  shots         : int     — number of measurement shots for Aer (default 1024)

Notes
-----
- Statevector backend computes exact expectation values analytically (no shots).
- Aer backends estimate expectation values from shot counts; more shots → less
  shot noise, at the cost of runtime.
- All random reservoir weights are drawn per-seed then the global seed is restored,
  so downstream experiments remain deterministic.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, SparsePauliOp

# Aer imports — guarded so the file still imports cleanly if qiskit-aer is absent
try:
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import (
        NoiseModel,
        depolarizing_error,
        ReadoutError,
    )
    _AER_AVAILABLE = True
except ImportError:
    _AER_AVAILABLE = False


# ============================================================
# OBSERVABLES
# ============================================================

def build_observables(n_qubits: int) -> list:
    """
    Build single-qubit Z and two-qubit ZZ Pauli observables for all qubits.

    Returns
    -------
    list[SparsePauliOp]
        Length = n_qubits  +  n_qubits*(n_qubits-1)//2
    """
    obs = []

    # Single-qubit Z
    for q in range(n_qubits):
        pauli_str    = ['I'] * n_qubits
        pauli_str[q] = 'Z'
        obs.append(SparsePauliOp(''.join(reversed(pauli_str))))

    # Two-qubit ZZ
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            pauli_str    = ['I'] * n_qubits
            pauli_str[i] = 'Z'
            pauli_str[j] = 'Z'
            obs.append(SparsePauliOp(''.join(reversed(pauli_str))))

    return obs


# ============================================================
# NOISE MODEL FACTORY
# ============================================================

def build_noise_model(
    depolar_error: float = 0.01,
    cx_error:      float = 0.02,
    readout_error: float = 0.01,
) -> "NoiseModel":
    """
    Build a simple, physically motivated noise model for AerSimulator.

    Noise channels applied
    ----------------------
    1. Depolarizing error after every single-qubit gate (RY, RZ, U)
       — probability `depolar_error`
    2. Depolarizing error after every two-qubit gate (CX)
       — probability `cx_error`
    3. Symmetric readout (bit-flip) error on every qubit measurement
       — probability `readout_error`

    Parameters
    ----------
    depolar_error : float   0 → no single-qubit noise
    cx_error      : float   0 → no two-qubit noise
    readout_error : float   0 → perfect measurements

    Returns
    -------
    NoiseModel
    """
    if not _AER_AVAILABLE:
        raise ImportError(
            "qiskit-aer is not installed. Run:  pip install qiskit-aer"
        )

    nm = NoiseModel()

    # 1. Single-qubit gate depolarizing noise
    if depolar_error > 0:
        err_1q = depolarizing_error(depolar_error, 1)
        nm.add_all_qubit_quantum_error(err_1q, ["ry", "rz", "u"])

    # 2. Two-qubit (CX) depolarizing noise
    if cx_error > 0:
        err_2q = depolarizing_error(cx_error, 2)
        nm.add_all_qubit_quantum_error(err_2q, ["cx"])

    # 3. Readout (measurement) error — symmetric bit-flip
    if readout_error > 0:
        p0g1 = readout_error          # P(read 0 | state is 1)
        p1g0 = readout_error          # P(read 1 | state is 0)
        ro_err = ReadoutError([[1 - p1g0, p1g0],
                               [p0g1,     1 - p0g1]])
        nm.add_all_qubit_readout_error(ro_err)

    return nm


# ============================================================
# QUANTUM RESERVOIR
# ============================================================

class QuantumReservoir:
    """
    Ensemble of fixed-weight quantum reservoirs.

    Parameters
    ----------
    n_qubits      : int
    n_layers      : int
    seeds         : list[int]
    global_seed   : int, optional
    backend       : str
        "statevector"  — exact simulation (original, default)
        "aer_ideal"    — Aer shot-based, zero noise
        "aer_noise"    — Aer shot-based + NoiseModel
    shots         : int     — shots for Aer backends (ignored for statevector)
    depolar_error : float   — single-qubit depolarizing prob (aer_noise only)
    cx_error      : float   — CX gate depolarizing prob       (aer_noise only)
    readout_error : float   — measurement bit-flip prob        (aer_noise only)
    """

    def __init__(
        self,
        n_qubits:      int,
        n_layers:      int,
        seeds:         list,
        global_seed:   int   = 42,
        backend:       str   = "statevector",
        shots:         int   = 1024,
        depolar_error: float = 0.0,
        cx_error:      float = 0.0,
        readout_error: float = 0.0,
    ):
        self.n_qubits      = n_qubits
        self.n_layers      = n_layers
        self.observables   = build_observables(n_qubits)
        self.backend_name  = backend
        self.shots         = shots
        self.depolar_error = depolar_error
        self.cx_error      = cx_error
        self.readout_error = readout_error

        # Validate backend
        valid_backends = {"statevector", "aer_ideal", "aer_noise"}
        if backend not in valid_backends:
            raise ValueError(f"backend must be one of {valid_backends}, got '{backend}'")
        if backend in ("aer_ideal", "aer_noise") and not _AER_AVAILABLE:
            raise ImportError(
                "qiskit-aer is not installed. Run:  pip install qiskit-aer"
            )

        # Build Aer simulator once (reused across all circuits)
        if backend == "aer_ideal":
            self._aer_sim = AerSimulator(method="statevector")
        elif backend == "aer_noise":
            noise_model    = build_noise_model(depolar_error, cx_error, readout_error)
            self._aer_sim  = AerSimulator(
                method="density_matrix",   # density_matrix handles mixed states from noise
                noise_model=noise_model,
            )
        else:
            self._aer_sim = None  # statevector path uses Qiskit directly

        # Build reservoir weight matrices
        self.reservoirs = []
        for seed in seeds:
            np.random.seed(seed)
            weights = np.random.uniform(
                0, 2 * np.pi,
                size=(n_layers, n_qubits)
            )
            self.reservoirs.append({"weights": weights, "seed": seed})

        np.random.seed(global_seed)   # restore global state

    # ----------------------------------------------------------
    # Circuit construction  (shared by all backends)
    # ----------------------------------------------------------

    def _encode_step(self, qc: QuantumCircuit, step) -> None:
        """Encode one input step (scalar or 2-vector) via RY gates."""
        if hasattr(step, '__len__') and len(step) >= 2:
            val     = float(step[0])
            aux_val = float(step[1])
        else:
            val     = float(step)
            aux_val = 0.0

        half = self.n_qubits // 2
        for q in range(half):
            qc.ry(np.pi * val, q)
        for q in range(half, self.n_qubits):
            qc.ry(np.pi * aux_val, q)

    def _apply_reservoir_layer(self, qc: QuantumCircuit, weights: np.ndarray) -> None:
        """Apply all Rz rotations and CNOT entanglement for one reservoir."""
        for layer in range(self.n_layers):
            for q in range(self.n_qubits):
                qc.rz(weights[layer, q], q)
            for q in range(self.n_qubits - 1):
                qc.cx(q, q + 1)

    def build_circuit(self, input_sequence, weights: np.ndarray) -> QuantumCircuit:
        """
        Build a full reservoir circuit for a time-ordered input sequence.
        Adds measurements when using an Aer backend.
        """
        qc = QuantumCircuit(self.n_qubits)
        for step in input_sequence:
            self._encode_step(qc, step)
            self._apply_reservoir_layer(qc, weights)

        # Aer backends need explicit measurements to produce counts
        if self.backend_name in ("aer_ideal", "aer_noise"):
            qc.measure_all()

        return qc

    # ----------------------------------------------------------
    # Feature extraction  (backend-specific)
    # ----------------------------------------------------------

    def extract_latent_state(self, qc: QuantumCircuit) -> np.ndarray:
        """
        Compute expectation values of all observables.

        Statevector: exact analytical expectation values (no shots, no noise).
        Aer:         estimated from measurement counts over `self.shots` shots.
                     Each Pauli expectation value  <Z_i>  is estimated as
                         (N_0 - N_1) / (N_0 + N_1)
                     where N_0, N_1 are counts of the corresponding bit being 0 or 1.

        Returns
        -------
        np.ndarray, shape (n_observables,)
        """
        if self.backend_name == "statevector":
            return self._extract_statevector(qc)
        else:
            return self._extract_aer(qc)

    def _extract_statevector(self, qc: QuantumCircuit) -> np.ndarray:
        """Exact expectation values via Statevector (original method)."""
        sv = Statevector(qc)
        return np.array(
            [float(sv.expectation_value(op).real) for op in self.observables]
        )

    def _extract_aer(self, qc: QuantumCircuit) -> np.ndarray:
        """
        Shot-based expectation values from AerSimulator.

        Strategy
        --------
        Run the circuit once with `shots` shots.
        From the resulting bitstring counts, estimate each Pauli expectation:

          <Z_q>  = (counts with bit q == 0  -  counts with bit q == 1) / total_shots
          <ZZ_ij> = (counts where bits i,j agree  -  counts where they disagree) / total_shots
        """
        from qiskit import transpile

        t_qc    = transpile(qc, self._aer_sim)
        job     = self._aer_sim.run(t_qc, shots=self.shots)
        counts  = job.result().get_counts()
        total   = sum(counts.values())

        # Parse each bitstring into a numpy array (bit order: rightmost = qubit 0)
        # bitstring "0110" → bit array [0, 1, 1, 0] where index 0 = qubit 0
        bit_matrix = np.zeros((len(counts), self.n_qubits), dtype=np.int8)
        weights_   = np.zeros(len(counts), dtype=np.float64)

        for row, (bitstr, cnt) in enumerate(counts.items()):
            # Qiskit counts keys: leftmost char = highest qubit index
            # Reverse so index 0 → qubit 0
            bits = [int(b) for b in reversed(bitstr.replace(" ", ""))]
            # Pad or trim to n_qubits (safety)
            bits = bits[:self.n_qubits] + [0] * max(0, self.n_qubits - len(bits))
            bit_matrix[row, :] = bits
            weights_[row]       = cnt / total

        # +1 for |0>, -1 for |1>  (eigenvalues of Z)
        z_vals = 1 - 2 * bit_matrix.astype(np.float64)   # shape: (n_bitstrings, n_qubits)

        expectations = []

        # Single-qubit Z observables
        for q in range(self.n_qubits):
            exp_val = np.dot(weights_, z_vals[:, q])
            expectations.append(exp_val)

        # Two-qubit ZZ observables
        for i in range(self.n_qubits):
            for j in range(i + 1, self.n_qubits):
                exp_val = np.dot(weights_, z_vals[:, i] * z_vals[:, j])
                expectations.append(exp_val)

        return np.array(expectations)

    # ----------------------------------------------------------
    # Ensemble  (unchanged)
    # ----------------------------------------------------------

    def get_combined_latent(self, input_sequence) -> np.ndarray:
        """
        Run all reservoirs in the ensemble and concatenate their latent vectors.

        Returns
        -------
        np.ndarray, shape (n_reservoirs × n_observables,)
        """
        all_features = []
        for res in self.reservoirs:
            qc  = self.build_circuit(input_sequence, res["weights"])
            z_t = self.extract_latent_state(qc)
            all_features.append(z_t)
        return np.concatenate(all_features)


# ============================================================
# FEATURE PREPARATION UTILITIES  (unchanged)
# ============================================================

def prepare_latent_history(
    inputs,
    reservoir: QuantumReservoir,
    lookback: int,
) -> np.ndarray:
    """
    Slide a lookback window over `inputs` and extract one latent vector per step.
    """
    inputs = np.asarray(inputs)
    latent = []
    for i in range(lookback, len(inputs)):
        seq = inputs[i - lookback:i]
        latent.append(reservoir.get_combined_latent(seq))
    return np.array(latent)


def apply_hankel(latent_history: np.ndarray, window: int) -> np.ndarray:
    """
    Stack consecutive latent vectors into Hankel (delay-embedding) features.
    """
    hankel = []
    for i in range(len(latent_history) - window + 1):
        hankel.append(latent_history[i:i + window].flatten())
    return np.array(hankel)


def prepare_hankel_features(
    inputs,
    reservoir: QuantumReservoir,
    lookback: int,
    window: int,
) -> np.ndarray:
    """
    Convenience wrapper: latent history extraction + Hankel stacking in one call.
    """
    latent_history = prepare_latent_history(inputs, reservoir, lookback)
    return apply_hankel(latent_history, window)
