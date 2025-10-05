import os
import io
import time
import random
import string
import hashlib
import hmac
import zipfile
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

# ---------------------------
# Configuración general
# ---------------------------
st.set_page_config(
    page_title="UD2 — Fundamentos técnicos: hash, bloque, cadena",
    page_icon="⛓️",
    layout="wide",
)

# Tema oscuro: pequeños ajustes de contraste
st.markdown("""
<style>
pre, code, .stCode, .stMarkdown code { background:#0f1b2d !important; color:#e5e7eb !important; }
[data-testid="stTable"] th, [data-testid="stDataFrame"] thead th { background:#0f172a !important; color:#e5e7eb !important; }
</style>
""", unsafe_allow_html=True)

os.makedirs("entregas", exist_ok=True)
os.makedirs("materiales", exist_ok=True)

# ---------------------------
# Utilidades comunes
# ---------------------------
def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

def alter_one_char(text: str) -> str:
    if not text:
        return text
    pos = random.randrange(len(text))
    alphabet = string.ascii_letters + string.digits + " .,-_:;()¿?¡!/'\""
    new_char = random.choice(alphabet)
    while new_char == text[pos]:
        new_char = random.choice(alphabet)
    return text[:pos] + new_char + text[pos+1:]

def download_csv_button(df: pd.DataFrame, label: str, filename: str):
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    st.download_button(label, buf.getvalue(), file_name=filename, mime="text/csv")

def _zip_folder_md(folder_path: str) -> io.BytesIO:
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.endswith(".md"):
                    full_path = os.path.join(root, file)
                    arcname = os.path.relpath(full_path, folder_path)
                    zf.write(full_path, arcname=arcname)
    mem.seek(0)
    return mem

def _list_md_files(folder: str):
    if not os.path.isdir(folder):
        return []
    return sorted([f for f in os.listdir(folder) if f.endswith(".md")])

def _delete_md_in_folder(folder: str) -> int:
    if not os.path.isdir(folder):
        return 0
    count = 0
    for f in os.listdir(folder):
        if f.endswith(".md"):
            try:
                os.remove(os.path.join(folder, f))
                count += 1
            except Exception:
                pass
    return count

# ---------------------------
# PoW helpers
# ---------------------------
def block_hash(index: int, timestamp_iso: str, data: str, prev_hash: str, nonce: int) -> str:
    payload = f"{index}|{timestamp_iso}|{data}|{prev_hash}|{nonce}"
    return sha256_hex(payload)

def mine_nonce(index: int, timestamp_iso: str, data: str, prev_hash: str, difficulty: int, max_iters: int = 2_000_000):
    """Busca un nonce tal que el hash empiece por '0'*difficulty. Devuelve (nonce, hash, iters, seconds)."""
    assert difficulty >= 1
    target = "0" * difficulty
    start = time.time()
    nonce = 0
    for i in range(max_iters):
        h = block_hash(index, timestamp_iso, data, prev_hash, nonce)
        if h.startswith(target):
            elapsed = time.time() - start
            return nonce, h, i + 1, elapsed
        nonce += 1
    elapsed = time.time() - start
    return None, None, max_iters, elapsed  # no encontrado en límite

# ---------------------------
# PoS (simulado) helpers
# ---------------------------
def ensure_validators():
    """Inicializa un conjunto de validadores con stake fijo (simulado)."""
    if "validators" in st.session_state:
        return
    names = ["ValA", "ValB", "ValC", "ValD"]
    stakes = [40, 30, 20, 10]  # total 100
    vals = []
    for n, s in zip(names, stakes):
        secret = os.urandom(16).hex()
        pub = sha256_hex(secret)  # simulación
        vals.append({"id": n, "name": n, "stake": s, "secret": secret, "pub": pub})
    st.session_state.validators = vals

def total_stake():
    return sum(v["stake"] for v in st.session_state.validators)

def pos_header(index: int, timestamp_iso: str, data: str, prev_hash: str) -> str:
    """Cabecera firmada en PoS (nonce fijo 0 para simplificar)."""
    return f"{index}|{timestamp_iso}|{data}|{prev_hash}|0"

def pos_hmac(header: str, secret_hex: str) -> str:
    return hmac.new(bytes.fromhex(secret_hex), header.encode("utf-8"), hashlib.sha256).hexdigest()

def pos_select_signers(threshold_frac: float):
    """Selecciona validadores por stake descendente hasta cubrir el umbral."""
    vals = sorted(st.session_state.validators, key=lambda x: -x["stake"])
    need = threshold_frac * total_stake()
    got = 0
    chosen = []
    for v in vals:
        if got >= need:
            break
        chosen.append(v)
        got += v["stake"]
    return chosen, got

def pos_sign_block(index: int, timestamp_iso: str, data: str, prev_hash: str, threshold_frac: float):
    """Devuelve (signers_ids, stake_signed, signatures_dict)."""
    header = pos_header(index, timestamp_iso, data, prev_hash)
    chosen, got = pos_select_signers(threshold_frac)
    sigs = {v["id"]: pos_hmac(header, v["secret"]) for v in chosen}
    return [v["id"] for v in chosen], got, sigs

def stake_of(ids):
    stake_map = {v["id"]: v["stake"] for v in st.session_state.validators}
    return sum(stake_map.get(i, 0) for i in ids)

# ---------------------------
# Datos demo
# ---------------------------
@st.cache_data
def load_tx_demo():
    try:
        return pd.read_csv("data/tx_demo.csv")
    except Exception:
        return pd.DataFrame({
            "id": [1, 2, 3],
            "descripcion": ["Pago colegiatura", "Tasa registral", "Honorarios"],
            "importe": [1200.0, 85.5, 350.0],
        })

tx_df = load_tx_demo()
ensure_validators()

# ---------------------------
# Estado inicial
# ---------------------------
if "s1_text" not in st.session_state:
    r = tx_df.iloc[0]
    st.session_state.s1_text = f"{int(r['id'])}: {r['descripcion']} — {float(r['importe']):.2f} EUR"

if "chain" not in st.session_state:
    genesis = {
        "index": 0,
        "timestamp": now_iso(),
        "data": "GENESIS",
        "prev_hash": "0" * 64,
        "nonce": 0,
        "hash": "",      # se calcula abajo
        "mode": "genesis",
        "difficulty": None,
        "pos_signers": [],
        "pos_threshold": None,
        "pos_signed_stake": 0
    }
    genesis["hash"] = block_hash(genesis["index"], genesis["timestamp"], genesis["data"], genesis["prev_hash"], genesis["nonce"])
    st.session_state.chain = [genesis]

if "consensus_mode" not in st.session_state:
    st.session_state.consensus_mode = "PoW-lite"

# ---------------------------
# Encabezado
# ---------------------------
st.title("UD2 — Fundamentos técnicos: hash, bloque, cadena")
st.caption("Objetivo: comprender cómo se construye la inmutabilidad (hash + nonce + timestamp + enlace)")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Hash", "SHA-256", delta="Efecto avalancha")
c2.metric("Nonce", "PoW-lite", delta="Dificultad")
c3.metric("Bloque", "Timestamp+PrevHash", delta="Enlace")
c4.metric("Cadena", "Validez", delta="Re-minado/Revalidación")
st.divider()

# Selector global de modo de consenso
st.subheader("Modo de consenso")
st.radio(
    "Elige cómo se validan los bloques que añadas en esta sesión",
    options=["PoW-lite", "PoS (simulado)"],
    index=0 if st.session_state.consensus_mode == "PoW-lite" else 1,
    key="consensus_mode",
    horizontal=True,
    help="PoW-lite: se busca nonce con dificultad de ceros. PoS: se requiere umbral de firmas por stake."
)

tabs = st.tabs([
    "1) Teoría",
    "2) S1 — Hash con nonce (PoW-lite)",
    "3) S2 — Construye una mini-cadena",
    "4) Visualización/Validación",
    "5) Lecturas guiadas",
    "6) Entregables y rúbrica"
])

# ---------------------------
# 1) Teoría
# ---------------------------
with tabs[0]:
    st.subheader("Qué hace inmutable a un bloque")
    st.markdown(
        """
- **Hash (SHA-256):** huella del contenido. Un cambio mínimo → hash totalmente distinto (**avalancha**).
- **Nonce (PoW):** número que ajustamos para que el hash cumpla un patrón (p. ej., empiece con *n* ceros).
- **Dificultad (PoW):** cuántos ceros iniciales exige la red. Mayor dificultad → más intentos esperados.
- **Timestamp:** fija el momento de creación del bloque.
- **Prev_hash:** cada bloque **enlaza** con el anterior. Si cambias el pasado, cambian los hashes futuros.
- **PoS (simulado):** la validez exige un **quórum de firmas** ponderadas por **stake** (no minado).
- **Inmutabilidad práctica:** alterar un bloque exige re-minar (PoW) o revalidar firmas (PoS) **ese** bloque y **todos los siguientes**.
"""
    )
    st.markdown("### Prueba rápida: hash de un texto")
    demo = st.text_area("Texto", "Bloque de ejemplo", height=90)
    st.code(sha256_hex(demo), language="bash")
    st.info("Cambia una letra y observa el hash. Eso es el **efecto avalancha**.")

# ---------------------------
# 2) S1 — Hash con nonce (PoW-lite)
# ---------------------------
with tabs[1]:
    st.header("S1 — Modifica el dato y observa el hash con/ sin nonce")
    left, right = st.columns([1, 1])

    with left:
        st.markdown("#### 2.1 Dato base (puedes editarlo)")
        pick = st.selectbox(
            "Cargar desde dataset demo:",
            options=tx_df["id"].tolist(),
            format_func=lambda i: f"ID {i} — {tx_df[tx_df['id']==i]['descripcion'].values[0]}"
        )
        if st.button("📥 Cargar selección"):
            r = tx_df[tx_df["id"] == pick].iloc[0]
            st.session_state.s1_text = f"{int(r['id'])}: {r['descripcion']} — {float(r['importe']):.2f} EUR"

        st.text_area("Dato (editable):", key="s1_text", height=100)
        if st.button("🔁 Alterar 1 carácter"):
            st.session_state.s1_text = alter_one_char(st.session_state.s1_text)

        st.markdown("#### 2.2 Hash simple (sin nonce)")
        st.code(sha256_hex(st.session_state.s1_text))

    with right:
        st.markdown("#### 2.3 Prueba de trabajo (lite)")
        difficulty = st.slider("Dificultad (ceros iniciales)", 1, 5, 3)
        ts = now_iso()
        idx = 1
        prev = "x" * 64  # en S1 no encadenamos; sólo demostración PoW

        if st.button("⛏️ Minar (encontrar nonce)"):
            nonce, h, iters, secs = mine_nonce(idx, ts, st.session_state.s1_text, prev, difficulty)
            if nonce is not None:
                st.success(f"Nonce: {nonce} · Hash: {h[:20]}... · Intentos: {iters:,} · Tiempo: {secs:.2f}s")
                st.session_state.s1_mined = {
                    "index": idx, "timestamp": ts, "data": st.session_state.s1_text,
                    "prev_hash": prev, "nonce": nonce, "hash": h,
                    "difficulty": difficulty, "iters": iters, "secs": secs
                }
            else:
                st.error("No se encontró nonce en el límite de iteraciones.")

        mined = st.session_state.get("s1_mined")
        if mined:
            st.json(mined, expanded=False)

    st.markdown("---")
    st.caption("**Logro S1:** Explica en 5 líneas cómo **nonce + dificultad** contribuyen a la **inmutabilidad práctica**.")

# ---------------------------
# 3) S2 — Construye una mini-cadena (PoW o PoS)
# ---------------------------
with tabs[2]:
    st.header("S2 — Construcción de bloque y encadenamiento")

    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("#### 3.1 Añadir un bloque")
        data_input = st.text_area("Datos del bloque (JSON/Texto libre):", "['tx1','tx2','tx3']", height=100, key="s2_data")

        if st.session_state.consensus_mode == "PoW-lite":
            difficulty2 = st.slider("Dificultad (ceros iniciales)", 1, 5, 3, key="diff_chain")
            if st.button("⛏️ Minar y añadir (PoW)"):
                prev_block = st.session_state.chain[-1]
                idx = prev_block["index"] + 1
                ts2 = now_iso()
                nonce, h, iters, secs = mine_nonce(idx, ts2, data_input, prev_block["hash"], difficulty2)
                if nonce is not None:
                    new_block = {
                        "index": idx, "timestamp": ts2, "data": data_input,
                        "prev_hash": prev_block["hash"], "nonce": nonce, "hash": h,
                        "mode": "pow", "difficulty": difficulty2,
                        "pos_signers": [], "pos_threshold": None, "pos_signed_stake": 0
                    }
                    st.session_state.chain.append(new_block)
                    st.success(f"Bloque {idx} (PoW) añadido. Hash {h[:20]}… · intentos {iters:,} · {secs:.2f}s")
                else:
                    st.error("No se encontró nonce en el límite de iteraciones.")

        else:
            pos_thr = st.slider("Umbral de stake (quórum PoS)", 0.50, 0.90, 0.67, 0.01, key="pos_thr")
            if st.button("🖋️ Proponer y validar (PoS)"):
                prev_block = st.session_state.chain[-1]
                idx = prev_block["index"] + 1
                ts2 = now_iso()
                # nonce fijo 0 en PoS simulado
                h = block_hash(idx, ts2, data_input, prev_block["hash"], 0)
                signers, signed_stake, sigs = pos_sign_block(idx, ts2, data_input, prev_block["hash"], pos_thr)
                if signed_stake >= pos_thr * total_stake():
                    new_block = {
                        "index": idx, "timestamp": ts2, "data": data_input,
                        "prev_hash": prev_block["hash"], "nonce": 0, "hash": h,
                        "mode": "pos", "difficulty": None,
                        "pos_signers": signers, "pos_threshold": float(pos_thr), "pos_signed_stake": int(signed_stake)
                    }
                    st.session_state.chain.append(new_block)
                    st.success(f"Bloque {idx} (PoS) añadido. Firmas: {signers} · stake firmado: {signed_stake}/{total_stake()}")
                else:
                    st.error("No se alcanzó el quórum de stake.")

    with c2:
        st.markdown("#### 3.2 Alterar un bloque existente")
        if len(st.session_state.chain) > 1:
            choices = [b["index"] for b in st.session_state.chain if b["index"] != 0]
            target = st.selectbox("Elegir bloque a alterar (≠ génesis)", options=choices)
            new_text = st.text_area("Nuevo dato para el bloque elegido:", height=80, key="s2_newdata")
            if st.button("✏️ Alterar dato del bloque"):
                # Modifica dato y reencadena sin re-minar/firmar
                start_idx = None
                for i, b in enumerate(st.session_state.chain):
                    if b["index"] == target:
                        st.session_state.chain[i]["data"] = new_text or alter_one_char(b["data"])
                        # Recalcula hash con mismo nonce (PoW) o 0 (PoS)
                        if b["mode"] == "pow":
                            n = st.session_state.chain[i]["nonce"]
                        else:
                            n = 0
                            st.session_state.chain[i]["pos_signers"] = []
                            st.session_state.chain[i]["pos_signed_stake"] = 0
                        st.session_state.chain[i]["hash"] = block_hash(
                            b["index"], b["timestamp"], st.session_state.chain[i]["data"], b["prev_hash"], n
                        )
                        start_idx = i + 1
                        break
                if start_idx is not None:
                    # Propaga cambios hacia delante (sin re-minar/revalidar)
                    for j in range(start_idx, len(st.session_state.chain)):
                        prev_h = st.session_state.chain[j-1]["hash"]
                        st.session_state.chain[j]["prev_hash"] = prev_h
                        if st.session_state.chain[j]["mode"] == "pow":
                            n = st.session_state.chain[j]["nonce"]  # mantiene nonce ⇒ probablemente inválido
                        else:
                            n = 0
                            st.session_state.chain[j]["pos_signers"] = []
                            st.session_state.chain[j]["pos_signed_stake"] = 0
                        st.session_state.chain[j]["hash"] = block_hash(
                            st.session_state.chain[j]["index"],
                            st.session_state.chain[j]["timestamp"],
                            st.session_state.chain[j]["data"],
                            st.session_state.chain[j]["prev_hash"],
                            n
                        )
                    st.warning("Bloque alterado. La cadena puede estar **inválida** hasta re-minar (PoW) o revalidar (PoS).")

            # Re-minar/revalidar desde el bloque seleccionado, respetando el modo de cada bloque
            if st.button("🔁 Reparar desde ese bloque (re-minado/revalidación)"):
                for j in range(len(st.session_state.chain)):
                    if st.session_state.chain[j]["index"] == target:
                        start_idx = j
                        break
                for j in range(start_idx, len(st.session_state.chain)):
                    prev_h = st.session_state.chain[j-1]["hash"] if j > 0 else "0"*64
                    b = st.session_state.chain[j]
                    idx = b["index"]
                    ts3 = b["timestamp"]  # mantenemos timestamp
                    data3 = b["data"]
                    if b["mode"] == "pow":
                        # usa dificultad histórica del bloque si existe; por defecto 3
                        diff = b.get("difficulty") or 3
                        nonce, h, iters, secs = mine_nonce(idx, ts3, data3, prev_h, diff)
                        if nonce is None:
                            st.error(f"No se pudo re-minar el bloque {idx} en el límite de iteraciones.")
                            break
                        b["prev_hash"], b["nonce"], b["hash"] = prev_h, nonce, h
                        b["difficulty"] = diff
                    elif b["mode"] == "pos":
                        thr = b.get("pos_threshold") or 0.67
                        h = block_hash(idx, ts3, data3, prev_h, 0)
                        signers, signed_stake, sigs = pos_sign_block(idx, ts3, data3, prev_h, thr)
                        if signed_stake < thr * total_stake():
                            st.error(f"No se alcanzó quórum en bloque {idx}.")
                            break
                        b["prev_hash"], b["nonce"], b["hash"] = prev_h, 0, h
                        b["pos_signers"], b["pos_threshold"], b["pos_signed_stake"] = signers, float(thr), int(signed_stake)
                    else:
                        # génesis
                        b["prev_hash"] = prev_h
                        b["hash"] = block_hash(idx, ts3, data3, prev_h, b["nonce"])
                else:
                    st.success("Cadena reparada desde el bloque seleccionado.")
        else:
            st.info("Añade primero algún bloque en 3.1.")

    st.markdown("---")
    st.caption("**Logro S2:** Demuestra que alterar un bloque obliga a **re-minar** (PoW) o **revalidar firmas** (PoS) ese bloque y los posteriores.")

# ---------------------------
# 4) Visualización/Validación
# ---------------------------
with tabs[3]:
    st.header("Estado de la cadena y validación")

    df = pd.DataFrame(st.session_state.chain)
    st.dataframe(df, width="stretch")

    st.subheader("Validación")
    problems = []
    ok = True

    # Validación de enlace
    for i, b in enumerate(st.session_state.chain):
        # Recalcula hash esperado
        expected = block_hash(b["index"], b["timestamp"], b["data"], b["prev_hash"], b["nonce"])
        if expected != b["hash"]:
            ok = False
            problems.append(f"Bloque {b['index']}: hash no corresponde a sus campos.")
        if i == 0:
            continue
        prev_ok = (b["prev_hash"] == st.session_state.chain[i-1]["hash"])
        if not prev_ok:
            ok = False
            problems.append(f"Bloque {b['index']}: prev_hash no coincide con el hash del anterior.")

    # Validación específica por modo
    for i, b in enumerate(st.session_state.chain):
        if i == 0:
            continue
        if b["mode"] == "pow":
            diff = b.get("difficulty") or 3
            if not b["hash"].startswith("0" * diff):
                ok = False
                problems.append(f"Bloque {b['index']} (PoW): no cumple dificultad {diff}.")
        elif b["mode"] == "pos":
            thr = b.get("pos_threshold") or 0.67
            signed = stake_of(b.get("pos_signers", []))
            if signed < thr * total_stake():
                ok = False
                problems.append(f"Bloque {b['index']} (PoS): stake firmado {signed} < umbral {thr*total_stake():.0f}.")

    if ok:
        st.success("✅ Cadena válida (enlace correcto y reglas de consenso satisfechas por bloque).")
    else:
        st.error("❌ Cadena inválida.")
        for p in problems:
            st.caption(f"• {p}")

    st.markdown("#### Validadores (PoS simulado)")
    vals_df = pd.DataFrame(st.session_state.validators)[["id", "stake", "pub"]]
    st.dataframe(vals_df, width="stretch")

# ---------------------------
# 5) Lecturas guiadas
# ---------------------------
with tabs[4]:
    st.header("Lectura y guía de estudio — Tapscott & Tapscott (2016), cap. 2")
    st.markdown(
        """
**Tapscott & Tapscott, *Blockchain Revolution*, cap. 2: “La confianza en la era digital”**  
- Tesis: la confianza se reconfigura mediante redes distribuidas y criptografía.
- Foco UD2: cómo la **arquitectura técnica** (hash+nonce+dificultad+réplica o firmas por stake) respalda esa confianza.

**Preguntas guía**
1) ¿Qué papel juega la *prueba de trabajo* (o su alternativa) en la creación de confianza sin intermediario?
2) ¿Por qué el **enlace por `prev_hash`** hace “caro” reescribir la historia?
3) ¿Cómo afecta la **dificultad/umbral** al coste de ataque y a la latencia?
"""
    )

    if st.button("📄 Generar y guardar guía (MD)"):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"materiales/UD2_lecturas_{ts}.md"
        content = """# Guía de lectura — UD2

## Tapscott & Tapscott (2016), cap. 2 — La confianza en la era digital
- Tesis: la confianza se reconfigura con redes distribuidas y criptografía.
- Claves UD2: hash, nonce, dificultad, timestamp, enlace de bloques, firmas por stake (PoS).
- Preguntas:
  1) Rol de PoW/PoS en la confianza sin intermediario.
  2) Por qué el prev_hash encadenado dificulta reescritura histórica.
  3) Cómo inciden dificultad/umbral en coste de ataque y latencia/energía.
"""
        with open(fname, "w", encoding="utf-8") as f:
            f.write(content)
        st.success(f"Guía guardada en {fname}")
        with open(fname, "r", encoding="utf-8") as fh:
            st.download_button(
                "⬇️ Descargar ahora (Guía UD2)",
                fh.read(),
                file_name=os.path.basename(fname),
                mime="text/markdown",
                key=f"dl_ud2_{ts}"
            )

    st.markdown("---")
    st.info("ℹ️ Las guías se guardan en `./materiales`. Abajo puedes descargar cualquiera o todas en ZIP.")

    mats = _list_md_files("materiales")
    if mats:
        for idx, f in enumerate(mats):
            path = os.path.join("materiales", f)
            with open(path, "r", encoding="utf-8") as fh:
                st.download_button(
                    f"⬇️ Descargar {f}",
                    fh.read(),
                    file_name=f,
                    mime="text/markdown",
                    key=f"dl_mat_{idx}_{f}"
                )
        memzip_mat = _zip_folder_md("materiales")
        st.download_button(
            "⬇️ Descargar TODO (ZIP)",
            memzip_mat,
            file_name="materiales_ud2.zip",
            mime="application/zip",
            key="zip_mat_ud2"
        )
    else:
        st.caption("No hay materiales .md generados aún.")

# ---------------------------
# 6) Entregables y rúbrica
# ---------------------------
with tabs[5]:
    st.header("Entregables (UD2) y rúbrica")
    st.info(
        "ℹ️ Al guardar, los archivos se crean en el **servidor** dentro de `./entregas`. "
        "Desde aquí puedes **descargar** cada entrega o **todas en ZIP** y, si quieres, **borrarlas**."
    )

    st.subheader("Entrega S1 — 5 líneas")
    s1_text = st.text_area(
        "Explica cómo **nonce + dificultad** sostienen la inmutabilidad práctica.",
        height=110, key="e_s1_text"
    )
    if st.button("💾 Guardar Entrega S1 (MD)"):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"entregas/UD2_S1_{ts}.md"
        with open(fname, "w", encoding="utf-8") as f:
            f.write("# UD2 — Entrega S1\n\n")
            f.write(f"**Fecha:** {ts}\n\n")
            f.write("## Respuesta (máx. 5 líneas)\n\n")
            f.write((s1_text or "").strip() + "\n")
        st.success(f"Entrega guardada en {fname}")
        with open(fname, "r", encoding="utf-8") as fh:
            st.download_button(
                "⬇️ Descargar ahora (UD2 S1)",
                fh.read(),
                file_name=os.path.basename(fname),
                mime="text/markdown",
                key=f"dl_ud2s1_{ts}"
            )

    st.subheader("Entrega S2 — Justifica la inmutabilidad encadenada")
    s2_text = st.text_area(
        "Explica por qué **alterar un bloque** obliga a **re-minar** (PoW) o **revalidar firmas** (PoS) ese bloque y todos los posteriores.",
        height=110, key="e_s2_text"
    )
    if st.button("💾 Guardar Entrega S2 (MD)"):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"entregas/UD2_S2_{ts}.md"
        with open(fname, "w", encoding="utf-8") as f:
            f.write("# UD2 — Entrega S2\n\n")
            f.write(f"**Fecha:** {ts}\n\n")
            f.write("## Justificación\n\n")
            f.write((s2_text or "").strip() + "\n")
        st.success(f"Entrega guardada en {fname}")
        with open(fname, "r", encoding="utf-8") as fh:
            st.download_button(
                "⬇️ Descargar ahora (UD2 S2)",
                fh.read(),
                file_name=os.path.basename(fname),
                mime="text/markdown",
                key=f"dl_ud2s2_{ts}"
            )

    st.markdown("---")
    st.markdown("#### Entregas guardadas")
    files = _list_md_files("entregas")
    if files:
        for idx, f in enumerate(files):
            p = os.path.join("entregas", f)
            with open(p, "r", encoding="utf-8") as fh:
                st.download_button(
                    f"⬇️ Descargar {f}",
                    fh.read(),
                    file_name=f,
                    mime="text/markdown",
                    key=f"dl_ent_{idx}_{f}"
                )
        memzip = _zip_folder_md("entregas")
        st.download_button(
            "⬇️ Descargar TODO (ZIP)",
            memzip,
            file_name="entregas_ud2.zip",
            mime="application/zip",
            key="zip_ent_ud2"
        )
    else:
        st.caption("Aún no hay entregas guardadas.")

    st.markdown("#### Borrado tras descarga")
    confirm = st.checkbox("He descargado mis entregas y quiero borrarlas del servidor")
    if st.button("🧹 Borrar todas las entregas (.md)", disabled=not confirm):
        removed = _delete_md_in_folder("entregas")
        if removed > 0:
            st.success(f"Se borraron {removed} archivo(s) .md de 'entregas'.")
        else:
            st.warning("No había archivos .md que borrar.")

    st.markdown("---")
    st.caption("Rúbrica general: precisión técnica (40%), claridad (30%), aplicación/ejemplo (30%).")
