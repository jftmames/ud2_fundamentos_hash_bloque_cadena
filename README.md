# UD2 — Fundamentos técnicos: hash, bloque, cadena

## Objetivo
Comprender **cómo se construye la inmutabilidad**: hash, nonce, timestamp y **enlace** entre bloques.

## Contenido en la app
- Teoría: hash, nonce, dificultad, timestamp y encadenamiento.
- Práctica S1 (PoW-lite): modificar dato y comprobar cambio de hash con/ sin nonce.
- Práctica S2: construir una **mini-blockchain**; minar, alterar un bloque, invalidar y re-minar.
- Lectura: **Tapscott & Tapscott (2016), cap. 2** con preguntas guía.
- Entregables: explicación breve (S1) y justificativo sobre inmutabilidad encadenada (S2), con descarga/ZIP/borrado.

## Ejecución
```bash
pip install -r requirements.txt
streamlit run app.py
