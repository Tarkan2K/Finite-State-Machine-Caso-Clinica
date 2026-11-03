"""
Entry point for serving the clinical interface on port 3000.
Uses FastAPI + Uvicorn when installed, or falls back to Python's built-in HTTP server
for preview environments without third-party pip packages.
"""

import csv
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
from pathlib import Path
from urllib.parse import unquote

BASE_DIR = Path(__file__).resolve().parent
STATIC_INDEX = BASE_DIR / "src" / "api" / "static" / "index.html"
PATIENTS_CSV = BASE_DIR / "Base_de_datos_ficticia" / "pacientes.csv"


def _load_csv_patients() -> list[dict]:
    if not PATIENTS_CSV.exists():
        return []
    patients = []
    with PATIENTS_CSV.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            def split_col(val: str) -> list[str]:
                if not val or val.strip().lower() in ("ninguna", "ninguno", "ninguna conocida"):
                    return []
                return [s.strip() for s in val.split(";") if s.strip()]

            patients.append(
                {
                    "rut": row["rut"].strip(),
                    "nombre": row["nombre"].strip(),
                    "edad": int(row["edad"].strip()),
                    "ciudad": row["ciudad"].strip(),
                    "enfermedades_cronicas": split_col(row["enfermedades_cronicas"]),
                    "alergias": split_col(row["alergias"]),
                    "farmacos_actuales": split_col(row["farmacos_actuales"]),
                }
            )
    return patients


class _FallbackPreviewHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args) -> None:
        pass

    def _send_json(self, status: int, data: object) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/" or self.path.startswith("/?"):
            html = STATIC_INDEX.read_text(encoding="utf-8").encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html)))
            self.end_headers()
            self.wfile.write(html)
            return

        if self.path == "/health":
            self._send_json(200, {"status": "healthy", "engine": "fsm-agent-runtime"})
            return

        if self.path == "/api/v1/patients":
            self._send_json(200, _load_csv_patients())
            return

        if self.path.startswith("/api/v1/patients/"):
            rut = unquote(self.path.removeprefix("/api/v1/patients/")).strip().upper()
            for p in _load_csv_patients():
                if p["rut"].upper() == rut:
                    self._send_json(200, p)
                    return
            self._send_json(404, {"detail": f"Paciente con RUT '{rut}' no encontrado."})
            return

        self._send_json(404, {"detail": "Not found"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 3000))
    try:
        import uvicorn
        from src.api.routes import app

        uvicorn.run(app, host="0.0.0.0", port=port)
    except ImportError:
        server = HTTPServer(("0.0.0.0", port), _FallbackPreviewHandler)
        print(f"Serving clinical dashboard on http://0.0.0.0:{port}")
        server.serve_forever()
