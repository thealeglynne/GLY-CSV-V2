from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import tempfile
import uvicorn
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# ========================
# Configuración inicial
# ========================
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("❌ No hay API key en el .env")

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=api_key,
    temperature=0.3,
)

app = FastAPI(
    title="API Analizador de CSV",
    description="Genera un informe ejecutivo narrativo basado en los datos del CSV y descripción del usuario",
    version="2.0.0"
)

origins = [
    "https://glynne-sst-ai-hsiy.vercel.app",  # tu frontend
    "http://localhost:3000",  # para pruebas locales
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========================
# Funciones auxiliares
# ========================
def analizar_csv(file_path: str) -> str:
    """Genera un análisis técnico simple del CSV"""
    df = pd.read_csv(file_path)
    df = df.dropna(axis=1, how='all')

    numericas = df.select_dtypes(include=['number'])
    categoricas = df.select_dtypes(include=['object'])
    fechas = df.select_dtypes(include=['datetime64', 'datetime'])

    reporte = []

    if not numericas.empty:
        desc_num = numericas.describe().T
        reporte.append("RESUMEN NUMÉRICO:")
        for col, row in desc_num.iterrows():
            reporte.append(f"- {col}: media={row['mean']:.2f}, min={row['min']}, max={row['max']}, nulos={df[col].isna().sum()}")
    else:
        reporte.append("No hay columnas numéricas")

    if not categoricas.empty:
        reporte.append("RESUMEN CATEGÓRICO:")
        for col in categoricas.columns:
            top_val = df[col].value_counts().head(3)
            reporte.append(f"- {col}: {df[col].nunique()} valores únicos. Top 3: {', '.join(top_val.index.astype(str))}")
    else:
        reporte.append("No hay columnas categóricas")

    if not fechas.empty:
        reporte.append("RESUMEN FECHAS:")
        for col in fechas.columns:
            reporte.append(f"- {col}: rango {df[col].min()} → {df[col].max()}, nulos={df[col].isna().sum()}")

    nulls = df.isna().sum()
    cols_with_nulls = nulls[nulls > 0]
    if not cols_with_nulls.empty:
        reporte.append("VALORES FALTANTES:")
        for col, val in cols_with_nulls.items():
            reporte.append(f"- {col}: {val} nulos ({val/len(df)*100:.1f}%)")

    reporte.append(f"ESTADÍSTICAS GENERALES:")
    reporte.append(f"- Filas: {len(df)}")
    reporte.append(f"- Columnas: {len(df.columns)}")
    reporte.append(f"- Memoria usada: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    return "\n".join(reporte)


def generar_informe_llm(analisis_tecnico: str, descripcion: str) -> str:
    """Llama al LLM para generar un informe ejecutivo narrativo considerando la descripción del usuario"""
    prompt_final = f"""
Eres un analista de datos senior con enfoque consultivo. Analiza el siguiente dataset y genera un **informe ejecutivo narrativo** de 6–8 párrafos.
Toma en cuenta la descripción que da el usuario para contextualizar la información.

**DESCRIPCIÓN DEL DATASET:**
{descripcion}

**DATOS TÉCNICOS DEL DATASET:**
{analisis_tecnico}

**INSTRUCCIONES:**
- No te enfoques en explicar cada tabla o estadística.
- Infiera sobre estado general, procesos, riesgos, oportunidades y recomendaciones.
- Usa un lenguaje claro, profesional y consultivo.
- Responde únicamente con el informe, sin encabezados.
"""
    respuesta = llm.invoke(prompt_final)
    return respuesta.content


def separar_columnas_csv(file_path: str):
    """Devuelve matrices separadas (numéricas / no numéricas)"""
    df = pd.read_csv(file_path)
    columnas_numericas = df.select_dtypes(include=["number"]).columns
    columnas_no_numericas = df.select_dtypes(exclude=["number"]).columns
    matriz_numerica = df[columnas_numericas].to_dict(orient="records")
    matriz_no_numerica = df[columnas_no_numericas].to_dict(orient="records")
    return {"numericas": matriz_numerica, "no_numericas": matriz_no_numerica}


# ========================
# Endpoint principal
# ========================
@app.post("/procesar-csv")
async def procesar_csv(
    file: UploadFile = File(...),
    descripcion: str = Form(...)
):
    """
    Recibe un CSV y una descripción corta del dataset.
    La descripción sirve como contexto adicional para el LLM.
    """
    try:
        if not file.filename.endswith(".csv"):
            raise HTTPException(status_code=400, detail="Solo se permiten archivos .csv")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            contents = await file.read()
            tmp.write(contents)
            tmp_path = tmp.name

        # Analizar CSV
        analisis_tecnico = analizar_csv(tmp_path)

        # Generar informe LLM con contexto
        informe = generar_informe_llm(analisis_tecnico, descripcion)

        # Separar columnas para el frontend
        tablas = separar_columnas_csv(tmp_path)

        os.remove(tmp_path)

        return {
            "tablas": tablas,
            "analisis_tecnico": analisis_tecnico,
            "informe_ejecutivo": informe
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al procesar el archivo: {str(e)}")


# ========================
# Run server
# ========================
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
