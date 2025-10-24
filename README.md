# Analizador Léxico y Sintáctico (FORTRAN-77) - Tarea 3

Este repositorio contiene el código fuente de un **analizador léxico y sintáctico** desarrollado en Python para la Tarea 3 del curso de Teoría de la Computación.

El script `fortran_ll1_final.py` (o el nombre que le hayas puesto) es un \\textit{front-end} de compilador que procesa un subconjunto del lenguaje FORTRAN-77. Implementa un analizador léxico para la tokenización y un parser LL(1) de descenso recursivo para validar la gramática y construir un **Árbol Sintáctico Abstracto (AST)**.

El programa puede ejecutarse en modo consola (CLI) o a través de una Interfaz Gráfica de Usuario (GUI).

## 🌟 Características Principales

  * **Analizador Léxico:** Reconoce palabras clave, identificadores, números (enteros y reales), operadores y delimitadores de FORTRAN-77.
  * **Analizador Sintáctico:** Implementa un parser LL(1) de descenso recursivo que valida la estructura de declaraciones, asignaciones, expresiones aritméticas, `IF-THEN-ELSE` y bucles `DO`.
  * **Construcción de AST:** Genera un Árbol Sintáctico Abstracto que representa la jerarquía lógica del código fuente.
  * **Exportación de Resultados:**
      * Imprime el AST en formato **JSON** en la consola (modo CLI).
      * Genera un **árbol visual** (ej. `.png`, `.svg`, `.dot`) usando Graphviz.
  * **Detección de Errores:** Implementa un modo de recuperación de errores (modo pánico) para reportar múltiples errores sintácticos en una sola ejecución.
  * **Interfaz Gráfica (GUI):** Incluye una GUI (hecha con Tkinter) para una interacción más amigable.

## 🛠️ Requisitos

  * Python 3.x
  * **Graphviz:** Este es un requisito **externo** y **obligatorio** para poder generar las imágenes del árbol visual (`.png`, `.svg`, etc.).
      * Puedes descargarlo desde [graphviz.org/download/](https://graphviz.org/download/).
      * *Asegúrate de que el ejecutable `dot` esté en el PATH de tu sistema después de la instalación.*

## 🚀 Uso

El programa tiene dos modos de ejecución:

### 1\. Modo Gráfico (GUI) (Recomendado)

Es la forma más sencilla de usarlo. Ejecuta el script con el flag `--gui`:

```bash
python3 fortran_ll1_final.py --gui
```

Se abrirá una ventana donde podrás:

  * Abrir archivos `.f` o pegar tu código.
  * Analizar el código y ver los errores o el AST (JSON) en el panel de resultados.
  * Exportar el árbol visual en formato `.png`, `.svg` o `.dot`.

### 2\. Modo Consola (CLI)

Puedes pasar el archivo fuente como argumento.

**Para analizar y ver el AST (JSON) en la consola:**

```bash
python3 fortran_ll1_final.py demo.f
```

**Para analizar y exportar el árbol visual a un archivo:**

Usa el flag `-o` o `--out` para especificar el archivo de salida. El formato se detecta por la extensión.

```bash
# Exportar como PNG
python3 fortran_ll1_final.py demo.f -o demo_ast.png

# Exportar como SVG
python3 fortran_ll1_final.py demo.f -o demo_ast.svg

# Exportar solo el archivo .dot
python3 fortran_ll1_final.py demo.f -o demo_ast.dot
```

## 📜 Ejemplo de Ejecución (CLI)

Supongamos que tienes un archivo llamado `demo.f` con el siguiente contenido:

```fortran
PROGRAM DEMO
INTEGER I, N
REAL X
I = 1;
N = 10;
X = (I + 3) * 2.5;
IF (I < N) THEN
  I = I + 1;
ELSE
  X = X - 1.0;
ENDIF
DO I = 1, N, 1
  X = X + I;
ENDDO
END
```

Para analizar este archivo y generar una imagen del AST, ejecutarías:

```bash
python3 fortran_ll1_final.py demo.f -o demo_ast.png
```

**Salida en Consola (JSON):**

El programa imprimirá en la consola el AST en formato JSON:

```json
{
  "_node_type": "Program",
  "name": "DEMO",
  "decls": [
    {
      "_node_type": "Decl",
      "dtype": "INTEGER",
      "ids": [ "I", "N" ]
    },
    {
      "_node_type": "Decl",
      "dtype": "REAL",
      "ids": [ "X" ]
    }
  ],
  "stmts": [
    {
      "_node_type": "Assign",
      "id": "I",
      "expr": { "_node_type": "Literal", "value": "1" }
    },
    {
      "_node_type": "Assign",
      "id": "N",
      "expr": { "_node_type": "Literal", "value": "10" }
    },
    // ... (resto del AST) ...
  ]
}
```

**Salida Gráfica (Imagen):**

Además, este comando generará un archivo `demo_ast.png` con el árbol sintáctico visual:

## 🧑‍💻 Autores

  * **Joaquín Aguilar**
  * **Rigoberto Aravena**
  * **Benjamín Sánchez**
