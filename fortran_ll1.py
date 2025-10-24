#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fortran_ll1.py — Analizador léxico + sintáctico LL(1)
Subconjunto tipo FORTRAN-77 (free form simplificado) con:
  - PROGRAM ... END
  - Declaraciones: INTEGER/REAL id[,id]*
  - Asignaciones: ID = expr ;
  - IF (...) THEN stmt* [ELSE stmt*] ENDIF
  - DO i=expr,expr[,expr] ... ENDDO
  - READ()/WRITE() (formas simples)
  - Expresiones con + - * / y paréntesis, relacionales: == <> < <= > >=

Decisiones prácticas para la tarea:
  - Terminamos sentencias con ';' para sincronización y errores más claros.
  - Comentarios: línea que inicia con 'C' o '!' al inicio de línea.
  - AST en JSON y export opcional a Graphviz DOT.

Uso (CLI):
  python fortran_ll1_final.py archivo.f
  python fortran_ll1_final.py archivo.f --out arbol.png
  python fortran_ll1_final.py archivo.f --out arbol.svg
  python fortran_ll1_final.py archivo.f --out arbol.dot

Uso (GUI):
  python fortran_ll1_final.py --gui
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict, fields
from enum import Enum, auto
from typing import Optional, List, Tuple, Dict, Any
import sys, json, argparse, os, subprocess # <-- subprocess AÑADIDO


# Importaciones para la GUI
try:
    import tkinter as tk
    from tkinter import scrolledtext, filedialog, messagebox
except ImportError:
    tk = None


# -----------------------------
# Tokens y utilidades de Lexer
# -----------------------------
class TokenType(Enum):
    PROGRAM = auto(); END = auto(); INTEGER = auto(); REAL = auto()
    IF = auto(); THEN = auto(); ELSE = auto(); ENDIF = auto()
    DO = auto(); ENDDO = auto(); CONTINUE = auto()
    READ = auto(); WRITE = auto(); STOP = auto(); RETURN = auto()
    ID = auto(); INT = auto(); REALNUM = auto()
    ASSIGN = auto(); EQ = auto(); NE = auto(); LT = auto(); LE = auto()
    GT = auto(); GE = auto(); PLUS = auto(); MINUS = auto()
    MUL = auto(); DIV = auto(); LP = auto(); RP = auto()
    COMMA = auto(); SEMI = auto(); EOF = auto(); INVALID = auto()
KEYWORDS = {
    "PROGRAM": TokenType.PROGRAM, "END": TokenType.END, "INTEGER": TokenType.INTEGER,
    "REAL": TokenType.REAL, "IF": TokenType.IF, "THEN": TokenType.THEN,
    "ELSE": TokenType.ELSE, "ENDIF": TokenType.ENDIF, "DO": TokenType.DO,
    "ENDDO": TokenType.ENDDO, "CONTINUE": TokenType.CONTINUE, "READ": TokenType.READ,
    "WRITE": TokenType.WRITE, "STOP": TokenType.STOP, "RETURN": TokenType.RETURN,
}
@dataclass
class Token:
    type: TokenType; lexeme: str; line: int; col: int
    def __repr__(self) -> str: return f"{self.type.name}('{self.lexeme}',{self.line}:{self.col})"
class Lexer:
    def __init__(self, text: str):
        self.text = text; self.i = 0; self.line = 1; self.col = 1; self.n = len(text)
    def _peek(self, k: int = 0) -> str:
        j = self.i + k; return '\0' if j < 0 or j >= self.n else self.text[j]
    def _advance(self) -> str:
        ch = self._peek(0)
        if ch == '\n': self.line += 1; self.col = 1
        else: self.col += 1
        self.i += 1; return ch
    def _skip_ws_and_comments(self):
        while True:
            moved = False
            while self._peek().isspace(): self._advance(); moved = True
            if self.col == 1 and (self._peek(0) in ('C', '!')):
                while self._peek(0) not in ('\n', '\0'): self._advance()
                moved = True; continue
            if not moved: break
    def next_token(self) -> Token:
        self._skip_ws_and_comments()
        start_line, start_col = self.line, self.col
        ch = self._peek(0)
        if ch == '\0': return Token(TokenType.EOF, "", start_line, start_col)
        if ch.isalpha() or ch == '_':
            lex = []
            while self._peek(0).isalnum() or self._peek(0) == '_': lex.append(self._advance())
            lexeme = ''.join(lex).upper()
            ttype = KEYWORDS.get(lexeme, TokenType.ID)
            return Token(ttype, lexeme, start_line, start_col)
        if ch.isdigit():
            lex = []; is_real = False
            while self._peek(0).isdigit(): lex.append(self._advance())
            if self._peek(0) == '.' and self._peek(1).isdigit():
                is_real = True; lex.append(self._advance())
                while self._peek(0).isdigit(): lex.append(self._advance())
            if self._peek(0) in ('e', 'E'):
                is_real = True; lex.append(self._advance())
                if self._peek(0) in ('+', '-'): lex.append(self._advance())
                if self._peek(0).isdigit():
                    while self._peek(0).isdigit(): lex.append(self._advance())
            lexeme = ''.join(lex)
            return Token(TokenType.REALNUM if is_real else TokenType.INT, lexeme, start_line, start_col)
        op_map = {'==': TokenType.EQ, '<>': TokenType.NE, '<=': TokenType.LE, '>=': TokenType.GE}
        for op, tt in op_map.items():
            if ch == op[0] and self._peek(1) == op[1]:
                self._advance(); self._advance(); return Token(tt, op, start_line, start_col)
        single_char_map = {
            '=': TokenType.ASSIGN, '<': TokenType.LT, '>': TokenType.GT,
            '+': TokenType.PLUS, '-': TokenType.MINUS, '*': TokenType.MUL, '/': TokenType.DIV,
            '(': TokenType.LP, ')': TokenType.RP, ',': TokenType.COMMA, ';': TokenType.SEMI,
        }
        if ch in single_char_map:
            self._advance(); return Token(single_char_map[ch], ch, start_line, start_col)
        bad = self._advance(); return Token(TokenType.INVALID, bad, start_line, start_col)
@dataclass
class ASTNode:
    def to_json_dict(self) -> Any:
        def convert(o):
            if isinstance(o, ASTNode): return o.to_json_dict()
            if isinstance(o, list): return [convert(x) for x in o]
            return o
        data = asdict(self)
        data["_node_type"] = self.__class__.__name__ 
        for k, v in data.items(): data[k] = convert(v)
        return data
@dataclass
class Program(ASTNode): name: str; decls: List['Decl'] = field(default_factory=list); stmts: List['Stmt'] = field(default_factory=list)
@dataclass
class Decl(ASTNode): dtype: str; ids: List[str]
@dataclass
class Stmt(ASTNode): pass
@dataclass
class Assign(Stmt): id: str; expr: 'Expr'
@dataclass
class If(Stmt): cond: 'Expr'; then_stmts: List[Stmt]; else_stmts: Optional[List[Stmt]] = None
@dataclass
class Do(Stmt): var: str; start: 'Expr'; end: 'Expr'; step: Optional['Expr']; body: List[Stmt]
@dataclass
class Read(Stmt): ids: List[str]
@dataclass
class Write(Stmt): exprs: List['Expr']
@dataclass
class Continue(Stmt): pass
@dataclass
class Stop(Stmt): pass
@dataclass
class Return(Stmt): pass
@dataclass
class Expr(ASTNode): pass
@dataclass
class BinOp(Expr): op: str; left: Expr; right: Expr
@dataclass
class UnaryOp(Expr): op: str; expr: Expr
@dataclass
class Literal(Expr): value: str
@dataclass
class Var(Expr): name: str
class ParseError(Exception): pass
class Parser:
    def __init__(self, lexer: Lexer):
        self.lexer = lexer; self.la = self.lexer.next_token(); self.errors: List[str] = []
        self.SYNC: Dict[str, Tuple[TokenType, ...]] = {
            "Program": (TokenType.EOF,),
            "DeclList": (TokenType.ID, TokenType.IF, TokenType.DO, TokenType.CONTINUE,
                         TokenType.READ, TokenType.WRITE, TokenType.STOP, TokenType.RETURN,
                         TokenType.END, TokenType.ENDIF, TokenType.ENDDO),
            "StmtList": (TokenType.END, TokenType.ELSE, TokenType.ENDIF, TokenType.ENDDO),
            "Stmt": (TokenType.SEMI, TokenType.END, TokenType.ELSE, TokenType.ENDIF, TokenType.ENDDO),
            "IfStmt": (TokenType.ENDIF,), "DoStmt": (TokenType.ENDDO,),
            "Expr": (TokenType.SEMI, TokenType.COMMA, TokenType.RP, TokenType.THEN),
        }
    def _error(self, msg: str): self.errors.append(f"[L{self.la.line}:C{self.la.col}] {msg}")
    def _match(self, t: TokenType) -> Token:
        if self.la.type == t:
            cur = self.la
            if cur.type != TokenType.EOF: self.la = self.lexer.next_token()
            return cur
        self._error(f"Se esperaba {t.name} pero se encontró {self.la.type.name} ('{self.la.lexeme}')")
        raise ParseError()
    def _sync(self, prod: str):
        syncset = self.SYNC.get(prod, (TokenType.SEMI, TokenType.END, TokenType.EOF))
        while self.la.type not in syncset and self.la.type != TokenType.EOF:
            self.la = self.lexer.next_token()
    def parse_program(self) -> Program:
        name = "UNKNOWN"; decls = []; stmts = []
        try:
            self._match(TokenType.PROGRAM); name_tok = self._match(TokenType.ID); name = name_tok.lexeme
        except ParseError: self._sync("Program")
        decls = self.parse_decl_list(); stmts = self.parse_stmt_list()
        try: self._match(TokenType.END); self._match(TokenType.EOF)
        except ParseError: self._sync("Program")
        return Program(name=name, decls=decls, stmts=stmts)
    def parse_decl_list(self) -> List[Decl]:
        decls: List[Decl] = []
        while self.la.type in (TokenType.INTEGER, TokenType.REAL): decls.append(self.parse_decl())
        return decls
    def parse_decl(self) -> Decl:
        try:
            dtype = ""
            if self.la.type == TokenType.INTEGER: dtype = self._match(TokenType.INTEGER).lexeme
            elif self.la.type == TokenType.REAL: dtype = self._match(TokenType.REAL).lexeme
            else: self._error("Tipo de declaración esperado"); raise ParseError()
            ids = self.parse_id_list(); return Decl(dtype=dtype, ids=ids)
        except ParseError: self._sync("DeclList"); return Decl(dtype="ERROR", ids=[])
    def parse_id_list(self) -> List[str]:
        ids: List[str] = []
        try:
            t = self._match(TokenType.ID); ids.append(t.lexeme)
            while self.la.type == TokenType.COMMA:
                self._match(TokenType.COMMA); t = self._match(TokenType.ID); ids.append(t.lexeme)
        except ParseError: self._sync("DeclList")
        return ids
    def parse_stmt_list(self) -> List[Stmt]:
        stmts: List[Stmt] = []
        FIRST_STMT = (TokenType.ID, TokenType.IF, TokenType.DO, TokenType.CONTINUE,
                      TokenType.READ, TokenType.WRITE, TokenType.STOP, TokenType.RETURN)
        while self.la.type in FIRST_STMT: stmts.append(self.parse_stmt())
        return stmts
    def parse_stmt(self) -> Stmt:
        try:
            if self.la.type == TokenType.ID:
                asg = self.parse_assign(); self._match(TokenType.SEMI); return asg
            if self.la.type == TokenType.IF: return self.parse_if()
            if self.la.type == TokenType.DO: return self.parse_do()
            if self.la.type == TokenType.CONTINUE:
                self._match(TokenType.CONTINUE); self._match(TokenType.SEMI); return Continue()
            if self.la.type == TokenType.STOP:
                self._match(TokenType.STOP); self._match(TokenType.SEMI); return Stop()
            if self.la.type == TokenType.RETURN:
                self._match(TokenType.RETURN); self._match(TokenType.SEMI); return Return()
            if self.la.type in (TokenType.READ, TokenType.WRITE):
                io = self.parse_io(); self._match(TokenType.SEMI); return io
            self._error(f"Inicio de sentencia no reconocido: {self.la.type.name}"); raise ParseError()
        except ParseError:
            self._sync("Stmt")
            if self.la.type == TokenType.SEMI: self.la = self.lexer.next_token()
            return Continue()
    def parse_assign(self) -> Assign:
        try:
            ident = self._match(TokenType.ID).lexeme; self._match(TokenType.ASSIGN); e = self.parse_expr()
            return Assign(id=ident, expr=e)
        except ParseError: self._sync("Stmt"); return Assign(id="ERROR", expr=Literal("0"))
    def parse_if(self) -> If:
        try:
            self._match(TokenType.IF); self._match(TokenType.LP); cond = self.parse_rel_expr()
            self._match(TokenType.RP); self._match(TokenType.THEN); then_stmts = self.parse_stmt_list()
            else_stmts = None
            if self.la.type == TokenType.ELSE:
                self._match(TokenType.ELSE); else_stmts = self.parse_stmt_list()
            self._match(TokenType.ENDIF)
            return If(cond=cond, then_stmts=then_stmts, else_stmts=else_stmts)
        except ParseError:
            self._sync("IfStmt")
            if self.la.type == TokenType.ENDIF: self._match(TokenType.ENDIF)
            return If(cond=Literal("ERROR"), then_stmts=[], else_stmts=None)
    def parse_do(self) -> Do:
        try:
            self._match(TokenType.DO); var = self._match(TokenType.ID).lexeme
            self._match(TokenType.ASSIGN); start = self.parse_expr(); self._match(TokenType.COMMA)
            end = self.parse_expr(); step = None
            if self.la.type == TokenType.COMMA: self._match(TokenType.COMMA); step = self.parse_expr()
            body = self.parse_stmt_list(); self._match(TokenType.ENDDO)
            return Do(var=var, start=start, end=end, step=step, body=body)
        except ParseError:
            self._sync("DoStmt")
            if self.la.type == TokenType.ENDDO: self._match(TokenType.ENDDO)
            return Do(var="ERROR", start=Literal("0"), end=Literal("0"), step=None, body=[])
    def parse_io(self) -> Stmt:
        try:
            if self.la.type == TokenType.READ:
                self._match(TokenType.READ); self._match(TokenType.LP); self._match(TokenType.RP)
                ids = self.parse_id_list(); return Read(ids=ids)
            elif self.la.type == TokenType.WRITE:
                self._match(TokenType.WRITE); self._match(TokenType.LP); self._match(TokenType.RP)
                exprs = self.parse_expr_list(); return Write(exprs=exprs)
            else: self._error("Se esperaba READ o WRITE"); raise ParseError()
        except ParseError: self._sync("Stmt"); return Continue()
    def parse_expr_list(self) -> List[Expr]:
        exprs = []
        try:
            exprs.append(self.parse_expr())
            while self.la.type == TokenType.COMMA:
                self._match(TokenType.COMMA); exprs.append(self.parse_expr())
        except ParseError: self._sync("Expr")
        return exprs
    def parse_rel_expr(self) -> Expr:
        left = self.parse_expr()
        if self.la.type in (TokenType.EQ, TokenType.NE, TokenType.LT, TokenType.LE, TokenType.GT, TokenType.GE):
            op_tok = self.la; self.la = self.lexer.next_token(); right = self.parse_expr()
            return BinOp(op=op_tok.lexeme, left=left, right=right)
        return left
    def parse_expr(self) -> Expr:
        left = self.parse_term()
        while self.la.type in (TokenType.PLUS, TokenType.MINUS):
            op = self.la.lexeme; self.la = self.lexer.next_token(); right = self.parse_term()
            left = BinOp(op=op, left=left, right=right)
        return left
    def parse_term(self) -> Expr:
        left = self.parse_factor()
        while self.la.type in (TokenType.MUL, TokenType.DIV):
            op = self.la.lexeme; self.la = self.lexer.next_token(); right = self.parse_factor()
            left = BinOp(op=op, left=left, right=right)
        return left
    def parse_factor(self) -> Expr:
        if self.la.type == TokenType.MINUS:
            op = self._match(TokenType.MINUS).lexeme; factor = self.parse_factor()
            return UnaryOp(op=op, expr=factor)
        if self.la.type == TokenType.LP:
            self._match(TokenType.LP); e = self.parse_expr()
            try: self._match(TokenType.RP)
            except ParseError: self._sync("Expr")
            return e
        if self.la.type == TokenType.ID:
            name = self._match(TokenType.ID).lexeme; return Var(name=name)
        if self.la.type == TokenType.INT:
            val = self._match(TokenType.INT).lexeme; return Literal(value=val)
        if self.la.type == TokenType.REALNUM:
            val = self._match(TokenType.REALNUM).lexeme; return Literal(value=val)
        self._error(f"Factor inválido: se esperaba ( ID | INT | REAL | - | LP ) y llegó {self.la.type.name} '{self.la.lexeme}'")
        self.la = self.lexer.next_token(); return Literal(value="0")
    

# -------------------------------------
# Función para convertir AST a DOT
# -------------------------------------
def ast_to_dot(ast_root: ASTNode) -> str:
    """Recorre el AST y genera una cadena en formato DOT (Graphviz)."""
    dot_lines = ['digraph AST {', '  node [shape=box, fontname="Arial"];', '  edge [fontname="Arial"];']
    counter = [0] # Usamos lista para que sea mutable por referencia

    def _walk(node: ASTNode, parent_id: Optional[str]):
        my_id = f"node{counter[0]}"; counter[0] += 1
        
        label_parts = [node.__class__.__name__]
        child_nodes: List[Tuple[str, ASTNode]] = []

        for f in fields(node):
            val = getattr(node, f.name)
            
            if isinstance(val, ASTNode):
                child_nodes.append((f.name, val))
            elif isinstance(val, list) and val and isinstance(val[0], ASTNode):
                # Es una lista de nodos AST (ej. stmts, decls)
                for i, item in enumerate(val):
                    child_nodes.append((f"{f.name}[{i}]", item))
            elif val is None or (isinstance(val, list) and not val):
                # Omitir Nones (como step o else_stmts) y listas vacías
                continue
            else:
                # Atributo simple (str, int, bool) O lista de simples (como ids)
                label_parts.append(f"{f.name} = {repr(val)}")

        # Arreglo del bug del "\n" (quitamos la 'r' de antes)
        dot_lines.append(f'  {my_id} [label="{"\\n".join(label_parts)}"];')
        
        if parent_id:
            dot_lines.append(f'  {parent_id} -> {my_id};')
        
        # Recurrir sobre hijos
        for _, child in child_nodes:
            _walk(child, my_id)

    _walk(ast_root, None)
    dot_lines.append('}')
    return '\n'.join(dot_lines)


# -------------------------------------
# Lógica de exportación de árbol
# -------------------------------------
def export_tree(dot_string: str, output_file: str, stderr_callback=print):
    """
    Guarda el árbol. Si la extensión es .dot, guarda el texto.
    Si es .png, .svg, .pdf, etc., usa subprocess para llamar a Graphviz.
    """
    _root, ext = os.path.splitext(output_file)
    ext = ext.lower()

    if ext == ".dot":
        # Guardar el archivo DOT de texto
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(dot_string)
            stderr_callback(f"Árbol (DOT) guardado en: {output_file}")
        except Exception as e:
            stderr_callback(f"Error al guardar archivo DOT: {e}")
    else:
        # Usar Graphviz para generar la imagen
        format_type = ext.lstrip('.') # "png", "svg", "pdf"
        try:
            subprocess.run(
                ['dot', f'-T{format_type}', '-o', output_file],
                input=dot_string,
                encoding='utf-8',
                check=True,
                stderr=subprocess.PIPE # Captura stderr de 'dot'
            )
            stderr_callback(f"Imagen del árbol ({format_type}) guardada en: {output_file}")
        
        except FileNotFoundError:
            stderr_callback(
                "Error: Comando 'dot' (Graphviz) no encontrado.\n"
                "Asegúrate de que Graphviz esté instalado y en el PATH del sistema.\n"
                "Descarga: https://graphviz.org/download/\n"
                f"(Mientras tanto, el archivo .dot se guardó en {_root}.dot)"
            )
            # Guardar el .dot como fallback
            export_tree(dot_string, f"{_root}.dot", stderr_callback)
            
        except subprocess.CalledProcessError as e:
            stderr_callback(f"Error al ejecutar Graphviz 'dot':\n{e.stderr.decode()}")
        except Exception as e:
            stderr_callback(f"Error inesperado al exportar imagen: {e}")


# -------------------------------------
# Interfaz Gráfica (GUI)
# -------------------------------------
if tk:
    class FortranParserApp:
        def __init__(self, root):
            self.root = root; self.root.title("Analizador FORTRAN-77 (Tarea 3)"); self.root.geometry("1000x700")
            self.last_ast: Optional[ASTNode] = None; self.current_filepath: Optional[str] = None
            self.menu = tk.Menu(self.root); self.root.config(menu=self.menu)
            self.file_menu = tk.Menu(self.menu, tearoff=0); self.menu.add_cascade(label="Archivo", menu=self.file_menu)
            self.file_menu.add_command(label="Abrir archivo...", command=self.open_file)
            self.file_menu.add_command(label="Guardar resultados (JSON/Errores)...", command=self.save_results)
            self.file_menu.add_separator(); self.file_menu.add_command(label="Salir", command=self.root.quit)
            self.help_menu = tk.Menu(self.menu, tearoff=0); self.menu.add_cascade(label="Ayuda", menu=self.help_menu)
            self.help_menu.add_command(label="Acerca de...", command=self.show_about)
            self.paned_window = tk.PanedWindow(self.root, orient=tk.HORIZONTAL, sashrelief=tk.RAISED)
            self.paned_window.pack(fill=tk.BOTH, expand=True, pady=5, padx=5)
            self.left_frame = tk.Frame(self.paned_window, relief=tk.SUNKEN, borderwidth=1)
            tk.Label(self.left_frame, text="Código Fuente FORTRAN", font=("Arial", 10, "bold")).pack(pady=2)
            self.source_text = scrolledtext.ScrolledText(self.left_frame, wrap=tk.WORD, undo=True, font=("Consolas", 10))
            self.source_text.pack(fill=tk.BOTH, expand=True, padx=2, pady=2); self.paned_window.add(self.left_frame, minsize=300)
            self.right_frame = tk.Frame(self.paned_window, relief=tk.SUNKEN, borderwidth=1)
            tk.Label(self.right_frame, text="Resultados (AST / Errores)", font=("Arial", 10, "bold")).pack(pady=2)
            self.result_text = scrolledtext.ScrolledText(self.right_frame, wrap=tk.WORD, state=tk.DISABLED, font=("Consolas", 10))
            self.result_text.pack(fill=tk.BOTH, expand=True, padx=2, pady=2); self.paned_window.add(self.right_frame, minsize=300)
            self.button_frame = tk.Frame(self.root, pady=5); self.button_frame.pack(fill=tk.X, side=tk.BOTTOM)
            self.analyze_button = tk.Button(self.button_frame, text="▶ Analizar Código", font=("Arial", 12, "bold"), command=self.analyze_code, bg="#4CAF50", fg="white")
            self.analyze_button.pack(side=tk.LEFT, padx=10, pady=5)
            
            self.export_button = tk.Button(self.button_frame, text="Exportar Árbol Visual (PNG/DOT)...", font=("Arial", 12), command=self.export_visual_tree, state=tk.DISABLED)
            self.export_button.pack(side=tk.LEFT, padx=10, pady=5)
            
            self.clear_button = tk.Button(self.button_frame, text="Limpiar", font=("Arial", 12), command=self.clear_all)
            self.clear_button.pack(side=tk.RIGHT, padx=10, pady=5)

        def open_file(self):
            path = filedialog.askopenfilename(filetypes=[("FORTRAN files", "*.f *.f77 *.for"), ("All files", "*.*")])
            if not path: return
            self.current_filepath = path
            try:
                with open(path, "r", encoding="utf-8") as f: content = f.read()
                self.source_text.delete("1.0", tk.END); self.source_text.insert("1.0", content)
                self.root.title(f"Analizador FORTRAN-77 - {path}"); self.clear_results()
            except Exception as e: messagebox.showerror("Error al abrir", f"No se pudo leer el archivo:\n{e}")

        def save_results(self):
            content = self.result_text.get("1.0", tk.END).strip()
            if not content: messagebox.showwarning("Nada que guardar", "No hay resultados para guardar."); return
            path = filedialog.asksaveasfilename(filetypes=[("Text files", "*.txt"), ("All files", "*.*")], defaultextension=".txt")
            if not path: return
            try:
                with open(path, "w", encoding="utf-8") as f: f.write(content)
                messagebox.showinfo("Éxito", "Resultados guardados.")
            except Exception as e: messagebox.showerror("Error al guardar", f"No se pudo guardar el archivo:\n{e}")

        def clear_results(self):
            self.result_text.config(state=tk.NORMAL); self.result_text.delete("1.0", tk.END)
            self.result_text.config(state=tk.DISABLED); self.export_button.config(state=tk.DISABLED); self.last_ast = None
        
        def clear_all(self):
            self.source_text.delete("1.0", tk.END); self.clear_results()
            self.current_filepath = None; self.root.title("Analizador FORTRAN-77 (Tarea 3)")
        
        def show_about(self):
            messagebox.showinfo("Acerca de", "Analizador Léxico y Sintáctico para FORTRAN-77\n\nTarea 3 - Teoría de la Computación")
        
        def analyze_code(self):
            src = self.source_text.get("1.0", "end-1c")
            if not src.strip(): messagebox.showwarning("Código vacío", "No hay código fuente para analizar."); return
            self.clear_results(); self.result_text.config(state=tk.NORMAL)
            
            lexer = Lexer(src); parser = Parser(lexer); ast = parser.parse_program()
            self.last_ast = ast
            
            if parser.errors:
                self.result_text.insert(tk.END, "=== ERRORES ENCONTRADOS ===\n\n", "error_title")
                self.result_text.tag_config("error_title", font=("Arial", 11, "bold"), foreground="red")
                for e in parser.errors: self.result_text.insert(tk.END, f"{e}\n")
                self.result_text.insert(tk.END, "\n" + "="*30 + "\n\n")
            
            self.result_text.insert(tk.END, "=== ÁRBOL SINTÁCTICO (JSON) ===\n\n", "ast_title")
            self.result_text.tag_config("ast_title", font=("Arial", 11, "bold"), foreground="blue")
            try:
                json_ast = json.dumps(ast.to_json_dict(), indent=2)
                self.result_text.insert(tk.END, json_ast)
            except Exception as e: self.result_text.insert(tk.END, f"Error al serializar AST a JSON:\n{e}")
            
            self.result_text.config(state=tk.DISABLED); self.export_button.config(state=tk.NORMAL)

        def export_visual_tree(self):
            if not self.last_ast:
                messagebox.showerror("Error", "No hay un AST para exportar. Analice un archivo primero."); return
            
            try:
                dot_content = ast_to_dot(self.last_ast)
            except Exception as e:
                messagebox.showerror("Error", f"No se pudo generar el string DOT:\n{e}"); return

            default_name = "ast.png"
            if self.current_filepath:
                base_name = os.path.splitext(os.path.basename(self.current_filepath))[0]
                default_name = f"{base_name}_ast.png"

            path = filedialog.asksaveasfilename(
                title="Guardar Árbol Visual como...",
                filetypes=[
                    ("PNG Image", "*.png"),
                    ("SVG Vector Image", "*.svg"),
                    ("PDF Document", "*.pdf"),
                    ("DOT Source File", "*.dot"),
                    ("All Files", "*.*")
                ],
                defaultextension=".png",
                initialfile=default_name
            )
            
            if not path: return
            
            # Función de callback para mostrar errores/éxito en un messagebox
            def gui_callback(message):
                if "Error" in message:
                    messagebox.showerror("Error de Exportación", message)
                else:
                    messagebox.showinfo("Éxito", message)

            export_tree(dot_content, path, stderr_callback=gui_callback)


# ---------------------------------
# CLI (Lógica de Línea de Comandos)
# ---------------------------------
def run_cli(args):
    try:
        src = open(args.file, "r", encoding="utf-8").read()
    except Exception as e:
        print(f"Error al abrir '{args.file}': {e}", file=sys.stderr); sys.exit(2)

    lexer = Lexer(src); parser = Parser(lexer); ast = parser.parse_program()

    # Argumento --out (reemplaza a --dot)
    if args.out:
        try:
            dot_str = ast_to_dot(ast)
            export_tree(dot_str, args.out, stderr_callback=print) # Usa la nueva función
        except Exception as e:
            print(f"Error al generar el árbol: {e}", file=sys.stderr)

    # Imprimir AST en JSON a stdout (salida principal)
    print(json.dumps(ast.to_json_dict(), indent=2))

    if parser.errors:
        print("\n=== ERRORES ===", file=sys.stderr)
        for e in parser.errors: print(e, file=sys.stderr)
        sys.exit(1)


# ---------------------------------
# Función para ejecutar la GUI
# ---------------------------------
def run_gui():
    if not tk:
        print("Error: No se pudo importar el módulo 'tkinter'.", file=sys.stderr)
        print("La funcionalidad GUI no está disponible.", file=sys.stderr)
        sys.exit(3)
    root = tk.Tk(); app = FortranParserApp(root); root.mainloop()


# ---------------------------------
# Punto de entrada principal
# ---------------------------------
def main():
    ap = argparse.ArgumentParser(description="Parser LL(1) para subconjunto FORTRAN (tarea).")
    group = ap.add_mutually_exclusive_group()
    group.add_argument("file", nargs='?', help="Archivo fuente a analizar (modo CLI)")
    group.add_argument("--gui", action="store_true", help="Ejecutar la interfaz gráfica (modo GUI)")

    # Argumento MODIFICADO
    ap.add_argument("-o", "--out", help="Ruta de salida para el árbol visual (ej: ast.png, ast.svg, ast.dot). Solo modo CLI.")

    args = ap.parse_args()
    if args.gui: run_gui()
    elif args.file: run_cli(args)
    else: ap.print_help()

if __name__ == "__main__":
    main()