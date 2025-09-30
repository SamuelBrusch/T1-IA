import tkinter as tk
from tkinter import ttk, messagebox
import os
import joblib
import pandas as pd
import numpy as np
import random

from config import MODELS_DIR, METADATA_PATH, RESULTS_PATH
from data_and_train import winner_on_board, is_board_full, board_state

ALGORITHM_LABELS = {
	"knn": "k-NN",
	"mlp": "MLP",
	"tree": "Árvore de Decisão",
	"forest": "Random Forest",
}

def get_model_paths_for_key(key: str):
	model_path = os.path.join(MODELS_DIR, f"{key}_model.pkl")
	scaler_path = os.path.join(MODELS_DIR, f"{key}_scaler.pkl")
	return model_path, scaler_path if os.path.exists(scaler_path) else None

class TicTacToeGUI(tk.Tk):
	def __init__(self):
		super().__init__()
		self.title("Jogo da Velha - IA")
		self.resizable(False, False)

		self.board = ['b'] * 9
		self.current_player = 'x'
		self.model = None
		self.scaler = None
		self.correct = 0
		self.total = 0

		self._build_ui()
		self._build_metrics_ui()
		self._load_selected_model()
		self._update_prediction()
		self._load_metrics()

	def _build_ui(self):
		top = ttk.Frame(self, padding=10)
		top.grid(row=0, column=0, sticky="nsew")

		# Algorithm selection
		sel_frame = ttk.Frame(top)
		sel_frame.grid(row=0, column=0, columnspan=3, pady=(0,10), sticky="w")
		ttk.Label(sel_frame, text="Algoritmo:").grid(row=0, column=0, padx=(0,8))
		self.algo_var = tk.StringVar(value="mlp")
		self.algo_combo = ttk.Combobox(sel_frame, textvariable=self.algo_var, state="readonly",
									   values=[f"{k} - {v}" for k,v in ALGORITHM_LABELS.items()])
		self.algo_combo.grid(row=0, column=1)
		self.algo_combo.bind('<<ComboboxSelected>>', lambda e: self._load_selected_model())

		# Board buttons
		self.buttons = []
		grid = ttk.Frame(top)
		grid.grid(row=1, column=0, columnspan=3)
		for i in range(3):
			for j in range(3):
				idx = i*3 + j
				btn = ttk.Button(grid, text=" ", width=6, command=lambda ix=idx: self._on_click(ix))
				btn.grid(row=i, column=j, padx=4, pady=4)
				self.buttons.append(btn)

		# Status and metrics
		self.pred_var = tk.StringVar(value="Predição: -")
		self.real_var = tk.StringVar(value="Real: -")
		self.acc_var = tk.StringVar(value="Acurácia: 0.0% (0/0)")

		ttk.Label(top, textvariable=self.pred_var).grid(row=2, column=0, sticky="w", pady=(10,0))
		ttk.Label(top, textvariable=self.real_var).grid(row=2, column=1, sticky="w", pady=(10,0))
		ttk.Label(top, textvariable=self.acc_var).grid(row=2, column=2, sticky="e", pady=(10,0))

		# Controls
		ctrl = ttk.Frame(top)
		ctrl.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(10,0))
		ttk.Button(ctrl, text="Novo Jogo", command=self._reset).grid(row=0, column=0)

	def _build_metrics_ui(self):
		wrapper = ttk.Labelframe(self, text="Métricas dos Algoritmos", padding=10)
		wrapper.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0,10))

		# Best model summary
		self.best_var = tk.StringVar(value="Melhor modelo: -")
		ttk.Label(wrapper, textvariable=self.best_var).grid(row=0, column=0, sticky="w")

		# Table of results
		cols = ("Algoritmo/Config", "Val")
		self.metrics_table = ttk.Treeview(wrapper, columns=cols, show='headings', height=6)
		self.metrics_table.heading(cols[0], text=cols[0])
		self.metrics_table.heading(cols[1], text=cols[1])
		self.metrics_table.column(cols[0], width=380)
		self.metrics_table.column(cols[1], width=80, anchor='e')
		self.metrics_table.grid(row=1, column=0, sticky="nsew", pady=(6,0))

		# Scrollbar
		sb = ttk.Scrollbar(wrapper, orient="vertical", command=self.metrics_table.yview)
		self.metrics_table.configure(yscroll=sb.set)
		sb.grid(row=1, column=1, sticky="ns")

		# Refresh button
		ttk.Button(wrapper, text="Recarregar métricas", command=self._load_metrics).grid(row=2, column=0, sticky="e", pady=(6,0))

	def _encode_board(self):
		columns = ['tl','tm','tr','ml','mm','mr','bl','bm','br']
		row = [{'x':2,'o':1,'b':0}[v] for v in self.board]
		df = pd.DataFrame([row], columns=columns)
		if self.scaler:
			scaled = self.scaler.transform(df.values.astype(float))
			df = pd.DataFrame(scaled, columns=columns)
		return df

	def _predict(self):
		if not self.model:
			return None
		enc = self._encode_board()
		if hasattr(self.model, 'feature_names_in_'):
			try:
				cols = list(self.model.feature_names_in_)
				if hasattr(enc, 'reindex'):
					enc = enc.reindex(columns=cols)
				else:
					enc = pd.DataFrame(enc, columns=cols)
				X = enc
			except Exception:
				X = enc.values if hasattr(enc, 'values') else enc
		else:
			X = enc.values if hasattr(enc, 'values') else enc
		return self.model.predict(X)[0]

	def _update_prediction(self):
		pred = self._predict()
		real = board_state(self.board)
		if pred is not None:
			self.total += 1
			if pred == real:
				self.correct += 1
		self.pred_var.set(f"Predição: {pred}")
		self.real_var.set(f"Real: {real}")
		acc = (self.correct / self.total * 100) if self.total else 0.0
		self.acc_var.set(f"Acurácia: {acc:.1f}% ({self.correct}/{self.total})")

	def _load_selected_model(self):
		key = self.algo_var.get().split(' ')[0] if ' ' in self.algo_var.get() else self.algo_var.get()
		if key not in ALGORITHM_LABELS:
			key = 'mlp'
			self.algo_var.set('mlp - MLP')
		model_path, scaler_path = get_model_paths_for_key(key)
		if not os.path.exists(model_path):
			messagebox.showinfo("Modelo ausente", f"Modelo '{key}' não encontrado. Treine com 'python main.py --train'.")
			self.model = None
			self.scaler = None
			return
		self.model = joblib.load(model_path)
		self.scaler = joblib.load(scaler_path) if scaler_path and os.path.exists(scaler_path) else None

	def _load_metrics(self):
		for i in self.metrics_table.get_children():
			self.metrics_table.delete(i)
		best_text = "Melhor modelo: -"
		try:
			if os.path.exists(METADATA_PATH):
				meta = joblib.load(METADATA_PATH)
				best_text = f"Melhor modelo: {meta.get('best_model','-')} (Score: {meta.get('best_score',0):.3f})"
		except Exception:
			best_text = "Melhor modelo: (erro ao carregar)"
		self.best_var.set(best_text)

		try:
			if os.path.exists(RESULTS_PATH):
				df = pd.read_csv(RESULTS_PATH)
				if 'Validation_Score' in df.columns:
					df = df.sort_values('Validation_Score', ascending=False)
				for _, row in df.iterrows():
					algo = str(row.get('Algorithm',''))
					val = row.get('Validation_Score', '')
					try:
						val_str = f"{float(val):.3f}"
					except Exception:
						val_str = str(val)
					self.metrics_table.insert('', 'end', values=(algo, val_str))
		except Exception:
			pass

	def _on_click(self, idx: int):
		if self.board[idx] != 'b' or self._game_over():
			return
		self.board[idx] = 'x'
		self._render()
		if self._game_over():
			self._finalize()
			return
		empty = [i for i,v in enumerate(self.board) if v == 'b']
		if empty:
			ai_pos = random.choice(empty)
			self.board[ai_pos] = 'o'
		self._render()
		self._update_prediction()
		if self._game_over():
			self._finalize()

	def _render(self):
		for i in range(9):
			ch = ' '
			if self.board[i] == 'x':
				ch = 'X'
			elif self.board[i] == 'o':
				ch = 'O'
			self.buttons[i]["text"] = ch

	def _reset(self):
		self.board = ['b'] * 9
		self.current_player = 'x'
		self.correct = 0
		self.total = 0
		self._render()
		self._update_prediction()

	def _game_over(self):
		return winner_on_board(self.board) is not None or is_board_full(self.board)

	def _finalize(self):
		win = winner_on_board(self.board)
		if win:
			messagebox.showinfo("Fim de Jogo", f"Vencedor: {'X' if win=='x' else 'O'}")
		else:
			messagebox.showinfo("Fim de Jogo", "Empate")

def run_gui():
	app = TicTacToeGUI()
	app.mainloop()
