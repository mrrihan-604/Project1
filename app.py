import os
import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


def make_one_hot_encoder():
    """Create an encoder compatible with old and new scikit-learn releases."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


# ==========================================
# 1. MODEL TRAINING PIPELINE
# ==========================================
def load_and_train_model():
    dataset_name = os.path.join(os.path.dirname(__file__), "house_price_prediction_dataset.csv")

    if os.path.exists(dataset_name):
        try:
            df = pd.read_csv(dataset_name)
        except Exception:
            df = create_dummy_dataset()
    else:
        df = create_dummy_dataset()

    possible_targets = ["SalePrice", "Price", "selling_price", "Selling_Price", "house_price"]
    target_col = next((c for c in possible_targets if c in df.columns), None)

    if not target_col:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        target_col = numeric_cols[-1] if numeric_cols else df.columns[-1]

    df = df.dropna(subset=[target_col])
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Exclude the identifier and use every available property feature.
    features = [column for column in X.columns if column != "House_ID"]
    X = X[features]

    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    transformers = []
    if num_cols:
        transformers.append((
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]),
            num_cols
        ))

    if cat_cols:
        transformers.append((
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", make_one_hot_encoder())
            ]),
            cat_cols
        ))

    model_pipeline = Pipeline([
        ("preprocessor", ColumnTransformer(transformers=transformers)),
        ("regressor", LinearRegression())
    ])

    model_pipeline.fit(X, y)
    return model_pipeline, features, num_cols, cat_cols, df


def create_dummy_dataset():
    np.random.seed(42)
    n_samples = 300
    data = {
        "OverallQual": np.random.randint(3, 10, size=n_samples),
        "GrLivArea": np.random.randint(800, 3600, size=n_samples),
        "TotalBsmtSF": np.random.randint(400, 2200, size=n_samples),
        "GarageCars": np.random.randint(1, 4, size=n_samples),
        "YearBuilt": np.random.randint(1970, 2024, size=n_samples),
    }
    df = pd.DataFrame(data)
    df["SalePrice"] = (
        df["OverallQual"] * 19500
        + df["GrLivArea"] * 90
        + df["TotalBsmtSF"] * 48
        + df["GarageCars"] * 13500
        + (df["YearBuilt"] - 1950) * 380
        + np.random.normal(0, 7000, size=n_samples)
    )
    return df


pipeline, feature_names, numeric_features, categorical_features, source_df = load_and_train_model()


# ==========================================
# 2. ENHANCED MODERN TKINTER GUI
# ==========================================
class ModernHousePredictor:
    BG_MAIN = "#0F172A"
    BG_CARD = "#FFFFFF"
    BG_MUTED = "#F8FAFC"
    PRIMARY = "#3B82F6"
    PRIMARY_HOVER = "#2563EB"
    TEXT_DARK = "#1E293B"
    TEXT_MUTED = "#64748B"
    ACCENT_GREEN = "#10B981"
    BORDER_COLOR = "#E2E8F0"

    def __init__(self, root):
        self.root = root
        self.root.title("Real Estate Valuation Suite")
        self.root.geometry("560x780")
        self.root.configure(bg=self.BG_MAIN)
        self.root.resizable(False, False)

        self.entries = {}
        self.build_ui()

    def build_ui(self):
        header = tk.Frame(self.root, bg=self.BG_MAIN, pady=18)
        header.pack(fill="x")

        title = tk.Label(
            header,
            text="🏡 Real Estate Intelligence",
            font=("Segoe UI", 18, "bold"),
            fg="#FFFFFF",
            bg=self.BG_MAIN
        )
        title.pack()

        subtitle = tk.Label(
            header,
            text="Linear Regression Property Pricing Engine",
            font=("Segoe UI", 9),
            fg="#94A3B8",
            bg=self.BG_MAIN
        )
        subtitle.pack(pady=(3, 0))

        self.card = tk.Frame(self.root, bg=self.BG_CARD, padx=26, pady=22, relief="flat")
        self.card.pack(fill="both", expand=True, padx=18, pady=(0, 18))

        section_title = tk.Label(
            self.card,
            text="Property Characteristics",
            font=("Segoe UI", 11, "bold"),
            fg=self.TEXT_DARK,
            bg=self.BG_CARD
        )
        section_title.pack(anchor="w", pady=(0, 12))

        form_frame = tk.Frame(self.card, bg=self.BG_CARD)
        form_frame.pack(fill="x")

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TCombobox", fieldbackground="#F1F5F9", background="#CBD5E1")

        for idx, feat in enumerate(feature_names):
            lbl = tk.Label(
                form_frame,
                text=feat,
                font=("Segoe UI", 9, "bold"),
                fg=self.TEXT_MUTED,
                bg=self.BG_CARD
            )
            lbl.grid(row=idx, column=0, sticky="w", pady=7)

            if feat in categorical_features:
                unique_vals = list(source_df[feat].dropna().unique().astype(str))
                entry_widget = ttk.Combobox(form_frame, values=unique_vals, width=24, state="readonly")
                if unique_vals:
                    entry_widget.current(0)
            else:
                entry_widget = tk.Entry(
                    form_frame,
                    font=("Segoe UI", 9),
                    bg="#F1F5F9",
                    fg=self.TEXT_DARK,
                    relief="flat",
                    bd=6,
                    highlightthickness=1,
                    highlightcolor=self.PRIMARY,
                    highlightbackground=self.BORDER_COLOR,
                    width=25
                )
                default_val = round(float(source_df[feat].median()), 1)
                entry_widget.insert(0, str(default_val))

            entry_widget.grid(row=idx, column=1, sticky="e", pady=7, padx=(20, 0))
            self.entries[feat] = entry_widget

        btn_box = tk.Frame(self.card, bg=self.BG_CARD)
        btn_box.pack(fill="x", pady=(18, 14))

        self.btn_predict = tk.Button(
            btn_box,
            text="Estimate Selling Price 🚀",
            command=self.predict_price,
            font=("Segoe UI", 10, "bold"),
            bg=self.PRIMARY,
            fg="#FFFFFF",
            activebackground=self.PRIMARY_HOVER,
            activeforeground="#FFFFFF",
            relief="flat",
            bd=0,
            pady=10,
            cursor="hand2"
        )
        self.btn_predict.pack(fill="x")
        self.btn_predict.bind("<Enter>", lambda e: self.btn_predict.config(bg=self.PRIMARY_HOVER))
        self.btn_predict.bind("<Leave>", lambda e: self.btn_predict.config(bg=self.PRIMARY))

        btn_reset = tk.Button(
            btn_box,
            text="Reset Defaults",
            command=self.reset_inputs,
            font=("Segoe UI", 8),
            bg=self.BG_CARD,
            fg=self.TEXT_MUTED,
            relief="flat",
            cursor="hand2"
        )
        btn_reset.pack(anchor="e", pady=(6, 0))

        self.result_card = tk.Frame(
            self.card,
            bg=self.BG_MUTED,
            padx=16,
            pady=14,
            relief="flat",
            highlightthickness=1,
            highlightbackground=self.BORDER_COLOR
        )
        self.result_card.pack(fill="both", expand=True)

        self.lbl_result_tag = tk.Label(
            self.result_card,
            text="PREDICTED SALE VALUE",
            font=("Segoe UI", 8, "bold"),
            fg=self.TEXT_MUTED,
            bg=self.BG_MUTED
        )
        self.lbl_result_tag.pack()

        self.lbl_price = tk.Label(
            self.result_card,
            text="—",
            font=("Segoe UI", 22, "bold"),
            fg=self.TEXT_DARK,
            bg=self.BG_MUTED
        )
        self.lbl_price.pack(pady=(4, 6))

        self.details_box = tk.Frame(self.result_card, bg=self.BG_MUTED)
        self.details_box.pack(fill="x", pady=(4, 0))

    def reset_inputs(self):
        for feat in feature_names:
            if feat in numeric_features:
                self.entries[feat].config(bg="#F1F5F9")
                self.entries[feat].delete(0, tk.END)
                self.entries[feat].insert(0, str(round(float(source_df[feat].median()), 1)))
        self.lbl_price.config(text="—", fg=self.TEXT_DARK)
        self.clear_details()

    def clear_details(self):
        for widget in self.details_box.winfo_children():
            widget.destroy()

    def predict_price(self):
        input_data = {}
        has_error = False

        for feat in feature_names:
            raw_val = self.entries[feat].get().strip()
            if feat in numeric_features:
                try:
                    input_data[feat] = [float(raw_val)]
                    self.entries[feat].config(bg="#F1F5F9")
                except ValueError:
                    self.entries[feat].config(bg="#FEE2E2")
                    has_error = True
            else:
                input_data[feat] = [raw_val]

        if has_error:
            messagebox.showerror("Invalid Input", "Highlighted fields must be valid numeric values.")
            return

        try:
            input_df = pd.DataFrame(input_data)
            pred = pipeline.predict(input_df)[0]
            val = max(0.0, pred)

            self.lbl_price.config(text=f"${val:,.2f}", fg=self.ACCENT_GREEN)

            self.clear_details()
            row = 0
            for k, v in input_data.items():
                chip = tk.Label(
                    self.details_box,
                    text=f"{k}: {v[0]}",
                    font=("Segoe UI", 8),
                    bg="#E2E8F0",
                    fg=self.TEXT_DARK,
                    padx=6,
                    pady=2
                )
                chip.grid(row=row // 2, column=row % 2, padx=4, pady=3, sticky="ew")
                row += 1

            self.details_box.grid_columnconfigure(0, weight=1)
            self.details_box.grid_columnconfigure(1, weight=1)

        except Exception as err:
            messagebox.showerror("Prediction Error", str(err))


# ==========================================
# 3. RUNTIME
# ==========================================
if __name__ == "__main__":
    root = tk.Tk()
    app = ModernHousePredictor(root)
    root.mainloop()