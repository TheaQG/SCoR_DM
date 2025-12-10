import os
import zarr # type: ignore
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from sklearn.feature_selection import mutual_info_regression
from sklearn.cross_decomposition import CCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from typing import List, Tuple


class DataCorrelationAnalyzer:
    def __init__(self,
                 hr_model: str,
                 lr_model: str,
                 hr_var: str,
                 lr_vars: List[str],
                 split_type: str = 'test',
                 path_data: str = './',
                 cutout: Tuple[int, int, int, int] = (170, 170+180, 340, 340+180),
                 save_figs: bool = False,
                 fig_path: str = './correlation_figs'):

        self.hr_model = hr_model
        self.lr_model = lr_model
        self.hr_var = hr_var
        self.lr_vars = lr_vars
        self.split_type = split_type
        self.path_data = path_data
        self.cutout = cutout
        self.save_figs = save_figs
        self.fig_path = fig_path

        os.makedirs(self.fig_path, exist_ok=True)

    def _build_path(self, model: str, var: str):
        return os.path.join(self.path_data,
                            f'data_{model}',
                            'size_589x789',
                            f'{var}_589x789',
                            'zarr_files',
                            f'{self.split_type}.zarr')

    def _load_zarr_stack(self, path: str, var_str: str):
        print(path)
        zgroup = zarr.open_group(path, mode='r')
        keys = sorted(zgroup.keys())
        data_list = []
        for k in keys:
            arr = None
            if var_str in zgroup[k]:
                arr = zgroup[k][var_str][:].squeeze() # type: ignore
            elif 'arr_0' in zgroup[k]:
                arr = zgroup[k]['arr_0'][:].squeeze() # type: ignore
            if arr is None:
                continue
            cut = arr[self.cutout[0]:self.cutout[1], self.cutout[2]:self.cutout[3]]
            data_list.append(cut)
        return np.stack(data_list)  # (T, H, W)

    def _map_var(self, var: str, model: str):
        if var == 'temp': return 't'
        if var == 'prcp': return 'tp'
        return var

    def load_data(self):
        print("Loading HR data...")
        hr_path = self._build_path(self.hr_model, self.hr_var)
        self.hr_data = self._load_zarr_stack(hr_path, self._map_var(self.hr_var, self.hr_model))

        self.lr_data = {}
        for var in self.lr_vars:
            print(f"Loading LR variable: {var}...")
            lr_path = self._build_path(self.lr_model, var)
            self.lr_data[var] = self._load_zarr_stack(lr_path, self._map_var(var, self.lr_model))

    def compute_spatial_correlation_maps(self):
        T, H, W = self.hr_data.shape
        corr_maps = {}
        for var, lr in self.lr_data.items():
            corr_map = np.zeros((H, W))
            for i in range(H):
                for j in range(W):
                    r, _ = pearsonr(lr[:, i, j], self.hr_data[:, i, j])
                    corr_map[i, j] = r
            corr_maps[var] = corr_map
        self.corr_maps = corr_maps
        return corr_maps

    def compute_temporal_correlations(self, method='pearson'):
        T, H, W = self.hr_data.shape
        correlations = {}
        for var, lr in self.lr_data.items():
            if method == 'pearson':
                func = pearsonr
            elif method == 'spearman':
                func = spearmanr
            else:
                raise ValueError("Unsupported method")

            ts_corr = np.zeros((H, W))
            for i in range(H):
                for j in range(W):
                    r, _ = func(lr[:, i, j], self.hr_data[:, i, j])
                    ts_corr[i, j] = r
            correlations[var] = ts_corr
        return correlations

    def compute_lagged_correlation(self, max_lag=5):
        T, H, W = self.hr_data.shape
        lag_corrs = {}
        for var, lr in self.lr_data.items():
            corrs = []
            for lag in range(-max_lag, max_lag + 1):
                if lag < 0:
                    shifted = lr[:lag]
                    target = self.hr_data[-lag:]
                elif lag > 0:
                    shifted = lr[lag:]
                    target = self.hr_data[:-lag]
                else:
                    shifted = lr
                    target = self.hr_data
                r = np.corrcoef(shifted.reshape(shifted.shape[0], -1),
                                target.reshape(target.shape[0], -1))[0, 1]
                corrs.append(r)
            lag_corrs[var] = corrs
        return lag_corrs

    def compute_mutual_information(self):
        mi_maps = {}
        T, H, W = self.hr_data.shape
        y = self.hr_data.reshape(T, -1).mean(axis=1)
        for var, lr in self.lr_data.items():
            X = lr.reshape(T, -1)
            mi = mutual_info_regression(X, y)
            mi_maps[var] = mi.reshape(H, W)
        return mi_maps

    def compute_eof_analysis(self, n_modes=3):
        T, H, W = self.hr_data.shape
        eof_maps = {}
        for var, lr in self.lr_data.items():
            X = lr.reshape(T, -1)
            pca = PCA(n_components=n_modes)
            pca.fit(X)
            eof_maps[var] = pca.components_.reshape((n_modes, H, W))
        return eof_maps

    def compute_composite_maps(self, threshold=0.9):
        T, H, W = self.hr_data.shape
        composites = {}
        target = self.hr_data.reshape(T, -1).mean(axis=1)
        thresh_val = np.quantile(target, threshold)
        mask = target > thresh_val
        for var, lr in self.lr_data.items():
            comp_map = lr[mask].mean(axis=0)
            composites[var] = comp_map
        return composites

    def compute_cca(self, n_components=2):
        cca_results = {}
        T, H, W = self.hr_data.shape
        Y = self.hr_data.reshape(T, -1)
        for var, lr in self.lr_data.items():
            X = lr.reshape(T, -1)
            cca = CCA(n_components=n_components)
            cca.fit(X, Y)
            X_c, Y_c = cca.transform(X, Y)
            cca_results[var] = (X_c, Y_c)
        return cca_results

    def compute_rf_feature_importance(self):
        rf_results = {}
        T, H, W = self.hr_data.shape
        y = self.hr_data.reshape(T, -1).mean(axis=1)
        for var, lr in self.lr_data.items():
            X = lr.reshape(T, -1)
            rf = RandomForestRegressor(n_estimators=50)
            rf.fit(X, y)
            rf_results[var] = rf.feature_importances_.reshape(H, W)
        return rf_results

    def visualize_correlation_maps(self):
        for var, cmap in self.corr_maps.items():
            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(cmap, cmap='RdBu_r', vmin=-1, vmax=1)
            plt.colorbar(im, ax=ax)
            ax.set_title(f"Spatial correlation: {var} vs {self.hr_var}")
            plt.tight_layout()
            if self.save_figs:
                out_path = os.path.join(self.fig_path, f'{var}_vs_{self.hr_var}_corr.png')
                fig.savefig(out_path, dpi=300)
            plt.show()

    def run(self):
        self.load_data()
        self.compute_spatial_correlation_maps()
        self.visualize_correlation_maps()


if __name__ == '__main__':
    analyzer = DataCorrelationAnalyzer(
        hr_model='DANRA',
        lr_model='ERA5',
        hr_var='prcp',
        lr_vars=['temp', 'prcp'],
        path_data='/scratch/project_xxxxxxxxx/user/Data/Data_DiffMod_small/',
        save_figs=True
    )
    analyzer.run()
