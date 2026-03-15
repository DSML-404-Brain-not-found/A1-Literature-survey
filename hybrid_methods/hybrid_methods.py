"""
hybrid_methods.py
=================
處理不平衡二元分類資料的混合方法（Hybrid Methods）實作。
共包含 5 種方法，分屬三大架構族：

  StandardAdaBoost 家族（共用 StandardAdaBoost 迴圈）
    1. SMOTEBoost  ── Chawla et al. (2003)
    2. RUSBoost    ── Seiffert et al. (2010)
    3. RHSBoost    ── Gong & Kim (2017)

  特化演算法（獨立修改 Alpha 計算）
    4. SUBoost     ── Baghmishe et al. (2025)

  Pipeline 家族（單次 SMOTE + Classifier）
    5. SMOTECSL    ── Thai-Nghe et al. (2010)

重要設計原則
-----------
* 訓練弱分類器使用重採樣後的 (X_res, y_res, w_res)。
* 計算錯誤率、Alpha 以及更新樣本權重，**一律使用原始 (X_t, y_t, w_t)**，
  確保標準 AdaBoost 理論正確性（weight isolation）。

引擎版本說明
-----------
在二元分類情境下，AdaBoost.M2 的偽損失與權重矩陣退化為標準
AdaBoost（AdaBoost.M1 / SAMME）。為確保實驗控制變因絕對嚴謹，
本實作直接採用 StandardAdaBoost 作為所有 Data-level 方法的底層引擎。

- 弱分類器設定：預設使用 `max_depth=5` 的決策樹（近似 C4.5 演算法）。
- 學術依據：雖然原始 AdaBoost 理論多採用 `max_depth=1`，但為應付不平衡資料的複雜邊界，
本實作對齊 Chawla (2003) SMOTEBoost 與 Seiffert (2010) RUSBoost 等文獻之設定採用較深的決策樹。
- 實驗控制變因：Baseline 與所有重採樣混合方法統一採用此預設分類器，確保實驗效能之差異 100% 
源自於「資料抽樣策略」，而非底層模型結構之不同。
"""

from __future__ import annotations

import numpy as np
from collections import Counter
from typing import Optional, Tuple

from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler


# ============================================================
# 工具函式
# ============================================================

def _safe_log(x: float) -> float:
    """避免 log(0) 的保護性對數計算。"""
    return np.log(max(x, 1e-10))


def _normalize(w: np.ndarray) -> np.ndarray:
    """將權重陣列正規化使其總和為 1。"""
    total = w.sum()
    if total == 0:
        return np.ones_like(w) / len(w)
    return w / total


# ============================================================
# 核心引擎：StandardAdaBoost
# ============================================================

class StandardAdaBoost(BaseEstimator, ClassifierMixin):
    """
    標準 AdaBoost（SAMME）基底類別。

    文獻依據
    --------
    Freund, Y., & Schapire, R. E. (1997).
    "A decision-theoretic generalization of on-line learning and an
    application to boosting." Journal of Computer and System Sciences,
    55(1), 119–139.

    在二元分類情境下，SAMME 與 AdaBoost.M1 數學上完全等價：
        α_t = 0.5 * ln((1 - ε_t) / ε_t)
    其中 ε_t 為加權分類錯誤率。

    架構設計
    --------
    子類別只需覆寫 ``_resample(X, y, w)`` 方法，即可插入不同的
    重採樣策略（SMOTE、RUS、ROSE 等），而不改動任何 Alpha 計算
    或權重更新邏輯。

    本類別同時作為實驗的 Baseline（無重採樣）模型。

    所有具有隨機性的元件均透過 random_state 統一控制，
    確保實驗可重現性。

    權重防呆機制（weight isolation）
    ---------------------------------
    1. 弱分類器訓練：使用重採樣後的 ``(X_res, y_res, w_res)``。
    2. 錯誤率計算、Alpha 計算、權重更新：**只使用原始 (X_t, y_t, w_t)**。
    這確保重採樣不會汙染 AdaBoost 的理論收斂性。
    """

    def __init__(
        self,
        n_estimators: int = 100,
        base_estimator: Optional[BaseEstimator] = None,
        random_state: Optional[int] = None,
    ) -> None:
        self.n_estimators = n_estimators
        # 預設使用 max_depth=5 的決策樹（近似 C4.5 演算法），作為所有不平衡學習混合方法的控制變因基準
        self.base_estimator = (
            base_estimator
            if base_estimator is not None
            else DecisionTreeClassifier(max_depth=5, random_state=random_state)
        )
        self.random_state = random_state

    # ----------------------------------------------------------
    # 重採樣接口（子類別覆寫此方法）
    # ----------------------------------------------------------

    def _resample(
        self,
        X: np.ndarray,
        y: np.ndarray,
        w: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        對 (X, y, w) 進行重採樣並回傳 (X_res, y_res, w_res)。
        預設不做任何處理（直接回傳原資料），供子類別覆寫。
        """
        return X, y, w

    # ----------------------------------------------------------
    # 主訓練迴圈
    # ----------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "StandardAdaBoost":
        """
        執行標準 AdaBoost 主迴圈。

        Parameters
        ----------
        X : shape (n_samples, n_features)
        y : shape (n_samples,)，二元標籤（0 / 1）
        """
        n_samples = len(y)
        self.classes_ = np.unique(y)

        # 初始化均勻樣本權重
        w_t: np.ndarray = np.full(n_samples, 1.0 / n_samples)

        self.estimators_: list = []
        self.estimator_weights_: list[float] = []

        # 儲存原始資料副本，供 weight isolation 使用
        X_t, y_t = X.copy(), y.copy()

        for t in range(self.n_estimators):
            # -----------------------------------------------
            # Step 1：重採樣（呼叫子類別接口）
            # 回傳重採樣後的訓練資料及對應權重
            # -----------------------------------------------
            X_res, y_res, w_res = self._resample(X_t, y_t, w_t)

            # -----------------------------------------------
            # Step 2：訓練弱分類器（使用重採樣後的資料）
            # -----------------------------------------------
            clf = clone(self.base_estimator)
            # 確保每輪使用不同但可重現的隨機種子
            if hasattr(clf, "random_state"):
                clf.set_params(
                    random_state=(
                        None if self.random_state is None
                        else self.random_state + t
                    )
                )
            clf.fit(X_res, y_res, sample_weight=w_res)

            # -----------------------------------------------
            # Step 3：在「原始」資料上預測（weight isolation）
            # 計算加權錯誤率 ε_t
            # -----------------------------------------------
            y_pred_orig = clf.predict(X_t)
            incorrect = (y_pred_orig != y_t).astype(float)
            epsilon_t = float(np.dot(w_t, incorrect))

            # 若錯誤率 >= 0.5 或 == 0，提前終止（標準 AdaBoost 做法）
            if epsilon_t >= 0.5 or epsilon_t == 0.0:
                break

            # -----------------------------------------------
            # Step 4：計算分類器權重 α_t（SAMME 公式）
            # α_t = 0.5 * ln((1 - ε_t) / ε_t)
            # -----------------------------------------------
            alpha_t = 0.5 * _safe_log((1.0 - epsilon_t) / epsilon_t)

            self.estimators_.append(clf)
            self.estimator_weights_.append(alpha_t)

            # -----------------------------------------------
            # Step 5：更新原始樣本權重（weight isolation）
            # 正確分類 → exp(-α)，錯誤分類 → exp(+α)
            # -----------------------------------------------
            y_signed = np.where(y_t == y_pred_orig, 1.0, -1.0)
            w_t = w_t * np.exp(-alpha_t * y_signed)
            w_t = _normalize(w_t)

        return self

    # ----------------------------------------------------------
    # 預測方法
    # ----------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        集成預測：對每個弱分類器的預測結果加權投票。
        """
        if not self.estimators_:
            raise RuntimeError("模型尚未訓練，請先呼叫 fit()。")

        # 累計加權投票分數（shape: n_samples）
        scores = np.zeros(len(X))
        for clf, alpha in zip(self.estimators_, self.estimator_weights_):
            # 將預測標籤由 {0,1} 轉為 {-1,+1} 後加權
            y_pred = clf.predict(X)
            scores += alpha * np.where(y_pred == 1, 1.0, -1.0)

        return (scores >= 0.0).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        回傳軟性機率（以加權投票分數透過 sigmoid 轉換）。
        """
        if not self.estimators_:
            raise RuntimeError("模型尚未訓練，請先呼叫 fit()。")

        scores = np.zeros(len(X))
        total_alpha = sum(self.estimator_weights_) or 1.0
        for clf, alpha in zip(self.estimators_, self.estimator_weights_):
            y_pred = clf.predict(X)
            scores += alpha * np.where(y_pred == 1, 1.0, -1.0)

        # 正規化至 [-1, 1] 再透過 sigmoid
        prob_pos = 1.0 / (1.0 + np.exp(-2.0 * scores / total_alpha))
        return np.column_stack([1.0 - prob_pos, prob_pos])


# ============================================================
# 方法一：SMOTEBoost
# ============================================================

class SMOTEBoost(StandardAdaBoost):
    """
    SMOTEBoost：在每輪 AdaBoost 迴圈前以 SMOTE 過採樣少數類。

    文獻依據
    --------
    Chawla, N. V., Lazarevic, A., Hall, L. O., & Bowyer, K. W. (2003).
    "SMOTEBoost: Improving Prediction of the Minority Class in Boosting."
    PKDD 2003. LNCS 2838, pp. 107–119.

    論文對應重點
    -----------
    * 演算法基於 AdaBoost（論文 Algorithm 1）。
    * 每輪開始時，以 SMOTE 對少數類進行過採樣，生成合成樣本。
    * 合成樣本賦予「均勻初始權重」（論文 Section 3.1），
      與原始樣本的現有權重合併後，重新正規化。
    * 錯誤率計算與權重更新僅使用原始樣本（StandardAdaBoost 保證此點）。
    """

    def __init__(
        self,
        n_estimators: int = 100,
        base_estimator: Optional[BaseEstimator] = None,
        k_neighbors: int = 5,
        random_state: Optional[int] = None,
    ) -> None:
        super().__init__(
            n_estimators=n_estimators,
            base_estimator=base_estimator,
            random_state=random_state,
        )
        self.k_neighbors = k_neighbors

    def _resample(
        self,
        X: np.ndarray,
        y: np.ndarray,
        w: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        使用 SMOTE 過採樣，保留原始樣本現有權重，
        並為合成樣本賦予均勻初始權重，最後合併正規化。
        """
        n_orig = len(y)
        smote = SMOTE(
            k_neighbors=self.k_neighbors,
            random_state=self.random_state,
        )

        try:
            X_res, y_res = smote.fit_resample(X, y)
        except ValueError:
            # 少數類樣本數不足以執行 SMOTE 時，直接回傳原資料
            return X, y, w

        n_synthetic = len(y_res) - n_orig

        if n_synthetic <= 0:
            return X, y, w

        # 合成樣本的均勻初始權重（以整體樣本數計算）
        uniform_w = 1.0 / len(y_res)
        w_synthetic = np.full(n_synthetic, uniform_w)

        # 合併原始權重與合成樣本均勻權重
        w_res = np.concatenate([w, w_synthetic])
        w_res = _normalize(w_res)

        return X_res, y_res, w_res


# ============================================================
# 方法二：RUSBoost
# ============================================================

class RUSBoost(StandardAdaBoost):
    """
    RUSBoost：在每輪 AdaBoost 迴圈前以隨機欠採樣去除多數類樣本。

    文獻依據
    --------
    Seiffert, C., Khoshgoftaar, T. M., Van Hulse, J., & Napolitano, A.
    (2010). "RUSBoost: A Hybrid Approach to Alleviating Class Imbalance."
    IEEE Transactions on Systems, Man, and Cybernetics, 40(1), 185–197.

    論文對應重點
    -----------
    * 演算法同樣基於 AdaBoost（論文 Algorithm 1）。
    * 每輪對多數類隨機欠採樣（RUS），並透過 ``sample_indices_``
      取出對應保留的原始權重（論文 Section II-B 3rd step）。
    * 保留樣本的權重直接繼承原始權重並正規化（不重新初始化）。
    * 錯誤率 / Alpha / 權重更新仍使用完整原始集（StandardAdaBoost 保證）。
    """

    def __init__(
        self,
        n_estimators: int = 100,
        base_estimator: Optional[BaseEstimator] = None,
        sampling_strategy: float | str = "auto",
        random_state: Optional[int] = None,
    ) -> None:
        super().__init__(
            n_estimators=n_estimators,
            base_estimator=base_estimator,
            random_state=random_state,
        )
        self.sampling_strategy = sampling_strategy

    def _resample(
        self,
        X: np.ndarray,
        y: np.ndarray,
        w: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        隨機欠採樣多數類，並從原始樣本繼承對應的樣本權重。
        """
        rus = RandomUnderSampler(
            sampling_strategy=self.sampling_strategy,
            random_state=self.random_state,
            replacement=False,
        )
        rus.fit_resample(X, y)

        # sample_indices_：RandomUnderSampler 保留的原始樣本索引
        indices = rus.sample_indices_
        X_res = X[indices]
        y_res = y[indices]

        # 繼承原始權重並正規化
        w_res = w[indices]
        w_res = _normalize(w_res)

        return X_res, y_res, w_res


# ============================================================
# 方法三：RHSBoost
# ============================================================

class RHSBoost(RUSBoost):
    """
    RHSBoost：RUS + ROSE 結合的混合採樣 Boosting 方法。
    本實作以 SMOTE 模擬論文中的 ROSE（兩者均為基於核密度的合成過採樣）。

    為避免開源套件預設之 `auto` 參數導致混合採樣失效，本實作顯式地（Explicitly）分離了欠採樣與過採樣比例。預設先以 RUS (`rus_strategy=0.5`) 保留適度比例的多數類資訊，再以 SMOTE (`smote_strategy='auto'`) 補足少數類，確保決策邊界能真正融合兩者優勢。

    文獻依據
    --------
    Gong, J., & Kim, H. (2017).
    "RHSBoost: Improving classification performance in imbalanced data."
    Computational Statistics & Data Analysis, 111, 1–13.

    論文對應重點
    -----------
    * RHSBoost = RUSBoost 欠採樣後再做 ROSE 過採樣（論文 Section 3）。
    * 步驟：
        1. 先呼叫 RUSBoost._resample() 取得欠採樣結果與繼承權重。
        2. 對欠採樣後的資料執行 SMOTE 過採樣。
        3. 合成樣本給予均勻初始權重（如 SMOTEBoost）。
        4. 原始保留樣本繼承 RUS 後的正規化權重。
        5. 合併後再次正規化。
    """

    def __init__(
        self,
        n_estimators: int = 100,
        base_estimator: Optional[BaseEstimator] = None,
        rus_strategy: float = 0.5,
        smote_strategy: float | str = "auto",
        k_neighbors: int = 5,
        random_state: Optional[int] = None,
    ) -> None:
        super().__init__(
            n_estimators=n_estimators,
            base_estimator=base_estimator,
            sampling_strategy=rus_strategy,
            random_state=random_state,
        )
        self.smote_strategy = smote_strategy
        self.k_neighbors = k_neighbors

    def _resample(
        self,
        X: np.ndarray,
        y: np.ndarray,
        w: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        先 RUS 欠採樣（繼承 RUSBoost._resample），
        再對欠採樣結果執行 SMOTE 過採樣。
        """
        # Step 1：執行 RUS，取得欠採樣後的資料與繼承權重
        X_rus, y_rus, w_rus = super()._resample(X, y, w)

        n_rus = len(y_rus)

        # Step 2：對欠採樣後的資料嘗試 SMOTE 過採樣
        smote = SMOTE(
            sampling_strategy=self.smote_strategy,
            k_neighbors=self.k_neighbors,
            random_state=self.random_state,
        )
        try:
            X_res, y_res = smote.fit_resample(X_rus, y_rus)
        except ValueError:
            # 少數類數量不足 SMOTE 所需時，直接使用 RUS 結果
            return X_rus, y_rus, w_rus

        n_synthetic = len(y_res) - n_rus

        if n_synthetic <= 0:
            return X_rus, y_rus, w_rus

        # Step 3：合成樣本使用均勻初始權重
        uniform_w = 1.0 / len(y_res)
        w_synthetic = np.full(n_synthetic, uniform_w)

        # Step 4：合併 RUS 保留樣本的繼承權重與合成樣本均勻權重
        w_res = np.concatenate([w_rus, w_synthetic])
        w_res = _normalize(w_res)

        return X_res, y_res, w_res


# ============================================================
# 方法四：SUBoost（獨立實作，不繼承 StandardAdaBoost）
# ============================================================

class SUBoost(BaseEstimator, ClassifierMixin):
    """
    SUBoost：選擇性欠採樣 Boosting，將 Alpha 拆分為少數/多數類。

    文獻依據
    --------
    Baghmishe, M. R., et al. (2025).
    "SUBoost: A Novel Boosting-Based Selective Undersampling for
    Handling Imbalanced Data."

    論文對應重點
    -----------
    此方法在演算法層次修改了標準 AdaBoost 的 Alpha 計算，
    因此**必須獨立撰寫 fit 迴圈**，不能繼承 StandardAdaBoost。

    1. 拆分錯誤率：
       - ε_min：少數類的加權分類錯誤率
       - ε_maj：多數類的加權分類錯誤率

    2. 拆分分類器權重：
       - α_min = 0.5 * ln((1 - ε_min) / ε_min)
       - α_maj = 0.5 * ln((1 - ε_maj) / ε_maj)

    3. 不對稱權重更新：
       - 少數類樣本：w *= exp(-α_min * margin)
       - 多數類樣本：w *= exp(-α_maj * margin)

    4. 選擇性欠採樣（每輪結尾）：
       依當前樣本權重分佈，以 rng.choice 從原始樣本中
       重採樣，讓高權重（難分類）樣本更容易被選上，實現
       「自動剃除容易分類的多數類」。

    與原始文獻對齊說明
    ------------------
    * 演算法框架：完全對齊文獻的 Algorithm 1。
    * 基分類器：文獻使用 C4.5（未修剪或淺層修剪），為對齊其實際複雜度，
      此處與 StandardAdaBoost 及所有其他重採樣混合方法保持一致，
      統一使用 sklearn.tree.DecisionTreeClassifier(max_depth=5) 作為逼近。
      同樣保留 random_state 以確保實驗控制變因可重現。
    """

    def __init__(
        self,
        n_estimators: int = 100,
        base_estimator: Optional[BaseEstimator] = None,
        subsample_size: Optional[int] = None,
        random_state: Optional[int] = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.base_estimator = (
            base_estimator
            if base_estimator is not None
            else DecisionTreeClassifier(max_depth=5, random_state=random_state)
        )
        # 每輪選擇性欠採樣保留的樣本數；None 表示保留原始樣本數的一半
        self.subsample_size = subsample_size
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SUBoost":
        """
        執行 SUBoost 主訓練迴圈。
        """
        # 使用 np.random.RandomState 確保可重現性，避免污染全域隨機狀態
        rng = np.random.RandomState(self.random_state)

        n_samples = len(y)
        self.classes_ = np.unique(y)

        minority_label = min(Counter(y), key=lambda c: Counter(y)[c])
        majority_label = 1 - minority_label

        min_idx = np.where(y == minority_label)[0]
        maj_idx = np.where(y == majority_label)[0]

        # 初始均勻權重
        w_t: np.ndarray = np.full(n_samples, 1.0 / n_samples)

        self.estimators_: list = []
        self.alphas_min_: list[float] = []
        self.alphas_maj_: list[float] = []

        # 當前使用的訓練集（起初為全量資料）
        X_curr, y_curr, w_curr = X.copy(), y.copy(), w_t.copy()
        curr_min_mask = y_curr == minority_label
        curr_maj_mask = y_curr == majority_label

        # 欠採樣目標大小
        subsample_size = self.subsample_size or max(
            int(n_samples * 0.5), len(min_idx) * 2
        )

        for t in range(self.n_estimators):
            # -----------------------------------------------
            # Step 1：訓練弱分類器（使用當前訓練集）
            # -----------------------------------------------
            clf = clone(self.base_estimator)
            if hasattr(clf, "random_state"):
                clf.set_params(
                    random_state=(
                        None if self.random_state is None
                        else self.random_state + t
                    )
                )
            clf.fit(X_curr, y_curr, sample_weight=w_curr)

            y_pred = clf.predict(X_curr)
            incorrect = (y_pred != y_curr)

            # -----------------------------------------------
            # Step 2：拆分計算少數類 / 多數類加權錯誤率
            # -----------------------------------------------
            # 少數類錯誤率
            w_min = w_curr[curr_min_mask]
            err_min_mask = incorrect[curr_min_mask]
            w_min_sum = w_min.sum() or 1e-10
            epsilon_min = float(np.dot(w_min / w_min_sum, err_min_mask.astype(float)))

            # 多數類錯誤率
            w_maj = w_curr[curr_maj_mask]
            err_maj_mask = incorrect[curr_maj_mask]
            w_maj_sum = w_maj.sum() or 1e-10
            epsilon_maj = float(np.dot(w_maj / w_maj_sum, err_maj_mask.astype(float)))

            # 邊界保護
            epsilon_min = np.clip(epsilon_min, 1e-10, 1.0 - 1e-10)
            epsilon_maj = np.clip(epsilon_maj, 1e-10, 1.0 - 1e-10)

            # -----------------------------------------------
            # Step 3：計算拆分後的 Alpha
            # α_min / α_maj = 0.5 * ln((1 - ε) / ε)
            # -----------------------------------------------
            alpha_min = 0.5 * _safe_log((1.0 - epsilon_min) / epsilon_min)
            alpha_maj = 0.5 * _safe_log((1.0 - epsilon_maj) / epsilon_maj)

            self.estimators_.append(clf)
            self.alphas_min_.append(alpha_min)
            self.alphas_maj_.append(alpha_maj)

            # -----------------------------------------------
            # Step 4：不對稱權重更新
            # 少數類：w *= exp(-α_min * margin)
            # 多數類：w *= exp(-α_maj * margin)
            # -----------------------------------------------
            margin = np.where(y_pred == y_curr, 1.0, -1.0)

            new_w = np.zeros_like(w_curr)
            new_w[curr_min_mask] = w_curr[curr_min_mask] * np.exp(
                -alpha_min * margin[curr_min_mask]
            )
            new_w[curr_maj_mask] = w_curr[curr_maj_mask] * np.exp(
                -alpha_maj * margin[curr_maj_mask]
            )
            new_w = _normalize(new_w)

            # -----------------------------------------------
            # Step 5：選擇性欠採樣
            # 依更新後的權重分佈，以加權抽樣選取樣本，
            # 讓難以分類（高權重）的樣本優先被保留。
            # 這會自動剃除容易分類（低權重）的多數類樣本。
            # -----------------------------------------------
            sample_n = min(subsample_size, len(y_curr))
            chosen_idx = rng.choice(
                len(y_curr),
                size=sample_n,
                replace=False,
                p=new_w,
            )
            chosen_idx = np.sort(chosen_idx)

            X_curr = X_curr[chosen_idx]
            y_curr = y_curr[chosen_idx]
            w_curr = _normalize(new_w[chosen_idx])

            curr_min_mask = y_curr == minority_label
            curr_maj_mask = y_curr == majority_label

            # 若少數類樣本已耗盡，提前終止
            if curr_min_mask.sum() == 0 or curr_maj_mask.sum() == 0:
                break

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        集成預測：少數類與多數類分別使用各自的 Alpha 加權投票。
        """
        if not self.estimators_:
            raise RuntimeError("模型尚未訓練，請先呼叫 fit()。")

        scores = np.zeros(len(X))
        for clf, a_min, a_maj in zip(
            self.estimators_, self.alphas_min_, self.alphas_maj_
        ):
            y_pred = clf.predict(X)
            # 少數類(1)預測使用 alpha_min，多數類(0)預測使用 alpha_maj
            alpha = np.where(y_pred == 1, a_min, a_maj)
            scores += alpha * np.where(y_pred == 1, 1.0, -1.0)

        return (scores >= 0.0).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """軟性機率輸出（sigmoid 轉換）。"""
        if not self.estimators_:
            raise RuntimeError("模型尚未訓練，請先呼叫 fit()。")

        scores = np.zeros(len(X))
        total_alpha = sum(self.alphas_min_) or 1.0
        for clf, a_min, a_maj in zip(
            self.estimators_, self.alphas_min_, self.alphas_maj_
        ):
            y_pred = clf.predict(X)
            alpha = np.where(y_pred == 1, a_min, a_maj)
            scores += alpha * np.where(y_pred == 1, 1.0, -1.0)

        prob_pos = 1.0 / (1.0 + np.exp(-2.0 * scores / total_alpha))
        return np.column_stack([1.0 - prob_pos, prob_pos])


# ============================================================
# 方法五：SMOTECSL（純 Pipeline，無 Boosting 迴圈）
# ============================================================

class SMOTECSL(BaseEstimator, ClassifierMixin):
    """
    SMOTE + Cost-Sensitive Learning（CSL）Pipeline。

    文獻依據
    --------
    Thai-Nghe, N., Gantner, Z., & Schmidt-Thieme, L. (2010).
    "Cost-sensitive learning methods for imbalanced data."
    IJCNN 2010.

    論文對應重點
    -----------
    為避免 SMOTE 平衡資料後導致 CSL 失效，本實作在重採樣前先依據「原始資料分佈」
    計算出真實的 class_weight，並強制注入分類器中。此舉不僅保留了 SMOTE 的
    特徵擴充優勢，更嚴格恪守了 CSL 基於真實先驗機率進行代價懲罰的數學定義。
    """

    def __init__(
        self,
        base_estimator: Optional[BaseEstimator] = None,
        k_neighbors: int = 5,
        random_state: Optional[int] = None,
    ) -> None:
        # 預設為帶成本敏感的 SVM，貼合論文設定
        self.base_estimator = (
            base_estimator
            if base_estimator is not None
            else SVC(
                probability=True,
                class_weight="balanced",
                random_state=random_state,
            )
        )
        self.k_neighbors = k_neighbors
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SMOTECSL":
        """
        執行 SMOTE 過採樣後，訓練 cost-sensitive 分類器。

        Parameters
        ----------
        X : shape (n_samples, n_features)
        y : shape (n_samples,)
        """
        self.classes_ = np.unique(y)

        # 在執行 SMOTE 之前，先針對原始的 y 計算出真實的 class_weight
        original_weights = compute_class_weight(
            class_weight="balanced",
            classes=self.classes_,
            y=y
        )
        weight_dict = dict(zip(self.classes_, original_weights))

        # Step 1：SMOTE 過採樣（還原標準設定，補至 1:1）
        smote = SMOTE(
            sampling_strategy="auto",
            k_neighbors=self.k_neighbors,
            random_state=self.random_state,
        )
        X_res, y_res = smote.fit_resample(X, y)

        # Step 2：clone 並強制將原始的真實代價矩陣注入分類器中
        self.estimator_ = clone(self.base_estimator)
        self.estimator_.set_params(class_weight=weight_dict)
        self.estimator_.fit(X_res, y_res)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """直接呼叫底層分類器預測。"""
        if not hasattr(self, "estimator_"):
            raise RuntimeError("模型尚未訓練，請先呼叫 fit()。")
        return self.estimator_.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """回傳類別機率（需要 base_estimator 支援 predict_proba）。"""
        if not hasattr(self, "estimator_"):
            raise RuntimeError("模型尚未訓練，請先呼叫 fit()。")
        return self.estimator_.predict_proba(X)
