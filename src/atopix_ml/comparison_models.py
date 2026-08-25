"""Serializable full-cohort estimators for the manuscript comparison models."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import SVC, LinearSVC

from atopix_ml.documented_proposed import rank_average

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

COMPARISON_MODEL_IDS: Final = (
    "curated_logistic_five_c",
    "conventional_logistic_c1",
    "single_multitask_learner",
    "tabular_foundation_model",
    "gradient_boosted_trees",
    "linear_svm",
    "adaboosted_logistic",
    "extremely_randomized_trees",
    "rbf_svm",
    "gaussian_naive_bayes",
    "knn",
    "random_forest",
    "lightgbm",
    "xgboost",
    "mlp",
)


def _validate_X(X: FloatArray, feature_names: tuple[str, ...]) -> FloatArray:
    values = np.asarray(X, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(feature_names):
        raise ValueError("X must be two-dimensional and match the stored feature order")
    if np.any(np.isinf(values)):
        raise ValueError("X may contain missing values but not infinity")
    return values


def _positive_probability(model: Any, X: FloatArray) -> FloatArray:
    return np.asarray(model.predict_proba(X)[:, 1], dtype=np.float64)


@dataclass(slots=True)
class ProbabilityMeanEnsemble:
    """Uniformly average probabilities from fitted comparison-model members."""

    model_id: str
    feature_names: tuple[str, ...]
    members: list[Any]
    contract: str

    def predict_proba(self, X: FloatArray) -> FloatArray:
        values = _validate_X(X, self.feature_names)
        if not self.members:
            raise ValueError("comparison ensemble has no fitted members")
        positive = np.mean(
            [_positive_probability(member, values) for member in self.members],
            axis=0,
        )
        return np.column_stack((1.0 - positive, positive))


@dataclass(slots=True)
class SingleEstimatorComparisonModel:
    """Attach the shared comparison-model contract to one fitted estimator."""

    model_id: str
    feature_names: tuple[str, ...]
    estimator: Any
    contract: str

    def predict_proba(self, X: FloatArray) -> FloatArray:
        values = _validate_X(X, self.feature_names)
        return np.asarray(self.estimator.predict_proba(values), dtype=np.float64)


@dataclass(slots=True)
class MultiTaskLinearComparisonModel:
    """Serializable full-cohort form of ``v5_multitask_easi``."""

    model_id: str
    feature_names: tuple[str, ...]
    threshold_models: dict[int, dict[str, ProbabilityMeanEnsemble]]
    contract: str

    def predict_proba(self, X: FloatArray) -> FloatArray:
        values = _validate_X(X, self.feature_names)
        if len(values) < 2:
            raise ValueError("historical batch-rank scoring requires at least two rows")
        task_scores = {}
        for threshold, groups in self.threshold_models.items():
            task_scores[threshold] = rank_average(
                groups["logistic"].predict_proba(values)[:, 1],
                groups["huber"].predict_proba(values)[:, 1],
            )
        weighted = 0.25 * task_scores[65] + 0.60 * task_scores[75] + 0.15 * task_scores[85]
        ordinal_rank = rank_average(
            task_scores[75],
            task_scores[65],
            task_scores[85],
        )
        positive = 0.5 * weighted + 0.5 * ordinal_rank
        return np.column_stack((1.0 - positive, positive))


@dataclass(slots=True)
class MultiTaskLightGBMComparisonModel:
    """Serializable full-cohort form of ``v5_mt_lgbm``."""

    model_id: str
    feature_names: tuple[str, ...]
    threshold_models: dict[int, ProbabilityMeanEnsemble]
    contract: str

    def predict_proba(self, X: FloatArray) -> FloatArray:
        values = _validate_X(X, self.feature_names)
        if len(values) < 2:
            raise ValueError("historical batch-rank scoring requires at least two rows")
        positive = rank_average(
            *[
                self.threshold_models[threshold].predict_proba(values)[:, 1]
                for threshold in (65, 75, 85)
            ]
        )
        return np.column_stack((1.0 - positive, positive))


def _fit_calibrated(base: Any, X: FloatArray, y: IntArray) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = CalibratedClassifierCV(base, method="isotonic", cv=3)
        model.fit(X, y)
    return model


def _linear_pipeline(classifier: Any, *, standard: bool = False) -> Pipeline:
    scaler = StandardScaler() if standard else RobustScaler()
    return Pipeline(
        [
            ("imp", SimpleImputer(strategy="median")),
            ("sc", scaler),
            ("clf", classifier),
        ]
    )


def _fit_curated_logistic(
    X: FloatArray,
    y: IntArray,
    feature_names: tuple[str, ...],
) -> ProbabilityMeanEnsemble:
    members = [
        _fit_calibrated(
            _linear_pipeline(
                LogisticRegression(
                    C=c_value,
                    penalty="l2",
                    solver="lbfgs",
                    max_iter=4000,
                    random_state=42,
                )
            ),
            X,
            y,
        )
        for c_value in (0.1, 0.3, 1.0, 3.0, 10.0)
    ]
    return ProbabilityMeanEnsemble(
        model_id="curated_logistic_five_c",
        feature_names=feature_names,
        members=members,
        contract="median imputation; RobustScaler; five-C L2 logistic bag; isotonic CV3",
    )


def _fit_conventional_logistic(
    X: FloatArray,
    y: IntArray,
    feature_names: tuple[str, ...],
) -> ProbabilityMeanEnsemble:
    member = _fit_calibrated(
        _linear_pipeline(
            LogisticRegression(
                C=1.0,
                penalty="l2",
                solver="lbfgs",
                max_iter=4000,
                random_state=42,
            )
        ),
        X,
        y,
    )
    return ProbabilityMeanEnsemble(
        model_id="conventional_logistic_c1",
        feature_names=feature_names,
        members=[member],
        contract="median imputation; RobustScaler; C=1 L2 logistic; isotonic CV3",
    )


def _fit_linear_svm(
    X: FloatArray,
    y: IntArray,
    feature_names: tuple[str, ...],
) -> ProbabilityMeanEnsemble:
    members = [
        _fit_calibrated(
            _linear_pipeline(
                LinearSVC(
                    C=c_value,
                    max_iter=5000,
                    dual=True,
                    random_state=42,
                )
            ),
            X,
            y,
        )
        for c_value in (0.1, 0.3, 1.0, 3.0)
    ]
    return ProbabilityMeanEnsemble(
        model_id="linear_svm",
        feature_names=feature_names,
        members=members,
        contract="median imputation; RobustScaler; four-C LinearSVC bag; isotonic CV3",
    )


def _fit_adaboosted_logistic(
    X: FloatArray,
    y: IntArray,
    feature_names: tuple[str, ...],
) -> ProbabilityMeanEnsemble:
    members = []
    for n_estimators in (10, 30, 60):
        for learning_rate in (0.3, 1.0):
            classifier = AdaBoostClassifier(
                estimator=LogisticRegression(
                    C=0.5,
                    max_iter=1000,
                    random_state=42,
                ),
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                algorithm="SAMME",
                random_state=42,
            )
            members.append(_fit_calibrated(_linear_pipeline(classifier), X, y))
    return ProbabilityMeanEnsemble(
        model_id="adaboosted_logistic",
        feature_names=feature_names,
        members=members,
        contract="median imputation; RobustScaler; six AdaBoost-LR variants; isotonic CV3",
    )


def _fit_rbf_svm(
    X: FloatArray,
    y: IntArray,
    feature_names: tuple[str, ...],
) -> ProbabilityMeanEnsemble:
    members = []
    for c_value in (0.5, 1.0, 3.0):
        for gamma in ("scale", 0.05, 0.1):
            classifier = SVC(
                C=c_value,
                kernel="rbf",
                gamma=gamma,
                probability=True,
                random_state=42,
            )
            members.append(_fit_calibrated(_linear_pipeline(classifier), X, y))
    return ProbabilityMeanEnsemble(
        model_id="rbf_svm",
        feature_names=feature_names,
        members=members,
        contract="median imputation; RobustScaler; nine RBF-SVC variants; isotonic CV3",
    )


def _fit_knn(
    X: FloatArray,
    y: IntArray,
    feature_names: tuple[str, ...],
) -> ProbabilityMeanEnsemble:
    members = []
    for n_neighbors in (5, 9, 13, 17, 21):
        for metric in ("euclidean", "manhattan"):
            classifier = KNeighborsClassifier(
                n_neighbors=n_neighbors,
                metric=metric,
                weights="distance",
            )
            members.append(_fit_calibrated(_linear_pipeline(classifier), X, y))
    return ProbabilityMeanEnsemble(
        model_id="knn",
        feature_names=feature_names,
        members=members,
        contract="median imputation; RobustScaler; ten distance-KNN variants; isotonic CV3",
    )


def _fit_gaussian_nb(
    X: FloatArray,
    y: IntArray,
    feature_names: tuple[str, ...],
) -> ProbabilityMeanEnsemble:
    member = _fit_calibrated(_linear_pipeline(GaussianNB()), X, y)
    return ProbabilityMeanEnsemble(
        model_id="gaussian_naive_bayes",
        feature_names=feature_names,
        members=[member],
        contract="median imputation; RobustScaler; GaussianNB; isotonic CV3",
    )


def _fit_tree_bag(
    model_id: str,
    X: FloatArray,
    y: IntArray,
    feature_names: tuple[str, ...],
) -> ProbabilityMeanEnsemble:
    members = []
    for seed in (42, 7, 123):
        if model_id == "random_forest":
            classifier = RandomForestClassifier(
                n_estimators=400,
                max_depth=6,
                min_samples_leaf=4,
                max_features="sqrt",
                n_jobs=2,
                random_state=seed,
            )
        else:
            classifier = ExtraTreesClassifier(
                n_estimators=400,
                max_depth=6,
                min_samples_leaf=4,
                max_features="sqrt",
                n_jobs=2,
                random_state=seed,
            )
        base = Pipeline(
            [
                ("imp", SimpleImputer(strategy="median")),
                ("clf", classifier),
            ]
        )
        members.append(_fit_calibrated(base, X, y))
    return ProbabilityMeanEnsemble(
        model_id=model_id,
        feature_names=feature_names,
        members=members,
        contract=(
            "median imputation; three-seed "
            f"{'RandomForest' if model_id == 'random_forest' else 'ExtraTrees'} bag; "
            "isotonic CV3"
        ),
    )


def _fit_lightgbm(
    X: FloatArray,
    y: IntArray,
    feature_names: tuple[str, ...],
) -> ProbabilityMeanEnsemble:
    import lightgbm as lgb

    members = []
    for seed in (42, 7, 123):
        classifier = lgb.LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=15,
            min_child_samples=10,
            reg_alpha=0.1,
            reg_lambda=0.5,
            subsample=0.9,
            subsample_freq=1,
            colsample_bytree=0.9,
            random_state=seed,
            verbose=-1,
            n_jobs=1,
        )
        base = Pipeline(
            [
                ("imp", SimpleImputer(strategy="median")),
                ("clf", classifier),
            ]
        )
        members.append(_fit_calibrated(base, X, y))
    return ProbabilityMeanEnsemble(
        model_id="lightgbm",
        feature_names=feature_names,
        members=members,
        contract="median imputation; three-seed LightGBM bag; isotonic CV3",
    )


def _fit_xgboost(
    X: FloatArray,
    y: IntArray,
    feature_names: tuple[str, ...],
) -> ProbabilityMeanEnsemble:
    import xgboost as xgb

    members = []
    for seed in (42, 7, 123):
        classifier = xgb.XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            min_child_weight=5,
            reg_alpha=0.1,
            reg_lambda=1.0,
            subsample=0.9,
            colsample_bytree=0.9,
            tree_method="hist",
            n_jobs=1,
            random_state=seed,
            eval_metric="logloss",
            verbosity=0,
        )
        base = Pipeline(
            [
                ("imp", SimpleImputer(strategy="median")),
                ("clf", classifier),
            ]
        )
        members.append(_fit_calibrated(base, X, y))
    return ProbabilityMeanEnsemble(
        model_id="xgboost",
        feature_names=feature_names,
        members=members,
        contract="median imputation; three-seed XGBoost bag; isotonic CV3",
    )


def _fit_mlp(
    X: FloatArray,
    y: IntArray,
    feature_names: tuple[str, ...],
) -> ProbabilityMeanEnsemble:
    members = []
    for seed in (42, 7, 123):
        for hidden_layers in ((32, 16), (16, 8), (64, 32, 16)):
            classifier = MLPClassifier(
                hidden_layer_sizes=hidden_layers,
                activation="relu",
                solver="adam",
                alpha=1e-3,
                learning_rate_init=1e-3,
                max_iter=400,
                early_stopping=True,
                validation_fraction=0.2,
                n_iter_no_change=20,
                random_state=seed,
            )
            members.append(
                _fit_calibrated(
                    _linear_pipeline(classifier, standard=True),
                    X,
                    y,
                )
            )
    return ProbabilityMeanEnsemble(
        model_id="mlp",
        feature_names=feature_names,
        members=members,
        contract="median imputation; StandardScaler; nine MLP variants; isotonic CV3",
    )


def _fit_multitask_linear(
    X: FloatArray,
    y_pct: FloatArray,
    feature_names: tuple[str, ...],
) -> MultiTaskLinearComparisonModel:
    thresholds = {}
    for threshold in (65, 75, 85):
        labels = (y_pct >= threshold).astype(np.int64)
        logistic = [
            _fit_calibrated(
                _linear_pipeline(
                    LogisticRegression(
                        C=c_value,
                        max_iter=4000,
                        random_state=42,
                    )
                ),
                X,
                labels,
            )
            for c_value in (0.3, 1.0, 3.0)
        ]
        huber = []
        for seed in (42, 7, 123):
            for alpha in (1e-4, 1e-3):
                huber.append(
                    _fit_calibrated(
                        _linear_pipeline(
                            SGDClassifier(
                                loss="modified_huber",
                                alpha=alpha,
                                max_iter=2000,
                                random_state=seed,
                            ),
                            standard=True,
                        ),
                        X,
                        labels,
                    )
                )
        thresholds[threshold] = {
            "logistic": ProbabilityMeanEnsemble(
                model_id=f"single_multitask_lr_{threshold}",
                feature_names=feature_names,
                members=logistic,
                contract="three-C logistic threshold bag",
            ),
            "huber": ProbabilityMeanEnsemble(
                model_id=f"single_multitask_huber_{threshold}",
                feature_names=feature_names,
                members=huber,
                contract="three-seed two-alpha modified-Huber threshold bag",
            ),
        }
    return MultiTaskLinearComparisonModel(
        model_id="single_multitask_learner",
        feature_names=feature_names,
        threshold_models=thresholds,
        contract=(
            "EASI-65/75/85 LR+modified-Huber task ranks; weighted probability "
            "and ordinal-rank blend from v5_multitask_easi"
        ),
    )


def _fit_multitask_lightgbm(
    X: FloatArray,
    y_pct: FloatArray,
    feature_names: tuple[str, ...],
) -> MultiTaskLightGBMComparisonModel:
    import lightgbm as lgb

    thresholds = {}
    for threshold in (65, 75, 85):
        labels = (y_pct >= threshold).astype(np.int64)
        members = []
        for num_leaves in (4, 8, 16):
            for n_estimators in (30, 100):
                classifier = lgb.LGBMClassifier(
                    n_estimators=n_estimators,
                    num_leaves=num_leaves,
                    learning_rate=0.05,
                    min_child_samples=8,
                    reg_alpha=0.1,
                    reg_lambda=0.1,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    random_state=42,
                    verbose=-1,
                )
                members.append(_fit_calibrated(classifier, X, labels))
        thresholds[threshold] = ProbabilityMeanEnsemble(
            model_id=f"gradient_boosted_trees_{threshold}",
            feature_names=feature_names,
            members=members,
            contract="six calibrated LightGBM variants for one EASI threshold",
        )
    return MultiTaskLightGBMComparisonModel(
        model_id="gradient_boosted_trees",
        feature_names=feature_names,
        threshold_models=thresholds,
        contract="EASI-65/75/85 six-model LightGBM means followed by task rank-average",
    )


def _fit_tabpfn(
    X: FloatArray,
    y: IntArray,
    feature_names: tuple[str, ...],
    *,
    device: str,
) -> SingleEstimatorComparisonModel:
    from tabpfn import TabPFNClassifier

    estimator = TabPFNClassifier(
        device=device,
        n_estimators=16,
        random_state=42,
        ignore_pretraining_limits=True,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        estimator.fit(X, y)
    return SingleEstimatorComparisonModel(
        model_id="tabular_foundation_model",
        feature_names=feature_names,
        estimator=estimator,
        contract="raw 23-feature TabPFN v2 classifier; 16 estimators; random_state=42",
    )


def fit_comparison_model(
    model_id: str,
    X: FloatArray,
    y: IntArray,
    y_pct: FloatArray,
    *,
    feature_names: tuple[str, ...],
    tabpfn_device: str = "cuda",
) -> Any:
    """Fit one unique full-cohort comparison estimator from its preserved contract."""
    values = _validate_X(X, feature_names)
    labels = np.asarray(y, dtype=np.int64)
    improvement = np.asarray(y_pct, dtype=np.float64)
    if values.shape[0] != labels.size or labels.size != improvement.size:
        raise ValueError("X, y, and y_pct must be row-aligned")
    if model_id not in COMPARISON_MODEL_IDS:
        raise ValueError(f"unknown comparison model: {model_id}")

    if model_id == "curated_logistic_five_c":
        return _fit_curated_logistic(values, labels, feature_names)
    if model_id == "conventional_logistic_c1":
        return _fit_conventional_logistic(values, labels, feature_names)
    if model_id == "single_multitask_learner":
        return _fit_multitask_linear(values, improvement, feature_names)
    if model_id == "tabular_foundation_model":
        return _fit_tabpfn(
            values,
            labels,
            feature_names,
            device=tabpfn_device,
        )
    if model_id == "gradient_boosted_trees":
        return _fit_multitask_lightgbm(values, improvement, feature_names)
    if model_id == "linear_svm":
        return _fit_linear_svm(values, labels, feature_names)
    if model_id == "adaboosted_logistic":
        return _fit_adaboosted_logistic(values, labels, feature_names)
    if model_id == "extremely_randomized_trees":
        return _fit_tree_bag(model_id, values, labels, feature_names)
    if model_id == "rbf_svm":
        return _fit_rbf_svm(values, labels, feature_names)
    if model_id == "gaussian_naive_bayes":
        return _fit_gaussian_nb(values, labels, feature_names)
    if model_id == "knn":
        return _fit_knn(values, labels, feature_names)
    if model_id == "random_forest":
        return _fit_tree_bag(model_id, values, labels, feature_names)
    if model_id == "lightgbm":
        return _fit_lightgbm(values, labels, feature_names)
    if model_id == "xgboost":
        return _fit_xgboost(values, labels, feature_names)
    if model_id == "mlp":
        return _fit_mlp(values, labels, feature_names)
    raise AssertionError("comparison model dispatch is incomplete")
