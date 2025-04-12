import numpy as np
from .utils.feature_engineering import CustomFeatureEngineering
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from category_encoders import TargetEncoder


class PreprocessingPipeline:
    
	def __init__(self):
		self._pipeline = None

		self.numeric_features = []
		self.low_cardinality_features = []
		self.high_cardinality_features = []

	def _infer_columns_types(self, df):
		"""
		Infer the columns types for the preprocessing pipeline
		Parameters:
		- X (pd.DataFrame): Input data.
		"""
		cols_to_exclude = ["id", "created_at", "version", "no_doors", "color", "seller", "link", "price_cash", "price_financed"]
		self.numeric_features = [col for col in df.columns if (df[col].dtype in [np.int64, np.float64]) and col not in cols_to_exclude]
        # Separating high and low cardinality features
		self.high_cardinality_features = ["manufacturer", "model"]
		self.low_cardinality_features = [col for col in df.columns if col not in self.numeric_features and col not in self.high_cardinality_features and col not in cols_to_exclude]

	def _create_feature_engineering_pipeline(self):
			return Pipeline(steps=[
				("feature_engineering", CustomFeatureEngineering())
			])

	def _create_preprocessing_transformers(self):
		"""
		Create the preprocessing pipeline

		Parameters:
		- y_train (pd.Series): Target variable.
		"""
		# Numeric features
		numeric_transformer = Pipeline(steps=[
			("scaler", StandardScaler())
		])
		# One hot encoding for low cardinality features
		low_cardinality_transformer = Pipeline(steps=[
			("onehot", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"))
		])
		# Target encoding for high cardinality features
		high_cardinality_transformer = Pipeline(steps=[
			("target", TargetEncoder(smoothing=10))
		])
		# Define the transformers
		transformers = [
			("num", numeric_transformer, self.numeric_features),
			("low_card", low_cardinality_transformer, self.low_cardinality_features),
			("high_card", high_cardinality_transformer, self.high_cardinality_features)
		]
		return ColumnTransformer(
			transformers=transformers,
			# remainder="passthrough"
		)

	def create_pipeline(self, df):
		
		# Infer the columns types
		self._infer_columns_types(df)

		self._pipeline = Pipeline(steps=[
			("feature_engineering", self._create_feature_engineering_pipeline()),
			("preprocessing", self._create_preprocessing_transformers())
		])

		return self._pipeline