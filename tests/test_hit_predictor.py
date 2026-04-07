import pytest
import numpy as np
from unittest.mock import Mock, patch
from src.ml.hit_predictor import HitPredictor
from src.metrics.collector import MetricsCollector


class TestHitPredictor:
    @pytest.fixture
    def metrics_collector(self):
        return Mock(spec=MetricsCollector)
    
    @pytest.fixture
    def predictor(self, metrics_collector):
        return HitPredictor(metrics_collector)
    
    def test_init_creates_empty_features(self, predictor):
        assert predictor.feature_history == []
        assert predictor.model is None
    
    def test_extract_features_returns_correct_shape(self, predictor):
        key = "test_key"
        current_time = 1000.0
        
        features = predictor._extract_features(key, current_time)
        
        assert len(features) == 4
        assert isinstance(features, list)
    
    def test_extract_features_includes_key_hash(self, predictor):
        key = "test_key"
        features = predictor._extract_features(key, 1000.0)
        
        # Key hash should be first feature
        expected_hash = hash(key) % 1000
        assert features[0] == expected_hash
    
    @patch('time.time')
    def test_predict_probability_no_model_returns_default(self, mock_time, predictor):
        mock_time.return_value = 1000.0
        
        prob = predictor.predict_probability("test_key")
        
        assert prob == 0.5
    
    def test_update_features_adds_to_history(self, predictor):
        predictor._update_features("key", True, 1000.0)
        
        assert len(predictor.feature_history) == 1
        assert predictor.feature_history[0]['hit'] is True
    
    def test_should_retrain_false_insufficient_data(self, predictor):
        # Add less than minimum samples
        for i in range(90):
            predictor._update_features(f"key_{i}", i % 2 == 0, float(i))
        
        assert not predictor._should_retrain()
    
    def test_should_retrain_true_sufficient_data(self, predictor):
        # Add sufficient samples
        for i in range(110):
            predictor._update_features(f"key_{i}", i % 2 == 0, float(i))
        
        assert predictor._should_retrain()
    
    @patch('sklearn.ensemble.RandomForestClassifier')
    def test_train_model_creates_model(self, mock_rf, predictor):
        # Setup training data
        for i in range(110):
            predictor._update_features(f"key_{i}", i % 2 == 0, float(i))
        
        predictor._train_model()
        
        mock_rf.assert_called_once()
        assert predictor.model is not None
    
    @patch('time.time')
    def test_predict_probability_with_model_returns_prediction(self, mock_time, predictor):
        mock_time.return_value = 1000.0
        
        # Mock model
        mock_model = Mock()
        mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])
        predictor.model = mock_model
        
        prob = predictor.predict_probability("test_key")
        
        assert prob == 0.7
        mock_model.predict_proba.assert_called_once()