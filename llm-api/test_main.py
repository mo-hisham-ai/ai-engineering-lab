import pytest
from pydantic import ValidationError

from main import Answer


def test_valid_answer():
    data = '{"answer": "RAG combines retrieval with generation.", "difficulty": "beginner", "key_points": ["a", "b", "c"]}'
    result = Answer.model_validate_json(data)
    assert result.difficulty == "beginner"
    assert len(result.key_points) == 3


def test_missing_field_raises():
    data = '{"answer": "text", "difficulty": "beginner"}'  # key_points ناقص
    with pytest.raises(ValidationError):
        Answer.model_validate_json(data)


def test_invalid_difficulty_raises():
    data = '{"answer": "text", "difficulty": "hard", "key_points": ["a"]}'  # hard غير مسموح
    with pytest.raises(ValidationError):
        Answer.model_validate_json(data)