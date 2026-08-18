import unittest

from utils.quiz import generate_quiz
from utils.qa import answer_question


class GenerationQualityTests(unittest.TestCase):
    def test_quiz_is_lecture_grounded(self):
        transcript = (
            "Machine learning is a field of artificial intelligence that gives computers the ability to learn from data. "
            "Supervised learning uses labeled examples to train a model. In classification, the model predicts a category such as spam or not spam. "
            "In regression, the model predicts a continuous value such as house price. Overfitting happens when a model memorizes the training data instead of generalizing. "
            "Regularization reduces overfitting by penalizing large weights."
        )
        summary = (
            "Machine learning teaches computers to learn from data. Supervised learning uses labeled examples. "
            "Classification predicts categories and regression predicts continuous values. Overfitting happens when a model memorizes training data, and regularization reduces overfitting by penalizing large weights."
        )
        quiz = generate_quiz(summary, transcript, [
            {"concept": "Machine learning", "explanation": "A field of AI that learns from data."},
            {"concept": "Supervised learning", "explanation": "Uses labeled examples."},
            {"concept": "Regularization", "explanation": "Penalizes large weights to reduce overfitting."},
        ])

        self.assertGreater(len(quiz), 0)
        all_text = " ".join(
            q["question"] + " " + " ".join(q["options"]) + " " + q["correct_answer"]
            for q in quiz
        ).lower()
        self.assertIn("regularization", all_text)
        self.assertNotIn("random unrelated topic", all_text)
        self.assertNotIn("not mentioned in the lecture", all_text)

    def test_qa_returns_relevant_answer(self):
        transcript = (
            "Machine learning is a field of artificial intelligence that gives computers the ability to learn from data. "
            "Supervised learning uses labeled examples to train a model. In classification, the model predicts a category such as spam or not spam. "
            "In regression, the model predicts a continuous value such as house price. Overfitting happens when a model memorizes the training data instead of generalizing. "
            "Regularization reduces overfitting by penalizing large weights."
        )
        answer = answer_question(transcript, "What is regularization used for?")
        self.assertTrue(answer)
        answer_lower = answer.lower()
        self.assertTrue(
            "reduce overfitting" in answer_lower
            or "penalizing large weights" in answer_lower
            or "overfitting" in answer_lower,
            msg=f"Unexpected answer: {answer!r}"
        )


if __name__ == "__main__":
    unittest.main()
