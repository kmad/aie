import os
import dspy
import pandas as pd
from textwrap import dedent
from dotenv import load_dotenv

instructions = dedent(
    """
    Step 1: Fix any spelling errors.
    Step 2: Make sure the time entry is in present tense. E.g., use "update" instead of "updated".
    Step 3: Remove references to the following words or phrases: work on, assist, various, multiple, attend to, address, attention to, miscellaneous
    Step 4: Remove all capitalization within the descriptions, EXCEPT for: 
        - Defined terms such as "Company", "Firm"
        - Proper nouns and names
        - Abbreviated names, for example, "J. Smith", "C. Pope", "M. Mouse", etc. MUST retain their capitalization.
    Step 5: Expand all law firm and consulting firm names or common shortnames. For example, 'K&E' becomes 'Kirkland & Ellis', etc.
    """
)


class TimeEntryCorrectionAssessment(dspy.Signature):
    """Assess the quality of time entry corrections based on process_entry instructions."""

    original_description = dspy.InputField(desc="Original time entry description")
    corrected_description = dspy.InputField(desc="Corrected time entry description")

    # Assessment criteria aligned with process_entry steps
    spelling_grammar_improvement = dspy.OutputField(
        desc="Have spelling and grammar errors been fixed? (Yes/No). If none are applicable, then Yes."
    )
    present_tense_used = dspy.OutputField(
        desc="Was the tense improved from the original to present tense? (Yes/No) Compare original vs corrected: if original had past tense verbs (like 'reviewed', 'updated') and corrected has present tense verbs (like 'review', 'update'), then Yes. If original was already present tense then Yes. If the corrected made it worse (past tense), then No."
    )
    prohibited_phrases_removed = dspy.OutputField(
        desc="Have prohibited phrases (work on, assist, various, multiple, attend to, address, attention to, miscellaneous) been removed? (Yes/No)"
    )
    capitalization_correct = dspy.OutputField(
        desc="Is capitalization correct (no unnecessary caps except for defined terms, first word, and proper nouns)? (Yes/No)"
    )
    abbreviated_names_capitalized = dspy.OutputField(
        desc="Have abbreviated names (like 'J. Smith', 'C. Pope', 'M. Mouse') retained their proper capitalization? (Yes/No). If no abbreviated names are present, then Yes."
    )
    law_firm_names_expanded = dspy.OutputField(
        desc="Have law firm and consulting firm abbreviations been expanded to full names? (Yes/No). If none are applicable, then Yes."
    )
    correction_improved_quality = dspy.OutputField(
        desc="Did the correction follow the rules per the instructions? (Yes/No) If the correction made things worse (like changing present tense to past tense), then No."
    )
    overall_quality_score = dspy.OutputField(
        desc=f"Overall quality score from 1-10, per the instructions: \n### Instructions ### \n{instructions}"
    )


class ProcessEntryModule(dspy.Module):
    """DSPy module that implements the process_entry function logic."""

    def __init__(self):
        super().__init__()
        # Create the change_tense predictor
        change_tense_signature = dspy.Signature(
            "time_entry: str -> time_entry_corrected: str"
        ).with_instructions(instructions)
        self.change_tense = dspy.Predict(change_tense_signature)

    def forward(self, original_description):
        # Step 1: Apply the change_tense correction
        entry = self.change_tense(time_entry=original_description)
        entry = entry.time_entry_corrected.rstrip(".")

        # Step 2: Apply additional text replacements
        entry = entry.replace("(all AcmeCo)", "(AcmeCo)")
        entry = entry.replace("(both AcmeCo)", "(AcmeCo)")
        entry = entry.replace("work streams", "workstreams")
        entry = entry.replace("regarding", "re:")

        # Step 3: Ensure first letter is capitalized
        if entry and len(entry) > 0:
            entry = entry[0].upper() + entry[1:]

        return dspy.Prediction(corrected_description=entry)


# Define separate assessment signatures for different types of evaluations
class SpellingGrammarAssessment(dspy.Signature):
    """Assess spelling and grammar improvements."""

    original_description = dspy.InputField(desc="Original time entry description")
    corrected_description = dspy.InputField(desc="Corrected time entry description")
    spelling_grammar_improvement = dspy.OutputField(
        desc="Have spelling and grammar errors been fixed? (Yes/No). If none are applicable, then Yes. In general if the text is terse and not a full sentence, then Yes."
    )


class TenseAssessment(dspy.Signature):
    """Assess present tense usage."""

    original_description = dspy.InputField(desc="Original time entry description")
    corrected_description = dspy.InputField(desc="Corrected time entry description")
    present_tense_used = dspy.OutputField(
        desc="Was the tense improved from the original to present tense? (Yes/No) Compare original vs corrected: if original had past tense verbs (like 'reviewed', 'updated') and corrected has present tense verbs (like 'review', 'update'), then Yes. If original was already present tense then Yes. If the corrected made it worse (past tense), then No."
    )


class ProhibitedPhrasesAssessment(dspy.Signature):
    """Assess removal of prohibited phrases."""

    original_description = dspy.InputField(desc="Original time entry description")
    corrected_description = dspy.InputField(desc="Corrected time entry description")
    prohibited_phrases_removed = dspy.OutputField(
        desc="Have prohibited phrases (work on, assist, various, multiple, attend to, address, attention to, miscellaneous) been removed? (Yes/No)"
    )


class CapitalizationAssessment(dspy.Signature):
    """Assess general capitalization rules."""

    original_description = dspy.InputField(desc="Original time entry description")
    corrected_description = dspy.InputField(desc="Corrected time entry description")
    capitalization_correct = dspy.OutputField(
        desc="Is capitalization correct (no unnecessary caps except for defined terms, first word, and proper nouns)? (Yes/No)"
    )


class AbbreviatedNamesAssessment(dspy.Signature):
    """Assess abbreviated names capitalization retention."""

    original_description = dspy.InputField(desc="Original time entry description")
    corrected_description = dspy.InputField(desc="Corrected time entry description")
    abbreviated_names_capitalized = dspy.OutputField(
        desc="Have abbreviated names (like 'J. Smith', 'M. Mouse', 'M. Jordan') retained their proper capitalization? (Yes/No). If no abbreviated names are present, then Yes."
    )


class LawFirmNamesAssessment(dspy.Signature):
    """Assess law firm name expansions."""

    original_description = dspy.InputField(desc="Original time entry description")
    corrected_description = dspy.InputField(desc="Corrected time entry description")
    law_firm_names_expanded = dspy.OutputField(
        desc="Have law firm and consulting firm abbreviations been expanded to full names? (Yes/No). If none are applicable, then Yes."
    )


class OverallQualityAssessment(dspy.Signature):
    """Assess overall quality score."""

    original_description = dspy.InputField(desc="Original time entry description")
    corrected_description = dspy.InputField(desc="Corrected time entry description")
    overall_quality_score = dspy.OutputField(
        desc=f"Overall quality score from 1-10, per the instructions: \n### Instructions ### \n{instructions}"
    )


# Create separate assessors
def spelling_grammar_metric(example, pred, trace=None):
    """Evaluate spelling and grammar improvements."""

    assessor = dspy.Predict(SpellingGrammarAssessment)
    assessment = assessor(
        original_description=str(example.original_description),
        corrected_description=str(pred.corrected_description),
    )
    return 1.0 if assessment.spelling_grammar_improvement.lower() == "yes" else 0.0


def present_tense_metric(example, pred, trace=None):
    """Evaluate present tense usage."""
    assessor = dspy.Predict(TenseAssessment)
    assessment = assessor(
        original_description=str(example.original_description),
        corrected_description=str(pred.corrected_description),
    )
    return 1.0 if assessment.present_tense_used.lower() == "yes" else 0.0


def prohibited_phrases_metric(example, pred, trace=None):
    """Evaluate removal of prohibited phrases."""
    assessor = dspy.Predict(ProhibitedPhrasesAssessment)
    assessment = assessor(
        original_description=str(example.original_description),
        corrected_description=str(pred.corrected_description),
    )
    return 1.0 if assessment.prohibited_phrases_removed.lower() == "yes" else 0.0


def capitalization_metric(example, pred, trace=None):
    """Evaluate general capitalization rules."""
    assessor = dspy.Predict(CapitalizationAssessment)
    assessment = assessor(
        original_description=str(example.original_description),
        corrected_description=str(pred.corrected_description),
    )
    return 1.0 if assessment.capitalization_correct.lower() == "yes" else 0.0


def abbreviated_names_metric(example, pred, trace=None):
    """Evaluate abbreviated names capitalization retention."""
    assessor = dspy.Predict(AbbreviatedNamesAssessment)
    assessment = assessor(
        original_description=str(example.original_description),
        corrected_description=str(pred.corrected_description),
    )
    return 1.0 if assessment.abbreviated_names_capitalized.lower() == "yes" else 0.0


def law_firm_names_metric(example, pred, trace=None):
    """Evaluate law firm name expansions."""
    assessor = dspy.Predict(LawFirmNamesAssessment)
    assessment = assessor(
        original_description=str(example.original_description),
        corrected_description=str(pred.corrected_description),
    )
    return 1.0 if assessment.law_firm_names_expanded.lower() == "yes" else 0.0


def overall_quality_metric(example, pred, trace=None):
    """Evaluate overall quality score from 1-10."""
    assessor = dspy.Predict(OverallQualityAssessment)
    assessment = assessor(
        original_description=str(example.original_description),
        corrected_description=str(pred.corrected_description),
    )
    try:
        return float(assessment.overall_quality_score) / 10.0
    except:
        return 0.5  # Default if parsing fails


def create_examples_from_dataframe(df):
    """Create DSPy examples from the joined dataframe."""
    examples = []

    for idx, row in df.iterrows():
        # Only create examples where both descriptions exist
        if pd.notna(row["Description_x"]) and pd.notna(row["Description_y"]):
            example = dspy.Example(
                original_description=str(row["Description_x"]),
                corrected_description=str(row["Description_y"]),
            ).with_inputs("original_description")
            examples.append(example)

    return examples


# Define metrics for comparing descriptions
def semantic_similarity_metric(example, pred, trace=None):
    """Simple semantic similarity metric using word overlap."""
    expected = str(example.corrected_description).lower().split()
    predicted = str(pred.corrected_description).lower().split()

    if not expected or not predicted:
        return 0.0

    # Calculate word overlap
    expected_set = set(expected)
    predicted_set = set(predicted)

    intersection = len(expected_set.intersection(predicted_set))
    union = len(expected_set.union(predicted_set))

    if union == 0:
        return 0.0

    # Jaccard similarity
    jaccard = intersection / union

    # Also consider word order similarity (simplified)
    common_words = expected_set.intersection(predicted_set)
    if len(common_words) == 0:
        return 0.0

    # Calculate position similarity for common words
    position_scores = []
    for word in common_words:
        try:
            expected_pos = expected.index(word)
            predicted_pos = predicted.index(word)
            # Normalize position difference
            max_pos = max(len(expected), len(predicted))
            if max_pos > 0:
                pos_diff = abs(expected_pos - predicted_pos) / max_pos
                position_scores.append(1.0 - pos_diff)
            else:
                position_scores.append(1.0)
        except ValueError:
            position_scores.append(0.0)

    avg_position_score = (
        sum(position_scores) / len(position_scores) if position_scores else 0.0
    )

    # Combine Jaccard and position similarity
    final_score = (jaccard + avg_position_score) / 2.0
    return final_score


if __name__ == "__main__":
    load_dotenv()
    LM_CONFIG = {
        "model": os.getenv("DSPY_MODEL"),
        "api_key": os.getenv("DSPY_API_KEY"),
        "api_base": os.getenv("DSPY_ENDPOINT"),
        "api_version": os.getenv("DSPY_API_VERSION"),
        "max_tokens": int(os.getenv("DSPY_MAX_TOKENS", 50_000)),
        "temperature": float(os.getenv("DSPY_TEMPERATURE", 1.0)),
        "cache": os.getenv("DSPY_CACHE", "True").lower() == "true",
    }
    LM_CONFIG_FAST = {
        "model": os.getenv("FAST_DSPY_MODEL"),
        "api_key": os.getenv("FAST_DSPY_API_KEY"),
        "api_base": os.getenv("FAST_DSPY_ENDPOINT"),
        "api_version": os.getenv("FAST_DSPY_API_VERSION"),
        "max_tokens": int(os.getenv("FAST_DSPY_MAX_TOKENS", 50_000)),
    }

    # Configure with fast model for optimization
    lm_fast = dspy.LM(**LM_CONFIG_FAST)
    dspy.settings.configure(lm=lm_fast, track_usage=True, cache=False)

    df_original = pd.read_excel("time_entries.xlsx", sheet_name="Time from SAP")
    df_corrected = pd.read_excel("time_entries.xlsx", sheet_name="May Time - Corrected")

    df_joined = pd.merge(
        df_original,
        df_corrected,
        on=[
            "Reference Document",
            "Time Package",
            "Date",
            "Name",
        ],
        how="left",
    )

    # Create examples from the dataframe
    examples = create_examples_from_dataframe(df_joined)
    print(f"Created {len(examples)} examples for evaluation")

    # Display first few examples
    print("\nFirst 3 examples:")
    for i, example in enumerate(examples[:3]):
        print(f"\nExample {i + 1}:")
        print(f"Original: {example.original_description}")
        print(f"Corrected: {example.corrected_description}")

    # ===== EVALUATION SECTION =====
    print("\n" + "=" * 50)
    print("RUNNING EVALUATION")
    print("=" * 50)

    # Create our prediction module using the ProcessEntryModule (with fast model)
    time_entry_corrector = ProcessEntryModule()

    # Split examples into train/dev sets (80/20 split)
    train_size = int(0.8 * len(examples))
    train_examples = examples[:train_size]
    dev_examples = examples[train_size:]

    print(f"Training set size: {len(train_examples)}")
    print(f"Development set size: {len(dev_examples)}")

    # Define evaluation metrics
    def length_ratio_metric(example, pred, trace=None):
        """Check if the length ratio is reasonable (not too short/long)."""
        expected_len = len(str(example.corrected_description))
        predicted_len = len(str(pred.corrected_description))

        if expected_len == 0:
            return 0.0

        ratio = predicted_len / expected_len
        # Penalize if ratio is too far from 1.0
        if ratio < 0.5 or ratio > 2.0:
            return 0.0
        return 1.0

    # Run evaluation using DSPy's Evaluate
    from dspy.evaluate import Evaluate

    print("\nRunning evaluation on development set...")
    evaluator = Evaluate(
        devset=dev_examples, num_threads=2, display_progress=True, display_table=5
    )

    # Evaluate with length ratio metric (no LM needed)
    length_ratio_score = evaluator(time_entry_corrector, metric=length_ratio_metric)
    print(f"Length Ratio Score: {length_ratio_score:.3f}")

    # Evaluate with semantic similarity metric (no LM needed)
    semantic_score = evaluator(time_entry_corrector, metric=semantic_similarity_metric)
    print(f"Semantic Similarity Score: {semantic_score:.3f}")

    # Evaluate with individual quality metrics using the good model
    print("\n" + "-" * 30)
    print("INDIVIDUAL QUALITY METRICS (using good model)")
    print("-" * 30)

    # Create good model for evaluation
    lm_good = dspy.LM(**LM_CONFIG)

    # Wrap evaluation metrics with the good model
    with dspy.context(lm=lm_good):
        spelling_score = evaluator(time_entry_corrector, metric=spelling_grammar_metric)
        print(f"Spelling & Grammar: {spelling_score:.3f}")

        tense_score = evaluator(time_entry_corrector, metric=present_tense_metric)
        print(f"Present Tense Usage: {tense_score:.3f}")

        phrases_score = evaluator(
            time_entry_corrector, metric=prohibited_phrases_metric
        )
        print(f"Prohibited Phrases Removed: {phrases_score:.3f}")

        caps_score = evaluator(time_entry_corrector, metric=capitalization_metric)
        print(f"General Capitalization: {caps_score:.3f}")

        abbrev_score = evaluator(time_entry_corrector, metric=abbreviated_names_metric)
        print(f"Abbreviated Names Capitalized: {abbrev_score:.3f}")

        law_firm_score = evaluator(time_entry_corrector, metric=law_firm_names_metric)
        print(f"Law Firm Names Expanded: {law_firm_score:.3f}")

        overall_quality_score = evaluator(
            time_entry_corrector, metric=overall_quality_metric
        )
        print(f"Overall Quality (1-10): {overall_quality_score:.3f}")

    # Calculate average of all quality metrics
    quality_metrics = [
        spelling_score,
        tense_score,
        phrases_score,
        caps_score,
        abbrev_score,
        law_firm_score,
        overall_quality_score,
    ]
    avg_quality_score = sum(quality_metrics) / len(quality_metrics)
    print(f"Average Quality Score: {avg_quality_score:.3f}")

    # Calculate overall score (average of all metrics)
    overall_score = (length_ratio_score + semantic_score + avg_quality_score) / 3.0
    print(f"\nOverall Evaluation Score: {overall_score:.3f}")

    # Show some example predictions
    print("\n" + "=" * 50)
    print("SAMPLE PREDICTIONS")
    print("=" * 50)

    for i, example in enumerate(dev_examples[:5]):
        print(f"\nExample {i + 1}:")
        print(f"Original: {example.original_description}")
        print(f"Expected: {example.corrected_description}")

        try:
            # Use fast model for prediction
            prediction = time_entry_corrector(
                original_description=example.original_description
            )
            print(f"Predicted: {prediction.corrected_description}")

            # Check individual metrics for this example using good model
            with dspy.context(lm=lm_good):
                print("Individual scores:")
                print(f"  Spelling: {spelling_grammar_metric(example, prediction):.1f}")
                print(f"  Tense: {present_tense_metric(example, prediction):.1f}")
                print(
                    f"  Phrases: {prohibited_phrases_metric(example, prediction):.1f}"
                )
                print(
                    f"  Capitalization: {capitalization_metric(example, prediction):.1f}"
                )
                print(
                    f"  Abbreviated Names: {abbreviated_names_metric(example, prediction):.1f}"
                )
                print(
                    f"  Law Firm Names: {law_firm_names_metric(example, prediction):.1f}"
                )

        except Exception as e:
            print(f"Error in prediction: {e}")

    # Optional: Save evaluation results
    evaluation_results = {
        "length_ratio_score": length_ratio_score,
        "semantic_similarity_score": semantic_score,
        "spelling_grammar_score": spelling_score,
        "present_tense_score": tense_score,
        "prohibited_phrases_score": phrases_score,
        "capitalization_score": caps_score,
        "abbreviated_names_score": abbrev_score,
        "law_firm_names_score": law_firm_score,
        "overall_quality_score": overall_quality_score,
        "average_quality_score": avg_quality_score,
        "overall_score": overall_score,
        "num_examples": len(dev_examples),
    }

    print(f"\nEvaluation Results Summary:")
    for metric, score in evaluation_results.items():
        if metric != "num_examples":
            print(f"  {metric}: {score:.3f}")
    print(f"  num_examples: {evaluation_results['num_examples']}")

    # Print usage statistics for both models
    print(f"\n" + "=" * 50)
    print("USAGE STATISTICS")
    print("=" * 50)
