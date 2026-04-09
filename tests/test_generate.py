from annotation_pipeline.generate import generate_probes


def test_financial_advice_policy():
    result = generate_probes("Do not generate financial advice", num_generations=5)
    print(result)


if __name__ == "__main__":
    test_financial_advice_policy()
