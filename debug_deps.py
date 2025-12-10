try:
    print("Importing json...")
    import json
    print("Importing sentence_transformers...")
    from sentence_transformers import CrossEncoder, SentenceTransformer
    print("Success sentence_transformers")
    print("Importing langchain...")
    import langchain
    print("Importing langgraph...")
    import langgraph
    print("ALL IMPORTS SUCCESS")
except Exception as e:
    print(f"FAIL: {e}")
