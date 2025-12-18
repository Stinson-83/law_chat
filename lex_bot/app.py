import os
import sys
import logging
from typing import Optional, List
from dotenv import load_dotenv

# --- 1. Configuration & Logging Setup ---

# Load env relative to this file
current_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(current_dir, ".env"))

# Configure Console Logging
# This sets up a logger that prints info/errors to your terminal with timestamps
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(name)s - %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("LexBot")

# --- 2. Path Setup ---

# Add parent directory to path to allow 'lex_bot.graph' imports
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
    logger.info(f"Added {parent_dir} to sys.path for module resolution.")

# --- 3. Internal Imports ---

try:
    from lex_bot.graph import LegalWorkflow
    from lex_bot.state import AgentState
    logger.info("Successfully imported LegalWorkflow and AgentState.")
except ImportError as e:
    logger.critical(f"Failed to import internal modules: {e}")
    sys.exit(1)

# --- 4. Main Application Logic ---

def process_query(query: str):
    """
    Handles the workflow execution for a single query.
    """
    logger.info(f"📨 Processing new query: '{query}'")

    # Initialize state
    initial_state = AgentState(
        messages=[],
        original_query=query,
        law_context=[],
        case_context=[],
        errors=[]
    )

    try:
        # Initialize Workflow
        logger.info("⚙️  Initializing Legal Workflow...")
        workflow = LegalWorkflow()
        
        # Invoke the graph
        logger.info("🏃 Running workflow graph...")
        result = workflow.run(initial_state)
        
        # Extract Results
        answer = result.get("final_answer", "No answer generated.")
        law_q = result.get("law_query")
        case_q = result.get("case_query")

        # Log specific details about the execution
        if law_q:
            logger.info(f"⚖️  Generated Law Query: {law_q}")
        if case_q:
            logger.info(f"🏛️  Generated Case Query: {case_q}")

        return answer

    except Exception as e:
        logger.error(f"❌ Workflow Execution Error: {e}", exc_info=True)
        return f"An error occurred while processing your request: {str(e)}"

def run_terminal_mode():
    """
    Main interactive loop for the terminal.
    """
    print("\n" + "="*60)
    print("🤖 Lex Bot - Advanced Agentic Indian Law Research (Terminal Mode)")
    print("   Type 'exit', 'quit', or 'q' to stop.")
    print("="*60 + "\n")

    while True:
        try:
            user_input = input("\n👉 Enter your legal query: ").strip()

            # Exit conditions
            if user_input.lower() in ["exit", "quit", "q"]:
                logger.info("Shutting down Lex Bot. Goodbye!")
                print("\n👋 Exiting...")
                break

            if not user_input:
                logger.warning("Empty input received. Please type a query.")
                continue

            # Process
            response = process_query(user_input)

            # Display Output cleanly
            print("\n" + "-"*30 + " RESPONSE " + "-"*30)
            print(response)
            print("-"*70)

        except KeyboardInterrupt:
            # Handle Ctrl+C gracefully
            print("\n\n👋 Forced exit detected.")
            break
        except Exception as e:
            logger.critical(f"Unexpected system error: {e}")

if __name__ == "__main__":
    run_terminal_mode()