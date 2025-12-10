Write-Host "Installing dependencies if needed..."
pip install -r requirements.txt

Write-Host "Running Agent Verification..."
python test_agents.py

Write-Host "Done."
